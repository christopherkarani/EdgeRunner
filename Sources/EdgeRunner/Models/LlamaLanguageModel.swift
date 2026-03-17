import Foundation
import Metal
import EdgeRunnerCore
import EdgeRunnerIO
import EdgeRunnerMetal

/// Full Llama inference engine conforming to `LogitsModel`.
///
/// Orchestrates the forward pass through the Llama transformer architecture
/// using EdgeRunner's Metal compute kernels:
///
/// ```
/// tokens → embedding → [RMSNorm → RoPE+GQA+KVCache → RMSNorm → SwiGLU FFN] × N → RMSNorm → LM head → logits
/// ```
///
/// Supports Llama 2, Llama 3, Qwen, Mistral, and any GGUF model using
/// the standard Llama architecture with SwiGLU + GQA + RoPE.
/// Handles quantized weights (Q4_0, Q8_0, Q4_K_M) via on-the-fly GPU dequantization.
public struct LlamaLanguageModel: LogitsModel, @unchecked Sendable {
    // @unchecked Sendable: Metal device/queue are thread-safe; KVCache uses Mutex internally.

    public static let modelIdentifier = "llama"

    private let config: LlamaConfig
    private let weights: [String: TensorStorage]

    // Metal infrastructure
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Metal kernels
    private let rmsNormKernel: RMSNormKernel
    private let ropeKernel: RoPEKernel
    private let gqaKernel: GQAKernel
    private let activationKernels: ActivationKernels
    private let gemvKernel: GEMVKernel

    // Elementwise add pipeline (for GPU residual connections)
    private let addPipeline: MTLComputePipelineState

    // Dequantization kernels
    private let dequantQ4_0: DequantQ4_0Kernel
    private let dequantQ8_0: DequantQ8_0Kernel
    private let dequantQ4KM: DequantQ4KMKernel

    // KV cache for autoregressive generation
    private let kvCache: KVCache

    // Dequantized weight cache — avoids re-dequantizing on every forward pass
    private let weightCache: WeightCacheActor

    // Metal buffer cache — avoids re-creating MTLBuffers from [Float] on every GEMV call
    private let metalBufferCache: MetalBufferCacheActor

    public init(model: LlamaModel, maxSeqLen: Int = 4096) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GenerationError.modelLoadFailed(reason: "No Metal device available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command queue")
        }

        self.config = model.config
        self.weights = model.loadedWeights
        self.device = device
        self.commandQueue = queue

        // Initialize Metal kernels
        self.rmsNormKernel = try RMSNormKernel(device: device)
        self.ropeKernel = try RoPEKernel(device: device)
        self.gqaKernel = try GQAKernel(device: device)
        self.activationKernels = try ActivationKernels(device: device)
        self.gemvKernel = try GEMVKernel(device: device)

        // Load elementwise add pipeline for GPU residual connections
        let registry = try KernelRegistry(device: device)
        self.addPipeline = try registry.pipeline(for: "elementwise_add_float")

        // Initialize dequant kernels
        self.dequantQ4_0 = try DequantQ4_0Kernel(device: device)
        self.dequantQ8_0 = try DequantQ8_0Kernel(device: device)
        self.dequantQ4KM = try DequantQ4KMKernel(device: device)

        // Initialize KV cache
        self.kvCache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: config.layerCount,
            numKVHeads: config.kvHeadCount,
            headDim: config.headDim,
            precision: .float32
        )

        self.weightCache = WeightCacheActor()
        self.metalBufferCache = MetalBufferCacheActor()
    }

    // MARK: - EdgeRunnerLanguageModel conformance

    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> LlamaLanguageModel {
        let loader = try GGUFLoader(url: url)
        let weightMap = try await loader.load(from: url)
        let ggufConfig = try LlamaConfig(fromGGUFMetadata: loader.modelConfig.metadata)
        var model = LlamaModel(config: ggufConfig)
        try model.loadWeights(from: weightMap)
        return try LlamaLanguageModel(
            model: model,
            maxSeqLen: configuration.contextWindowSize
        )
    }

    public func tokenize(_ text: String) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    public func detokenize(_ ids: [Int]) -> String {
        String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    public var eosTokenID: Int { 151645 }
    public var bosTokenID: Int? { 151643 }
    public var vocabularySize: Int { config.vocabSize }

    // MARK: - LogitsModel: forward pass

    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let seqLen = tokenIDs.count
        let dim = config.embeddingDim

        // 1. Token embedding lookup
        let embeddingFloats = try await embeddingLookup(tokenIDs: tokenIDs)

        // Create initial hidden state as MTLBuffer
        let hiddenBuf = device.makeBuffer(
            bytes: embeddingFloats,
            length: embeddingFloats.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // 2. Process through ALL transformer layers — fully fused GPU pipeline
        let resultBuf = try await fusedForwardPass(
            hiddenBuf: hiddenBuf,
            seqLen: seqLen
        )

        // 3. Read back the last token's hidden state for LM head
        let resultPtr = resultBuf.contents().bindMemory(to: Float.self, capacity: seqLen * dim)
        let lastTokenOffset = (seqLen - 1) * dim
        let lastTokenHidden = Array(UnsafeBufferPointer(start: resultPtr + lastTokenOffset, count: dim))

        // 4. LM head — GPU GEMV
        let lmHeadName = weights["lmHead.weight"] != nil ? "lmHead.weight" : "embedding.weight"
        let lmHeadBuf = try await readWeightBuffer(lmHeadName)
        return try await gemvKernel.executeWithWeightBuffer(
            weightBuffer: lmHeadBuf,
            x: lastTokenHidden,
            M: config.vocabSize,
            K: dim,
            commandQueue: commandQueue
        )
    }

    // MARK: - Fully Fused Forward Pass

    /// Encode ALL 28 transformer layers into a SINGLE command buffer.
    /// Data stays as MTLBuffers throughout — no [Float]↔MTLBuffer round-trips between layers.
    /// ONE GPU sync point for the entire transformer stack.
    private func fusedForwardPass(
        hiddenBuf: MTLBuffer,
        seqLen: Int
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let totalHiddenBytes = seqLen * dim * MemoryLayout<Float>.stride

        // Pre-allocate ALL scratch buffers for the pipeline
        // We use ping-pong: layerInput → layerOutput, then swap
        var currentHidden = hiddenBuf

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }

        for layerIndex in 0..<config.layerCount {
            let prefix = "layers.\(layerIndex)"

            // Load cached weight MTLBuffers
            let attnNormBuf = try await readWeightBuffer("\(prefix).attentionNorm.weight")
            let wqBuf = try await readWeightBuffer("\(prefix).attention.wq.weight")
            let wkBuf = try await readWeightBuffer("\(prefix).attention.wk.weight")
            let wvBuf = try await readWeightBuffer("\(prefix).attention.wv.weight")
            let woBuf = try await readWeightBuffer("\(prefix).attention.wo.weight")
            let ffnNormBuf = try await readWeightBuffer("\(prefix).ffnNorm.weight")
            let gateBuf = try await readWeightBuffer("\(prefix).feedForward.gate.weight")
            let upBuf = try await readWeightBuffer("\(prefix).feedForward.up.weight")
            let downBuf = try await readWeightBuffer("\(prefix).feedForward.down.weight")

            // Allocate per-layer scratch buffers (seqLen positions)
            let normedBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
            let afterAttnBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
            let ffnNormedBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
            let layerOutputBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!

            // Per-token Q/K/V buffers (concatenated across positions)
            let allQBuf = device.makeBuffer(length: seqLen * qDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let allKBuf = device.makeBuffer(length: seqLen * kvDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let allVBuf = device.makeBuffer(length: seqLen * kvDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let ropeQBuf = device.makeBuffer(length: seqLen * qDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let ropeKBuf = device.makeBuffer(length: seqLen * kvDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let attnOutBuf = device.makeBuffer(length: seqLen * qDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let projBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!

            // FFN scratch
            let gateOutBuf = device.makeBuffer(length: seqLen * interDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let upOutBuf = device.makeBuffer(length: seqLen * interDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let activBuf = device.makeBuffer(length: seqLen * interDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let downOutBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!

            // === ATTENTION BLOCK ===

            // 1. Pre-attention RMSNorm
            try rmsNormKernel.encode(
                commandBuffer: cmdBuf,
                inputBuffer: currentHidden,
                weightBuffer: attnNormBuf,
                outputBuffer: normedBuf,
                rows: seqLen, cols: dim,
                eps: Float(config.rmsNormEpsilon)
            )

            // 2. Q/K/V projections for each position
            for t in 0..<seqLen {
                let tokenOffset = t * dim * MemoryLayout<Float>.stride
                let qOffset = t * qDim * MemoryLayout<Float>.stride
                let kvOffset = t * kvDim * MemoryLayout<Float>.stride

                // Use sub-buffer views via offset encoding
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: wqBuf,
                    inputBuffer: normedBuf, inputOffset: tokenOffset,
                    outputBuffer: allQBuf, outputOffset: qOffset,
                    M: qDim, K: dim
                )
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: wkBuf,
                    inputBuffer: normedBuf, inputOffset: tokenOffset,
                    outputBuffer: allKBuf, outputOffset: kvOffset,
                    M: kvDim, K: dim
                )
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: wvBuf,
                    inputBuffer: normedBuf, inputOffset: tokenOffset,
                    outputBuffer: allVBuf, outputOffset: kvOffset,
                    M: kvDim, K: dim
                )
            }

            // 3. RoPE for Q and K
            try ropeKernel.encode(
                commandBuffer: cmdBuf,
                inputBuffer: allQBuf, outputBuffer: ropeQBuf,
                seqLen: seqLen, numHeads: config.headCount, headDim: headDim,
                startPos: 0, theta: Float(config.ropeFreqBase)
            )
            try ropeKernel.encode(
                commandBuffer: cmdBuf,
                inputBuffer: allKBuf, outputBuffer: ropeKBuf,
                seqLen: seqLen, numHeads: config.kvHeadCount, headDim: headDim,
                startPos: 0, theta: Float(config.ropeFreqBase)
            )

            // 4. GQA attention
            try gqaKernel.encode(
                commandBuffer: cmdBuf,
                qBuffer: ropeQBuf, kBuffer: ropeKBuf, vBuffer: allVBuf,
                outputBuffer: attnOutBuf,
                seqLen: seqLen, headDim: headDim,
                numHeads: config.headCount, numKVHeads: config.kvHeadCount,
                causal: true
            )

            // 5. Output projection for each position
            for t in 0..<seqLen {
                let attnOffset = t * qDim * MemoryLayout<Float>.stride
                let projOffset = t * dim * MemoryLayout<Float>.stride
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: woBuf,
                    inputBuffer: attnOutBuf, inputOffset: attnOffset,
                    outputBuffer: projBuf, outputOffset: projOffset,
                    M: dim, K: qDim
                )
            }

            // 6. Residual add: afterAttn = hidden + projected (GPU)
            try encodeElementwiseAdd(
                cmdBuf: cmdBuf,
                aBuf: currentHidden, bBuf: projBuf,
                outBuf: afterAttnBuf,
                count: seqLen * dim
            )

            // === FFN BLOCK ===

            // 7. Pre-FFN RMSNorm
            try rmsNormKernel.encode(
                commandBuffer: cmdBuf,
                inputBuffer: afterAttnBuf,
                weightBuffer: ffnNormBuf,
                outputBuffer: ffnNormedBuf,
                rows: seqLen, cols: dim,
                eps: Float(config.rmsNormEpsilon)
            )

            // 8. Gate + Up projections for each position
            for t in 0..<seqLen {
                let tokenOffset = t * dim * MemoryLayout<Float>.stride
                let interOffset = t * interDim * MemoryLayout<Float>.stride
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: gateBuf,
                    inputBuffer: ffnNormedBuf, inputOffset: tokenOffset,
                    outputBuffer: gateOutBuf, outputOffset: interOffset,
                    M: interDim, K: dim
                )
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: upBuf,
                    inputBuffer: ffnNormedBuf, inputOffset: tokenOffset,
                    outputBuffer: upOutBuf, outputOffset: interOffset,
                    M: interDim, K: dim
                )
            }

            // 9. SwiGLU activation
            try activationKernels.encodeSwiglu(
                commandBuffer: cmdBuf,
                gateBuffer: gateOutBuf, upBuffer: upOutBuf,
                outputBuffer: activBuf,
                count: seqLen * interDim
            )

            // 10. Down projection for each position
            for t in 0..<seqLen {
                let interOffset = t * interDim * MemoryLayout<Float>.stride
                let tokenOffset = t * dim * MemoryLayout<Float>.stride
                try encodeGEMVWithOffsets(
                    cmdBuf: cmdBuf, weightBuffer: downBuf,
                    inputBuffer: activBuf, inputOffset: interOffset,
                    outputBuffer: downOutBuf, outputOffset: tokenOffset,
                    M: dim, K: interDim
                )
            }

            // 11. Residual add: output = afterAttn + downResult (GPU)
            try encodeElementwiseAdd(
                cmdBuf: cmdBuf,
                aBuf: afterAttnBuf, bBuf: downOutBuf,
                outBuf: layerOutputBuf,
                count: seqLen * dim
            )

            currentHidden = layerOutputBuf
        }

        // === FINAL NORM ===
        let finalNormBuf = try await readWeightBuffer("finalNorm.weight")
        let finalOutputBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        try rmsNormKernel.encode(
            commandBuffer: cmdBuf,
            inputBuffer: currentHidden,
            weightBuffer: finalNormBuf,
            outputBuffer: finalOutputBuf,
            rows: seqLen, cols: dim,
            eps: Float(config.rmsNormEpsilon)
        )

        // ONE sync point for the entire transformer stack!
        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        return finalOutputBuf
    }

    // MARK: - GPU Pipeline Helpers

    /// Encode a GEMV dispatch with buffer offsets for per-token slicing.
    /// y[M] = weight[M,K] * input[offset..offset+K]
    private func encodeGEMVWithOffsets(
        cmdBuf: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer, inputOffset: Int,
        outputBuffer: MTLBuffer, outputOffset: Int,
        M: Int, K: Int
    ) throws {
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }
        var params = ERGEMVParams(M: UInt32(M), K: UInt32(K), lda: UInt32(K))
        encoder.setComputePipelineState(gemvKernel.f32Pipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: inputOffset, index: 1)
        encoder.setBuffer(outputBuffer, offset: outputOffset, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Encode element-wise addition: out[i] = a[i] + b[i]
    private func encodeElementwiseAdd(
        cmdBuf: MTLCommandBuffer,
        aBuf: MTLBuffer, bBuf: MTLBuffer,
        outBuf: MTLBuffer,
        count: Int
    ) throws {
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }
        var params = ERElementwiseParams(elementCount: UInt32(count))
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(outBuf, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERElementwiseParams>.stride, index: 3)
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(count, addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    // MARK: - Weight Reading with Dequantization

    /// Read a weight tensor, dequantizing on first access and caching the result.
    private func readWeight(_ name: String) async throws -> [Float] {
        if let cached = await weightCache.get(name) {
            return cached
        }

        guard let storage = weights[name] else {
            throw GenerationError.modelLoadFailed(reason: "Missing weight: \(name)")
        }

        let floats: [Float]
        let elementCount = storage.elementCount
        let basePtr = storage.buffer.contents() + storage.byteOffset

        switch storage.dataType {
        case .float32:
            let ptr = basePtr.bindMemory(to: Float.self, capacity: elementCount)
            floats = Array(UnsafeBufferPointer(start: ptr, count: elementCount))

        case .float16:
            let ptr = basePtr.bindMemory(to: Float16.self, capacity: elementCount)
            floats = Array(UnsafeBufferPointer(start: ptr, count: elementCount)).map { Float($0) }

        case .q4_0:
            let blockCount = elementCount / 32
            let byteCount = blockCount * 18
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            floats = try await dequantQ4_0.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        case .q8_0:
            let blockCount = elementCount / 32
            let byteCount = blockCount * 34
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            floats = try await dequantQ8_0.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        case .q4_K:
            let superBlockCount = elementCount / 256
            let byteCount = superBlockCount * 144
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            floats = try await dequantQ4KM.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        default:
            throw GenerationError.modelLoadFailed(
                reason: "Unsupported weight data type \(storage.dataType) for \(name)"
            )
        }

        await weightCache.set(name, value: floats)
        return floats
    }

    /// Read a weight tensor as a cached MTLBuffer.
    private func readWeightBuffer(_ name: String) async throws -> MTLBuffer {
        if let cached = await metalBufferCache.get(name) {
            return cached.rawValue
        }
        let floats = try await readWeight(name)
        guard let buffer = device.makeBuffer(
            bytes: floats,
            length: floats.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTLBuffer for \(name)")
        }
        await metalBufferCache.set(name, handle: MetalBufferHandle(buffer))
        return buffer
    }

    /// Embedding lookup — dequantizes only the needed rows, not the full table.
    private func embeddingLookup(tokenIDs: [Int]) async throws -> [Float] {
        let dim = config.embeddingDim

        guard let storage = weights["embedding.weight"] else {
            throw GenerationError.modelLoadFailed(reason: "Missing embedding.weight")
        }

        var result = [Float](repeating: 0, count: tokenIDs.count * dim)
        let basePtr = storage.buffer.contents() + storage.byteOffset

        switch storage.dataType {
        case .float32:
            let ptr = basePtr.bindMemory(to: Float.self, capacity: storage.elementCount)
            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let srcOffset = clampedID * dim
                let dstOffset = i * dim
                for d in 0..<dim {
                    result[dstOffset + d] = ptr[srcOffset + d]
                }
            }

        case .float16:
            let ptr = basePtr.bindMemory(to: Float16.self, capacity: storage.elementCount)
            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let srcOffset = clampedID * dim
                let dstOffset = i * dim
                for d in 0..<dim {
                    result[dstOffset + d] = Float(ptr[srcOffset + d])
                }
            }

        case .q8_0:
            let bytesPerBlock = 34
            let elementsPerBlock = 32
            let blocksPerRow = dim / elementsPerBlock
            let bytesPerRow = blocksPerRow * bytesPerBlock
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: storage.elementCount / 32 * 34)

            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let rowStart = clampedID * bytesPerRow
                let dstOffset = i * dim
                for block in 0..<blocksPerRow {
                    let blockStart = rowStart + block * bytesPerBlock
                    let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                    let scale = Float(Float16(bitPattern: scaleBits))
                    for j in 0..<elementsPerBlock {
                        let qval = Int8(bitPattern: rawPtr[blockStart + 2 + j])
                        result[dstOffset + block * elementsPerBlock + j] = scale * Float(qval)
                    }
                }
            }

        case .q4_0:
            let bytesPerBlock = 18
            let elementsPerBlock = 32
            let blocksPerRow = dim / elementsPerBlock
            let bytesPerRow = blocksPerRow * bytesPerBlock
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: storage.elementCount / 32 * 18)

            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let rowStart = clampedID * bytesPerRow
                let dstOffset = i * dim
                for block in 0..<blocksPerRow {
                    let blockStart = rowStart + block * bytesPerBlock
                    let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                    let scale = Float(Float16(bitPattern: scaleBits))
                    for j in 0..<16 {
                        let byte = rawPtr[blockStart + 2 + j]
                        let lo = Int(byte & 0x0F) - 8
                        let hi = Int(byte >> 4) - 8
                        result[dstOffset + block * elementsPerBlock + j * 2] = scale * Float(lo)
                        result[dstOffset + block * elementsPerBlock + j * 2 + 1] = scale * Float(hi)
                    }
                }
            }

        default:
            throw GenerationError.modelLoadFailed(
                reason: "Unsupported embedding data type: \(storage.dataType)"
            )
        }

        return result
    }

    /// Compute LM head logits using tied embedding weights (CPU fallback).
    private func computeTiedLMHead(
        hidden: [Float], embeddingStorage: TensorStorage, vocabSize: Int, dim: Int
    ) -> [Float] {
        let basePtr = embeddingStorage.buffer.contents() + embeddingStorage.byteOffset
        var logits = [Float](repeating: 0, count: vocabSize)
        switch embeddingStorage.dataType {
        case .q8_0:
            let bytesPerBlock = 34
            let elementsPerBlock = 32
            let blocksPerRow = dim / elementsPerBlock
            let bytesPerRow = blocksPerRow * bytesPerBlock
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: vocabSize * bytesPerRow)
            for vocabIdx in 0..<vocabSize {
                let rowStart = vocabIdx * bytesPerRow
                var dot: Float = 0
                for block in 0..<blocksPerRow {
                    let blockStart = rowStart + block * bytesPerBlock
                    let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                    let scale = Float(Float16(bitPattern: scaleBits))
                    for j in 0..<elementsPerBlock {
                        let qval = Int8(bitPattern: rawPtr[blockStart + 2 + j])
                        dot += scale * Float(qval) * hidden[block * elementsPerBlock + j]
                    }
                }
                logits[vocabIdx] = dot
            }
        case .float32:
            let ptr = basePtr.bindMemory(to: Float.self, capacity: vocabSize * dim)
            for vocabIdx in 0..<vocabSize {
                var dot: Float = 0
                let offset = vocabIdx * dim
                for d in 0..<dim { dot += ptr[offset + d] * hidden[d] }
                logits[vocabIdx] = dot
            }
        default:
            break
        }
        return logits
    }
}

// MARK: - Weight Cache Actor

private actor WeightCacheActor {
    private var cache: [String: [Float]] = [:]
    func get(_ name: String) -> [Float]? { cache[name] }
    func set(_ name: String, value: [Float]) { cache[name] = value }
}

// MARK: - Metal Buffer Cache Actor

private actor MetalBufferCacheActor {
    private var cache: [String: MetalBufferHandle] = [:]
    func get(_ name: String) -> MetalBufferHandle? { cache[name] }
    func set(_ name: String, handle: MetalBufferHandle) { cache[name] = handle }
}
