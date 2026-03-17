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

    // Pre-loaded weight buffers — eliminates async actor hops during forward pass
    private let preloadedWeights: PreloadedWeightsStore

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
        self.preloadedWeights = PreloadedWeightsStore()
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
        let floatStride = MemoryLayout<Float>.stride

        // 1. Fast embedding lookup — read directly from pre-dequantized MTLBuffer
        //    (the embedding weight is the same buffer as lmHead for tied embeddings,
        //    already dequantized and cached as Float32 MTLBuffer)
        let hiddenBuf: MTLBuffer

        if preloadedWeights.isLoaded {
            // FAST PATH: direct memcpy from cached embedding MTLBuffer (no Q8_0 decode)
            let embBuf = preloadedWeights.lmHead! // tied embeddings = lmHead
            let embPtr = embBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize * dim)
            hiddenBuf = device.makeBuffer(length: seqLen * dim * floatStride, options: .storageModeShared)!
            let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: seqLen * dim)
            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                memcpy(dstPtr + i * dim, embPtr + clampedID * dim, dim * floatStride)
            }
        } else {
            // COLD PATH: first call, weights not yet loaded — use CPU Q8_0 decode
            let embeddingFloats = try await embeddingLookup(tokenIDs: tokenIDs)
            hiddenBuf = device.makeBuffer(
                bytes: embeddingFloats,
                length: embeddingFloats.count * floatStride,
                options: .storageModeShared
            )!
        }

        // 2-4. Fully fused: ALL layers + final norm + LM head in ONE command buffer
        let logitsBuf = try await fusedForwardPass(
            hiddenBuf: hiddenBuf,
            seqLen: seqLen
        )

        // Read back logits
        let ptr = logitsBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize)
        return Array(UnsafeBufferPointer(start: ptr, count: config.vocabSize))
    }

    // MARK: - Fully Fused Forward Pass

    /// Encode ALL 28 transformer layers + final norm + LM head into a SINGLE command buffer.
    /// Returns MTLBuffer containing vocab-sized logits. ONE GPU sync for the entire forward pass.
    private func fusedForwardPass(
        hiddenBuf: MTLBuffer,
        seqLen: Int
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let floatStride = MemoryLayout<Float>.stride
        let totalHiddenBytes = seqLen * dim * floatStride

        var currentHidden = hiddenBuf

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }

        // Pre-allocate scratch buffers ONCE — reused across all 28 layers
        let normedBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        let afterAttnBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        let ffnNormedBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        // Ping-pong output buffers for alternating layers
        let outputBufA = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        let outputBufB = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!

        let allQBuf = device.makeBuffer(length: seqLen * qDim * floatStride, options: .storageModeShared)!
        let allKBuf = device.makeBuffer(length: seqLen * kvDim * floatStride, options: .storageModeShared)!
        let allVBuf = device.makeBuffer(length: seqLen * kvDim * floatStride, options: .storageModeShared)!
        let ropeQBuf = device.makeBuffer(length: seqLen * qDim * floatStride, options: .storageModeShared)!
        let ropeKBuf = device.makeBuffer(length: seqLen * kvDim * floatStride, options: .storageModeShared)!
        let attnOutBuf = device.makeBuffer(length: seqLen * qDim * floatStride, options: .storageModeShared)!
        let projBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
        let gateOutBuf = device.makeBuffer(length: seqLen * interDim * floatStride, options: .storageModeShared)!
        let upOutBuf = device.makeBuffer(length: seqLen * interDim * floatStride, options: .storageModeShared)!
        let activBuf = device.makeBuffer(length: seqLen * interDim * floatStride, options: .storageModeShared)!
        let downOutBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!

        // Pre-load ALL weight buffers on first call — eliminates 254 actor hops per subsequent call
        if !preloadedWeights.isLoaded {
            var layers = [LayerWeightBuffers]()
            layers.reserveCapacity(config.layerCount)
            for i in 0..<config.layerCount {
                let p = "layers.\(i)"
                layers.append(LayerWeightBuffers(
                    attnNorm: try await readWeightBuffer("\(p).attentionNorm.weight"),
                    wq: try await readWeightBuffer("\(p).attention.wq.weight"),
                    wk: try await readWeightBuffer("\(p).attention.wk.weight"),
                    wv: try await readWeightBuffer("\(p).attention.wv.weight"),
                    wo: try await readWeightBuffer("\(p).attention.wo.weight"),
                    ffnNorm: try await readWeightBuffer("\(p).ffnNorm.weight"),
                    gate: try await readWeightBuffer("\(p).feedForward.gate.weight"),
                    up: try await readWeightBuffer("\(p).feedForward.up.weight"),
                    down: try await readWeightBuffer("\(p).feedForward.down.weight")
                ))
            }
            let lmHeadName = weights["lmHead.weight"] != nil ? "lmHead.weight" : "embedding.weight"
            preloadedWeights.load(
                layers: layers,
                finalNorm: try await readWeightBuffer("finalNorm.weight"),
                lmHead: try await readWeightBuffer(lmHeadName)
            )
        }

        // === SINGLE ENCODER for the ENTIRE forward pass ===
        // Metal guarantees sequential execution + implicit barriers between dispatches
        // within the same encoder. Using 1 encoder instead of 422 eliminates encoder
        // creation overhead (~10μs × 422 = 4.2ms per forward pass).
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }

        let rmsNormPSO = rmsNormKernel.pipeline
        let gemvPSO = gemvKernel.f32Pipeline
        let ropePSO = ropeKernel.pipelineF32
        let gqaPSO = gqaKernel.pipelineF32
        let swigluPSO = activationKernels.swigluPipeline
        let addPSO = addPipeline
        let rmsEps = Float(config.rmsNormEpsilon)
        let ropeTheta = Float(config.ropeFreqBase)
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let gqaBlockSize = GQAKernel.blockSize

        for layerIndex in 0..<config.layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB

            // 1. RMSNorm (attention)
            do {
                var p = ERRMSNormParams(rows: UInt32(seqLen), cols: UInt32(dim), eps: rmsEps)
                enc.setComputePipelineState(rmsNormPSO)
                enc.setBuffer(currentHidden, offset: 0, index: 0)
                enc.setBuffer(lw.attnNorm, offset: 0, index: 1)
                enc.setBuffer(normedBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(seqLen, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 2. Q/K/V projections
            enc.setComputePipelineState(gemvPSO)
            for t in 0..<seqLen {
                let tokOff = t * dim * floatStride
                let qOff = t * qDim * floatStride
                let kvOff = t * kvDim * floatStride
                // Q
                var qP = ERGEMVParams(M: UInt32(qDim), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lw.wq, offset: 0, index: 0)
                enc.setBuffer(normedBuf, offset: tokOff, index: 1)
                enc.setBuffer(allQBuf, offset: qOff, index: 2)
                enc.setBytes(&qP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: qDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // K
                var kvP = ERGEMVParams(M: UInt32(kvDim), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lw.wk, offset: 0, index: 0)
                enc.setBuffer(allKBuf, offset: kvOff, index: 2)
                enc.setBytes(&kvP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: kvDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // V
                enc.setBuffer(lw.wv, offset: 0, index: 0)
                enc.setBuffer(allVBuf, offset: kvOff, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: kvDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // 3. RoPE Q
            do {
                var p = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numHeads),
                    headDim: UInt32(headDim), startPos: 0, theta: ropeTheta, scalingFactor: 1)
                enc.setComputePipelineState(ropePSO)
                enc.setBuffer(allQBuf, offset: 0, index: 0)
                enc.setBuffer(ropeQBuf, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                let halfDim = headDim / 2
                enc.dispatchThreads(MTLSize(width: halfDim, height: numHeads, depth: seqLen),
                    threadsPerThreadgroup: MTLSize(width: min(halfDim, ropePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }
            // RoPE K
            do {
                var p = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numKVHeads),
                    headDim: UInt32(headDim), startPos: 0, theta: ropeTheta, scalingFactor: 1)
                enc.setBuffer(allKBuf, offset: 0, index: 0)
                enc.setBuffer(ropeKBuf, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                let halfDim = headDim / 2
                enc.dispatchThreads(MTLSize(width: halfDim, height: numKVHeads, depth: seqLen),
                    threadsPerThreadgroup: MTLSize(width: min(halfDim, ropePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 4. GQA
            do {
                let groupSize = numHeads / numKVHeads
                var p = ERGQAParams(seqLen: UInt32(seqLen), headDim: UInt32(headDim),
                    numHeads: UInt32(numHeads), numKVHeads: UInt32(numKVHeads),
                    groupSize: UInt32(groupSize), scale: 1.0 / sqrt(Float(headDim)),
                    causal: 1, kvBlockSize: UInt32(gqaBlockSize), qBlockSize: UInt32(gqaBlockSize))
                enc.setComputePipelineState(gqaPSO)
                enc.setBuffer(ropeQBuf, offset: 0, index: 0)
                enc.setBuffer(ropeKBuf, offset: 0, index: 1)
                enc.setBuffer(allVBuf, offset: 0, index: 2)
                enc.setBuffer(attnOutBuf, offset: 0, index: 3)
                enc.setBytes(&p, length: MemoryLayout<ERGQAParams>.stride, index: 4)
                let qBlockCount = (seqLen + gqaBlockSize - 1) / gqaBlockSize
                enc.dispatchThreadgroups(MTLSize(width: qBlockCount, height: numHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: gqaBlockSize, height: 1, depth: 1))
            }

            // 5. Output projection
            enc.setComputePipelineState(gemvPSO)
            for t in 0..<seqLen {
                var p = ERGEMVParams(M: UInt32(dim), K: UInt32(qDim), lda: UInt32(qDim))
                enc.setBuffer(lw.wo, offset: 0, index: 0)
                enc.setBuffer(attnOutBuf, offset: t * qDim * floatStride, index: 1)
                enc.setBuffer(projBuf, offset: t * dim * floatStride, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // 6. Residual add (attention)
            do {
                var p = ERElementwiseParams(elementCount: UInt32(seqLen * dim))
                enc.setComputePipelineState(addPSO)
                enc.setBuffer(currentHidden, offset: 0, index: 0)
                enc.setBuffer(projBuf, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERElementwiseParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: seqLen * dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(seqLen * dim, addPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 7. RMSNorm (FFN)
            do {
                var p = ERRMSNormParams(rows: UInt32(seqLen), cols: UInt32(dim), eps: rmsEps)
                enc.setComputePipelineState(rmsNormPSO)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 0)
                enc.setBuffer(lw.ffnNorm, offset: 0, index: 1)
                enc.setBuffer(ffnNormedBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(seqLen, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 8. Gate + Up projections
            enc.setComputePipelineState(gemvPSO)
            for t in 0..<seqLen {
                let tokOff = t * dim * floatStride
                let intOff = t * interDim * floatStride
                var p = ERGEMVParams(M: UInt32(interDim), K: UInt32(dim), lda: UInt32(dim))
                // gate
                enc.setBuffer(lw.gate, offset: 0, index: 0)
                enc.setBuffer(ffnNormedBuf, offset: tokOff, index: 1)
                enc.setBuffer(gateOutBuf, offset: intOff, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: interDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // up
                enc.setBuffer(lw.up, offset: 0, index: 0)
                enc.setBuffer(upOutBuf, offset: intOff, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: interDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // 9. SwiGLU
            do {
                var p = ERActivationParams(count: UInt32(seqLen * interDim))
                enc.setComputePipelineState(swigluPSO)
                enc.setBuffer(gateOutBuf, offset: 0, index: 0)
                enc.setBuffer(upOutBuf, offset: 0, index: 1)
                enc.setBuffer(activBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERActivationParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: seqLen * interDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(seqLen * interDim, swigluPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 10. Down projection
            enc.setComputePipelineState(gemvPSO)
            for t in 0..<seqLen {
                var p = ERGEMVParams(M: UInt32(dim), K: UInt32(interDim), lda: UInt32(interDim))
                enc.setBuffer(lw.down, offset: 0, index: 0)
                enc.setBuffer(activBuf, offset: t * interDim * floatStride, index: 1)
                enc.setBuffer(downOutBuf, offset: t * dim * floatStride, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // 11. Residual add (FFN)
            do {
                var p = ERElementwiseParams(elementCount: UInt32(seqLen * dim))
                enc.setComputePipelineState(addPSO)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 0)
                enc.setBuffer(downOutBuf, offset: 0, index: 1)
                enc.setBuffer(layerOutputBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERElementwiseParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: seqLen * dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(seqLen * dim, addPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            currentHidden = layerOutputBuf
        }

        // === FINAL NORM ===
        do {
            let finalOutputBuf = device.makeBuffer(length: totalHiddenBytes, options: .storageModeShared)!
            var p = ERRMSNormParams(rows: UInt32(seqLen), cols: UInt32(dim), eps: rmsEps)
            enc.setComputePipelineState(rmsNormPSO)
            enc.setBuffer(currentHidden, offset: 0, index: 0)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 1)
            enc.setBuffer(finalOutputBuf, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
            enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(seqLen, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            currentHidden = finalOutputBuf
        }

        // === LM HEAD ===
        let lmHeadBuf = preloadedWeights.lmHead!
        let logitsBuf = device.makeBuffer(length: config.vocabSize * floatStride, options: .storageModeShared)!
        do {
            var p = ERGEMVParams(M: UInt32(config.vocabSize), K: UInt32(dim), lda: UInt32(dim))
            enc.setComputePipelineState(gemvPSO)
            enc.setBuffer(lmHeadBuf, offset: 0, index: 0)
            enc.setBuffer(currentHidden, offset: (seqLen - 1) * dim * floatStride, index: 1)
            enc.setBuffer(logitsBuf, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: config.vocabSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        enc.endEncoding()

        // ONE sync point for the ENTIRE forward pass (layers + norm + LM head)!
        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        return logitsBuf
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

// MARK: - Pre-loaded Weight Buffers

/// Holds all layer weight MTLBuffers for zero-async access during forward pass.
private struct LayerWeightBuffers {
    let attnNorm: MTLBuffer
    let wq: MTLBuffer
    let wk: MTLBuffer
    let wv: MTLBuffer
    let wo: MTLBuffer
    let ffnNorm: MTLBuffer
    let gate: MTLBuffer
    let up: MTLBuffer
    let down: MTLBuffer
}

/// Thread-safe store for pre-loaded weights. Loaded once on first forward pass,
/// then accessed directly (zero actor hops) on all subsequent passes.
private final class PreloadedWeightsStore: @unchecked Sendable {
    var layers: [LayerWeightBuffers] = []
    var finalNorm: MTLBuffer?
    var lmHead: MTLBuffer?
    var isLoaded = false

    func load(layers: [LayerWeightBuffers], finalNorm: MTLBuffer, lmHead: MTLBuffer) {
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.isLoaded = true
    }
}

// MARK: - Metal Buffer Cache Actor

private actor MetalBufferCacheActor {
    private var cache: [String: MetalBufferHandle] = [:]
    func get(_ name: String) -> MetalBufferHandle? { cache[name] }
    func set(_ name: String, handle: MetalBufferHandle) { cache[name] = handle }
}
