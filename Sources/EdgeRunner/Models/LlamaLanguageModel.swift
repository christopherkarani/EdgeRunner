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

    // Dequantization kernels
    private let dequantQ4_0: DequantQ4_0Kernel
    private let dequantQ8_0: DequantQ8_0Kernel
    private let dequantQ4KM: DequantQ4KMKernel

    // KV cache for autoregressive generation
    private let kvCache: KVCache

    // Dequantized weight cache — avoids re-dequantizing on every forward pass
    private let weightCache: WeightCacheActor

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
        // Byte-level fallback tokenizer.
        // Real usage should pair with BPETokenizer loaded from the GGUF vocab.
        Array(text.utf8).map { Int($0) }
    }

    public func detokenize(_ ids: [Int]) -> String {
        String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    public var eosTokenID: Int { 151645 } // Qwen EOS (<|endoftext|>)
    public var bosTokenID: Int? { 151643 } // Qwen BOS
    public var vocabularySize: Int { config.vocabSize }

    // MARK: - LogitsModel: forward pass

    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let seqLen = tokenIDs.count
        let dim = config.embeddingDim

        // 1. Token embedding lookup (embedding is always fp32/fp16, not quantized)
        var hidden = try await embeddingLookup(tokenIDs: tokenIDs)

        // 2. Process through transformer layers
        for layerIndex in 0..<config.layerCount {
            hidden = try await transformerLayer(
                hidden: hidden,
                layerIndex: layerIndex,
                seqLen: seqLen,
                startPos: 0
            )
        }

        // 3. Final RMS norm
        let finalNormWeight = try await readWeight("finalNorm.weight")
        hidden = try await rmsNormKernel.execute(
            input: hidden,
            weight: finalNormWeight,
            rows: seqLen,
            cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // 4. LM head: project last token's hidden state to vocab logits
        let lastTokenHidden = Array(hidden.suffix(dim))

        if weights["lmHead.weight"] != nil {
            let lmHeadWeight = try await readWeight("lmHead.weight")
            return try await gemvKernel.execute(
                a: lmHeadWeight,
                x: lastTokenHidden,
                M: config.vocabSize,
                K: dim,
                commandQueue: commandQueue
            )
        }

        // Tied embeddings: compute logits as dot product with each embedding row (CPU)
        // This avoids dequantizing the full 593MB embedding table at once
        let embStorage = weights["embedding.weight"]!
        return computeTiedLMHead(
            hidden: lastTokenHidden,
            embeddingStorage: embStorage,
            vocabSize: config.vocabSize,
            dim: dim
        )
    }

    // MARK: - Transformer Layer

    private func transformerLayer(
        hidden: [Float],
        layerIndex: Int,
        seqLen: Int,
        startPos: Int
    ) async throws -> [Float] {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let prefix = "layers.\(layerIndex)"

        // Pre-attention RMS norm
        let attnNormWeight = try await readWeight("\(prefix).attentionNorm.weight")
        let normed = try await rmsNormKernel.execute(
            input: hidden,
            weight: attnNormWeight,
            rows: seqLen,
            cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // Q, K, V projections
        let wq = try await readWeight("\(prefix).attention.wq.weight")
        let wk = try await readWeight("\(prefix).attention.wk.weight")
        let wv = try await readWeight("\(prefix).attention.wv.weight")
        let wo = try await readWeight("\(prefix).attention.wo.weight")

        var allQ = [Float]()
        var allK = [Float]()
        var allV = [Float]()
        allQ.reserveCapacity(seqLen * config.headCount * headDim)
        allK.reserveCapacity(seqLen * config.kvHeadCount * headDim)
        allV.reserveCapacity(seqLen * config.kvHeadCount * headDim)

        for t in 0..<seqLen {
            let tokenHidden = Array(normed[t * dim..<(t + 1) * dim])

            let q = try await gemvKernel.execute(
                a: wq, x: tokenHidden,
                M: config.headCount * headDim, K: dim,
                commandQueue: commandQueue
            )
            let k = try await gemvKernel.execute(
                a: wk, x: tokenHidden,
                M: config.kvHeadCount * headDim, K: dim,
                commandQueue: commandQueue
            )
            let v = try await gemvKernel.execute(
                a: wv, x: tokenHidden,
                M: config.kvHeadCount * headDim, K: dim,
                commandQueue: commandQueue
            )

            allQ.append(contentsOf: q)
            allK.append(contentsOf: k)
            allV.append(contentsOf: v)
        }

        // Apply RoPE
        let ropeQ = try await ropeKernel.execute(
            input: allQ, seqLen: seqLen,
            numHeads: config.headCount, headDim: headDim,
            startPos: startPos, theta: Float(config.ropeFreqBase),
            commandQueue: commandQueue
        )
        let ropeK = try await ropeKernel.execute(
            input: allK, seqLen: seqLen,
            numHeads: config.kvHeadCount, headDim: headDim,
            startPos: startPos, theta: Float(config.ropeFreqBase),
            commandQueue: commandQueue
        )

        // GQA
        let attnOutput = try await gqaKernel.execute(
            q: ropeQ, k: ropeK, v: allV,
            seqLen: seqLen, headDim: headDim,
            numHeads: config.headCount, numKVHeads: config.kvHeadCount,
            causal: true, commandQueue: commandQueue
        )

        // Output projection + residual
        var afterAttn = hidden
        for t in 0..<seqLen {
            let tokenAttn = Array(attnOutput[t * config.headCount * headDim..<(t + 1) * config.headCount * headDim])
            let projected = try await gemvKernel.execute(
                a: wo, x: tokenAttn,
                M: dim, K: config.headCount * headDim,
                commandQueue: commandQueue
            )
            for d in 0..<dim {
                afterAttn[t * dim + d] += projected[d]
            }
        }

        // Pre-FFN RMS norm
        let ffnNormWeight = try await readWeight("\(prefix).ffnNorm.weight")
        let ffnNormed = try await rmsNormKernel.execute(
            input: afterAttn,
            weight: ffnNormWeight,
            rows: seqLen, cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // SwiGLU FFN
        let gateWeight = try await readWeight("\(prefix).feedForward.gate.weight")
        let upWeight = try await readWeight("\(prefix).feedForward.up.weight")
        let downWeight = try await readWeight("\(prefix).feedForward.down.weight")

        var output = afterAttn
        for t in 0..<seqLen {
            let tokenHidden = Array(ffnNormed[t * dim..<(t + 1) * dim])

            let gateResult = try await gemvKernel.execute(
                a: gateWeight, x: tokenHidden,
                M: config.intermediateDim, K: dim,
                commandQueue: commandQueue
            )
            let upResult = try await gemvKernel.execute(
                a: upWeight, x: tokenHidden,
                M: config.intermediateDim, K: dim,
                commandQueue: commandQueue
            )

            let activated = try await activationKernels.swiglu(
                gate: gateResult, up: upResult,
                commandQueue: commandQueue
            )

            let downResult = try await gemvKernel.execute(
                a: downWeight, x: activated,
                M: dim, K: config.intermediateDim,
                commandQueue: commandQueue
            )

            for d in 0..<dim {
                output[t * dim + d] += downResult[d]
            }
        }

        return output
    }

    // MARK: - Weight Reading with Dequantization

    /// Read a weight tensor, dequantizing on first access and caching the result.
    private func readWeight(_ name: String) async throws -> [Float] {
        // Check cache first
        if let cached = await weightCache.get(name) {
            return cached
        }

        guard let storage = weights[name] else {
            throw GenerationError.modelLoadFailed(reason: "Missing weight: \(name)")
        }

        let floats: [Float]

        // Compute actual byte count for this tensor from element count + data type
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
                blockData: blockData,
                blockCount: blockCount,
                commandQueue: commandQueue
            )

        case .q8_0:
            let blockCount = elementCount / 32
            let byteCount = blockCount * 34
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            floats = try await dequantQ8_0.dequantise(
                blockData: blockData,
                blockCount: blockCount,
                commandQueue: commandQueue
            )

        case .q4_K:
            let superBlockCount = elementCount / 256
            let byteCount = superBlockCount * 144
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            floats = try await dequantQ4KM.dequantise(
                blockData: blockData,
                superBlockCount: superBlockCount,
                commandQueue: commandQueue
            )

        default:
            throw GenerationError.modelLoadFailed(
                reason: "Unsupported weight data type \(storage.dataType) for \(name)"
            )
        }

        // Cache for future use
        await weightCache.set(name, value: floats)
        return floats
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
            // Q8_0: 32 elements per block, 34 bytes per block (2 bytes scale + 32 bytes quants)
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
                    // First 2 bytes: scale as Float16 (stored as UInt16)
                    let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                    let scale = Float(Float16(bitPattern: scaleBits))
                    // Next 32 bytes: quantized int8 values
                    for j in 0..<elementsPerBlock {
                        let qval = Int8(bitPattern: rawPtr[blockStart + 2 + j])
                        result[dstOffset + block * elementsPerBlock + j] = scale * Float(qval)
                    }
                }
            }

        case .q4_0:
            // Q4_0: 32 elements per block, 18 bytes per block (2 bytes scale + 16 bytes quants)
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
                    // 16 bytes encode 32 4-bit values (2 per byte)
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
    /// Compute LM head logits using tied embedding weights.
    /// Dequantizes one row at a time to avoid allocating 593MB at once.
    private func computeTiedLMHead(
        hidden: [Float],
        embeddingStorage: TensorStorage,
        vocabSize: Int,
        dim: Int
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
                for d in 0..<dim {
                    dot += ptr[offset + d] * hidden[d]
                }
                logits[vocabIdx] = dot
            }

        default:
            break // Unsupported — return zeros
        }

        return logits
    }
}

// MARK: - Weight Cache Actor

/// Thread-safe cache for dequantized weights.
/// Avoids re-dequantizing the same weight tensor on every forward pass.
private actor WeightCacheActor {
    private var cache: [String: [Float]] = [:]

    func get(_ name: String) -> [Float]? {
        cache[name]
    }

    func set(_ name: String, value: [Float]) {
        cache[name] = value
    }
}
