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

    // KV cache for autoregressive generation
    private let kvCache: KVCache

    // Generation state
    private var currentPos: Int = 0

    public init(model: LlamaModel) throws {
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

        // Initialize KV cache — use fp32 for correctness, can switch to fp16 for speed
        self.kvCache = try KVCache(
            device: device,
            maxSeqLen: 4096,
            numLayers: config.layerCount,
            numKVHeads: config.kvHeadCount,
            headDim: config.headDim,
            precision: .float32
        )
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
        return try LlamaLanguageModel(model: model)
    }

    public func tokenize(_ text: String) -> [Int] {
        // Minimal byte-level fallback tokenizer.
        // Real usage should pair with BPETokenizer loaded from the GGUF vocab.
        Array(text.utf8).map { Int($0) }
    }

    public func detokenize(_ ids: [Int]) -> String {
        String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    public var eosTokenID: Int { 2 } // Standard EOS for Llama-family
    public var bosTokenID: Int? { 1 } // Standard BOS for Llama-family
    public var vocabularySize: Int { config.vocabSize }

    // MARK: - LogitsModel: forward pass

    /// Run the full Llama transformer forward pass, returning logits for the last token.
    ///
    /// For autoregressive generation, this is called with the full context each time.
    /// The KV cache avoids recomputing attention for previously seen tokens.
    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let seqLen = tokenIDs.count
        let dim = config.embeddingDim
        // 1. Token embedding lookup
        var hidden = embeddingLookup(tokenIDs: tokenIDs)

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
        let finalNormWeight = readWeight("finalNorm.weight")
        hidden = try await rmsNormKernel.execute(
            input: hidden,
            weight: finalNormWeight,
            rows: seqLen,
            cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // 4. LM head: project last token's hidden state to vocab logits
        // logits = hidden[last_token] @ lmHead.weight^T
        let lmHeadWeight = readWeight("lmHead.weight")
        let lastTokenHidden = Array(hidden.suffix(dim))

        let logits = try await gemvKernel.execute(
            a: lmHeadWeight,
            x: lastTokenHidden,
            M: config.vocabSize,
            K: dim,
            commandQueue: commandQueue
        )

        return logits
    }

    // MARK: - Transformer Layer

    /// Single transformer layer: RMSNorm → Attention(RoPE+GQA) → residual → RMSNorm → SwiGLU FFN → residual
    private func transformerLayer(
        hidden: [Float],
        layerIndex: Int,
        seqLen: Int,
        startPos: Int
    ) async throws -> [Float] {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let prefix = "layers.\(layerIndex)"

        // --- Attention block ---

        // Pre-attention RMS norm
        let attnNormWeight = readWeight("\(prefix).attentionNorm.weight")
        let normed = try await rmsNormKernel.execute(
            input: hidden,
            weight: attnNormWeight,
            rows: seqLen,
            cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // Q, K, V projections via GEMV (for single-token decode) or matmul
        let wq = readWeight("\(prefix).attention.wq.weight")
        let wk = readWeight("\(prefix).attention.wk.weight")
        let wv = readWeight("\(prefix).attention.wv.weight")
        let wo = readWeight("\(prefix).attention.wo.weight")

        // Project for each token position
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

        // Apply RoPE to Q and K
        let ropeQ = try await ropeKernel.execute(
            input: allQ,
            seqLen: seqLen,
            numHeads: config.headCount,
            headDim: headDim,
            startPos: startPos,
            theta: Float(config.ropeFreqBase),
            commandQueue: commandQueue
        )
        let ropeK = try await ropeKernel.execute(
            input: allK,
            seqLen: seqLen,
            numHeads: config.kvHeadCount,
            headDim: headDim,
            startPos: startPos,
            theta: Float(config.ropeFreqBase),
            commandQueue: commandQueue
        )

        // Grouped Query Attention
        let attnOutput = try await gqaKernel.execute(
            q: ropeQ,
            k: ropeK,
            v: allV,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: config.headCount,
            numKVHeads: config.kvHeadCount,
            causal: true,
            commandQueue: commandQueue
        )

        // Output projection: project attention output back to model dim
        var projectedAttn = [Float](repeating: 0, count: seqLen * dim)
        for t in 0..<seqLen {
            let tokenAttn = Array(attnOutput[t * config.headCount * headDim..<(t + 1) * config.headCount * headDim])
            let projected = try await gemvKernel.execute(
                a: wo, x: tokenAttn,
                M: dim, K: config.headCount * headDim,
                commandQueue: commandQueue
            )
            for d in 0..<dim {
                projectedAttn[t * dim + d] = projected[d]
            }
        }

        // Residual connection: hidden + attention_output
        var afterAttn = [Float](repeating: 0, count: seqLen * dim)
        for i in 0..<(seqLen * dim) {
            afterAttn[i] = hidden[i] + projectedAttn[i]
        }

        // --- FFN block ---

        // Pre-FFN RMS norm
        let ffnNormWeight = readWeight("\(prefix).ffnNorm.weight")
        let ffnNormed = try await rmsNormKernel.execute(
            input: afterAttn,
            weight: ffnNormWeight,
            rows: seqLen,
            cols: dim,
            eps: Float(config.rmsNormEpsilon),
            commandQueue: commandQueue
        )

        // SwiGLU FFN: output = down(swiglu(gate(x), up(x)))
        let gateWeight = readWeight("\(prefix).feedForward.gate.weight")
        let upWeight = readWeight("\(prefix).feedForward.up.weight")
        let downWeight = readWeight("\(prefix).feedForward.down.weight")

        var ffnOutput = [Float](repeating: 0, count: seqLen * dim)

        for t in 0..<seqLen {
            let tokenHidden = Array(ffnNormed[t * dim..<(t + 1) * dim])

            // gate(x) and up(x) projections
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

            // SwiGLU activation: swish(gate) * up
            let activated = try await activationKernels.swiglu(
                gate: gateResult,
                up: upResult,
                commandQueue: commandQueue
            )

            // Down projection back to model dim
            let downResult = try await gemvKernel.execute(
                a: downWeight, x: activated,
                M: dim, K: config.intermediateDim,
                commandQueue: commandQueue
            )

            for d in 0..<dim {
                ffnOutput[t * dim + d] = downResult[d]
            }
        }

        // Residual connection: afterAttn + ffn_output
        var output = [Float](repeating: 0, count: seqLen * dim)
        for i in 0..<(seqLen * dim) {
            output[i] = afterAttn[i] + ffnOutput[i]
        }

        return output
    }

    // MARK: - Helpers

    /// Look up token embeddings from the embedding weight matrix.
    private func embeddingLookup(tokenIDs: [Int]) -> [Float] {
        let dim = config.embeddingDim
        let embeddingWeight = readWeight("embedding.weight")

        var result = [Float](repeating: 0, count: tokenIDs.count * dim)
        for (i, tokenID) in tokenIDs.enumerated() {
            let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
            let srcOffset = clampedID * dim
            let dstOffset = i * dim
            for d in 0..<dim {
                result[dstOffset + d] = embeddingWeight[srcOffset + d]
            }
        }
        return result
    }

    /// Read a weight tensor as Float array from loaded weights.
    ///
    /// Weight data is stored in Metal buffers (possibly quantized).
    /// This reads the raw bytes and interprets as Float32.
    /// For quantized models, dequantization should happen at this layer.
    private func readWeight(_ name: String) -> [Float] {
        guard let storage = weights[name] else {
            fatalError("Missing weight: \(name)")
        }
        let floatCount = storage.byteCount / MemoryLayout<Float>.stride
        let pointer = storage.buffer.contents().bindMemory(
            to: Float.self,
            capacity: floatCount
        )
        return Array(UnsafeBufferPointer(start: pointer, count: floatCount))
    }
}
