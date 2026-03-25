import Foundation
import Metal
import Synchronization
import Accelerate
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
    private let tokenizer: (any Tokenizer)?
    private var tiedEmbeddingWeightName: String {
        weights["lmHead.weight"] != nil ? "lmHead.weight" : "embedding.weight"
    }

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

    // Fused dequant+GEMV pipeline for Q8_0 weights (3.8x bandwidth reduction)
    private let fusedQ8GemvPipeline: MTLComputePipelineState
    private let fusedQ8GemvTiledPipeline: MTLComputePipelineState  // Exp 35: Tile-based with coalesced access
    private let fusedQ8GemvF16OutPipeline: MTLComputePipelineState
    private let fusedQKVPipeline: MTLComputePipelineState
    private let fusedGateUpSiluPipeline: MTLComputePipelineState
    private let convertF32ToF16Pipeline: MTLComputePipelineState
    private let ropeNeoXF16OutPipeline: MTLComputePipelineState
    private let fusedQKNormRoPEPipeline: MTLComputePipelineState
    private let fusedNormRoPEGQAPipeline: MTLComputePipelineState
    private let fusedFinalNormGemvPipeline: MTLComputePipelineState
    private let gemvAddPipeline: MTLComputePipelineState

    // Dequantization kernels
    private let dequantQ4_0: DequantQ4_0Kernel
    private let dequantQ8_0: DequantQ8_0Kernel
    private let dequantQ4KM: DequantQ4KMKernel
    private let dequantQ5K: DequantQ5KKernel
    private let dequantQ6K: DequantQ6KKernel
    private let dequantQ3K: DequantQ3KKernel
    private let dequantQ2K: DequantQ2KKernel
    private let dequantQ5_0: DequantQ5_0Kernel
    private let dequantQ5_1: DequantQ5_1Kernel

    // KV cache for autoregressive generation
    private let kvCache: KVCache

    // Decoder state for KV cache: tracks previously processed tokens for incremental decode
    private let decoderState: DecoderStateStore

    // Per-layer KV cache MTLBuffers — written directly by GPU kernels
    private let layerKCaches: [MTLBuffer]
    private let layerVCaches: [MTLBuffer]

    // Pre-loaded weight buffers — eliminates async actor hops during forward pass
    // Uses raw Q8_0 buffers directly; no float32 caching to save memory
    private let preloadedWeights: PreloadedWeightsStore

    // Pre-allocated scratch buffers — eliminates per-call buffer allocations
    private let scratch: ScratchBuffers

    // Metal 4 state — argument table dispatch with minimal per-dispatch overhead
    private let metal4State: Metal4State?

    // Params buffer for optimized Metal 3 decode path
    private let decodeParamsBuffer: MTLBuffer?
    private let decodeDebugOptions: DecodeDebugOptions

    public init(model: LlamaModel, maxSeqLen: Int = 4096) throws {
        try self.init(model: model, maxSeqLen: maxSeqLen, decodeDebugOptions: nil, tokenizer: nil)
    }

    fileprivate init(
        model: LlamaModel,
        maxSeqLen: Int = 4096,
        decodeDebugOptions: DecodeDebugOptions?,
        tokenizer: (any Tokenizer)? = nil
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GenerationError.modelLoadFailed(reason: "No Metal device available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command queue")
        }

        self.config = model.config
        self.weights = model.loadedWeights
        self.tokenizer = tokenizer
        self.device = device
        self.commandQueue = queue
        self.decodeDebugOptions = decodeDebugOptions ?? DecodeDebugOptions(
            environment: ProcessInfo.processInfo.environment,
            config: model.config,
            overrides: nil
        )

        // Initialize Metal kernels
        self.rmsNormKernel = try RMSNormKernel(device: device)
        self.ropeKernel = try RoPEKernel(device: device)
        self.gqaKernel = try GQAKernel(device: device)
        self.activationKernels = try ActivationKernels(device: device)
        self.gemvKernel = try GEMVKernel(device: device)

        // Load elementwise add pipeline for GPU residual connections
        let registry = try KernelRegistry(device: device)
        self.addPipeline = try registry.pipeline(for: "elementwise_add_float")
        self.fusedQ8GemvPipeline = try registry.pipeline(for: "dequant_q8_0_gemv")
        self.fusedQ8GemvTiledPipeline = try registry.pipeline(for: "dequant_q8_0_gemv_tiled")
        self.fusedQ8GemvF16OutPipeline = try registry.pipeline(for: "dequant_q8_0_gemv_f16out")
        self.fusedQKVPipeline = try registry.pipeline(for: "dequant_q8_0_fused_qkv")
        self.fusedGateUpSiluPipeline = try registry.pipeline(for: "dequant_q8_0_fused_gate_up_silu")
        self.convertF32ToF16Pipeline = try registry.pipeline(for: "convert_f32_to_f16")
        self.ropeNeoXF16OutPipeline = try registry.pipeline(for: "rope_neox_f32_to_f16")
        self.fusedQKNormRoPEPipeline = try registry.pipeline(for: "fused_qk_norm_rope_neox")
        self.fusedNormRoPEGQAPipeline = try registry.pipeline(for: "fused_qk_norm_rope_gqa")
        self.fusedFinalNormGemvPipeline = try registry.pipeline(for: "dequant_q8_0_fused_final_norm_gemv")
        self.gemvAddPipeline = try registry.pipeline(for: "dequant_q8_0_gemv_add")

        // Initialize dequant kernels
        self.dequantQ4_0 = try DequantQ4_0Kernel(device: device)
        self.dequantQ8_0 = try DequantQ8_0Kernel(device: device)
        self.dequantQ4KM = try DequantQ4KMKernel(device: device)
        self.dequantQ5K = try DequantQ5KKernel(device: device)
        self.dequantQ6K = try DequantQ6KKernel(device: device)
        self.dequantQ3K = try DequantQ3KKernel(device: device)
        self.dequantQ2K = try DequantQ2KKernel(device: device)
        self.dequantQ5_0 = try DequantQ5_0Kernel(device: device)
        self.dequantQ5_1 = try DequantQ5_1Kernel(device: device)

        // Initialize KV cache
        let effectiveMaxSeq = maxSeqLen
        self.kvCache = try KVCache(
            device: device,
            maxSeqLen: effectiveMaxSeq,
            numLayers: config.layerCount,
            numKVHeads: config.kvHeadCount,
            headDim: config.headDim,
            precision: .float16
        )

        // Extract per-layer KV cache MTLBuffers for direct GPU writes
        var kCaches = [MTLBuffer]()
        var vCaches = [MTLBuffer]()
        kCaches.reserveCapacity(config.layerCount)
        vCaches.reserveCapacity(config.layerCount)
        for i in 0..<config.layerCount {
            let (kBuf, vBuf) = try kvCache.metalBuffers(layer: i)
            kCaches.append(kBuf)
            vCaches.append(vBuf)
        }
        self.layerKCaches = kCaches
        self.layerVCaches = vCaches

        // Initialize decoder state for incremental decode
        self.decoderState = DecoderStateStore()

        self.preloadedWeights = PreloadedWeightsStore()

        // Pre-allocate scratch buffers for max seqLen
        let maxSeq = effectiveMaxSeq
        let fs = MemoryLayout<Float>.stride
        let qDim = config.headCount * config.headDim
        let kvDim = config.kvHeadCount * config.headDim
        let dim = config.embeddingDim
        let interDim = config.intermediateDim

        func allocBuffer(_ size: Int) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: size, options: .storageModeShared) else {
                throw GenerationError.modelLoadFailed(reason: "GPU buffer allocation failed (\(size) bytes)")
            }
            return buf
        }

        self.scratch = try ScratchBuffers(
            normed: allocBuffer(maxSeq * dim * fs),
            afterAttn: allocBuffer(maxSeq * dim * fs),
            ffnNormed: allocBuffer(maxSeq * dim * fs),
            outputA: allocBuffer(maxSeq * dim * fs),
            outputB: allocBuffer(maxSeq * dim * fs),
            allQ: allocBuffer(maxSeq * qDim * fs),
            allK: allocBuffer(maxSeq * kvDim * fs),
            allV: allocBuffer(maxSeq * kvDim * fs),
            ropeQ: allocBuffer(maxSeq * qDim * fs),
            ropeK: allocBuffer(maxSeq * kvDim * fs),
            attnOut: allocBuffer(maxSeq * qDim * fs),
            proj: allocBuffer(maxSeq * dim * fs),
            gateOut: allocBuffer(maxSeq * interDim * fs),
            upOut: allocBuffer(maxSeq * interDim * fs),
            activ: allocBuffer(maxSeq * interDim * fs),
            downOut: allocBuffer(maxSeq * dim * fs),
            finalOut: allocBuffer(maxSeq * dim * fs),
            logits: allocBuffer(config.vocabSize * fs),
            decodeHidden: allocBuffer(dim * fs)
        )

        // Initialize Metal 4 state if available (macOS 26+)
        if #available(macOS 26.0, iOS 26.0, *) {
            self.metal4State = try? Metal4State(device: device)
        } else {
            self.metal4State = nil
        }

        // Pre-allocated params buffer for optimized Metal 3 decode.
        // 8 slots x 256 bytes each: QKV(0), Mega(256), Wo(512), GUS(768), Down(1024),
        // FinalNorm(1280), LMHead(1536), FusedFinalNormLMHead(1792)
        self.decodeParamsBuffer = device.makeBuffer(length: 8 * 256, options: .storageModeShared)
    }

    // MARK: - EdgeRunnerLanguageModel conformance

    /// Loads a Llama-family model from a GGUF file.
    ///
    /// This method loads the model weights using memory-mapped I/O for efficient
    /// memory usage and fast loading. The model is immediately ready for inference
    /// after loading completes.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let modelURL = URL(fileURLWithPath: "/path/to/model.gguf")
    /// let config = ModelConfiguration(contextWindowSize: 2048)
    /// let model = try await LlamaLanguageModel.load(from: modelURL, configuration: config)
    /// ```
    ///
    /// - Parameters:
    ///   - url: File URL pointing to a `.gguf` model file.
    ///   - configuration: Configuration for loading and behavior.
    /// - Returns: A loaded `LlamaLanguageModel` ready for inference.
    /// - Throws: `GenerationError.modelLoadFailed` if loading fails.
    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> LlamaLanguageModel {
        let loader = try GGUFLoader(url: url)
        let weightMap = try await loader.load(from: url)

        // Validate quantization types before proceeding
        try validateQuantizationTypes(weightMap)

        let ggufConfig = try LlamaConfig(fromGGUFMetadata: loader.modelConfig.metadata)
        var model = LlamaModel(config: ggufConfig)
        try model.loadWeights(from: weightMap)

        // Create tokenizer from GGUF metadata (BPE or SentencePiece)
        let loadedTokenizer: (any Tokenizer)?
        do {
            let tokenizerMetadata = try loader.modelConfig.tokenizerMetadata()
            loadedTokenizer = try TokenizerFactory.create(from: tokenizerMetadata)
        } catch {
            // Fall back to nil tokenizer if metadata is missing or unsupported
            loadedTokenizer = nil
        }

        return try LlamaLanguageModel(
            model: model,
            maxSeqLen: configuration.contextWindowSize,
            decodeDebugOptions: DecodeDebugOptions(
                environment: ProcessInfo.processInfo.environment,
                config: ggufConfig,
                overrides: configuration.llamaDecodeOverrides
            ),
            tokenizer: loadedTokenizer
        )
    }

    /// Tokenizes text into token IDs using the tokenizer loaded from GGUF metadata.
    /// Falls back to byte-level encoding if no tokenizer is available.
    public func tokenize(_ text: String) -> [Int] {
        if let tokenizer {
            return tokenizer.encode(text, addBOS: tokenizer.shouldAddBOS)
        }
        return Array(text.utf8).map { Int($0) }
    }

    /// Detokenizes token IDs back to text using the tokenizer.
    /// Falls back to byte-level decoding if no tokenizer is available.
    public func detokenize(_ ids: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(ids, skipSpecialTokens: true)
        }
        return String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    /// The end-of-sequence token ID.
    public var eosTokenID: Int { tokenizer?.eosTokenID ?? 151645 }

    /// The beginning-of-sequence token ID.
    public var bosTokenID: Int? { tokenizer?.bosTokenID ?? 151643 }

    /// The total number of tokens in the model's vocabulary.
    public var vocabularySize: Int { tokenizer?.vocabularySize ?? config.vocabSize }

    /// Formats conversation messages using the model's chat template.
    public func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool
    ) -> String? {
        try? tokenizer?.applyChatTemplate(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt
        )
    }

    // MARK: - Validation

    /// Validates that all tensors in the weight map use supported quantization types.
    /// Throws at load time instead of crashing at dequantization time.
    private static func validateQuantizationTypes(_ weightMap: WeightMap) throws {
        let supportedTypes: Set<TensorDataType> = [
            .float32, .float16, .q4_0, .q8_0, .q4_K,
            .q6_K, .q5_K, .q3_K, .q2_K, .q5_0, .q5_1
        ]

        var unsupportedTypes = Set<TensorDataType>()
        var unsupportedTensors = [String]()

        for name in weightMap.tensorNames {
            guard let tensor = weightMap[name] else { continue }
            if !supportedTypes.contains(tensor.dataType) {
                unsupportedTypes.insert(tensor.dataType)
                if unsupportedTensors.count < 3 {
                    unsupportedTensors.append("\(name) (\(tensor.dataType))")
                }
            }
        }

        guard unsupportedTypes.isEmpty else {
            let typeNames = unsupportedTypes.map { type -> String in
                switch type {
                case .q4_1: return "Q4_1"
                case .q5_0: return "Q5_0"
                case .q5_1: return "Q5_1"
                case .q8_1: return "Q8_1"
                case .q2_K: return "Q2_K"
                case .q3_K: return "Q3_K"
                case .q5_K: return "Q5_K"
                case .q6_K: return "Q6_K"
                case .q8_K: return "Q8_K"
                default: return "type(\(type.rawValue))"
                }
            }.sorted().joined(separator: ", ")

            let examples = unsupportedTensors.joined(separator: ", ")
            throw GenerationError.modelLoadFailed(
                reason: "Model uses unsupported quantization: \(typeNames). "
                    + "Supported: Q2_K, Q3_K, Q4_0, Q4_K_M, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, F16, F32. "
                    + "Examples: \(examples)"
            )
        }
    }

    // MARK: - LogitsModel: forward pass

    /// Generates the next token using the provided sampling configuration.
    ///
    /// This method performs a forward pass through the model and applies
    /// the specified sampling strategy (greedy, temperature, top-p, etc.).
    ///
    /// - Parameters:
    ///   - tokenIDs: The input token sequence.
    ///   - sampling: Configuration for sampling strategy.
    /// - Returns: The ID of the generated next token.
    /// - Throws: `GenerationError` if generation fails.
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let isPureGreedy = sampling.temperature <= 0 && sampling.repetitionPenalty <= 1.0
        if isPureGreedy {
            let result = try await greedyToken(for: tokenIDs)
            if result.hasNonFinite {
                throw GenerationError.decodingFailed("Logits contain NaN/Inf during greedy decode")
            }
            return result.token
        }

        let logitsArray: [Float]
        if tokenIDs == decoderState.cachedLogitsInput, let cached = decoderState.cachedLogits {
            logitsArray = cached
        } else {
            logitsArray = try await self.logits(for: tokenIDs)
        }

        let pipeline = sampling.toPipeline()
        return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
    }

    /// Computes the raw logits for the next token given an input sequence.
    ///
    /// This is the core inference method. It performs a forward pass through
    /// the transformer and returns the unnormalized log probabilities for each
    /// token in the vocabulary.
    ///
    /// ## Performance
    ///
    /// For autoregressive generation, this method automatically uses the KV cache
    /// for sequences longer than one token, significantly speeding up inference.
    ///
    /// ## Example
    ///
    /// ```swift
    /// var tokens = [1]  // Start with BOS token
    /// for _ in 0..<10 {
    ///     let logits = try await model.logits(for: tokens)
    ///     let nextToken = greedyArgmax(logits)  // Or use sampling
    ///     tokens.append(nextToken)
    /// }
    /// ```
    ///
    /// - Parameter tokenIDs: The input sequence of token IDs.
    /// - Returns: An array of logits with length equal to `vocabularySize`.
    /// - Throws: `GenerationError` if the forward pass fails.
    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        if tokenIDs == decoderState.cachedLogitsInput, let cached = decoderState.cachedLogits {
            return cached
        }
        let logitsBuf = try await forwardLogitsBuffer(for: tokenIDs)
        let result = materializeLogits(from: logitsBuf, count: config.vocabSize)
        decoderState.cachedLogits = result
        decoderState.cachedLogitsInput = tokenIDs
        return result
    }

    /// Greedy argmax without materializing logits to a Swift array.
    /// Returns the selected token and whether any non-finite values were present.
    func greedyToken(for tokenIDs: [Int]) async throws -> (token: Int, hasNonFinite: Bool) {
        let logitsBuf = try await forwardLogitsBuffer(for: tokenIDs)
        let token = greedyArgmax(logitsBuf: logitsBuf, count: config.vocabSize)
        let hasNonFinite = containsNonFinite(logitsBuf: logitsBuf, count: config.vocabSize)
        decoderState.cachedLogits = nil
        decoderState.cachedLogitsInput = tokenIDs
        return (token, hasNonFinite)
    }

    /// Measures average wall-clock latency (ms) of the final norm + LM head GPU path.
    /// Intended for profiling only; not used in normal decode.
    public func measureLMHeadLatency(samples: Int = 5) async throws -> Double {
        guard let finalNorm = preloadedWeights.finalNorm else {
            throw GenerationError.modelLoadFailed(reason: "Final norm weights not loaded")
        }

        let dim = config.embeddingDim
        let logitsBuf = scratch.logits
        let hiddenBuf = scratch.decodeHidden
        memset(hiddenBuf.contents(), 0, hiddenBuf.length)

        let clock = ContinuousClock()
        var totalMs: Double = 0

        for _ in 0..<samples {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else {
                throw GenerationError.modelLoadFailed(reason: "Failed to create LM head command encoder")
            }

            let rmsEps = Float(config.rmsNormEpsilon)
            if let lmRaw = preloadedWeights.lmHeadRaw {
                let blocksPerRow = preloadedWeights.lmHeadCols / 32
                var p = FusedGateUpSiluParams(
                    rows: UInt32(config.vocabSize),
                    cols: UInt32(dim),
                    blocksPerRow: UInt32(blocksPerRow),
                    rmsEps: rmsEps
                )
                enc.setComputePipelineState(fusedFinalNormGemvPipeline)
                enc.setBuffer(lmRaw, offset: 0, index: 0)
                enc.setBuffer(hiddenBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBuffer(finalNorm, offset: 0, index: 3)
                enc.setBytes(&p, length: MemoryLayout<FusedGateUpSiluParams>.stride, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            } else if let lmHead = preloadedWeights.lmHead {
                let finalOutputBuf = scratch.finalOut
                var normP = ERRMSNormParams(rows: 1, cols: UInt32(dim), eps: rmsEps)
                enc.setComputePipelineState(rmsNormKernel.pipeline)
                enc.setBuffer(hiddenBuf, offset: 0, index: 0)
                enc.setBuffer(finalNorm, offset: 0, index: 1)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 2)
                enc.setBytes(&normP, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                enc.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
                )

                let blocksPerRow = preloadedWeights.lmHeadCols / 32
                var p = ERDequantGEMVParams(
                    rows: UInt32(config.vocabSize),
                    cols: UInt32(dim),
                    blocksPerRow: UInt32(blocksPerRow)
                )
                enc.setComputePipelineState(fusedQ8GemvPipeline)
                enc.setBuffer(lmHead, offset: 0, index: 0)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            } else {
                throw GenerationError.modelLoadFailed(reason: "LM head weights not loaded")
            }

            enc.endEncoding()
            let start = clock.now
            cmdBuf.commit()
            await cmdBuf.completed()
            let end = clock.now
            if let error = cmdBuf.error { throw error }

            let duration = start.duration(to: end)
            totalMs += Double(duration.components.seconds) * 1000.0
                + Double(duration.components.attoseconds) * 1e-15
        }

        return totalMs / Double(max(samples, 1))
    }

    private func forwardLogitsBuffer(for tokenIDs: [Int]) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let floatStride = MemoryLayout<Float>.stride

        // Detect decode / prefix-reuse / full-prefill mode
        let previousTokenIDs = decoderState.previousTokenIDs

        // Count how many tokens at the start match between previous and current sequences
        let commonPrefixLen: Int = {
            let minLen = min(tokenIDs.count, previousTokenIDs.count)
            var i = 0
            while i < minLen && tokenIDs[i] == previousTokenIDs[i] { i += 1 }
            return i
        }()

        let isDecodeMode = commonPrefixLen == previousTokenIDs.count
            && tokenIDs.count == commonPrefixLen + 1
            && tokenIDs.count > 1

        if isDecodeMode, let newTokenID = tokenIDs.last {
            // DECODE MODE: process only the single new token using KV cache
            let currentPos = previousTokenIDs.count  // 0-indexed position of new token

            // Embedding lookup for single token — use pre-allocated buffer (zero allocation)
            let hiddenBuf = scratch.decodeHidden
            if let embBuf = preloadedWeights.lmHead {
                let embPtr = embBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize * dim)
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: dim)
                let clampedID = min(max(newTokenID, 0), config.vocabSize - 1)
                memcpy(dstPtr, embPtr + clampedID * dim, dim * floatStride)
            } else {
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: dim)
                try fillEmbeddings(tokenIDs: [newTokenID], into: dstPtr)
            }

            let logitsBuf = try await runDecodePass(hiddenBuf: hiddenBuf, currentPos: currentPos)

            // Update decoder state
            decoderState.previousTokenIDs = tokenIDs
            return logitsBuf
        } else if commonPrefixLen > 0
            && commonPrefixLen == previousTokenIDs.count
            && tokenIDs.count > commonPrefixLen + 1 {
            // PREFIX REUSE MODE: the new sequence strictly extends the cached sequence by
            // multiple tokens. KV cache positions 0..<commonPrefixLen are already valid.
            // Only prefill the suffix tokens with correct RoPE positions and causal masking.
            let suffixTokens = Array(tokenIDs[commonPrefixLen...])
            let suffixLen = suffixTokens.count

            // Embedding lookup for suffix tokens only
            let hiddenBuf: MTLBuffer
            if let embBuf = preloadedWeights.lmHead {
                let embPtr = embBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize * dim)
                guard let hBuf = device.makeBuffer(length: suffixLen * dim * floatStride, options: .storageModeShared) else {
                    throw GenerationError.modelLoadFailed(reason: "GPU buffer allocation failed for prefix-reuse embedding")
                }
                hiddenBuf = hBuf
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: suffixLen * dim)
                for (i, tokenID) in suffixTokens.enumerated() {
                    let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                    memcpy(dstPtr + i * dim, embPtr + clampedID * dim, dim * floatStride)
                }
            } else {
                guard let hBuf = device.makeBuffer(length: suffixLen * dim * floatStride, options: .storageModeShared) else {
                    throw GenerationError.modelLoadFailed(reason: "GPU buffer allocation failed for prefix-reuse embedding")
                }
                hiddenBuf = hBuf
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: suffixLen * dim)
                try fillEmbeddings(tokenIDs: suffixTokens, into: dstPtr)
            }

            // Run prefill on suffix only — RoPE positions offset by commonPrefixLen,
            // GQA attends over full KV cache (prefix + suffix), causal mask offset accordingly
            let logitsBuf = try await fusedPrefillPass(
                hiddenBuf: hiddenBuf,
                seqLen: suffixLen,
                startPosition: commonPrefixLen
            )

            // Update decoder state and KV cache position
            decoderState.previousTokenIDs = tokenIDs
            kvCache.setPosition(tokenIDs.count)
            return logitsBuf
        } else {
            // FULL PREFILL MODE: no useful prefix match — reset and recompute everything
            let seqLen = tokenIDs.count

            // Reset KV cache for new sequence
            kvCache.reset()

            // Embedding lookup for full sequence
            let hiddenBuf: MTLBuffer
            if let embBuf = preloadedWeights.lmHead {
                let embPtr = embBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize * dim)
                guard let hBuf = device.makeBuffer(length: seqLen * dim * floatStride, options: .storageModeShared) else {
                    throw GenerationError.modelLoadFailed(reason: "GPU buffer allocation failed for prefill embedding")
                }
                hiddenBuf = hBuf
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: seqLen * dim)
                for (i, tokenID) in tokenIDs.enumerated() {
                    let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                    memcpy(dstPtr + i * dim, embPtr + clampedID * dim, dim * floatStride)
                }
            } else {
                guard let hBuf = device.makeBuffer(length: seqLen * dim * floatStride, options: .storageModeShared) else {
                    throw GenerationError.modelLoadFailed(reason: "GPU buffer allocation failed for prefill embedding")
                }
                hiddenBuf = hBuf
                let dstPtr = hiddenBuf.contents().bindMemory(to: Float.self, capacity: seqLen * dim)
                try fillEmbeddings(tokenIDs: tokenIDs, into: dstPtr)
            }

            var logitsBuf = try await fusedPrefillPass(
                hiddenBuf: hiddenBuf,
                seqLen: seqLen
            )

            // Decode warmup: run 3 dummy decode passes to warm GPU pipeline for decode-specific
            // kernel variants (different kvSeqLen, currentPos). This eliminates the ~20% cold-start
            // penalty on the first real decode pass. Runs once, during the first prefill call.
            if !decoderState.decodeWarmedUp {
                decoderState.decodeWarmedUp = true
                let warmupHidden = scratch.decodeHidden
                let dummyPtr = warmupHidden.contents().bindMemory(to: Float.self, capacity: dim)
                memset(dummyPtr, 0, dim * floatStride)
                // Run 5 dummy decodes to warm GPU pipeline for decode-specific kernel variants.
                // 5 is the sweet spot: eliminates JIT cold-start penalty without excessive
                // first-call latency (~18ms warmup vs ~55ms for 15).
                for warmupIdx in 0..<5 {
                    let warmupPos = seqLen + warmupIdx
                    kvCache.setPosition(warmupPos)  // Set position so mega-kernel uses correct kvSeqLen
                    _ = try await runDecodePass(hiddenBuf: warmupHidden, currentPos: warmupPos)
                }
                // Zero ALL KV cache buffers to remove stale warmup data
                for i in 0..<config.layerCount {
                    memset(layerKCaches[i].contents(), 0, layerKCaches[i].length)
                    memset(layerVCaches[i].contents(), 0, layerVCaches[i].length)
                }
                kvCache.reset()
                // Re-run prefill to populate KV cache with correct data
                logitsBuf = try await fusedPrefillPass(hiddenBuf: hiddenBuf, seqLen: seqLen)
            }

            // Update decoder state and KV cache position
            decoderState.previousTokenIDs = tokenIDs
            kvCache.setPosition(seqLen)
            return logitsBuf
        }
    }

    private func runDecodePass(hiddenBuf: MTLBuffer, currentPos: Int) async throws -> MTLBuffer {
        if !decodeDebugOptions.requiresBaseDecodePath,
           decodeDebugOptions.preferMetal4DecodePath,
           #available(macOS 26.0, iOS 26.0, *),
           let m4 = metal4State {
            return try await fusedDecodePassMetal4(
                hiddenBuf: hiddenBuf,
                currentPos: currentPos,
                state: m4
            )
        }

        if !decodeDebugOptions.requiresBaseDecodePath,
           let paramsBuf = decodeParamsBuffer,
           preloadedWeights.layers.first?.wqRaw != nil {
            return try await fusedDecodePassOpt(
                hiddenBuf: hiddenBuf,
                currentPos: currentPos,
                paramsBuffer: paramsBuf
            )
        }

        if !decodeDebugOptions.requiresBaseDecodePath,
           #available(macOS 26.0, iOS 26.0, *),
           let m4 = metal4State {
            return try await fusedDecodePassMetal4(
                hiddenBuf: hiddenBuf,
                currentPos: currentPos,
                state: m4
            )
        }

        return try await fusedDecodePass(
            hiddenBuf: hiddenBuf,
            currentPos: currentPos
        )
    }

    private func materializeLogits(from logitsBuf: MTLBuffer, count: Int) -> [Float] {
        let ptr = logitsBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func greedyArgmax(_ logits: [Float]) -> Int {
        logits.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return 0 }
            return greedyArgmax(base, count: ptr.count)
        }
    }

    private func greedyArgmax(logitsBuf: MTLBuffer, count: Int) -> Int {
        let ptr = logitsBuf.contents().bindMemory(to: Float.self, capacity: count)
        return greedyArgmax(ptr, count: count)
    }

    private func containsNonFinite(logitsBuf: MTLBuffer, count: Int) -> Bool {
        let ptr = logitsBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            if !ptr[i].isFinite {
                return true
            }
        }
        return false
    }

    private func greedyArgmax(_ ptr: UnsafePointer<Float>, count: Int) -> Int {
        guard count > 0 else { return 0 }
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxValue, &maxIndex, vDSP_Length(count))
        return Int(maxIndex)
    }

    private func argmaxAndValidity(logitsBuf: MTLBuffer, count: Int) -> (Int, Bool) {
        let ptr = logitsBuf.contents().bindMemory(to: Float.self, capacity: count)
        var maxValue: Float = -.infinity
        var maxIndex = 0
        var hasNonFinite = false

        for i in 0..<count {
            let v = ptr[i]
            if !v.isFinite { hasNonFinite = true }
            if v > maxValue {
                maxValue = v
                maxIndex = i
            }
        }
        return (maxIndex, hasNonFinite)
    }

    // MARK: - Fully Fused Prefill Pass

    /// Encode ALL transformer layers + final norm + LM head into a SINGLE command buffer (prefill mode).
    /// Writes K/V data to per-layer cache buffers for subsequent decode steps.
    /// Returns MTLBuffer containing vocab-sized logits. ONE GPU sync for the entire forward pass.
    private func fusedPrefillPass(
        hiddenBuf: MTLBuffer,
        seqLen: Int,
        startPosition: Int = 0
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let floatStride = MemoryLayout<Float>.stride

        var currentHidden = hiddenBuf

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }

        // Use pre-allocated scratch buffers (zero allocation overhead)
        let normedBuf = scratch.normed
        let afterAttnBuf = scratch.afterAttn
        let ffnNormedBuf = scratch.ffnNormed
        let outputBufA = scratch.outputA
        let outputBufB = scratch.outputB
        let allQBuf = scratch.allQ
        let allKBuf = scratch.allK
        let ropeQBuf = scratch.ropeQ
        let ropeKBuf = scratch.ropeK
        let attnOutBuf = scratch.attnOut
        let projBuf = scratch.proj
        let gateOutBuf = scratch.gateOut
        let upOutBuf = scratch.upOut
        let activBuf = scratch.activ
        let downOutBuf = scratch.downOut

        // Pre-load ALL weight buffers on first call — eliminates 254 actor hops per subsequent call
        if !preloadedWeights.isLoaded {
            var layers = [LayerWeightBuffers]()
            layers.reserveCapacity(config.layerCount)
            for i in 0..<config.layerCount {
                let p = "layers.\(i)"
                let wqRaw = makeRawQ8BufferIfAvailable("\(p).attention.wq.weight")
                let wkRaw = makeRawQ8BufferIfAvailable("\(p).attention.wk.weight")
                let wvRaw = makeRawQ8BufferIfAvailable("\(p).attention.wv.weight")
                let woRaw = makeRawQ8BufferIfAvailable("\(p).attention.wo.weight")
                let gateRaw = makeRawQ8BufferIfAvailable("\(p).feedForward.gate.weight")
                let upRaw = makeRawQ8BufferIfAvailable("\(p).feedForward.up.weight")
                let downRaw = makeRawQ8BufferIfAvailable("\(p).feedForward.down.weight")

                let wqBuf = try await readWeightBufferIfNeeded("\(p).attention.wq.weight", rawBuffer: wqRaw)
                let wkBuf = try await readWeightBufferIfNeeded("\(p).attention.wk.weight", rawBuffer: wkRaw)
                let wvBuf = try await readWeightBufferIfNeeded("\(p).attention.wv.weight", rawBuffer: wvRaw)
                let woBuf = try await readWeightBufferIfNeeded("\(p).attention.wo.weight", rawBuffer: woRaw)
                let gateBuf = try await readWeightBufferIfNeeded("\(p).feedForward.gate.weight", rawBuffer: gateRaw)
                let upBuf = try await readWeightBufferIfNeeded("\(p).feedForward.up.weight", rawBuffer: upRaw)
                let downBuf = try await readWeightBufferIfNeeded("\(p).feedForward.down.weight", rawBuffer: downRaw)

                // Load per-head Q/K norm weights if present (Qwen3)
                let qNormName = "\(p).attention.qNorm.weight"
                let kNormName = "\(p).attention.kNorm.weight"
                let qNormBuf: MTLBuffer? = weights[qNormName] != nil ? try await readWeightBuffer(qNormName) : nil
                let kNormBuf: MTLBuffer? = weights[kNormName] != nil ? try await readWeightBuffer(kNormName) : nil

                layers.append(LayerWeightBuffers(
                    attnNorm: try await readWeightBuffer("\(p).attentionNorm.weight"),
                    wq: wqBuf, wk: wkBuf, wv: wvBuf, wo: woBuf,
                    qNorm: qNormBuf, kNorm: kNormBuf,
                    ffnNorm: try await readWeightBuffer("\(p).ffnNorm.weight"),
                    gate: gateBuf, up: upBuf, down: downBuf,
                    wqRaw: wqRaw, wkRaw: wkRaw, wvRaw: wvRaw, woRaw: woRaw,
                    gateRaw: gateRaw, upRaw: upRaw, downRaw: downRaw
                ))
            }
            let lmHeadName = tiedEmbeddingWeightName
            let lmHeadRawBuf = makeRawQ8BufferIfAvailable(lmHeadName)
            let lmHeadBuf = try await readWeightBufferIfNeeded(lmHeadName, rawBuffer: lmHeadRawBuf)
            let lmHeadCols = dim

            preloadedWeights.load(
                layers: layers,
                finalNorm: try await readWeightBuffer("finalNorm.weight"),
                lmHead: lmHeadBuf,
                lmHeadRaw: lmHeadRawBuf,
                lmHeadCols: lmHeadCols
            )

            // Populate Metal 4 residency set now that all weight buffers are available
            if #available(macOS 26.0, iOS 26.0, *), let m4 = metal4State {
                m4.populateResidencySet(
                    scratch: scratch,
                    layerKCaches: layerKCaches,
                    layerVCaches: layerVCaches,
                    preloadedWeights: preloadedWeights
                )
            }
        }

        // === SINGLE ENCODER for the ENTIRE forward pass ===
        // Metal guarantees sequential execution + implicit barriers between dispatches
        // within the same encoder. Using 1 encoder instead of 422 eliminates encoder
        // creation overhead (~10μs × 422 = 4.2ms per forward pass).
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }

        let rmsNormPSO = rmsNormKernel.pipeline
        let fusedFinalNormGemvPSO = fusedFinalNormGemvPipeline
        let gemvPSO = gemvKernel.f32Pipeline
        let fusedQ8PSO = fusedQ8GemvPipeline
        let fusedQ8TiledPSO = fusedQ8GemvTiledPipeline
        let ropePSO = ropeKernel.pipelineF32
        let gqaPSO = gqaKernel.pipelineF16KV
        let swigluPSO = activationKernels.swigluPipeline
        let addPSO = addPipeline
        let convertF32ToF16PSO = convertF32ToF16Pipeline
        let halfStride = MemoryLayout<Float16>.stride
        let rmsEps = Float(config.rmsNormEpsilon)
        let ropeTheta = Float(config.ropeFreqBase)
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let gqaBlockSize = GQAKernel.blockSize
        let allVBuf = scratch.allV

        for layerIndex in 0..<config.layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB

            // 1+2. FUSED RMSNorm + Q/K/V projections (saves 1 dispatch per layer)
            let useQ8Fused = lw.wqRaw != nil
            let blocksPerRowDim = dim / 32  // Q8_0: 32 elements per block
            let layerVCache = layerVCaches[layerIndex]
            let layerKCache = layerKCaches[layerIndex]

            if useQ8Fused && seqLen == 1 {
                // Fused RMSNorm + Q+K+V: RMSNorm is computed inline, no separate dispatch.
                let cacheWriteOffF16 = startPosition * kvDim * halfStride
                let fusedQKVPSO = fusedQKVPipeline
                enc.setComputePipelineState(fusedQKVPSO)
                enc.setBuffer(lw.wqRaw!, offset: 0, index: 0)
                enc.setBuffer(lw.wkRaw!, offset: 0, index: 1)
                enc.setBuffer(lw.wvRaw!, offset: 0, index: 2)
                enc.setBuffer(currentHidden, offset: 0, index: 3)  // raw hidden (RMSNorm applied inline)
                enc.setBuffer(allQBuf, offset: 0, index: 4)
                enc.setBuffer(allKBuf, offset: 0, index: 5)
                enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
                var qkvP = FusedQKVParams(qRows: UInt32(qDim), kvRows: UInt32(kvDim),
                    cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
                enc.setBytes(&qkvP, length: MemoryLayout<FusedQKVParams>.stride, index: 7)
                enc.setBuffer(lw.attnNorm, offset: 0, index: 8)  // RMSNorm weight
                let totalQKVRows = qDim + kvDim + kvDim
                enc.dispatchThreadgroups(MTLSize(width: (totalQKVRows + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate RMSNorm + QKV (for seqLen > 1 or non-Q8 weights)
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
                for t in 0..<seqLen {
                    let tokOff = t * dim * floatStride
                    let qOff = t * qDim * floatStride
                    let kvOff = t * kvDim * floatStride
                    let cacheWriteOffF16 = (t + startPosition) * kvDim * halfStride

                    if useQ8Fused {
                        enc.setComputePipelineState(fusedQ8PSO)
                        var qP = ERDequantGEMVParams(rows: UInt32(qDim), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim))
                        enc.setBuffer(lw.wqRaw!, offset: 0, index: 0)
                        enc.setBuffer(normedBuf, offset: tokOff, index: 1)
                        enc.setBuffer(allQBuf, offset: qOff, index: 2)
                        enc.setBytes(&qP, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (qDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        var kvP = ERDequantGEMVParams(rows: UInt32(kvDim), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim))
                        enc.setBuffer(lw.wkRaw!, offset: 0, index: 0)
                        enc.setBuffer(allKBuf, offset: kvOff, index: 2)
                        enc.setBytes(&kvP, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.setBuffer(lw.wvRaw!, offset: 0, index: 0)
                        enc.setBuffer(allVBuf, offset: kvOff, index: 2)
                        enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else {
                        enc.setComputePipelineState(gemvPSO)
                        var qP = ERGEMVParams(M: UInt32(qDim), K: UInt32(dim), lda: UInt32(dim))
                        enc.setBuffer(lw.wq, offset: 0, index: 0)
                        enc.setBuffer(normedBuf, offset: tokOff, index: 1)
                        enc.setBuffer(allQBuf, offset: qOff, index: 2)
                        enc.setBytes(&qP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (qDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        var kvP = ERGEMVParams(M: UInt32(kvDim), K: UInt32(dim), lda: UInt32(dim))
                        enc.setBuffer(lw.wk, offset: 0, index: 0)
                        enc.setBuffer(allKBuf, offset: kvOff, index: 2)
                        enc.setBytes(&kvP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        enc.setBuffer(lw.wv, offset: 0, index: 0)
                        enc.setBuffer(allVBuf, offset: kvOff, index: 2)
                        enc.setBytes(&kvP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }

                    // Convert V from f32 (allVBuf) → f16 (layerVCache)
                    enc.setComputePipelineState(convertF32ToF16PSO)
                    enc.setBuffer(allVBuf, offset: kvOff, index: 0)
                    enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 1)
                    var convCount = ERElementwiseParams(elementCount: UInt32(kvDim))
                    enc.setBytes(&convCount, length: MemoryLayout<ERElementwiseParams>.stride, index: 2)
                    enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(kvDim, convertF32ToF16PSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                // Convert V from f32 → f16 not needed for fused path (writes f16 directly)
            }

            // 3. FUSED Q/K per-head norm + RoPE Q + RoPE K→f16 (replaces 4-6 dispatches with 1)
            let hasQKNorm = lw.qNorm != nil && lw.kNorm != nil
            let ropeQOut: MTLBuffer  // used by GQA step below
            if hasQKNorm && seqLen == 1 {
                // MEGA-FUSED: Q/K norm + RoPE + GQA all in one dispatch
                let megaPSO = fusedNormRoPEGQAPipeline
                enc.setComputePipelineState(megaPSO)
                enc.setBuffer(allQBuf, offset: 0, index: 0)
                enc.setBuffer(allKBuf, offset: 0, index: 1)
                enc.setBuffer(lw.qNorm!, offset: 0, index: 2)
                enc.setBuffer(lw.kNorm!, offset: 0, index: 3)
                enc.setBuffer(attnOutBuf, offset: 0, index: 4)     // attention output directly!
                enc.setBuffer(layerKCache, offset: 0, index: 5)
                enc.setBuffer(layerVCache, offset: 0, index: 6)
                var p = (UInt32(numHeads), UInt32(numKVHeads), UInt32(headDim),
                         UInt32(startPosition), ropeTheta, Float(1.0), rmsEps,
                         UInt32(1 + startPosition), 1.0 / sqrt(Float(headDim)))  // kvSeqLen includes cached prefix
                enc.setBytes(&p, length: 9 * 4, index: 7)
                let totalHeads = numHeads + numKVHeads
                // 32 threads per head (single simdgroup) — eliminates all threadgroup barriers
                enc.dispatchThreads(MTLSize(width: 32, height: totalHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(32, megaPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                ropeQOut = attnOutBuf  // mega-kernel writes attention output directly
            } else {
                // Fallback: separate Q/K norm + RoPE (for seqLen > 1 or no Q/K norm)
                if hasQKNorm {
                    // Q norm: rows = seqLen * numHeads, cols = headDim
                    do {
                        var p = ERRMSNormParams(rows: UInt32(seqLen * numHeads), cols: UInt32(headDim), eps: rmsEps)
                        enc.setComputePipelineState(rmsNormPSO)
                        enc.setBuffer(allQBuf, offset: 0, index: 0)
                        enc.setBuffer(lw.qNorm!, offset: 0, index: 1)
                        enc.setBuffer(ropeQBuf, offset: 0, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                        let totalRows = seqLen * numHeads
                        enc.dispatchThreads(MTLSize(width: totalRows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(totalRows, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                    // K norm: rows = seqLen * numKVHeads, cols = headDim
                    do {
                        var p = ERRMSNormParams(rows: UInt32(seqLen * numKVHeads), cols: UInt32(headDim), eps: rmsEps)
                        enc.setComputePipelineState(rmsNormPSO)
                        enc.setBuffer(allKBuf, offset: 0, index: 0)
                        enc.setBuffer(lw.kNorm!, offset: 0, index: 1)
                        enc.setBuffer(ropeKBuf, offset: 0, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                        let totalRows = seqLen * numKVHeads
                        enc.dispatchThreads(MTLSize(width: totalRows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(totalRows, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                let useNeoXRoPE = hasQKNorm
                let activeRopePSO = useNeoXRoPE ? ropeKernel.pipelineNeoX : ropePSO
                let ropeQIn = hasQKNorm ? ropeQBuf : allQBuf
                let ropeQOutLocal = hasQKNorm ? allQBuf : ropeQBuf
                let ropeKIn = hasQKNorm ? ropeKBuf : allKBuf
                // RoPE Q
                do {
                    var p = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numHeads),
                        headDim: UInt32(headDim), startPos: UInt32(startPosition), theta: ropeTheta, scalingFactor: 1)
                    enc.setComputePipelineState(activeRopePSO)
                    enc.setBuffer(ropeQIn, offset: 0, index: 0)
                    enc.setBuffer(ropeQOutLocal, offset: 0, index: 1)
                    enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                    let halfDim = headDim / 2
                    enc.dispatchThreads(MTLSize(width: halfDim, height: numHeads, depth: seqLen),
                        threadsPerThreadgroup: MTLSize(width: min(halfDim, activeRopePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                // RoPE K → f32 temp, then convert to f16 cache
                let ropeKOut = hasQKNorm ? allKBuf : ropeKBuf
                do {
                    var p = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numKVHeads),
                        headDim: UInt32(headDim), startPos: UInt32(startPosition), theta: ropeTheta, scalingFactor: 1)
                    enc.setBuffer(ropeKIn, offset: 0, index: 0)
                    enc.setBuffer(ropeKOut, offset: 0, index: 1)
                    enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                    let halfDim = headDim / 2
                    enc.dispatchThreads(MTLSize(width: halfDim, height: numKVHeads, depth: seqLen),
                        threadsPerThreadgroup: MTLSize(width: min(halfDim, activeRopePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                // Convert RoPE'd K from f32 → f16 (layerKCache) for each position
                for t in 0..<seqLen {
                    let srcOff = t * kvDim * floatStride
                    let dstOff = (t + startPosition) * kvDim * halfStride
                    enc.setComputePipelineState(convertF32ToF16PSO)
                    enc.setBuffer(ropeKOut, offset: srcOff, index: 0)
                    enc.setBuffer(layerKCache, offset: dstOff, index: 1)
                    var convCount = ERElementwiseParams(elementCount: UInt32(kvDim))
                    enc.setBytes(&convCount, length: MemoryLayout<ERElementwiseParams>.stride, index: 2)
                    enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(kvDim, convertF32ToF16PSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                ropeQOut = hasQKNorm ? allQBuf : ropeQBuf
            }

            // 4. GQA (data is in [S,H,D] layout)
            //    Skip GQA if mega-kernel already computed attention (seqLen==1 && hasQKNorm)
            let megaKernelUsed = hasQKNorm && seqLen == 1
            if !megaKernelUsed {
                let groupSize = numHeads / numKVHeads
                let totalKVSeqLen = seqLen + startPosition
                var p = ERGQAParams(seqLen: UInt32(seqLen), headDim: UInt32(headDim),
                    numHeads: UInt32(numHeads), numKVHeads: UInt32(numKVHeads),
                    groupSize: UInt32(groupSize), scale: 1.0 / sqrt(Float(headDim)),
                    causal: 1, kvBlockSize: UInt32(gqaBlockSize), qBlockSize: UInt32(gqaBlockSize),
                    kvSeqLen: UInt32(totalKVSeqLen), qOffset: UInt32(startPosition))
                enc.setComputePipelineState(gqaPSO)
                enc.setBuffer(ropeQOut, offset: 0, index: 0)
                enc.setBuffer(layerKCache, offset: 0, index: 1)
                enc.setBuffer(layerVCache, offset: 0, index: 2)
                enc.setBuffer(attnOutBuf, offset: 0, index: 3)
                enc.setBytes(&p, length: MemoryLayout<ERGQAParams>.stride, index: 4)
                let qBlockCount = (seqLen + gqaBlockSize - 1) / gqaBlockSize
                enc.dispatchThreadgroups(MTLSize(width: qBlockCount, height: numHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: gqaBlockSize, height: 1, depth: 1))
            }

            // 5+6. Fused output projection + residual add (saves 1 dispatch for seqLen=1)
            let blocksPerRowQDim = qDim / 32
            if useQ8Fused && seqLen == 1, let woRaw = lw.woRaw {
                let gemvAddPSO = gemvAddPipeline
                enc.setComputePipelineState(gemvAddPSO)
                var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(qDim), blocksPerRow: UInt32(blocksPerRowQDim))
                enc.setBuffer(woRaw, offset: 0, index: 0)
                enc.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc.setBuffer(currentHidden, offset: 0, index: 2)  // residual
                enc.setBuffer(afterAttnBuf, offset: 0, index: 3)   // output
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate projection + add (for seqLen > 1 or non-Q8)
                for t in 0..<seqLen {
                    if useQ8Fused, let woRaw = lw.woRaw {
                        // Use tiled kernel for better memory coalescing in decode path
                        let useTiled = seqLen == 1
                        enc.setComputePipelineState(useTiled ? fusedQ8TiledPSO : fusedQ8PSO)
                        var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(qDim), blocksPerRow: UInt32(blocksPerRowQDim))
                        enc.setBuffer(woRaw, offset: 0, index: 0)
                        enc.setBuffer(attnOutBuf, offset: t * qDim * floatStride, index: 1)
                        enc.setBuffer(projBuf, offset: t * dim * floatStride, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else {
                        enc.setComputePipelineState(gemvPSO)
                        var p = ERGEMVParams(M: UInt32(dim), K: UInt32(qDim), lda: UInt32(qDim))
                        enc.setBuffer(lw.wo, offset: 0, index: 0)
                        enc.setBuffer(attnOutBuf, offset: t * qDim * floatStride, index: 1)
                        enc.setBuffer(projBuf, offset: t * dim * floatStride, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
                // Residual add (attention)
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
            }

            // 7+8+9. FUSED RMSNorm + Gate + Up + SwiGLU (saves 3 dispatches for seqLen=1)
            if useQ8Fused && seqLen == 1, let gateRaw = lw.gateRaw, let upRaw = lw.upRaw {
                let fusedGUSPSO = fusedGateUpSiluPipeline
                enc.setComputePipelineState(fusedGUSPSO)
                enc.setBuffer(gateRaw, offset: 0, index: 0)
                enc.setBuffer(upRaw, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)  // raw input (RMSNorm applied inline)
                enc.setBuffer(activBuf, offset: 0, index: 3)
                var fusedP = FusedGateUpSiluParams(rows: UInt32(interDim), cols: UInt32(dim),
                    blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
                enc.setBytes(&fusedP, length: MemoryLayout<FusedGateUpSiluParams>.stride, index: 4)
                enc.setBuffer(lw.ffnNorm, offset: 0, index: 5)  // RMSNorm weight
                enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate RMSNorm + Gate + Up + SwiGLU (for seqLen > 1 or non-Q8)
                // RMSNorm (FFN)
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
                // Gate + Up projections
                for t in 0..<seqLen {
                    let tokOff = t * dim * floatStride
                    let intOff = t * interDim * floatStride
                    if useQ8Fused, let gateRaw = lw.gateRaw, let upRaw = lw.upRaw {
                        // Use tiled kernel for better memory coalescing in decode path
                        let useTiled = seqLen == 1
                        enc.setComputePipelineState(useTiled ? fusedQ8TiledPSO : fusedQ8PSO)
                        var p = ERDequantGEMVParams(rows: UInt32(interDim), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim))
                        enc.setBuffer(gateRaw, offset: 0, index: 0)
                        enc.setBuffer(ffnNormedBuf, offset: tokOff, index: 1)
                        enc.setBuffer(gateOutBuf, offset: intOff, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.setBuffer(upRaw, offset: 0, index: 0)
                        enc.setBuffer(upOutBuf, offset: intOff, index: 2)
                        enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else {
                        enc.setComputePipelineState(gemvPSO)
                        var p = ERGEMVParams(M: UInt32(interDim), K: UInt32(dim), lda: UInt32(dim))
                        enc.setBuffer(lw.gate, offset: 0, index: 0)
                        enc.setBuffer(ffnNormedBuf, offset: tokOff, index: 1)
                        enc.setBuffer(gateOutBuf, offset: intOff, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        enc.setBuffer(lw.up, offset: 0, index: 0)
                        enc.setBuffer(upOutBuf, offset: intOff, index: 2)
                        enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
                // SwiGLU
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
            }

            // 10+11. Fused down projection + residual add (saves 1 dispatch for seqLen=1)
            let blocksPerRowInterDim = interDim / 32
            if useQ8Fused && seqLen == 1, let downRaw = lw.downRaw {
                let gemvAddPSO = gemvAddPipeline
                enc.setComputePipelineState(gemvAddPSO)
                var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(interDim), blocksPerRow: UInt32(blocksPerRowInterDim))
                enc.setBuffer(downRaw, offset: 0, index: 0)
                enc.setBuffer(activBuf, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)   // residual
                enc.setBuffer(layerOutputBuf, offset: 0, index: 3)  // output
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate down + add (for seqLen > 1 or non-Q8)
                for t in 0..<seqLen {
                    if useQ8Fused, let downRaw = lw.downRaw {
                        // Use tiled kernel for better memory coalescing in decode path
                        let useTiled = seqLen == 1
                        enc.setComputePipelineState(useTiled ? fusedQ8TiledPSO : fusedQ8PSO)
                        var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(interDim), blocksPerRow: UInt32(blocksPerRowInterDim))
                        enc.setBuffer(downRaw, offset: 0, index: 0)
                        enc.setBuffer(activBuf, offset: t * interDim * floatStride, index: 1)
                        enc.setBuffer(downOutBuf, offset: t * dim * floatStride, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else {
                        enc.setComputePipelineState(gemvPSO)
                        var p = ERGEMVParams(M: UInt32(dim), K: UInt32(interDim), lda: UInt32(interDim))
                        enc.setBuffer(lw.down, offset: 0, index: 0)
                        enc.setBuffer(activBuf, offset: t * interDim * floatStride, index: 1)
                        enc.setBuffer(downOutBuf, offset: t * dim * floatStride, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                        enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
                // Residual add (FFN)
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
            }

            currentHidden = layerOutputBuf
        }

        let logitsBuf = scratch.logits
        let lastHiddenOff = (seqLen - 1) * dim * floatStride
        if let lmRaw = preloadedWeights.lmHeadRaw {
            let blocksPerRow = preloadedWeights.lmHeadCols / 32
            var p = FusedGateUpSiluParams(
                rows: UInt32(config.vocabSize),
                cols: UInt32(dim),
                blocksPerRow: UInt32(blocksPerRow),
                rmsEps: rmsEps
            )
            enc.setComputePipelineState(fusedFinalNormGemvPSO)
            enc.setBuffer(lmRaw, offset: 0, index: 0)
            enc.setBuffer(currentHidden, offset: lastHiddenOff, index: 1)
            enc.setBuffer(logitsBuf, offset: 0, index: 2)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 3)
            enc.setBytes(&p, length: MemoryLayout<FusedGateUpSiluParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            let finalOutputBuf = scratch.finalOut
            var normP = ERRMSNormParams(rows: UInt32(seqLen), cols: UInt32(dim), eps: rmsEps)
            enc.setComputePipelineState(rmsNormPSO)
            enc.setBuffer(currentHidden, offset: 0, index: 0)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 1)
            enc.setBuffer(finalOutputBuf, offset: 0, index: 2)
            enc.setBytes(&normP, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
            enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(seqLen, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            let lmHeadBuf = preloadedWeights.lmHead!
            enc.setComputePipelineState(gemvPSO)
            var p = ERGEMVParams(M: UInt32(config.vocabSize), K: UInt32(dim), lda: UInt32(dim))
            enc.setBuffer(lmHeadBuf, offset: 0, index: 0)
            enc.setBuffer(finalOutputBuf, offset: lastHiddenOff, index: 1)
            enc.setBuffer(logitsBuf, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        enc.endEncoding()

        // ONE sync point for the ENTIRE forward pass (layers + norm + LM head)!
        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        return logitsBuf
    }

    // MARK: - Fully Fused Decode Pass (single token with KV cache)

    /// Process a SINGLE new token through all transformer layers using KV cache.
    /// K/V for all previous positions are read from per-layer cache buffers.
    /// New K/V for the current position are written to the cache.
    /// Returns MTLBuffer containing vocab-sized logits.
    private func fusedDecodePass(
        hiddenBuf: MTLBuffer,
        currentPos: Int
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim

        let interDim = config.intermediateDim
        let totalKVLen = currentPos + 1  // total positions including the new token

        var currentHidden = hiddenBuf

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }

        // Reuse scratch buffers — seqLen=1 so only first element needed
        let normedBuf = scratch.normed
        let afterAttnBuf = scratch.afterAttn
        let ffnNormedBuf = scratch.ffnNormed
        let outputBufA = scratch.outputA
        let outputBufB = scratch.outputB
        let allQBuf = scratch.allQ
        let allKBuf = scratch.allK
        let ropeQBuf = scratch.ropeQ
        let attnOutBuf = scratch.attnOut
        let projBuf = scratch.proj
        let gateOutBuf = scratch.gateOut
        let upOutBuf = scratch.upOut
        let activBuf = scratch.activ
        let downOutBuf = scratch.downOut

        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }

        let rmsNormPSO = rmsNormKernel.pipeline
        let fusedFinalNormGemvPSO = fusedFinalNormGemvPipeline
        _ = fusedQ8GemvPipeline  // Unused: we use tiled variant for decode
        let fusedQ8TiledPSO = fusedQ8GemvTiledPipeline
        let gemvPSO = gemvKernel.f32Pipeline
        let gqaPSO = gqaKernel.pipelineF16KV
        let swigluPSO = activationKernels.swigluPipeline
        let addPSO = addPipeline
        let convertF32ToF16PSO = convertF32ToF16Pipeline
        let halfStride = MemoryLayout<Float16>.stride
        let rmsEps = Float(config.rmsNormEpsilon)
        let ropeTheta = Float(config.ropeFreqBase)
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let gqaBlockSize = GQAKernel.blockSize
        let allVBuf = scratch.allV

        for layerIndex in 0..<config.layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB
            let layerKCache = layerKCaches[layerIndex]
            let layerVCache = layerVCaches[layerIndex]
            let cacheWriteOffF16 = currentPos * kvDim * halfStride

            // 1+2. FUSED RMSNorm + Q/K/V projections (saves 1 dispatch per layer)
            let useQ8Fused = lw.wqRaw != nil
            let blocksPerRowDim = dim / 32

            if useQ8Fused {
                // Fused RMSNorm + Q+K+V: RMSNorm is computed inline, no separate dispatch.
                let fusedQKVPSO = fusedQKVPipeline
                enc.setComputePipelineState(fusedQKVPSO)
                enc.setBuffer(lw.wqRaw!, offset: 0, index: 0)
                enc.setBuffer(lw.wkRaw!, offset: 0, index: 1)
                enc.setBuffer(lw.wvRaw!, offset: 0, index: 2)
                enc.setBuffer(currentHidden, offset: 0, index: 3)  // raw hidden (RMSNorm applied inline)
                enc.setBuffer(allQBuf, offset: 0, index: 4)
                enc.setBuffer(allKBuf, offset: 0, index: 5)
                enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
                var qkvP = FusedQKVParams(qRows: UInt32(qDim), kvRows: UInt32(kvDim),
                    cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
                enc.setBytes(&qkvP, length: MemoryLayout<FusedQKVParams>.stride, index: 7)
                enc.setBuffer(lw.attnNorm, offset: 0, index: 8)  // RMSNorm weight
                let totalQKVRows = qDim + kvDim + kvDim
                enc.dispatchThreadgroups(MTLSize(width: (totalQKVRows + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                enc.setComputePipelineState(gemvPSO)
                var qP = ERGEMVParams(M: UInt32(qDim), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lw.wq, offset: 0, index: 0)
                enc.setBuffer(normedBuf, offset: 0, index: 1)
                enc.setBuffer(allQBuf, offset: 0, index: 2)
                enc.setBytes(&qP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (qDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // K → allKBuf (will be RoPE'd to temp, then converted to f16 cache)
                var kvP = ERGEMVParams(M: UInt32(kvDim), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lw.wk, offset: 0, index: 0)
                enc.setBuffer(allKBuf, offset: 0, index: 2)
                enc.setBytes(&kvP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // V → f32 scratch (allVBuf), then convert to f16 cache
                enc.setBuffer(lw.wv, offset: 0, index: 0)
                enc.setBuffer(allVBuf, offset: 0, index: 2)
                enc.setBytes(&kvP, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (kvDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // Convert V from f32 → f16 only when using non-Q8 path (Q8 path writes f16 directly)
            if !useQ8Fused {
                enc.setComputePipelineState(convertF32ToF16PSO)
                enc.setBuffer(allVBuf, offset: 0, index: 0)
                enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 1)
                var vConvCount = ERElementwiseParams(elementCount: UInt32(kvDim))
                enc.setBytes(&vConvCount, length: MemoryLayout<ERElementwiseParams>.stride, index: 2)
                enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(kvDim, convertF32ToF16PSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // KV cache buffers disable hazard tracking, so decode must insert an explicit
            // barrier before GQA consumes the freshly-written K/V slices.
            if !decodeDebugOptions.disableKVCacheBarrier {
                enc.memoryBarrier(scope: .buffers)
            }

            // 3+4. MEGA-FUSED: Q/K norm + RoPE + GQA in SINGLE dispatch (saves 1 dispatch/layer)
            let hasQKNorm = lw.qNorm != nil && lw.kNorm != nil
            let useMegaKernel = hasQKNorm && !decodeDebugOptions.disableMegaKernel
            if useMegaKernel {
                let megaPSO = fusedNormRoPEGQAPipeline
                enc.setComputePipelineState(megaPSO)
                enc.setBuffer(allQBuf, offset: 0, index: 0)        // Q input
                enc.setBuffer(allKBuf, offset: 0, index: 1)        // K input
                enc.setBuffer(lw.qNorm!, offset: 0, index: 2)      // Q norm weight
                enc.setBuffer(lw.kNorm!, offset: 0, index: 3)      // K norm weight
                enc.setBuffer(attnOutBuf, offset: 0, index: 4)     // attention output (direct!)
                enc.setBuffer(layerKCache, offset: 0, index: 5)    // K cache (writes new K + reads all K)
                enc.setBuffer(layerVCache, offset: 0, index: 6)    // V cache (reads all V)
                var p = (UInt32(numHeads), UInt32(numKVHeads), UInt32(headDim),
                         UInt32(currentPos), ropeTheta, Float(1.0), rmsEps,
                         UInt32(totalKVLen), 1.0 / sqrt(Float(headDim)))
                enc.setBytes(&p, length: 9 * 4, index: 7)
                let totalHeads = numHeads + numKVHeads
                // 32 threads per head (single simdgroup) — eliminates all threadgroup barriers
                enc.dispatchThreads(MTLSize(width: 32, height: totalHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(32, megaPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            } else {
                // Fallback path used for models without Q/K norm, or when the mega decode
                // kernel is explicitly disabled during correctness bisects.
                let activeRopePSO = ropeKernel.pipelineNeoX
                let ropeKBuf = scratch.ropeK
                if hasQKNorm {
                    do {
                        var p = ERRMSNormParams(rows: UInt32(numHeads), cols: UInt32(headDim), eps: rmsEps)
                        enc.setComputePipelineState(rmsNormPSO)
                        enc.setBuffer(allQBuf, offset: 0, index: 0)
                        enc.setBuffer(lw.qNorm!, offset: 0, index: 1)
                        enc.setBuffer(ropeQBuf, offset: 0, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                        enc.dispatchThreads(MTLSize(width: numHeads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(numHeads, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                    do {
                        var p = ERRMSNormParams(rows: UInt32(numKVHeads), cols: UInt32(headDim), eps: rmsEps)
                        enc.setComputePipelineState(rmsNormPSO)
                        enc.setBuffer(allKBuf, offset: 0, index: 0)
                        enc.setBuffer(lw.kNorm!, offset: 0, index: 1)
                        enc.setBuffer(ropeKBuf, offset: 0, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
                        enc.dispatchThreads(MTLSize(width: numKVHeads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(numKVHeads, rmsNormPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                let ropeQInput = hasQKNorm ? ropeQBuf : allQBuf
                let ropeQOutput = hasQKNorm ? allQBuf : ropeQBuf
                let ropeKInput = hasQKNorm ? ropeKBuf : allKBuf
                do {
                    var p = ERRoPEParams(seqLen: 1, numHeads: UInt32(numHeads),
                        headDim: UInt32(headDim), startPos: UInt32(currentPos), theta: ropeTheta, scalingFactor: 1)
                    enc.setComputePipelineState(activeRopePSO)
                    enc.setBuffer(ropeQInput, offset: 0, index: 0)
                    enc.setBuffer(ropeQOutput, offset: 0, index: 1)
                    enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                    let halfDim = headDim / 2
                    enc.dispatchThreads(MTLSize(width: halfDim, height: numHeads, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(halfDim, activeRopePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                do {
                    var p = ERRoPEParams(seqLen: 1, numHeads: UInt32(numKVHeads),
                        headDim: UInt32(headDim), startPos: UInt32(currentPos), theta: ropeTheta, scalingFactor: 1)
                    enc.setComputePipelineState(ropeNeoXF16OutPipeline)
                    enc.setBuffer(ropeKInput, offset: 0, index: 0)
                    enc.setBuffer(layerKCache, offset: cacheWriteOffF16, index: 1)
                    enc.setBytes(&p, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
                    let halfDim = headDim / 2
                    enc.dispatchThreads(MTLSize(width: halfDim, height: numKVHeads, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(halfDim, activeRopePSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                if !decodeDebugOptions.disableKVCacheBarrier {
                    enc.memoryBarrier(scope: .buffers)
                }
                do {
                    let groupSize = numHeads / numKVHeads
                    var p = ERGQAParams(seqLen: 1, headDim: UInt32(headDim),
                        numHeads: UInt32(numHeads), numKVHeads: UInt32(numKVHeads),
                        groupSize: UInt32(groupSize), scale: 1.0 / sqrt(Float(headDim)),
                        causal: 1, kvBlockSize: UInt32(gqaBlockSize), qBlockSize: UInt32(gqaBlockSize),
                        kvSeqLen: UInt32(totalKVLen), qOffset: UInt32(currentPos))
                    enc.setComputePipelineState(gqaPSO)
                    enc.setBuffer(ropeQOutput, offset: 0, index: 0)
                    enc.setBuffer(layerKCache, offset: 0, index: 1)
                    enc.setBuffer(layerVCache, offset: 0, index: 2)
                    enc.setBuffer(attnOutBuf, offset: 0, index: 3)
                    enc.setBytes(&p, length: MemoryLayout<ERGQAParams>.stride, index: 4)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: numHeads, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: gqaBlockSize, height: 1, depth: 1))
                }
            }

            // 5+6. Fused output projection + residual add (saves 1 dispatch)
            //       afterAttn[i] = sum_k Wo[i,k]*attn[k] + hidden[i]
            let blocksPerRowQDim = qDim / 32
            if useQ8Fused, let woRaw = lw.woRaw {
                let gemvAddPSO = gemvAddPipeline
                enc.setComputePipelineState(gemvAddPSO)
                var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(qDim), blocksPerRow: UInt32(blocksPerRowQDim))
                enc.setBuffer(woRaw, offset: 0, index: 0)
                enc.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc.setBuffer(currentHidden, offset: 0, index: 2)  // residual
                enc.setBuffer(afterAttnBuf, offset: 0, index: 3)   // output
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate projection + add
                enc.setComputePipelineState(gemvPSO)
                var p = ERGEMVParams(M: UInt32(dim), K: UInt32(qDim), lda: UInt32(qDim))
                enc.setBuffer(lw.wo, offset: 0, index: 0)
                enc.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc.setBuffer(projBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                var addP = ERElementwiseParams(elementCount: UInt32(dim))
                enc.setComputePipelineState(addPSO)
                enc.setBuffer(currentHidden, offset: 0, index: 0)
                enc.setBuffer(projBuf, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)
                enc.setBytes(&addP, length: MemoryLayout<ERElementwiseParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(dim, addPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 7+8+9. FUSED RMSNorm + Gate + Up + SwiGLU (saves 1 dispatch per layer)
            if useQ8Fused, let gateRaw = lw.gateRaw, let upRaw = lw.upRaw {
                let fusedGUSPSO = fusedGateUpSiluPipeline
                enc.setComputePipelineState(fusedGUSPSO)
                enc.setBuffer(gateRaw, offset: 0, index: 0)
                enc.setBuffer(upRaw, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)  // raw input (RMSNorm applied inline)
                enc.setBuffer(activBuf, offset: 0, index: 3)
                var p = FusedGateUpSiluParams(rows: UInt32(interDim), cols: UInt32(dim),
                    blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
                enc.setBytes(&p, length: MemoryLayout<FusedGateUpSiluParams>.stride, index: 4)
                enc.setBuffer(lw.ffnNorm, offset: 0, index: 5)  // RMSNorm weight
                enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate gate + up + SwiGLU
                enc.setComputePipelineState(gemvPSO)
                var p = ERGEMVParams(M: UInt32(interDim), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lw.gate, offset: 0, index: 0)
                enc.setBuffer(ffnNormedBuf, offset: 0, index: 1)
                enc.setBuffer(gateOutBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.setBuffer(lw.up, offset: 0, index: 0)
                enc.setBuffer(upOutBuf, offset: 0, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // SwiGLU
                var sp = ERActivationParams(count: UInt32(interDim))
                enc.setComputePipelineState(swigluPSO)
                enc.setBuffer(gateOutBuf, offset: 0, index: 0)
                enc.setBuffer(upOutBuf, offset: 0, index: 1)
                enc.setBuffer(activBuf, offset: 0, index: 2)
                enc.setBytes(&sp, length: MemoryLayout<ERActivationParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: interDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(interDim, swigluPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            // 10+11. Fused down projection + residual add (saves 1 dispatch)
            //        layerOutput[i] = sum_k Wd[i,k]*activated[k] + afterAttn[i]
            let blocksPerRowInterDim = interDim / 32
            if useQ8Fused, let downRaw = lw.downRaw {
                let gemvAddPSO = gemvAddPipeline
                enc.setComputePipelineState(gemvAddPSO)
                var p = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(interDim), blocksPerRow: UInt32(blocksPerRowInterDim))
                enc.setBuffer(downRaw, offset: 0, index: 0)
                enc.setBuffer(activBuf, offset: 0, index: 1)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 2)   // residual
                enc.setBuffer(layerOutputBuf, offset: 0, index: 3)  // output
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Fallback: separate down + add
                enc.setComputePipelineState(gemvPSO)
                var p = ERGEMVParams(M: UInt32(dim), K: UInt32(interDim), lda: UInt32(interDim))
                enc.setBuffer(lw.down, offset: 0, index: 0)
                enc.setBuffer(activBuf, offset: 0, index: 1)
                enc.setBuffer(downOutBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (dim + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                var addP = ERElementwiseParams(elementCount: UInt32(dim))
                enc.setComputePipelineState(addPSO)
                enc.setBuffer(afterAttnBuf, offset: 0, index: 0)
                enc.setBuffer(downOutBuf, offset: 0, index: 1)
                enc.setBuffer(layerOutputBuf, offset: 0, index: 2)
                enc.setBytes(&addP, length: MemoryLayout<ERElementwiseParams>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(dim, addPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            }

            currentHidden = layerOutputBuf
        }

        let logitsBuf = scratch.logits
        if let lmRaw = preloadedWeights.lmHeadRaw, !decodeDebugOptions.disableFusedFinalNormLMHead {
            let blocksPerRow = preloadedWeights.lmHeadCols / 32
            var p = FusedGateUpSiluParams(
                rows: UInt32(config.vocabSize),
                cols: UInt32(dim),
                blocksPerRow: UInt32(blocksPerRow),
                rmsEps: rmsEps
            )
            enc.setComputePipelineState(fusedFinalNormGemvPSO)
            enc.setBuffer(lmRaw, offset: 0, index: 0)
            enc.setBuffer(currentHidden, offset: 0, index: 1)
            enc.setBuffer(logitsBuf, offset: 0, index: 2)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 3)
            enc.setBytes(&p, length: MemoryLayout<FusedGateUpSiluParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            let finalOutputBuf = scratch.finalOut
            var normP = ERRMSNormParams(rows: 1, cols: UInt32(dim), eps: rmsEps)
            enc.setComputePipelineState(rmsNormPSO)
            enc.setBuffer(currentHidden, offset: 0, index: 0)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 1)
            enc.setBuffer(finalOutputBuf, offset: 0, index: 2)
            enc.setBytes(&normP, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

            if let lmRaw = preloadedWeights.lmHeadRaw {
                let blocksPerRow = preloadedWeights.lmHeadCols / 32
                var p = ERDequantGEMVParams(rows: UInt32(config.vocabSize), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRow))
                // Use tiled kernel for LM head projection (always seqLen == 1 in decode)
                enc.setComputePipelineState(fusedQ8TiledPSO)
                enc.setBuffer(lmRaw, offset: 0, index: 0)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                let lmHeadBuf = preloadedWeights.lmHead!
                enc.setComputePipelineState(gemvPSO)
                var p = ERGEMVParams(M: UInt32(config.vocabSize), K: UInt32(dim), lda: UInt32(dim))
                enc.setBuffer(lmHeadBuf, offset: 0, index: 0)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }

        enc.endEncoding()

        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        return logitsBuf
    }

    // MARK: - Optimized Metal 3 Decode Pass (Params Buffer)

    /// Optimized Metal 3 decode using pre-allocated params buffer to eliminate setBytes.
    /// Constant params written once; only per-call varying params updated each decode.
    private func fusedDecodePassOpt(
        hiddenBuf: MTLBuffer,
        currentPos: Int,
        paramsBuffer: MTLBuffer
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let totalKVLen = currentPos + 1
        let halfStride = MemoryLayout<Float16>.stride
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let rmsEps = Float(config.rmsNormEpsilon)
        let ropeTheta = Float(config.ropeFreqBase)
        let blocksPerRowDim = dim / 32
        let layerCount = config.layerCount

        var currentHidden = hiddenBuf
        let afterAttnBuf = scratch.afterAttn
        let outputBufA = scratch.outputA
        let outputBufB = scratch.outputB
        let allQBuf = scratch.allQ
        let allKBuf = scratch.allK
        let attnOutBuf = scratch.attnOut
        let activBuf = scratch.activ

        let fusedQKVPSO = fusedQKVPipeline
        let megaPSO = fusedNormRoPEGQAPipeline
        let gemvAddPSO = gemvAddPipeline
        let fusedGUSPSO = fusedGateUpSiluPipeline
        let fusedFinalNormGemvPSO = fusedFinalNormGemvPipeline
        let rmsNormPSO = rmsNormKernel.pipeline
        let gemvPSO = gemvKernel.f32Pipeline

        let paramsBase = paramsBuffer.contents()

        let qkvGridWidth = (qDim + kvDim + kvDim + 1) / 2
        let megaThreadsPerHead = 32  // single simdgroup — zero barriers
        let megaTotalHeads = numHeads + numKVHeads
        let megaTGWidth = min(megaThreadsPerHead, megaPSO.maxTotalThreadsPerThreadgroup)
        let dimGridWidth = (dim + 1) / 2
        let interDimGridWidth = (interDim + 1) / 2
        let blocksPerRowQDim = qDim / 32
        let blocksPerRowInterDim = interDim / 32
        let cacheWriteOffF16 = currentPos * kvDim * halfStride

        // Write constant params on first call (slot 0 == 0 means uninitialized)
        let needsInit = paramsBase.load(as: UInt32.self) == 0
        if needsInit {
            var qkvP = FusedQKVParams(qRows: UInt32(qDim), kvRows: UInt32(kvDim), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
            memcpy(paramsBase, &qkvP, MemoryLayout<FusedQKVParams>.stride)
            var woP = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(qDim), blocksPerRow: UInt32(blocksPerRowQDim))
            memcpy(paramsBase + 512, &woP, MemoryLayout<ERDequantGEMVParams>.stride)
            var gusP = FusedGateUpSiluParams(rows: UInt32(interDim), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps)
            memcpy(paramsBase + 768, &gusP, MemoryLayout<FusedGateUpSiluParams>.stride)
            var downP = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(interDim), blocksPerRow: UInt32(blocksPerRowInterDim))
            memcpy(paramsBase + 1024, &downP, MemoryLayout<ERDequantGEMVParams>.stride)
            var normP = ERRMSNormParams(rows: 1, cols: UInt32(dim), eps: rmsEps)
            memcpy(paramsBase + 1280, &normP, MemoryLayout<ERRMSNormParams>.stride)
            if preloadedWeights.lmHeadRaw != nil {
                let bpr = preloadedWeights.lmHeadCols / 32
                var lmP = ERDequantGEMVParams(rows: UInt32(config.vocabSize), cols: UInt32(dim), blocksPerRow: UInt32(bpr))
                memcpy(paramsBase + 1536, &lmP, MemoryLayout<ERDequantGEMVParams>.stride)
                var fusedLmP = FusedGateUpSiluParams(
                    rows: UInt32(config.vocabSize),
                    cols: UInt32(dim),
                    blocksPerRow: UInt32(bpr),
                    rmsEps: rmsEps
                )
                memcpy(paramsBase + 1792, &fusedLmP, MemoryLayout<FusedGateUpSiluParams>.stride)
            } else {
                var lmP = ERGEMVParams(M: UInt32(config.vocabSize), K: UInt32(dim), lda: UInt32(dim))
                memcpy(paramsBase + 1536, &lmP, MemoryLayout<ERGEMVParams>.stride)
            }
            var megaP = (UInt32(numHeads), UInt32(numKVHeads), UInt32(headDim), UInt32(0), ropeTheta, Float(1.0), rmsEps, UInt32(0), 1.0 / sqrt(Float(headDim)))
            memcpy(paramsBase + 256, &megaP, 9 * 4)
        }

        // Per-call: update mega-kernel startPos + kvSeqLen
        (paramsBase + 256 + 12).storeBytes(of: UInt32(currentPos), as: UInt32.self)
        (paramsBase + 256 + 28).storeBytes(of: UInt32(totalKVLen), as: UInt32.self)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }

        for layerIndex in 0..<layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB
            let layerKCache = layerKCaches[layerIndex]
            let layerVCache = layerVCaches[layerIndex]

            // DISPATCH 1: Fused QKV
            enc.setComputePipelineState(fusedQKVPSO)
            enc.setBuffer(lw.wqRaw!, offset: 0, index: 0)
            enc.setBuffer(lw.wkRaw!, offset: 0, index: 1)
            enc.setBuffer(lw.wvRaw!, offset: 0, index: 2)
            enc.setBuffer(currentHidden, offset: 0, index: 3)
            enc.setBuffer(allQBuf, offset: 0, index: 4)
            enc.setBuffer(allKBuf, offset: 0, index: 5)
            enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
            enc.setBuffer(paramsBuffer, offset: 0, index: 7)
            enc.setBuffer(lw.attnNorm, offset: 0, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: qkvGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // KV cache buffers disable hazard tracking, so decode must insert an explicit
            // barrier before GQA consumes the freshly-written K/V slices.
            if !decodeDebugOptions.disableKVCacheBarrier {
                enc.memoryBarrier(scope: .buffers)
            }

            // DISPATCH 2: Mega-kernel Q/K norm + RoPE + GQA
            enc.setComputePipelineState(megaPSO)
            enc.setBuffer(allQBuf, offset: 0, index: 0)
            enc.setBuffer(allKBuf, offset: 0, index: 1)
            enc.setBuffer(lw.qNorm!, offset: 0, index: 2)
            enc.setBuffer(lw.kNorm!, offset: 0, index: 3)
            enc.setBuffer(attnOutBuf, offset: 0, index: 4)
            enc.setBuffer(layerKCache, offset: 0, index: 5)
            enc.setBuffer(layerVCache, offset: 0, index: 6)
            enc.setBuffer(paramsBuffer, offset: 256, index: 7)
            enc.dispatchThreads(MTLSize(width: megaThreadsPerHead, height: megaTotalHeads, depth: 1), threadsPerThreadgroup: MTLSize(width: megaTGWidth, height: 1, depth: 1))

            // DISPATCH 3: Wo + residual add
            enc.setComputePipelineState(gemvAddPSO)
            enc.setBuffer(lw.woRaw!, offset: 0, index: 0)
            enc.setBuffer(attnOutBuf, offset: 0, index: 1)
            enc.setBuffer(currentHidden, offset: 0, index: 2)
            enc.setBuffer(afterAttnBuf, offset: 0, index: 3)
            enc.setBuffer(paramsBuffer, offset: 512, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: dimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // DISPATCH 4: Gate + Up + SiLU
            enc.setComputePipelineState(fusedGUSPSO)
            enc.setBuffer(lw.gateRaw!, offset: 0, index: 0)
            enc.setBuffer(lw.upRaw!, offset: 0, index: 1)
            enc.setBuffer(afterAttnBuf, offset: 0, index: 2)
            enc.setBuffer(activBuf, offset: 0, index: 3)
            enc.setBuffer(paramsBuffer, offset: 768, index: 4)
            enc.setBuffer(lw.ffnNorm, offset: 0, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: interDimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // DISPATCH 5: Down + residual add
            enc.setComputePipelineState(gemvAddPSO)
            enc.setBuffer(lw.downRaw!, offset: 0, index: 0)
            enc.setBuffer(activBuf, offset: 0, index: 1)
            enc.setBuffer(layerOutputBuf, offset: 0, index: 3)
            enc.setBuffer(paramsBuffer, offset: 1024, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: dimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            currentHidden = layerOutputBuf
        }

        let logitsBuf = scratch.logits
        if let lmRaw = preloadedWeights.lmHeadRaw, !decodeDebugOptions.disableFusedFinalNormLMHead {
            enc.setComputePipelineState(fusedFinalNormGemvPSO)
            enc.setBuffer(lmRaw, offset: 0, index: 0)
            enc.setBuffer(currentHidden, offset: 0, index: 1)
            enc.setBuffer(logitsBuf, offset: 0, index: 2)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 3)
            enc.setBuffer(paramsBuffer, offset: 1792, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            let finalOutputBuf = scratch.finalOut
            enc.setComputePipelineState(rmsNormPSO)
            enc.setBuffer(currentHidden, offset: 0, index: 0)
            enc.setBuffer(preloadedWeights.finalNorm!, offset: 0, index: 1)
            enc.setBuffer(finalOutputBuf, offset: 0, index: 2)
            enc.setBuffer(paramsBuffer, offset: 1280, index: 3)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

            if let lmRaw = preloadedWeights.lmHeadRaw {
                enc.setComputePipelineState(fusedQ8GemvPipeline)
                enc.setBuffer(lmRaw, offset: 0, index: 0)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBuffer(paramsBuffer, offset: 1536, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                enc.setComputePipelineState(gemvPSO)
                enc.setBuffer(preloadedWeights.lmHead!, offset: 0, index: 0)
                enc.setBuffer(finalOutputBuf, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBuffer(paramsBuffer, offset: 1536, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }

        enc.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        return logitsBuf
    }

    // MARK: - Metal 4 Fused Decode Pass (Argument Table Dispatch)

    /// Metal 4 decode pass using argument table binding for minimal per-dispatch overhead.
    ///
    /// Key optimizations over the Metal 3 path:
    /// - `setArgumentTable` called ONCE; Metal snapshots at dispatch time
    /// - Only CHANGED buffer addresses updated between dispatches via `setAddress`
    /// - Execution-only barriers (`MTL4VisibilityOptionNone`) — no cache flushes on unified memory
    /// - Pre-allocated params buffer with 256-byte aligned slots — no `setBytes` copies
    /// - Single MTL4ComputeCommandEncoder for entire forward pass
    @available(macOS 26.0, iOS 26.0, *)
    private func fusedDecodePassMetal4(
        hiddenBuf: MTLBuffer,
        currentPos: Int,
        state: Metal4State
    ) async throws -> MTLBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let totalKVLen = currentPos + 1
        let halfStride = MemoryLayout<Float16>.stride
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let rmsEps = Float(config.rmsNormEpsilon)
        let ropeTheta = Float(config.ropeFreqBase)
        let blocksPerRowDim = dim / 32
        let layerCount = config.layerCount

        var currentHidden = hiddenBuf

        // Scratch buffers (same as Metal 3)
        let afterAttnBuf = scratch.afterAttn
        let outputBufA = scratch.outputA
        let outputBufB = scratch.outputB
        let allQBuf = scratch.allQ
        let allKBuf = scratch.allK
        let attnOutBuf = scratch.attnOut
        let activBuf = scratch.activ

        // Pipeline states
        let fusedQKVPSO = fusedQKVPipeline
        let megaPSO = fusedNormRoPEGQAPipeline
        let gemvAddPSO = gemvAddPipeline
        let fusedGUSPSO = fusedGateUpSiluPipeline
        let rmsNormPSO = rmsNormKernel.pipeline
        _ = fusedQ8GemvPipeline  // Unused: we use tiled variant for decode
        let fusedQ8TiledPSO = fusedQ8GemvTiledPipeline
        let gemvPSO = gemvKernel.f32Pipeline

        // MTL4 command buffer setup
        let m4Queue = state.commandQueue
        let m4CmdBuf = state.commandBuffer
        let allocator = state.allocator
        let argTable = state.argumentTable
        let paramsBuffer = state.paramsBuffer

        allocator.reset()
        m4CmdBuf.beginCommandBuffer(allocator: allocator)
        m4CmdBuf.useResidencySet(state.residencySet)

        guard let enc = m4CmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTL4 compute encoder")
        }

        // Set argument table ONCE — Metal snapshots at each dispatch
        enc.setArgumentTable(argTable)

        // Pre-populate CONSTANT scratch buffer addresses (never change between layers)
        // Index 4 in QKV = allQ, Index 5 in QKV = allK
        let allQAddr = allQBuf.gpuAddress
        let allKAddr = allKBuf.gpuAddress
        let attnOutAddr = attnOutBuf.gpuAddress
        let afterAttnAddr = afterAttnBuf.gpuAddress
        let activAddr = activBuf.gpuAddress

        let paramsBase = paramsBuffer.contents()
        let paramsGPUBase = paramsBuffer.gpuAddress

        // === PRE-WRITE ALL PARAMS INTO RING BUFFER (slots 0-6) ===
        // Written ONCE per decode pass — zero memcpy inside the layer loop.

        // Slot 0: QKV params
        var qkvP = FusedQKVParams(
            qRows: UInt32(qDim), kvRows: UInt32(kvDim),
            cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps
        )
        let qkvParamsAddr = paramsGPUBase
        memcpy(paramsBase, &qkvP, MemoryLayout<FusedQKVParams>.stride)

        // Slot 1: Mega-kernel params
        var megaP = (
            UInt32(numHeads), UInt32(numKVHeads), UInt32(headDim),
            UInt32(currentPos), ropeTheta, Float(1.0), rmsEps,
            UInt32(totalKVLen), 1.0 / sqrt(Float(headDim))
        )
        let megaParamsAddr = paramsGPUBase + 256
        memcpy(paramsBase + 256, &megaP, 9 * 4)

        // Slot 2: Wo+add params
        let blocksPerRowQDim = qDim / 32
        var woP = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(qDim), blocksPerRow: UInt32(blocksPerRowQDim))
        let woParamsAddr = paramsGPUBase + 512
        memcpy(paramsBase + 512, &woP, MemoryLayout<ERDequantGEMVParams>.stride)

        // Slot 3: Gate+Up+SiLU params
        var gusP = FusedGateUpSiluParams(
            rows: UInt32(interDim), cols: UInt32(dim),
            blocksPerRow: UInt32(blocksPerRowDim), rmsEps: rmsEps
        )
        let gusParamsAddr = paramsGPUBase + 768
        memcpy(paramsBase + 768, &gusP, MemoryLayout<FusedGateUpSiluParams>.stride)

        // Slot 4: Down+add params
        let blocksPerRowInterDim = interDim / 32
        var downP = ERDequantGEMVParams(rows: UInt32(dim), cols: UInt32(interDim), blocksPerRow: UInt32(blocksPerRowInterDim))
        let downParamsAddr = paramsGPUBase + 1024
        memcpy(paramsBase + 1024, &downP, MemoryLayout<ERDequantGEMVParams>.stride)

        // Slot 5: Final RMSNorm params
        var normP = ERRMSNormParams(rows: 1, cols: UInt32(dim), eps: rmsEps)
        let normParamsAddr = paramsGPUBase + 1280
        memcpy(paramsBase + 1280, &normP, MemoryLayout<ERRMSNormParams>.stride)

        // Slot 6: LM head params
        let lmParamsAddr = paramsGPUBase + 1536
        if preloadedWeights.lmHeadRaw != nil {
            let blocksPerRow = preloadedWeights.lmHeadCols / 32
            var lmP = ERDequantGEMVParams(rows: UInt32(config.vocabSize), cols: UInt32(dim), blocksPerRow: UInt32(blocksPerRow))
            memcpy(paramsBase + 1536, &lmP, MemoryLayout<ERDequantGEMVParams>.stride)
        } else {
            var lmP = ERGEMVParams(M: UInt32(config.vocabSize), K: UInt32(dim), lda: UInt32(dim))
            memcpy(paramsBase + 1536, &lmP, MemoryLayout<ERGEMVParams>.stride)
        }

        // === PRE-COMPUTE DISPATCH SIZES (hoisted out of layer loop) ===
        let qkvGridWidth = (qDim + kvDim + kvDim + 1) / 2
        let megaThreadsPerHead = 32  // single simdgroup — zero barriers
        let megaTotalHeads = numHeads + numKVHeads
        let megaTGWidth = min(megaThreadsPerHead, megaPSO.maxTotalThreadsPerThreadgroup)
        let dimGridWidth = (dim + 1) / 2
        let interDimGridWidth = (interDim + 1) / 2
        let cacheWriteOffF16 = currentPos * kvDim * halfStride

        // === LAYER LOOP — MINIMAL setAddress PER DISPATCH ===
        // Strategy: argument table state persists between dispatches.
        // Each dispatch sets ONLY the indices whose values differ from
        // what the table currently holds. Constant scratch/param addresses
        // are still re-set because different pipeline dispatches reuse
        // the same slot indices for different purposes.

        for layerIndex in 0..<layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB
            let layerKCache = layerKCaches[layerIndex]
            let layerVCache = layerVCaches[layerIndex]

            let useQ8Fused = lw.wqRaw != nil
            let hasQKNorm = lw.qNorm != nil && lw.kNorm != nil

            // ---- DISPATCH 1: Fused QKV (indices 0-8) ----
            if useQ8Fused {
                enc.setComputePipelineState(fusedQKVPSO)
                argTable.setAddress(lw.wqRaw!.gpuAddress, index: 0)
                argTable.setAddress(lw.wkRaw!.gpuAddress, index: 1)
                argTable.setAddress(lw.wvRaw!.gpuAddress, index: 2)
                argTable.setAddress(currentHidden.gpuAddress, index: 3)
                argTable.setAddress(allQAddr, index: 4)
                argTable.setAddress(allKAddr, index: 5)
                argTable.setAddress(layerVCache.gpuAddress + UInt64(cacheWriteOffF16), index: 6)
                argTable.setAddress(qkvParamsAddr, index: 7)
                argTable.setAddress(lw.attnNorm.gpuAddress, index: 8)

                enc.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: qkvGridWidth, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }

            // ---- BARRIER (execution-only — no cache flush on unified memory) ----
            enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

            // ---- DISPATCH 2: Mega-kernel Q/K norm + RoPE + GQA (indices 0-7) ----
            if hasQKNorm {
                enc.setComputePipelineState(megaPSO)
                argTable.setAddress(allQAddr, index: 0)
                argTable.setAddress(allKAddr, index: 1)
                argTable.setAddress(lw.qNorm!.gpuAddress, index: 2)
                argTable.setAddress(lw.kNorm!.gpuAddress, index: 3)
                argTable.setAddress(attnOutAddr, index: 4)
                argTable.setAddress(layerKCache.gpuAddress, index: 5)
                argTable.setAddress(layerVCache.gpuAddress, index: 6)
                argTable.setAddress(megaParamsAddr, index: 7)

                enc.dispatchThreads(
                    threadsPerGrid: MTLSize(width: megaThreadsPerHead, height: megaTotalHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: megaTGWidth, height: 1, depth: 1)
                )
            }

            // ---- BARRIER ----
            enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

            // ---- DISPATCH 3: Wo + residual add (indices 0-4) ----
            if useQ8Fused, let woRaw = lw.woRaw {
                enc.setComputePipelineState(gemvAddPSO)
                argTable.setAddress(woRaw.gpuAddress, index: 0)
                argTable.setAddress(attnOutAddr, index: 1)
                argTable.setAddress(currentHidden.gpuAddress, index: 2)
                argTable.setAddress(afterAttnAddr, index: 3)
                argTable.setAddress(woParamsAddr, index: 4)

                enc.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: dimGridWidth, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }

            // ---- BARRIER ----
            enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

            // ---- DISPATCH 4: Gate + Up + SiLU (indices 0-5) ----
            if useQ8Fused, let gateRaw = lw.gateRaw, let upRaw = lw.upRaw {
                enc.setComputePipelineState(fusedGUSPSO)
                argTable.setAddress(gateRaw.gpuAddress, index: 0)
                argTable.setAddress(upRaw.gpuAddress, index: 1)
                argTable.setAddress(afterAttnAddr, index: 2)
                argTable.setAddress(activAddr, index: 3)
                argTable.setAddress(gusParamsAddr, index: 4)
                argTable.setAddress(lw.ffnNorm.gpuAddress, index: 5)

                enc.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: interDimGridWidth, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }

            // ---- BARRIER ----
            enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

            // ---- DISPATCH 5: Down + residual add (indices 0-4) ----
            // Note: index 2 (afterAttnAddr) unchanged from GUS dispatch — skip setAddress
            if useQ8Fused, let downRaw = lw.downRaw {
                enc.setComputePipelineState(gemvAddPSO)
                argTable.setAddress(downRaw.gpuAddress, index: 0)
                argTable.setAddress(activAddr, index: 1)
                // index 2 = afterAttnAddr — already set by GUS dispatch
                argTable.setAddress(layerOutputBuf.gpuAddress, index: 3)
                argTable.setAddress(downParamsAddr, index: 4)

                enc.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: dimGridWidth, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }

            // ---- BARRIER (before next layer or final norm) ----
            enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

            currentHidden = layerOutputBuf
        }

        // === FINAL NORM (single token) — params pre-written in slot 5 ===
        let finalOutputBuf = scratch.finalOut
        do {
            enc.setComputePipelineState(rmsNormPSO)
            argTable.setAddress(currentHidden.gpuAddress, index: 0)
            argTable.setAddress(preloadedWeights.finalNorm!.gpuAddress, index: 1)
            argTable.setAddress(finalOutputBuf.gpuAddress, index: 2)
            argTable.setAddress(normParamsAddr, index: 3)

            enc.dispatchThreads(
                threadsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
        }

        // ---- BARRIER before LM head ----
        enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

        // === LM HEAD (single token) — params pre-written in slot 6 ===
        let logitsBuf = scratch.logits
        if let lmRaw = preloadedWeights.lmHeadRaw {
            // Use tiled kernel for LM head projection (always seqLen == 1 in decode)
            enc.setComputePipelineState(fusedQ8TiledPSO)
            argTable.setAddress(lmRaw.gpuAddress, index: 0)
            argTable.setAddress(finalOutputBuf.gpuAddress, index: 1)
            argTable.setAddress(logitsBuf.gpuAddress, index: 2)
            argTable.setAddress(lmParamsAddr, index: 3)

            enc.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        } else {
            let lmHeadBuf = preloadedWeights.lmHead!
            enc.setComputePipelineState(gemvPSO)
            argTable.setAddress(lmHeadBuf.gpuAddress, index: 0)
            argTable.setAddress(finalOutputBuf.gpuAddress, index: 1)
            argTable.setAddress(logitsBuf.gpuAddress, index: 2)
            argTable.setAddress(lmParamsAddr, index: 3)

            enc.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        enc.endEncoding()
        m4CmdBuf.endCommandBuffer()

        // Commit with feedback handler for async completion notification
        let commitOptions = MTL4CommitOptions()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commitOptions.addFeedbackHandler { _ in
                continuation.resume()
            }
            m4Queue.commit([m4CmdBuf], options: commitOptions)
        }

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

    /// Read a weight tensor as an MTLBuffer, dequantizing if necessary.
    /// Does NOT cache - creates a fresh buffer each time.
    /// For Q8_0 models, prefer makeRawQ8BufferIfAvailable to avoid float32 materialization.
    private func readWeightBuffer(_ name: String) async throws -> MTLBuffer {
        guard let storage = weights[name] else {
            throw GenerationError.modelLoadFailed(reason: "Missing weight: \(name)")
        }

        let floats = try await dequantizeToFloatArray(storage)
        guard let buffer = device.makeBuffer(
            bytes: floats,
            length: floats.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTLBuffer for \(name)")
        }
        return buffer
    }

    /// Dequantizes weight storage to a Float array (no caching).
    /// Copies data to ensure proper alignment for all data types.
    private func dequantizeToFloatArray(_ storage: TensorStorage) async throws -> [Float] {
        let elementCount = storage.elementCount
        let basePtr = storage.buffer.contents() + storage.byteOffset

        switch storage.dataType {
        case .float32:
            // Copy to ensure alignment - don't assume byteOffset is 4-byte aligned
            var result = [Float](repeating: 0, count: elementCount)
            result.withUnsafeMutableBytes { rawBuffer in
                rawBuffer.copyMemory(from: UnsafeRawBufferPointer(
                    start: basePtr,
                    count: elementCount * MemoryLayout<Float>.stride
                ))
            }
            return result

        case .float16:
            let byteCount = elementCount * MemoryLayout<Float16>.stride
            var f16Array = [Float16](repeating: 0, count: elementCount)
            f16Array.withUnsafeMutableBytes { rawBuffer in
                rawBuffer.copyMemory(from: UnsafeRawBufferPointer(start: basePtr, count: byteCount))
            }
            return f16Array.map { Float($0) }

        case .q4_0:
            let blockCount = elementCount / 32
            let byteCount = blockCount * 18
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ4_0.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        case .q8_0:
            let blockCount = elementCount / 32
            let byteCount = blockCount * 34
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ8_0.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        case .q4_K:
            let superBlockCount = elementCount / 256
            let byteCount = superBlockCount * 144
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ4KM.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        case .q6_K:
            let weightsPerBlock = 256
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q6_K"
                )
            }
            let superBlockCount = elementCount / weightsPerBlock
            let byteCount = superBlockCount * 210
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ6K.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        case .q5_K:
            let weightsPerBlock = 256
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q5_K"
                )
            }
            let superBlockCount = elementCount / weightsPerBlock
            let byteCount = superBlockCount * 176
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ5K.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        case .q3_K:
            let weightsPerBlock = 256
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q3_K"
                )
            }
            let superBlockCount = elementCount / weightsPerBlock
            let byteCount = superBlockCount * 110
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ3K.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        case .q2_K:
            let weightsPerBlock = 256
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q2_K"
                )
            }
            let superBlockCount = elementCount / weightsPerBlock
            let byteCount = superBlockCount * 84
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ2K.dequantise(
                blockData: blockData, superBlockCount: superBlockCount, commandQueue: commandQueue
            )

        case .q5_0:
            let weightsPerBlock = 32
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q5_0"
                )
            }
            let blockCount = elementCount / weightsPerBlock
            let byteCount = blockCount * 22
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ5_0.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        case .q5_1:
            let weightsPerBlock = 32
            guard elementCount % weightsPerBlock == 0 else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name): elementCount \(elementCount) not divisible by \(weightsPerBlock) for Q5_1"
                )
            }
            let blockCount = elementCount / weightsPerBlock
            let byteCount = blockCount * 24
            let ptr = basePtr.bindMemory(to: UInt8.self, capacity: byteCount)
            let blockData = Array(UnsafeBufferPointer(start: ptr, count: byteCount))
            return try await dequantQ5_1.dequantise(
                blockData: blockData, blockCount: blockCount, commandQueue: commandQueue
            )

        default:
            throw GenerationError.modelLoadFailed(
                reason: "Unsupported weight data type \(storage.dataType)"
            )
        }
    }

    /// Exposes the original Q8_0 payload without materializing a Float32 copy.
    /// This is the memory-efficient path - use this for matmul weights when available.
    private func makeRawQ8BufferIfAvailable(_ name: String) -> MTLBuffer? {
        guard let storage = weights[name], storage.dataType == .q8_0 else {
            return nil
        }

        let blockCount = storage.elementCount / 32
        let byteCount = blockCount * 34
        return device.makeBuffer(
            bytesNoCopy: storage.buffer.contents() + storage.byteOffset,
            length: byteCount,
            options: .storageModeShared,
            deallocator: nil
        )
    }

    /// Float32 fallback buffers are only needed when a fused raw Q8 path is unavailable.
    private func readWeightBufferIfNeeded(_ name: String, rawBuffer: MTLBuffer?) async throws -> MTLBuffer? {
        guard rawBuffer == nil else {
            return nil
        }
        return try await readWeightBuffer(name)
    }

    /// Fills embedding rows directly into the destination buffer, avoiding an intermediate array.
    private func fillEmbeddings(
        tokenIDs: [Int],
        into destination: UnsafeMutablePointer<Float>
    ) throws {
        let dim = config.embeddingDim

        let weightName = tiedEmbeddingWeightName
        guard let storage = weights[weightName] else {
            throw GenerationError.modelLoadFailed(reason: "Missing \(weightName)")
        }

        let basePtr = storage.buffer.contents() + storage.byteOffset
        let floatStride = MemoryLayout<Float>.stride

        switch storage.dataType {
        case .float32:
            let ptr = basePtr.bindMemory(to: Float.self, capacity: storage.elementCount)
            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let srcOffset = clampedID * dim
                let dstOffset = i * dim
                memcpy(destination + dstOffset, ptr + srcOffset, dim * floatStride)
            }

        case .float16:
            let ptr = basePtr.bindMemory(to: Float16.self, capacity: storage.elementCount)
            for (i, tokenID) in tokenIDs.enumerated() {
                let clampedID = min(max(tokenID, 0), config.vocabSize - 1)
                let srcOffset = clampedID * dim
                let dstOffset = i * dim
                for d in 0..<dim {
                    destination[dstOffset + d] = Float(ptr[srcOffset + d])
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
                        destination[dstOffset + block * elementsPerBlock + j] = scale * Float(qval)
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
                        destination[dstOffset + block * elementsPerBlock + j * 2] = scale * Float(lo)
                        destination[dstOffset + block * elementsPerBlock + j * 2 + 1] = scale * Float(hi)
                    }
                }
            }

        default:
            throw GenerationError.modelLoadFailed(
                reason: "Unsupported embedding data type: \(storage.dataType)"
            )
        }
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

// MARK: - Metal 4 State

/// Holds all Metal 4-specific objects: command queue, allocator, argument table,
/// residency set, params buffer, and shared event for completion signaling.
/// Created once at init; reused across all decode passes.
/// @unchecked Sendable: MTL4 objects are thread-safe for the usage pattern here
/// (single-writer during encode, no concurrent encode).
@available(macOS 26.0, iOS 26.0, *)
private final class Metal4State: @unchecked Sendable {
    let commandQueue: any MTL4CommandQueue
    let commandBuffer: any MTL4CommandBuffer
    let allocator: any MTL4CommandAllocator
    let argumentTable: any MTL4ArgumentTable
    let residencySet: any MTLResidencySet
    let paramsBuffer: MTLBuffer

    init(device: MTLDevice) throws {
        // Command queue
        guard let queue = device.makeMTL4CommandQueue() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTL4 command queue")
        }
        self.commandQueue = queue

        // Command allocator
        guard let alloc = device.makeCommandAllocator() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTL4 command allocator")
        }
        self.allocator = alloc

        // Command buffer — created from device, not queue
        guard let cmdBuf: (any MTL4CommandBuffer) = device.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create MTL4 command buffer")
        }
        self.commandBuffer = cmdBuf

        // Argument table — max 11 buffer slots (highest index used is 8 in QKV)
        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 11
        argDesc.initializeBindings = true
        self.argumentTable = try device.makeArgumentTable(descriptor: argDesc)

        // Params buffer: 7 slots (QKV, Mega, Wo, GUS, Down, FinalNorm, LMHead)
        // Each 256-byte aligned. Params are constant across layers, written once per decode pass.
        let paramsSize = 7 * 256
        guard let paramsBuf = device.makeBuffer(length: paramsSize, options: .storageModeShared) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create Metal 4 params buffer")
        }
        self.paramsBuffer = paramsBuf

        // Residency set
        let resDesc = MTLResidencySetDescriptor()
        resDesc.label = "EdgeRunner Metal4 Decode"
        resDesc.initialCapacity = 256
        self.residencySet = try device.makeResidencySet(descriptor: resDesc)

    }

    /// Populate the residency set with all buffers used during decode.
    /// Must be called after weights are preloaded.
    func populateResidencySet(
        scratch: ScratchBuffers,
        layerKCaches: [MTLBuffer],
        layerVCaches: [MTLBuffer],
        preloadedWeights: PreloadedWeightsStore
    ) {
        // Add scratch buffers
        let scratchBuffers: [MTLBuffer] = [
            scratch.normed, scratch.afterAttn, scratch.ffnNormed,
            scratch.outputA, scratch.outputB, scratch.allQ, scratch.allK,
            scratch.allV, scratch.ropeQ, scratch.attnOut, scratch.proj,
            scratch.gateOut, scratch.upOut, scratch.activ, scratch.downOut,
            scratch.finalOut, scratch.logits, scratch.decodeHidden
        ]
        for buf in scratchBuffers {
            residencySet.addAllocation(buf)
        }

        // Add KV caches
        for buf in layerKCaches { residencySet.addAllocation(buf) }
        for buf in layerVCaches { residencySet.addAllocation(buf) }

        // Add params buffer
        residencySet.addAllocation(paramsBuffer)

        // Add weight buffers
        for lw in preloadedWeights.layers {
            residencySet.addAllocation(lw.attnNorm)
            if let buffer = lw.wq { residencySet.addAllocation(buffer) }
            if let buffer = lw.wk { residencySet.addAllocation(buffer) }
            if let buffer = lw.wv { residencySet.addAllocation(buffer) }
            if let buffer = lw.wo { residencySet.addAllocation(buffer) }
            if let q = lw.qNorm { residencySet.addAllocation(q) }
            if let k = lw.kNorm { residencySet.addAllocation(k) }
            residencySet.addAllocation(lw.ffnNorm)
            if let buffer = lw.gate { residencySet.addAllocation(buffer) }
            if let buffer = lw.up { residencySet.addAllocation(buffer) }
            if let buffer = lw.down { residencySet.addAllocation(buffer) }
            if let b = lw.wqRaw { residencySet.addAllocation(b) }
            if let b = lw.wkRaw { residencySet.addAllocation(b) }
            if let b = lw.wvRaw { residencySet.addAllocation(b) }
            if let b = lw.woRaw { residencySet.addAllocation(b) }
            if let b = lw.gateRaw { residencySet.addAllocation(b) }
            if let b = lw.upRaw { residencySet.addAllocation(b) }
            if let b = lw.downRaw { residencySet.addAllocation(b) }
        }

        if let fn = preloadedWeights.finalNorm { residencySet.addAllocation(fn) }
        if let lm = preloadedWeights.lmHead { residencySet.addAllocation(lm) }
        if let lmRaw = preloadedWeights.lmHeadRaw { residencySet.addAllocation(lmRaw) }

        residencySet.commit()
        residencySet.requestResidency()
    }
}

// MARK: - Pre-allocated Scratch Buffers

/// Pre-allocated GPU buffers reused across all forward passes.
/// Eliminates ~17 MTLBuffer allocations per call.
/// @unchecked Sendable: all fields are immutable `let` MTLBuffer references.
/// MTLBuffer contents are mutated by GPU kernels (not Swift), synchronized via command buffer ordering.
private struct ScratchBuffers: @unchecked Sendable {
    let normed: MTLBuffer
    let afterAttn: MTLBuffer
    let ffnNormed: MTLBuffer
    let outputA: MTLBuffer
    let outputB: MTLBuffer
    let allQ: MTLBuffer
    let allK: MTLBuffer
    let allV: MTLBuffer
    let ropeQ: MTLBuffer
    let ropeK: MTLBuffer
    let attnOut: MTLBuffer
    let proj: MTLBuffer
    let gateOut: MTLBuffer
    let upOut: MTLBuffer
    let activ: MTLBuffer
    let downOut: MTLBuffer
    let finalOut: MTLBuffer
    let logits: MTLBuffer
    let decodeHidden: MTLBuffer  // Pre-allocated embedding buffer for decode (dim × float)
}

private struct DecodeDebugOptions: Sendable {
    let forceBaseDecodePath: Bool
    let disableMegaKernel: Bool
    let disableFusedFinalNormLMHead: Bool
    let disableKVCacheBarrier: Bool
    let preferMetal4DecodePath: Bool

    var requiresBaseDecodePath: Bool {
        forceBaseDecodePath || disableMegaKernel
    }

    init(
        environment: [String: String],
        config: LlamaConfig,
        overrides: LlamaDecodeOverrides?
    ) {
        let autoDisableMegaKernel = config.headCount + config.kvHeadCount > 24
        self.forceBaseDecodePath =
            overrides?.forceBaseDecodePath
            ?? Self.isEnabled(environment["EDGERUNNER_DECODE_FORCE_BASE"])
        self.disableMegaKernel =
            overrides?.disableMegaKernel
            ?? (autoDisableMegaKernel || Self.isEnabled(environment["EDGERUNNER_DECODE_DISABLE_MEGA_GQA"]))
        self.disableFusedFinalNormLMHead =
            overrides?.disableFusedFinalNormLMHead
            ?? Self.isEnabled(environment["EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD"])
        self.disableKVCacheBarrier =
            overrides?.disableKVCacheBarrier
            ?? Self.isEnabled(environment["EDGERUNNER_DECODE_DISABLE_KV_BARRIER"])
        self.preferMetal4DecodePath =
            overrides?.preferMetal4DecodePath
            ?? Self.isEnabled(environment["EDGERUNNER_DECODE_PREFER_METAL4"])
    }

    private static func isEnabled(_ value: String?) -> Bool {
        guard let value else { return false }
        switch value.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

}

// MARK: - Fused QKV Params (matches Metal ERFusedQKVParams layout)
/// Passed to `dequant_q8_0_fused_qkv` kernel.
private struct FusedQKVParams {
    var qRows: UInt32          // Q output rows (numHeads * headDim)
    var kvRows: UInt32         // K/V output rows (numKVHeads * headDim)
    var cols: UInt32           // input columns (dim)
    var blocksPerRow: UInt32   // Q8_0 blocks per row
    var rmsEps: Float          // RMSNorm epsilon (for fused RMSNorm+QKV)
}

private struct ERFusedNormRoPEParams {
    var numHeads: UInt32
    var numKVHeads: UInt32
    var headDim: UInt32
    var startPos: UInt32
    var theta: Float
    var scalingFactor: Float
    var rmsEps: Float
}

private struct FusedGateUpSiluParams {
    var rows: UInt32
    var cols: UInt32
    var blocksPerRow: UInt32
    var rmsEps: Float
}

// MARK: - Pre-loaded Weight Buffers

/// Holds all layer weight MTLBuffers for zero-async access during forward pass.
/// For Q8_0 layers, keep only the raw quantized payload and skip the redundant
/// Float32 materialization. Non-Q8 models still populate the Float32 fallback buffers.
private struct LayerWeightBuffers {
    let attnNorm: MTLBuffer
    let wq: MTLBuffer!
    let wk: MTLBuffer!
    let wv: MTLBuffer!
    let wo: MTLBuffer!
    let qNorm: MTLBuffer?  // Per-head Q RMSNorm weight [headDim] (Qwen3)
    let kNorm: MTLBuffer?  // Per-head K RMSNorm weight [headDim] (Qwen3)
    let ffnNorm: MTLBuffer
    let gate: MTLBuffer!
    let up: MTLBuffer!
    let down: MTLBuffer!

    // Raw Q8_0 quantized weight buffers (nil if not Q8_0)
    let wqRaw: MTLBuffer?
    let wkRaw: MTLBuffer?
    let wvRaw: MTLBuffer?
    let woRaw: MTLBuffer?
    let gateRaw: MTLBuffer?
    let upRaw: MTLBuffer?
    let downRaw: MTLBuffer?
}

/// Thread-safe store for pre-loaded weights. Write-once during first forward pass,
/// then read-only on all subsequent passes. Uses OSAllocatedUnfairLock for safe init.
/// @unchecked Sendable: MTLBuffer is thread-safe for read access after initialization.
/// Write access (load) happens exactly once before any concurrent reads (decode calls).
private final class PreloadedWeightsStore: @unchecked Sendable {
    // Write-once state protected by unfair lock for initialization
    private let lock = NSLock()
    private(set) var layers: [LayerWeightBuffers] = []
    private(set) var finalNorm: MTLBuffer?
    private(set) var lmHead: MTLBuffer?
    private(set) var isLoaded = false
    private(set) var lmHeadRaw: MTLBuffer?
    private(set) var lmHeadCols: Int = 0
    func load(layers: [LayerWeightBuffers], finalNorm: MTLBuffer, lmHead: MTLBuffer?,
              lmHeadRaw: MTLBuffer? = nil, lmHeadCols: Int = 0) {
        lock.lock()
        defer { lock.unlock() }
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.lmHeadRaw = lmHeadRaw
        self.lmHeadCols = lmHeadCols
        self.isLoaded = true
    }
}

// MARK: - Decoder State Store

/// Tracks the previously processed token sequence for KV cache decode detection.
/// When the new tokenIDs are the previous sequence plus one new token,
/// we can skip recomputing the full sequence and only process the new token.
/// @unchecked Sendable: accessed only from async logits(for:) which is called
/// sequentially (one token at a time). Lock protects against concurrent misuse.
private final class DecoderStateStore: @unchecked Sendable {
    private let lock = NSLock()
    private var _previousTokenIDs: [Int] = []
    private var _cachedLogits: [Float]?
    private var _cachedLogitsInput: [Int]?

    var previousTokenIDs: [Int] {
        get { lock.withLock { _previousTokenIDs } }
        set { lock.withLock { _previousTokenIDs = newValue } }
    }
    var cachedLogits: [Float]? {
        get { lock.withLock { _cachedLogits } }
        set { lock.withLock { _cachedLogits = newValue } }
    }
    var decodeWarmedUp: Bool = false
    var cachedLogitsInput: [Int]? {
        get { lock.withLock { _cachedLogitsInput } }
        set { lock.withLock { _cachedLogitsInput = newValue } }
    }
}
