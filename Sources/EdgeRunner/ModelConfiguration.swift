import Foundation

/// Public KV-cache compression policy.
///
/// `automatic` is the long-term default. Until TurboQuant clears its rollout gate,
/// it resolves to `disabled` at runtime and preserves today's FP16 cache path.
public enum KVCacheCompression: String, Sendable, Equatable, Codable {
    case disabled
    case automatic
    case q8_0
    case turboquantV2
    case turboQuantBalanced
    case turboQuantAggressive
}

enum ResolvedKVCacheCompression: Sendable, Equatable {
    case disabled
    case q8_0
    case turboquantV2
    case turboQuantBalanced
    case turboQuantAggressive
}

/// Internal prefill path overrides for benchmarking and staged rollout.
struct LlamaPrefillOverrides: Sendable, Equatable {
    var forceLegacyPrefillPath: Bool?
    var preferMetal4PrefillPath: Bool?
    var preferExactMatrixPrefillPath: Bool?

    init(
        forceLegacyPrefillPath: Bool? = nil,
        preferMetal4PrefillPath: Bool? = nil,
        preferExactMatrixPrefillPath: Bool? = nil
    ) {
        self.forceLegacyPrefillPath = forceLegacyPrefillPath
        self.preferMetal4PrefillPath = preferMetal4PrefillPath
        self.preferExactMatrixPrefillPath = preferExactMatrixPrefillPath
    }
}

/// Internal decode path overrides for debugging and compatibility.
struct LlamaDecodeOverrides: Sendable, Equatable {
    var forceBaseDecodePath: Bool?
    var disableMegaKernel: Bool?
    var disableFusedFinalNormLMHead: Bool?
    var disableKVCacheBarrier: Bool?
    var preferMetal4DecodePath: Bool?

    init(
        forceBaseDecodePath: Bool? = nil,
        disableMegaKernel: Bool? = nil,
        disableFusedFinalNormLMHead: Bool? = nil,
        disableKVCacheBarrier: Bool? = nil,
        preferMetal4DecodePath: Bool? = nil
    ) {
        self.forceBaseDecodePath = forceBaseDecodePath
        self.disableMegaKernel = disableMegaKernel
        self.disableFusedFinalNormLMHead = disableFusedFinalNormLMHead
        self.disableKVCacheBarrier = disableKVCacheBarrier
        self.preferMetal4DecodePath = preferMetal4DecodePath
    }
}

/// Configuration options for model loading and text generation.
///
/// Use `ModelConfiguration` to customize how models are loaded and how they
/// behave during inference. This includes context window size, memory usage,
/// and generation limits.
///
/// ## Example
///
/// ```swift
/// let config = ModelConfiguration(
///     maxTokens: 512,
///     contextWindowSize: 2048,
///     useMemoryMapping: true
/// )
/// ```
public struct ModelConfiguration: Sendable {
    /// The maximum number of tokens to generate in a single session.
    ///
    /// Default is 2048. Increase for longer outputs, but note that
    /// generation time scales linearly with token count.
    public var maxTokens: Int
    
    /// The maximum sequence length the model can process.
    ///
    /// This includes both the input prompt and generated tokens.
    /// Larger values use more memory for the KV cache.
    /// Default is 4096.
    public var contextWindowSize: Int
    
    /// Whether to use memory-mapped file I/O for loading model weights.
    ///
    /// When enabled, the operating system loads weight data on-demand
    /// rather than reading the entire file into RAM. This significantly
    /// reduces load time and memory pressure. Default is `true`.
    public var useMemoryMapping: Bool
    
    /// Optional URL to an external tokenizer file.
    ///
    /// If not provided, the tokenizer is loaded from the model file's
    /// embedded vocabulary (for GGUF models).
    public var tokenizerURL: URL?

    /// KV-cache compression policy.
    ///
    /// `automatic` is the public default. It currently resolves to `disabled`
    /// unless the runtime rollout gate is explicitly enabled.
    public var kvCacheCompression: KVCacheCompression
    
    var llamaDecodeOverrides: LlamaDecodeOverrides?
    var llamaPrefillOverrides: LlamaPrefillOverrides?

    /// Creates a new model configuration.
    ///
    /// - Parameters:
    ///   - maxTokens: Maximum tokens to generate (default: 2048).
    ///   - contextWindowSize: Maximum sequence length (default: 4096).
    ///   - useMemoryMapping: Enable memory-mapped loading (default: true).
    ///   - tokenizerURL: Optional external tokenizer file URL.
    public init(
        maxTokens: Int = 2048,
        contextWindowSize: Int = 4096,
        useMemoryMapping: Bool = true,
        tokenizerURL: URL? = nil,
        kvCacheCompression: KVCacheCompression = .automatic
    ) {
        self.maxTokens = maxTokens
        self.contextWindowSize = contextWindowSize
        self.useMemoryMapping = useMemoryMapping
        self.tokenizerURL = tokenizerURL
        self.kvCacheCompression = kvCacheCompression
        self.llamaDecodeOverrides = nil
        self.llamaPrefillOverrides = nil
    }
}

extension ModelConfiguration {
    static let automaticTurboQuantMinimumContextWindow = 8192
    static let automaticTurboQuantGateEnv = "EDGERUNNER_ENABLE_TURBOQUANT_AUTOMATIC"

    var resolvedKVCacheCompression: ResolvedKVCacheCompression {
        switch kvCacheCompression {
        case .disabled:
            return .disabled
        case .automatic:
            let automaticEnabled =
                ProcessInfo.processInfo.environment[Self.automaticTurboQuantGateEnv] == "1"
            guard automaticEnabled, contextWindowSize >= Self.automaticTurboQuantMinimumContextWindow else {
                return .disabled
            }
            return .turboQuantBalanced
        case .q8_0:
            return .q8_0
        case .turboquantV2:
            return .turboquantV2
        case .turboQuantBalanced:
            return .turboQuantBalanced
        case .turboQuantAggressive:
            return .turboQuantAggressive
        }
    }
}
