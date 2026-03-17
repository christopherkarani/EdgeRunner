import Foundation

public enum LlamaConfigError: Error, Sendable, Equatable {
    case missingMetadataKey(String)
    case invalidMetadataValue(key: String, description: String)
}

public struct LlamaConfig: Sendable, Equatable {
    public let embeddingDim: Int
    public let layerCount: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let vocabSize: Int
    public let intermediateDim: Int
    public let ropeFreqBase: Double
    public let rmsNormEpsilon: Double
    /// Explicit head dimension. Some models (Qwen 3) use headDim != embeddingDim/headCount.
    public let explicitHeadDim: Int?

    public var headDim: Int {
        explicitHeadDim ?? (embeddingDim / headCount)
    }

    public var gqaRatio: Int {
        headCount / kvHeadCount
    }

    public init(
        embeddingDim: Int,
        layerCount: Int,
        headCount: Int,
        kvHeadCount: Int,
        vocabSize: Int,
        intermediateDim: Int,
        ropeFreqBase: Double,
        rmsNormEpsilon: Double,
        explicitHeadDim: Int? = nil
    ) {
        self.embeddingDim = embeddingDim
        self.layerCount = layerCount
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.vocabSize = vocabSize
        self.intermediateDim = intermediateDim
        self.ropeFreqBase = ropeFreqBase
        self.rmsNormEpsilon = rmsNormEpsilon
        self.explicitHeadDim = explicitHeadDim
    }

    public init(fromGGUFMetadata metadata: [String: MetadataValue]) throws {
        // Detect architecture prefix: llama, qwen2, qwen3, mistral, etc.
        let arch = metadata["general.architecture"]?.stringValue ?? "llama"
        let prefix = arch

        self.embeddingDim = try Self.requireInt("\(prefix).embedding_length", from: metadata)
        self.layerCount = try Self.requireInt("\(prefix).block_count", from: metadata)
        self.headCount = try Self.requireInt("\(prefix).attention.head_count", from: metadata)
        self.kvHeadCount = try Self.requireInt("\(prefix).attention.head_count_kv", from: metadata)
        self.intermediateDim = try Self.requireInt("\(prefix).feed_forward_length", from: metadata)
        self.ropeFreqBase = try Self.requireDouble("\(prefix).rope.freq_base", from: metadata)
        self.rmsNormEpsilon = try Self.requireDouble(
            "\(prefix).attention.layer_norm_rms_epsilon",
            from: metadata
        )

        // Vocab size: try architecture-specific key, fall back to tokenizer count
        if let vocabVal = metadata["\(prefix).vocab_size"]?.intValue {
            self.vocabSize = vocabVal
        } else if let tokenizerCount = metadata["tokenizer.ggml.tokens"]?.arrayValue?.count {
            self.vocabSize = tokenizerCount
        } else {
            throw LlamaConfigError.missingMetadataKey("\(prefix).vocab_size")
        }

        // Explicit head dimension (Qwen 3 uses key_length != embedding_dim/head_count)
        if let keyLen = metadata["\(prefix).attention.key_length"]?.intValue {
            self.explicitHeadDim = keyLen
        } else {
            self.explicitHeadDim = nil
        }
    }

    private static func requireInt(_ key: String, from metadata: [String: MetadataValue]) throws -> Int {
        guard let value = metadata[key] else {
            throw LlamaConfigError.missingMetadataKey(key)
        }
        guard let integer = value.intValue else {
            throw LlamaConfigError.invalidMetadataValue(
                key: key,
                description: "Expected integer metadata value"
            )
        }
        return integer
    }

    private static func requireDouble(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) throws -> Double {
        guard let value = metadata[key] else {
            throw LlamaConfigError.missingMetadataKey(key)
        }
        if let integer = value.intValue {
            return Double(integer)
        }
        guard let float = value.floatValue else {
            throw LlamaConfigError.invalidMetadataValue(
                key: key,
                description: "Expected floating-point metadata value"
            )
        }
        return Double(float)
    }
}
