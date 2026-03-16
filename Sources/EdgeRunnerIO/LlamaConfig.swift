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

    public var headDim: Int {
        embeddingDim / headCount
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
        rmsNormEpsilon: Double
    ) {
        self.embeddingDim = embeddingDim
        self.layerCount = layerCount
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.vocabSize = vocabSize
        self.intermediateDim = intermediateDim
        self.ropeFreqBase = ropeFreqBase
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public init(fromGGUFMetadata metadata: [String: MetadataValue]) throws {
        self.embeddingDim = try Self.requireInt("llama.embedding_length", from: metadata)
        self.layerCount = try Self.requireInt("llama.block_count", from: metadata)
        self.headCount = try Self.requireInt("llama.attention.head_count", from: metadata)
        self.kvHeadCount = try Self.requireInt("llama.attention.head_count_kv", from: metadata)
        self.vocabSize = try Self.requireInt("llama.vocab_size", from: metadata)
        self.intermediateDim = try Self.requireInt("llama.feed_forward_length", from: metadata)
        self.ropeFreqBase = try Self.requireDouble("llama.rope.freq_base", from: metadata)
        self.rmsNormEpsilon = try Self.requireDouble(
            "llama.attention.layer_norm_rms_epsilon",
            from: metadata
        )
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
