import Foundation
import EdgeRunnerIO

public enum Gemma4LayerType: String, Sendable, Equatable {
    case sliding
    case global
}

public enum GGUFMetadataError: Error, Sendable, Equatable {
    case missingKey(String)
    case invalidValue(key: String, value: String)
}

/// Hyperparameter configuration for Gemma 4 (E4B) parsed from GGUF metadata.
///
/// Matches the `google/gemma-4-E4B` reference config. Includes Per-Layer
/// Embedding (PLE) dimensions, KV-cache sharing counts, and the
/// sliding/global attention pattern.
public struct Gemma4ModelConfig: Sendable, Equatable {
    public let numHiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let finalLogitSoftcapping: Float
    public let perLayerDim: Int
    public let perLayerVocabSize: Int
    public let numKVSharedLayers: Int
    public let slidingWindow: Int
    public let layerTypes: [Gemma4LayerType]
    public let ropeThetaLocal: Float
    public let ropeThetaGlobal: Float
    public let localRotaryDimension: Int
    public let globalRotaryDimension: Int
    public let partialRotaryFactor: Float

    public init(modelConfigMetadata metadata: [String: MetadataValue]) throws {
        self.numHiddenLayers = try Self.requireInt("gemma4.block_count", from: metadata)
        self.hiddenSize = try Self.requireInt("gemma4.embedding_length", from: metadata)
        self.intermediateSize = try Self.requireInt("gemma4.feed_forward_length", from: metadata)
        self.numAttentionHeads = try Self.requireInt("gemma4.attention.head_count", from: metadata)
        self.numKeyValueHeads = try Self.requireInt("gemma4.attention.head_count_kv", from: metadata)
        self.headDim = try Self.requireInt(
            keys: ["gemma4.attention.key_length_swa", "gemma4.attention.key_length"],
            from: metadata
        )
        self.globalHeadDim = try Self.requireInt(
            keys: ["gemma4.attention.key_length_global", "gemma4.attention.key_length"],
            from: metadata
        )
        self.vocabSize = try Self.requireVocabSize(from: metadata)
        self.maxPositionEmbeddings = try Self.requireInt("gemma4.context_length", from: metadata)
        self.rmsNormEps = try Self.requireFloat("gemma4.attention.layer_norm_rms_epsilon", from: metadata)
        self.finalLogitSoftcapping = try Self.requireFloat("gemma4.final_logit_softcapping", from: metadata)
        self.perLayerDim = try Self.requireInt(
            keys: ["gemma4.embedding_length_per_layer", "gemma4.embedding_length_per_layer_input"],
            from: metadata
        )
        self.perLayerVocabSize = Self.optionalInt("gemma4.per_layer_vocab_size", from: metadata)
            ?? self.vocabSize
        self.numKVSharedLayers = try Self.requireInt("gemma4.attention.shared_kv_layers", from: metadata)
        self.slidingWindow = try Self.requireInt("gemma4.attention.sliding_window", from: metadata)

        let parsedLayerTypes = try Self.parseLayerTypes(from: metadata)
        guard parsedLayerTypes.count == self.numHiddenLayers else {
            throw GGUFMetadataError.invalidValue(
                key: "gemma4.layer_types",
                value: "expected \(self.numHiddenLayers) entries, got \(parsedLayerTypes.count)"
            )
        }
        self.layerTypes = parsedLayerTypes

        self.ropeThetaLocal = Self.optionalFloat("gemma4.rope.freq_base_swa", from: metadata) ?? 10_000.0
        self.ropeThetaGlobal = Self.optionalFloat("gemma4.rope.freq_base", from: metadata) ?? 1_000_000.0
        self.localRotaryDimension = Self.optionalInt("gemma4.rope.dimension_count_swa", from: metadata)
            ?? self.headDim
        self.globalRotaryDimension = Self.optionalInt("gemma4.rope.dimension_count", from: metadata)
            ?? self.globalHeadDim
        try Self.validateRotaryDimension(self.localRotaryDimension, headDim: self.headDim, key: "gemma4.rope.dimension_count_swa")
        try Self.validateRotaryDimension(self.globalRotaryDimension, headDim: self.globalHeadDim, key: "gemma4.rope.dimension_count")
        self.partialRotaryFactor = Float(self.globalRotaryDimension) / Float(self.globalHeadDim)
    }

    public init(metadata: [String: GGUFMetadataValue]) throws {
        self.numHiddenLayers = try Self.requireInt("gemma4.block_count", from: metadata)
        self.hiddenSize = try Self.requireInt("gemma4.embedding_length", from: metadata)
        self.intermediateSize = try Self.requireInt("gemma4.feed_forward_length", from: metadata)
        self.numAttentionHeads = try Self.requireInt("gemma4.attention.head_count", from: metadata)
        self.numKeyValueHeads = try Self.requireInt("gemma4.attention.head_count_kv", from: metadata)
        self.headDim = try Self.requireInt(
            keys: ["gemma4.attention.key_length_swa", "gemma4.attention.key_length"],
            from: metadata
        )
        self.globalHeadDim = try Self.requireInt(
            keys: ["gemma4.attention.key_length_global", "gemma4.attention.key_length"],
            from: metadata
        )
        self.vocabSize = try Self.requireVocabSize(from: metadata)
        self.maxPositionEmbeddings = try Self.requireInt("gemma4.context_length", from: metadata)
        self.rmsNormEps = try Self.requireFloat("gemma4.attention.layer_norm_rms_epsilon", from: metadata)
        self.finalLogitSoftcapping = try Self.requireFloat("gemma4.final_logit_softcapping", from: metadata)
        self.perLayerDim = try Self.requireInt(
            keys: ["gemma4.embedding_length_per_layer", "gemma4.embedding_length_per_layer_input"],
            from: metadata
        )
        self.perLayerVocabSize = Self.optionalInt("gemma4.per_layer_vocab_size", from: metadata)
            ?? self.vocabSize
        self.numKVSharedLayers = try Self.requireInt("gemma4.attention.shared_kv_layers", from: metadata)
        self.slidingWindow = try Self.requireInt("gemma4.attention.sliding_window", from: metadata)

        let parsedLayerTypes = try Self.parseLayerTypes(from: metadata)
        guard parsedLayerTypes.count == self.numHiddenLayers else {
            throw GGUFMetadataError.invalidValue(
                key: "gemma4.layer_types",
                value: "expected \(self.numHiddenLayers) entries, got \(parsedLayerTypes.count)"
            )
        }
        self.layerTypes = parsedLayerTypes

        self.ropeThetaLocal = Self.optionalFloat("gemma4.rope.freq_base_swa", from: metadata) ?? 10_000.0
        self.ropeThetaGlobal = Self.optionalFloat("gemma4.rope.freq_base", from: metadata) ?? 1_000_000.0
        self.localRotaryDimension = Self.optionalInt("gemma4.rope.dimension_count_swa", from: metadata)
            ?? self.headDim
        self.globalRotaryDimension = Self.optionalInt("gemma4.rope.dimension_count", from: metadata)
            ?? self.globalHeadDim
        try Self.validateRotaryDimension(self.localRotaryDimension, headDim: self.headDim, key: "gemma4.rope.dimension_count_swa")
        try Self.validateRotaryDimension(self.globalRotaryDimension, headDim: self.globalHeadDim, key: "gemma4.rope.dimension_count")
        self.partialRotaryFactor = Float(self.globalRotaryDimension) / Float(self.globalHeadDim)
    }

    private static func validateRotaryDimension(_ value: Int, headDim: Int, key: String) throws {
        guard value > 0, value <= headDim, value % 2 == 0 else {
            throw GGUFMetadataError.invalidValue(
                key: key,
                value: "rotary dimension \(value) must be even and in 1...\(headDim)"
            )
        }
    }

    private static func requireInt(
        _ key: String,
        from metadata: [String: GGUFMetadataValue]
    ) throws -> Int {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        guard let integer = value.intValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return integer
    }

    private static func requireInt(
        keys: [String],
        from metadata: [String: GGUFMetadataValue]
    ) throws -> Int {
        for key in keys {
            if let value = metadata[key] {
                guard let integer = value.intValue else {
                    throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
                }
                return integer
            }
        }
        throw GGUFMetadataError.missingKey(keys.joined(separator: " or "))
    }

    private static func optionalInt(
        _ key: String,
        from metadata: [String: GGUFMetadataValue]
    ) -> Int? {
        metadata[key]?.intValue
    }

    private static func requireFloat(
        _ key: String,
        from metadata: [String: GGUFMetadataValue]
    ) throws -> Float {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        guard let float = value.floatValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return float
    }

    private static func optionalFloat(
        _ key: String,
        from metadata: [String: GGUFMetadataValue]
    ) -> Float? {
        metadata[key]?.floatValue
    }

    private static func requireString(
        _ key: String,
        from metadata: [String: GGUFMetadataValue]
    ) throws -> String {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        guard let string = value.stringValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return string
    }

    private static func requireInt(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) throws -> Int {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        guard let integer = value.intValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return integer
    }

    private static func requireInt(
        keys: [String],
        from metadata: [String: MetadataValue]
    ) throws -> Int {
        for key in keys {
            if let value = metadata[key] {
                guard let integer = value.intValue else {
                    throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
                }
                return integer
            }
        }
        throw GGUFMetadataError.missingKey(keys.joined(separator: " or "))
    }

    private static func optionalInt(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) -> Int? {
        metadata[key]?.intValue
    }

    private static func requireFloat(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) throws -> Float {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        if let float = value.floatValue {
            return float
        }
        guard let integer = value.intValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return Float(integer)
    }

    private static func optionalFloat(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) -> Float? {
        metadata[key]?.floatValue
    }

    private static func requireString(
        _ key: String,
        from metadata: [String: MetadataValue]
    ) throws -> String {
        guard let value = metadata[key] else {
            throw GGUFMetadataError.missingKey(key)
        }
        guard let string = value.stringValue else {
            throw GGUFMetadataError.invalidValue(key: key, value: "\(value)")
        }
        return string
    }

    /// Returns the layer whose KV cache layer `layer` should share.
    ///
    /// Layers in `[numHiddenLayers - numKVSharedLayers, numHiddenLayers)` reuse
    /// the KV cache of the nearest earlier layer with the same attention type
    /// that sits below the shared-layer boundary. Non-shared layers return
    /// themselves.
    public func kvSourceLayer(for layer: Int) -> Int {
        let firstSharedLayer = numHiddenLayers - numKVSharedLayers
        guard layer >= firstSharedLayer else { return layer }
        let targetType = layerTypes[layer]
        var probe = layer - 1
        while probe >= 0 {
            if layerTypes[probe] == targetType && probe < firstSharedLayer {
                return probe
            }
            probe -= 1
        }
        preconditionFailure(
            "Layer \(layer) (\(targetType)) has no same-type predecessor "
            + "below firstSharedLayer \(firstSharedLayer)"
        )
    }

    private static func requireVocabSize(from metadata: [String: GGUFMetadataValue]) throws -> Int {
        if let vocabSize = optionalInt("gemma4.vocab_size", from: metadata) {
            return vocabSize
        }
        guard let tokens = metadata["tokenizer.ggml.tokens"]?.arrayValue else {
            throw GGUFMetadataError.missingKey("gemma4.vocab_size or tokenizer.ggml.tokens")
        }
        return tokens.count
    }

    private static func requireVocabSize(from metadata: [String: MetadataValue]) throws -> Int {
        if let vocabSize = optionalInt("gemma4.vocab_size", from: metadata) {
            return vocabSize
        }
        guard let tokens = metadata["tokenizer.ggml.tokens"]?.arrayValue else {
            throw GGUFMetadataError.missingKey("gemma4.vocab_size or tokenizer.ggml.tokens")
        }
        return tokens.count
    }

    private static func parseLayerTypes(from metadata: [String: GGUFMetadataValue]) throws -> [Gemma4LayerType] {
        if let raw = metadata["gemma4.layer_types"]?.stringValue {
            return try parseLayerTypes(raw)
        }
        guard let pattern = metadata["gemma4.attention.sliding_window_pattern"]?.arrayValue else {
            throw GGUFMetadataError.missingKey(
                "gemma4.layer_types or gemma4.attention.sliding_window_pattern"
            )
        }
        return try parseLayerTypes(pattern.map(\.boolValue), key: "gemma4.attention.sliding_window_pattern")
    }

    private static func parseLayerTypes(from metadata: [String: MetadataValue]) throws -> [Gemma4LayerType] {
        if let raw = metadata["gemma4.layer_types"]?.stringValue {
            return try parseLayerTypes(raw)
        }
        guard let pattern = metadata["gemma4.attention.sliding_window_pattern"]?.arrayValue else {
            throw GGUFMetadataError.missingKey(
                "gemma4.layer_types or gemma4.attention.sliding_window_pattern"
            )
        }
        return try parseLayerTypes(pattern.map(\.boolValue), key: "gemma4.attention.sliding_window_pattern")
    }

    private static func parseLayerTypes(_ raw: String) throws -> [Gemma4LayerType] {
        let key = "gemma4.layer_types"
        var result: [Gemma4LayerType] = []
        result.reserveCapacity(64)
        for token in raw.split(separator: ",") {
            let trimmed = token.trimmingCharacters(in: .whitespaces)
            guard let layerType = Gemma4LayerType(rawValue: trimmed) else {
                throw GGUFMetadataError.invalidValue(key: key, value: trimmed)
            }
            result.append(layerType)
        }
        return result
    }

    private static func parseLayerTypes(_ pattern: [Bool?], key: String) throws -> [Gemma4LayerType] {
        var result: [Gemma4LayerType] = []
        result.reserveCapacity(pattern.count)
        for value in pattern {
            guard let isSliding = value else {
                throw GGUFMetadataError.invalidValue(key: key, value: "\(pattern)")
            }
            result.append(isSliding ? .sliding : .global)
        }
        return result
    }
}
