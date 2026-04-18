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
    public let partialRotaryFactor: Float

    public init(metadata: [String: GGUFMetadataValue]) throws {
        self.numHiddenLayers = try Self.requireInt("gemma4.block_count", from: metadata)
        self.hiddenSize = try Self.requireInt("gemma4.embedding_length", from: metadata)
        self.intermediateSize = try Self.requireInt("gemma4.feed_forward_length", from: metadata)
        self.numAttentionHeads = try Self.requireInt("gemma4.attention.head_count", from: metadata)
        self.numKeyValueHeads = try Self.requireInt("gemma4.attention.head_count_kv", from: metadata)
        self.headDim = try Self.requireInt("gemma4.attention.key_length", from: metadata)
        self.globalHeadDim = try Self.requireInt("gemma4.attention.key_length_global", from: metadata)
        self.vocabSize = try Self.requireInt("gemma4.vocab_size", from: metadata)
        self.maxPositionEmbeddings = try Self.requireInt("gemma4.context_length", from: metadata)
        self.rmsNormEps = try Self.requireFloat("gemma4.attention.layer_norm_rms_epsilon", from: metadata)
        self.finalLogitSoftcapping = try Self.requireFloat("gemma4.final_logit_softcapping", from: metadata)
        self.perLayerDim = try Self.requireInt("gemma4.embedding_length_per_layer", from: metadata)
        self.perLayerVocabSize = try Self.requireInt("gemma4.per_layer_vocab_size", from: metadata)
        self.numKVSharedLayers = try Self.requireInt("gemma4.attention.shared_kv_layers", from: metadata)
        self.slidingWindow = try Self.requireInt("gemma4.attention.sliding_window", from: metadata)

        let layerTypesRaw = try Self.requireString("gemma4.layer_types", from: metadata)
        let parsedLayerTypes = try Self.parseLayerTypes(layerTypesRaw)
        guard parsedLayerTypes.count == self.numHiddenLayers else {
            throw GGUFMetadataError.invalidValue(
                key: "gemma4.layer_types",
                value: "expected \(self.numHiddenLayers) entries, got \(parsedLayerTypes.count)"
            )
        }
        self.layerTypes = parsedLayerTypes

        self.ropeThetaLocal = 10_000.0
        self.ropeThetaGlobal = 1_000_000.0
        self.partialRotaryFactor = 0.25
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
}
