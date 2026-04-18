import Foundation
import Metal
import EdgeRunnerIO

/// Per-block tensor handles for a single Gemma 4 transformer layer.
///
/// `attnK` / `attnV` are `nil` when this layer shares its KV cache with an earlier
/// layer (see `Gemma4ModelConfig.kvSourceLayer(for:)`). In that case the layer
/// reuses the source layer's cached K/V — the GGUF file ships no dedicated
/// `attn_k.weight` / `attn_v.weight` tensors for that block.
public struct Gemma4BlockWeights: Sendable {
    public var inputNorm: TensorStorage
    public var attnQ: TensorStorage
    public var attnK: TensorStorage?
    public var attnV: TensorStorage?
    public var attnO: TensorStorage
    public var postAttentionNorm: TensorStorage
    public var ffnGate: TensorStorage
    public var ffnUp: TensorStorage
    public var ffnDown: TensorStorage
    public var postFFNNorm: TensorStorage
    public var perLayerInputGate: TensorStorage
    public var perLayerProjection: TensorStorage
    public var postPerLayerInputNorm: TensorStorage
}

public enum Gemma4LoadError: Error, Sendable, Equatable {
    /// A required model-level or per-layer tensor is missing from the weight map.
    case missingTensor(String)
    /// A required Per-Layer Embedding (PLE) tensor is missing from the weight map.
    case missingPLETensor(String)
    /// The PLE token-embedding tensor is stored in a quantization format the
    /// Gemma 4 runtime does not support (e.g. Q4_K_M, Q2_K, Q1_0). Community
    /// GGUFs never ship these for PLE.
    case unsupportedPLEQuant(String)
}

/// Bundle of tensor handles extracted from a weight map for Gemma 4 (E4B).
///
/// This type performs validation up front: missing tensors throw
/// `Gemma4LoadError.missingTensor` / `.missingPLETensor`, and an unsupported
/// PLE quantization throws `.unsupportedPLEQuant`. After construction, the
/// forward pass can read tensors directly from the typed fields without
/// further map lookups.
public struct Gemma4Weights: Sendable {
    public let tokenEmbedding: TensorStorage
    public let outputNorm: TensorStorage
    public let perLayerTokenEmbed: TensorStorage
    public let perLayerModelProjection: TensorStorage
    public let perLayerProjectionNorm: TensorStorage
    public let blocks: [Gemma4BlockWeights]

    public init(
        weightMap: WeightMap,
        config: Gemma4ModelConfig,
        device: MTLDevice
    ) throws {
        _ = device  // Reserved for future staging buffers; held to match factory signature.

        self.tokenEmbedding = try Self.require("token_embd.weight", from: weightMap)
        self.outputNorm = try Self.require("output_norm.weight", from: weightMap)

        let ple = try Self.requirePLE("per_layer_token_embd.weight", from: weightMap)
        guard Self.allowedPLEQuants.contains(ple.dataType) else {
            throw Gemma4LoadError.unsupportedPLEQuant(Self.quantName(for: ple.dataType))
        }
        self.perLayerTokenEmbed = ple
        self.perLayerModelProjection = try Self.requirePLE(
            "per_layer_model_proj.weight",
            from: weightMap
        )
        self.perLayerProjectionNorm = try Self.requirePLE(
            "per_layer_proj_norm.weight",
            from: weightMap
        )

        var blocks: [Gemma4BlockWeights] = []
        blocks.reserveCapacity(config.numHiddenLayers)
        for layer in 0..<config.numHiddenLayers {
            let prefix = "blk.\(layer)"
            let ownsKV = config.kvSourceLayer(for: layer) == layer
            let attnK: TensorStorage? = ownsKV
                ? try Self.require("\(prefix).attn_k.weight", from: weightMap)
                : nil
            let attnV: TensorStorage? = ownsKV
                ? try Self.require("\(prefix).attn_v.weight", from: weightMap)
                : nil

            let block = Gemma4BlockWeights(
                inputNorm: try Self.require("\(prefix).attn_norm.weight", from: weightMap),
                attnQ: try Self.require("\(prefix).attn_q.weight", from: weightMap),
                attnK: attnK,
                attnV: attnV,
                attnO: try Self.require("\(prefix).attn_output.weight", from: weightMap),
                postAttentionNorm: try Self.require(
                    "\(prefix).post_attention_norm.weight",
                    from: weightMap
                ),
                ffnGate: try Self.require("\(prefix).ffn_gate.weight", from: weightMap),
                ffnUp: try Self.require("\(prefix).ffn_up.weight", from: weightMap),
                ffnDown: try Self.require("\(prefix).ffn_down.weight", from: weightMap),
                postFFNNorm: try Self.require(
                    "\(prefix).post_ffw_norm.weight",
                    from: weightMap
                ),
                perLayerInputGate: try Self.require(
                    "\(prefix).inp_gate.weight",
                    from: weightMap
                ),
                perLayerProjection: try Self.require(
                    "\(prefix).proj.weight",
                    from: weightMap
                ),
                postPerLayerInputNorm: try Self.require(
                    "\(prefix).post_norm.weight",
                    from: weightMap
                )
            )
            blocks.append(block)
        }
        self.blocks = blocks
    }

    // MARK: - Private

    private static let allowedPLEQuants: Set<TensorDataType> = [
        .q8_0, .q5_0, .q5_1, .q4_0, .q4_1, .float16, .float32, .bfloat16
    ]

    private static func require(
        _ name: String,
        from weightMap: WeightMap
    ) throws -> TensorStorage {
        guard let tensor = weightMap[name] else {
            throw Gemma4LoadError.missingTensor(name)
        }
        return tensor
    }

    private static func requirePLE(
        _ name: String,
        from weightMap: WeightMap
    ) throws -> TensorStorage {
        guard let tensor = weightMap[name] else {
            throw Gemma4LoadError.missingPLETensor(name)
        }
        return tensor
    }

    private static func quantName(for dataType: TensorDataType) -> String {
        // Enum case names match the GGUF quant identifiers closely enough for
        // diagnostic use — e.g. `.q4_K` renders as "q4_K", `.float16` as "float16".
        // Callers consume this only for error messages, not wire-format matching.
        return String(describing: dataType)
    }
}
