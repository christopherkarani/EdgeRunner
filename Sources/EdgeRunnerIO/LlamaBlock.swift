import Foundation

public struct LlamaNorm: Sendable, Equatable {
    public let dim: Int
    public let epsilon: Double

    public init(dim: Int, epsilon: Double) {
        self.dim = dim
        self.epsilon = epsilon
    }
}

public struct LlamaAttention: Sendable, Equatable {
    public let config: LlamaConfig

    public init(config: LlamaConfig) {
        self.config = config
    }
}

public struct LlamaFeedForward: Sendable, Equatable {
    public let inputDim: Int
    public let hiddenDim: Int

    public init(inputDim: Int, hiddenDim: Int) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
    }
}

public final class LlamaBlock: Sendable {
    public let layerIndex: Int
    public let config: LlamaConfig
    public let attentionNorm: LlamaNorm
    public let ffnNorm: LlamaNorm
    public let attention: LlamaAttention
    public let feedForward: LlamaFeedForward

    private let prefix: String

    public init(config: LlamaConfig, layerIndex: Int) {
        self.config = config
        self.layerIndex = layerIndex
        self.prefix = "layers.\(layerIndex)"
        self.attentionNorm = LlamaNorm(dim: config.embeddingDim, epsilon: config.rmsNormEpsilon)
        self.ffnNorm = LlamaNorm(dim: config.embeddingDim, epsilon: config.rmsNormEpsilon)
        self.attention = LlamaAttention(config: config)
        self.feedForward = LlamaFeedForward(
            inputDim: config.embeddingDim,
            hiddenDim: config.intermediateDim
        )
    }

    public var parameterNames: [String] {
        [
            "\(prefix).attentionNorm.weight",
            "\(prefix).ffnNorm.weight",
            "\(prefix).attention.wq.weight",
            "\(prefix).attention.wk.weight",
            "\(prefix).attention.wv.weight",
            "\(prefix).attention.wo.weight",
            "\(prefix).feedForward.gate.weight",
            "\(prefix).feedForward.up.weight",
            "\(prefix).feedForward.down.weight",
        ]
    }
}

public enum LlamaWeightNameMapper: Sendable {
    public static func mapGGUFName(_ ggufName: String) -> String {
        switch ggufName {
        case "token_embd.weight":
            return "embedding.weight"
        case "output_norm.weight":
            return "finalNorm.weight"
        case "output.weight":
            return "lmHead.weight"
        default:
            break
        }

        guard ggufName.hasPrefix("blk.") else {
            return ggufName
        }

        let parts = ggufName.split(separator: ".", maxSplits: 2)
        guard parts.count == 3, let layerIndex = Int(parts[1]) else {
            return ggufName
        }

        let suffix = String(parts[2])
        let mappedSuffix: String
        switch suffix {
        case "attn_norm.weight":
            mappedSuffix = "attentionNorm.weight"
        case "ffn_norm.weight":
            mappedSuffix = "ffnNorm.weight"
        case "attn_q.weight":
            mappedSuffix = "attention.wq.weight"
        case "attn_k.weight":
            mappedSuffix = "attention.wk.weight"
        case "attn_v.weight":
            mappedSuffix = "attention.wv.weight"
        case "attn_output.weight":
            mappedSuffix = "attention.wo.weight"
        case "ffn_gate.weight":
            mappedSuffix = "feedForward.gate.weight"
        case "ffn_up.weight":
            mappedSuffix = "feedForward.up.weight"
        case "ffn_down.weight":
            mappedSuffix = "feedForward.down.weight"
        default:
            return ggufName
        }

        return "layers.\(layerIndex).\(mappedSuffix)"
    }
}
