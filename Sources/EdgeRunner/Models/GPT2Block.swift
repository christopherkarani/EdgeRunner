import Foundation

/// A single GPT-2 transformer block using pre-norm LayerNorm.
public struct GPT2Block: EdgeRunnerModule, Sendable {
    public typealias Input = TransformerBlockInput
    public typealias Output = TransformerBlockOutput

    private let config: GPT2Config
    private let attention: GPT2Attention
    private let feedForward: GPT2FeedForward
    private let ln1Gamma: [Float]
    private let ln1Beta: [Float]
    private let ln2Gamma: [Float]
    private let ln2Beta: [Float]

    public init(config: GPT2Config) throws {
        self.config = config
        self.attention = try GPT2Attention(config: config)
        self.feedForward = try GPT2FeedForward(config: config)
        self.ln1Gamma = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ln1Beta = [Float](repeating: 0.0, count: config.hiddenDim)
        self.ln2Gamma = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ln2Beta = [Float](repeating: 0.0, count: config.hiddenDim)
    }

    public func forward(_ input: TransformerBlockInput) async throws -> TransformerBlockOutput {
        let sequenceLength = input.seqLen
        let hiddenDim = config.hiddenDim
        let epsilon = config.layerNormEps

        let normalizedAttentionInput = cpuLayerNorm(
            input.hidden,
            gamma: ln1Gamma,
            beta: ln1Beta,
            rows: sequenceLength,
            cols: hiddenDim,
            eps: epsilon
        )
        let attentionOutput = try await attention.forward(
            AttentionInput(
                hidden: normalizedAttentionInput,
                seqLen: sequenceLength,
                startPos: input.startPos
            )
        )
        var hidden = zip(input.hidden, attentionOutput).map { $0 + $1 }

        let normalizedFeedForwardInput = cpuLayerNorm(
            hidden,
            gamma: ln2Gamma,
            beta: ln2Beta,
            rows: sequenceLength,
            cols: hiddenDim,
            eps: epsilon
        )
        let feedForwardOutput = try await feedForward.forward(normalizedFeedForwardInput)
        hidden = zip(hidden, feedForwardOutput).map { $0 + $1 }

        return TransformerBlockOutput(hidden: hidden)
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]
        parameters["ln_1.weight"] = ArrayTensorBox(data: ln1Gamma, shape: [config.hiddenDim])
        parameters["ln_1.bias"] = ArrayTensorBox(data: ln1Beta, shape: [config.hiddenDim])
        parameters["ln_2.weight"] = ArrayTensorBox(data: ln2Gamma, shape: [config.hiddenDim])
        parameters["ln_2.bias"] = ArrayTensorBox(data: ln2Beta, shape: [config.hiddenDim])
        for (key, value) in attention.parameters {
            parameters["attn.\(key)"] = value
        }
        for (key, value) in feedForward.parameters {
            parameters["mlp.\(key)"] = value
        }
        return parameters
    }

    private func cpuLayerNorm(
        _ input: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        cols: Int,
        eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for rowIndex in 0..<rows {
            let offset = rowIndex * cols
            let row = Array(input[offset..<(offset + cols)])
            let mean = row.reduce(0, +) / Float(cols)
            let variance = row.reduce(0) { partial, value in
                let delta = value - mean
                return partial + delta * delta
            } / Float(cols)
            let invStd = 1.0 / sqrt(variance + eps)
            for columnIndex in 0..<cols {
                output[offset + columnIndex] =
                    (row[columnIndex] - mean) * invStd * gamma[columnIndex] + beta[columnIndex]
            }
        }
        return output
    }
}
