import Foundation

/// Input to a single transformer block.
public struct TransformerBlockInput: Sendable {
    public let hidden: [Float]
    public let seqLen: Int
    public let startPos: Int

    public init(hidden: [Float], seqLen: Int, startPos: Int) {
        self.hidden = hidden
        self.seqLen = seqLen
        self.startPos = startPos
    }
}

/// Output from a single transformer block.
public struct TransformerBlockOutput: Sendable {
    public let hidden: [Float]

    public init(hidden: [Float]) {
        self.hidden = hidden
    }
}

/// Pre-norm transformer decoder block with residual connections.
public struct TransformerBlock: EdgeRunnerModule, Sendable {
    public typealias Input = TransformerBlockInput
    public typealias Output = TransformerBlockOutput

    private let config: TransformerConfig
    private let layerIndex: Int
    private let attention: MultiHeadAttention
    private let feedForward: FeedForward
    private let attentionNormWeight: [Float]
    private let ffnNormWeight: [Float]

    public init(
        config: TransformerConfig,
        layerIndex: Int,
        zeroWeights: Bool = false
    ) throws {
        self.config = config
        self.layerIndex = layerIndex
        self.attention = try MultiHeadAttention(config: config, zeroWeights: zeroWeights)
        self.feedForward = try FeedForward(config: config, zeroWeights: zeroWeights)
        self.attentionNormWeight = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ffnNormWeight = [Float](repeating: 1.0, count: config.hiddenDim)
    }

    public func forward(_ input: TransformerBlockInput) async throws -> TransformerBlockOutput {
        let sequenceLength = input.seqLen
        let hiddenDim = config.hiddenDim
        let epsilon = config.rmsNormEps

        let normalizedAttentionInput = cpuRMSNorm(
            input.hidden,
            weight: attentionNormWeight,
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

        let normalizedFeedForwardInput = cpuRMSNorm(
            hidden,
            weight: ffnNormWeight,
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
        parameters["attention_norm.weight"] = ArrayTensorBox(
            data: attentionNormWeight,
            shape: [config.hiddenDim]
        )
        parameters["ffn_norm.weight"] = ArrayTensorBox(
            data: ffnNormWeight,
            shape: [config.hiddenDim]
        )
        for (key, value) in attention.parameters {
            parameters["attention.\(key)"] = value
        }
        for (key, value) in feedForward.parameters {
            parameters["ffn.\(key)"] = value
        }
        return parameters
    }

    private func cpuRMSNorm(
        _ input: [Float],
        weight: [Float],
        rows: Int,
        cols: Int,
        eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for rowIndex in 0..<rows {
            let offset = rowIndex * cols
            let row = Array(input[offset..<(offset + cols)])
            let meanSquare = row.reduce(0) { $0 + $1 * $1 } / Float(cols)
            let scale = 1.0 / sqrt(meanSquare + eps)
            for columnIndex in 0..<cols {
                output[offset + columnIndex] = row[columnIndex] * scale * weight[columnIndex]
            }
        }
        return output
    }
}
