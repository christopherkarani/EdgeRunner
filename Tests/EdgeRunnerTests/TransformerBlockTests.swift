import Foundation
import Testing
@testable import EdgeRunner

private func ropeReference(_ values: [Float], position: Int, theta: Float) -> [Float] {
    var rotated = values
    let pairCount = values.count / 2

    for pairIndex in 0..<pairCount {
        let evenIndex = pairIndex * 2
        let oddIndex = evenIndex + 1
        let exponent = Float(evenIndex) / Float(values.count)
        let inverseFrequency = pow(theta, -exponent)
        let angle = Float(position) * inverseFrequency
        let cosAngle = cos(angle)
        let sinAngle = sin(angle)
        let evenValue = values[evenIndex]
        let oddValue = values[oddIndex]
        rotated[evenIndex] = evenValue * cosAngle - oddValue * sinAngle
        rotated[oddIndex] = evenValue * sinAngle + oddValue * cosAngle
    }

    return rotated
}

@Suite("Transformer Block")
struct TransformerBlockTests {
    static let tinyConfig = TransformerConfig(
        hiddenDim: 64,
        numHeads: 4,
        numKVHeads: 4,
        intermediateSize: 128,
        numLayers: 1,
        vocabSize: 32,
        maxSeqLen: 32,
        rmsNormEps: 1e-5,
        ropeTheta: 10_000.0
    )

    @Test func multiHeadAttentionOutputShape() async throws {
        let config = Self.tinyConfig
        let attention = try MultiHeadAttention(config: config)

        let sequenceLength = 4
        let input = (0..<(sequenceLength * config.hiddenDim)).map { _ in
            Float.random(in: -0.1...0.1)
        }
        let output = try await attention.forward(
            AttentionInput(hidden: input, seqLen: sequenceLength, startPos: 0)
        )

        #expect(output.count == sequenceLength * config.hiddenDim)
    }

    @Test func multiHeadAttentionParameters() throws {
        let config = Self.tinyConfig
        let attention = try MultiHeadAttention(config: config)
        let parameters = attention.parameters

        #expect(parameters.keys.contains("wq.weight"))
        #expect(parameters.keys.contains("wk.weight"))
        #expect(parameters.keys.contains("wv.weight"))
        #expect(parameters.keys.contains("wo.weight"))
    }

    @Test func multiHeadAttentionDeterministic() async throws {
        let config = Self.tinyConfig
        let attention = try MultiHeadAttention(config: config)
        let sequenceLength = 4
        let input = (0..<(sequenceLength * config.hiddenDim)).map { Float($0) * 0.01 }

        let output1 = try await attention.forward(
            AttentionInput(hidden: input, seqLen: sequenceLength, startPos: 0)
        )
        let output2 = try await attention.forward(
            AttentionInput(hidden: input, seqLen: sequenceLength, startPos: 0)
        )

        for index in 0..<output1.count {
            #expect(abs(output1[index] - output2[index]) < 1e-6)
        }
    }

    @Test func rotaryHelperUsesAbsolutePosition() {
        let head: [Float] = [1, 2, 3, 4]
        let rotated0 = MultiHeadAttention.applyRoPE(head, position: 0, theta: Self.tinyConfig.ropeTheta)
        let rotated1 = MultiHeadAttention.applyRoPE(head, position: 1, theta: Self.tinyConfig.ropeTheta)
        let rotated3 = MultiHeadAttention.applyRoPE(head, position: 3, theta: Self.tinyConfig.ropeTheta)

        #expect(rotated0 == head)
        #expect(rotated1 != rotated0)
        #expect(rotated3 != rotated1)
    }

    @Test func multiHeadAttentionAppliesRotaryEmbeddings() async throws {
        let config = TransformerConfig(
            hiddenDim: 4,
            numHeads: 1,
            numKVHeads: 1,
            intermediateSize: 8,
            numLayers: 1,
            vocabSize: 16,
            maxSeqLen: 16,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000.0
        )
        let identity: [Float] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        let wq = try LinearModule(inFeatures: 4, outFeatures: 4, weight: identity, bias: nil)
        let wk = try LinearModule(inFeatures: 4, outFeatures: 4, weight: identity, bias: nil)
        let wv = try LinearModule(inFeatures: 4, outFeatures: 4, weight: identity, bias: nil)
        let wo = try LinearModule(inFeatures: 4, outFeatures: 4, weight: identity, bias: nil)
        let attention = MultiHeadAttention(config: config, wq: wq, wk: wk, wv: wv, wo: wo)

        let token0: [Float] = [1, 2, 3, 4]
        let token1: [Float] = [4, 3, 2, 1]
        let output = try await attention.forward(
            AttentionInput(hidden: token0 + token1, seqLen: 2, startPos: 0)
        )

        let k0 = ropeReference(token0, position: 0, theta: config.ropeTheta)
        let q1 = ropeReference(token1, position: 1, theta: config.ropeTheta)
        let k1 = ropeReference(token1, position: 1, theta: config.ropeTheta)
        let scale = 1.0 / sqrt(Float(config.headDim))
        let score10 = zip(q1, k0).reduce(0) { $0 + $1.0 * $1.1 } * scale
        let score11 = zip(q1, k1).reduce(0) { $0 + $1.0 * $1.1 } * scale
        let maxScore = max(score10, score11)
        let exp10 = exp(score10 - maxScore)
        let exp11 = exp(score11 - maxScore)
        let prob10 = exp10 / (exp10 + exp11)
        let prob11 = exp11 / (exp10 + exp11)
        let expectedToken1 = zip(token0, token1).map { prob10 * $0 + prob11 * $1 }
        let expected = token0 + expectedToken1

        #expect(output.count == expected.count)
        for index in 0..<expected.count {
            #expect(abs(output[index] - expected[index]) < 1e-5)
        }
    }

    @Test func feedForwardOutputShape() async throws {
        let config = Self.tinyConfig
        let feedForward = try FeedForward(config: config)

        let sequenceLength = 4
        let input = (0..<(sequenceLength * config.hiddenDim)).map { _ in
            Float.random(in: -0.1...0.1)
        }
        let output = try await feedForward.forward(input)

        #expect(output.count == sequenceLength * config.hiddenDim)
    }

    @Test func feedForwardParameters() throws {
        let config = Self.tinyConfig
        let feedForward = try FeedForward(config: config)
        let parameters = feedForward.parameters

        #expect(parameters.keys.contains("gate.weight"))
        #expect(parameters.keys.contains("up.weight"))
        #expect(parameters.keys.contains("down.weight"))
    }

    @Test func transformerBlockOutputShape() async throws {
        let config = Self.tinyConfig
        let block = try TransformerBlock(config: config, layerIndex: 0)

        let sequenceLength = 4
        let input = (0..<(sequenceLength * config.hiddenDim)).map { _ in
            Float.random(in: -0.1...0.1)
        }
        let output = try await block.forward(
            TransformerBlockInput(hidden: input, seqLen: sequenceLength, startPos: 0)
        )

        #expect(output.hidden.count == sequenceLength * config.hiddenDim)
    }

    @Test func transformerBlockResidualConnection() async throws {
        let config = Self.tinyConfig
        let block = try TransformerBlock(config: config, layerIndex: 0, zeroWeights: true)

        let sequenceLength = 2
        let input = (0..<(sequenceLength * config.hiddenDim)).map { _ in
            Float.random(in: -1...1)
        }
        let output = try await block.forward(
            TransformerBlockInput(hidden: input, seqLen: sequenceLength, startPos: 0)
        )

        for index in 0..<input.count {
            #expect(abs(output.hidden[index] - input[index]) < 1e-4)
        }
    }

    @Test func transformerBlockParameters() throws {
        let config = Self.tinyConfig
        let block = try TransformerBlock(config: config, layerIndex: 0)
        let parameters = block.parameters

        let hasAttentionNorm = parameters.keys.contains { $0.hasPrefix("attention_norm") }
        let hasFFNNorm = parameters.keys.contains { $0.hasPrefix("ffn_norm") }
        let hasAttention = parameters.keys.contains { $0.hasPrefix("attention.") }
        let hasFeedForward = parameters.keys.contains { $0.hasPrefix("ffn.") }

        #expect(hasAttentionNorm)
        #expect(hasFFNNorm)
        #expect(hasAttention)
        #expect(hasFeedForward)
    }
}
