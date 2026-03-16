import Testing
@testable import EdgeRunner

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
