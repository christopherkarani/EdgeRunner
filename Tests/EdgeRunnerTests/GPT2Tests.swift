import Foundation
import Testing
@testable import EdgeRunner

@Suite("GPT-2 Reference Model")
struct GPT2Tests {
    static let gpt2Config = GPT2Config(
        vocabSize: 50_257,
        maxSeqLen: 1_024,
        numLayers: 12,
        numHeads: 12,
        hiddenDim: 768,
        layerNormEps: 1e-5
    )

    static let tinyConfig = GPT2Config(
        vocabSize: 32,
        maxSeqLen: 16,
        numLayers: 2,
        numHeads: 2,
        hiddenDim: 32,
        layerNormEps: 1e-5
    )

    @Test func gpt2ConfigProperties() {
        let config = Self.gpt2Config
        #expect(config.headDim == 64)
        #expect(config.intermediateSize == 3072)
    }

    @Test func embeddingLookup() async throws {
        let vocabSize = 8
        let dimension = 4
        var table = [Float](repeating: 0, count: vocabSize * dimension)
        for row in 0..<vocabSize {
            for column in 0..<dimension {
                table[row * dimension + column] = Float(row) * 0.1
            }
        }

        let embedding = Embedding(weight: table, vocabSize: vocabSize, dim: dimension)
        let result = try await embedding.forward([2, 5])

        #expect(result.count == 2 * dimension)
        for dimensionIndex in 0..<dimension {
            #expect(abs(result[dimensionIndex] - 0.2) < 1e-6)
            #expect(abs(result[dimension + dimensionIndex] - 0.5) < 1e-6)
        }
    }

    @Test func embeddingParameters() {
        let embedding = Embedding(
            weight: [Float](repeating: 0, count: 32 * 16),
            vocabSize: 32,
            dim: 16
        )
        let parameters = embedding.parameters
        #expect(parameters.keys.contains("weight"))
        #expect(parameters["weight"]?.elementCount == 32 * 16)
    }

    @Test func gpt2ForwardOutputShape() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIds = [1, 5, 10]
        let logits = try await model.forward(tokenIds)

        #expect(logits.count == tokenIds.count * config.vocabSize)
    }

    @Test func gpt2ForwardDeterministic() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIds = [1, 2, 3]
        let logits1 = try await model.forward(tokenIds)
        let logits2 = try await model.forward(tokenIds)

        for index in 0..<logits1.count {
            #expect(abs(logits1[index] - logits2[index]) < 1e-5)
        }
    }

    @Test func gpt2SingleToken() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let logits = try await model.forward([0])
        #expect(logits.count == config.vocabSize)
        for index in 0..<logits.count {
            #expect(logits[index].isFinite)
        }
    }

    @Test func gpt2ParameterCount() throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)
        let parameters = model.parameters
        let totalParameters = parameters.values.reduce(0) { $0 + $1.elementCount }

        #expect(totalParameters > 0)

        let hasTokenEmbedding = parameters.keys.contains { $0.contains("token_embedding") }
        let hasPositionEmbedding = parameters.keys.contains { $0.contains("position_embedding") }
        let hasBlocks = parameters.keys.contains { $0.contains("blocks.0") }
        let hasFinalLayerNorm = parameters.keys.contains { $0.contains("ln_final") }

        #expect(hasTokenEmbedding)
        #expect(hasPositionEmbedding)
        #expect(hasBlocks)
        #expect(hasFinalLayerNorm)
    }

    @Test func gpt2SoftmaxSumsToOne() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let logits = try await model.forward([1, 2])
        let lastTokenLogits = Array(logits[config.vocabSize..<(2 * config.vocabSize)])
        let maxLogit = lastTokenLogits.max() ?? 0
        let exponentials = lastTokenLogits.map { exp($0 - maxLogit) }
        let sumExp = exponentials.reduce(0, +)
        let probabilities = exponentials.map { $0 / sumExp }
        let probabilitySum = probabilities.reduce(0, +)

        #expect(abs(probabilitySum - 1.0) < 1e-5)
    }

    @Test func gpt2WeightShapes() throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)
        let parameters = model.parameters

        if let tokenEmbedding = parameters["token_embedding.weight"] {
            #expect(tokenEmbedding.shape == [config.vocabSize, config.hiddenDim])
        }

        if let positionEmbedding = parameters["position_embedding.weight"] {
            #expect(positionEmbedding.shape == [config.maxSeqLen, config.hiddenDim])
        }
    }
}
