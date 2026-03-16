import Foundation
import Testing
@testable import EdgeRunner

@Suite("Integration Tests")
struct IntegrationTests {
    static let tinyConfig = GPT2Config(
        vocabSize: 32,
        maxSeqLen: 16,
        numLayers: 2,
        numHeads: 2,
        hiddenDim: 32,
        layerNormEps: 1e-5
    )

    @Test func endToEndForwardPass() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIDs = [1, 5, 10, 15]
        let logits = try await model.forward(tokenIDs)

        #expect(logits.count == tokenIDs.count * config.vocabSize)
        for index in 0..<logits.count {
            #expect(logits[index].isFinite)
        }
    }

    @Test func endToEndWithPerplexity() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let inputIDs = [1, 5, 10, 15, 20]
        let targetIDs = [5, 10, 15, 20, 25]
        let logits = try await model.forward(inputIDs)

        var logitsPerToken: [[Float]] = []
        logitsPerToken.reserveCapacity(inputIDs.count)
        for tokenIndex in 0..<inputIDs.count {
            let offset = tokenIndex * config.vocabSize
            logitsPerToken.append(Array(logits[offset..<(offset + config.vocabSize)]))
        }

        let perplexity = Perplexity.compute(logitsPerToken: logitsPerToken, targetIds: targetIDs)
        #expect(perplexity.isFinite)
        #expect(perplexity > 0)
        #expect(perplexity > 1.0)
    }

    @Test func cpuReferenceSmallInput() async throws {
        let config = GPT2Config(
            vocabSize: 4,
            maxSeqLen: 4,
            numLayers: 1,
            numHeads: 1,
            hiddenDim: 4,
            layerNormEps: 1e-5
        )
        let model = try GPT2Model(config: config)

        let logits1 = try await model.forward([0, 1])
        let logits2 = try await model.forward([0, 1])

        #expect(logits1.count == logits2.count)
        for index in 0..<logits1.count {
            #expect(abs(logits1[index] - logits2[index]) < 1e-5)
        }
    }

    @Test func causalMaskingVerification() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let shortLogits = try await model.forward([5])
        let longLogits = try await model.forward([5, 10, 15])

        let shortFirst = Array(shortLogits[0..<config.vocabSize])
        let longFirst = Array(longLogits[0..<config.vocabSize])

        for index in 0..<config.vocabSize {
            #expect(abs(shortFirst[index] - longFirst[index]) < 1e-4)
        }
    }

    @Test func performanceBaseline() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIDs = Array(0..<config.maxSeqLen)
        let startTime = CFAbsoluteTimeGetCurrent()

        _ = try await model.forward(tokenIDs)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let tokensPerSecond = Double(tokenIDs.count) / elapsed

        #expect(elapsed < 60.0)
        let rateString = String(format: "%.1f", tokensPerSecond)
        let elapsedString = String(format: "%.3f", elapsed)
        print(
            "Performance baseline: \(rateString) tokens/sec " +
            "(\(tokenIDs.count) tokens in \(elapsedString)s)"
        )
    }

    @Test func gradientFreeForwardOnly() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let parametersBefore = model.parameters
        _ = try await model.forward([1, 2, 3])
        let parametersAfter = model.parameters

        #expect(parametersBefore.count == parametersAfter.count)
        for key in parametersBefore.keys {
            let before = parametersBefore[key]!.floatArray
            let after = parametersAfter[key]!.floatArray
            #expect(before.count == after.count)
            for index in 0..<before.count {
                #expect(before[index] == after[index])
            }
        }
    }
}
