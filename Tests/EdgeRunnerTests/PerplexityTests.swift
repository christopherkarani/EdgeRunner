import Foundation
import Testing
@testable import EdgeRunner

@Suite("Perplexity Computation")
struct PerplexityTests {
    @Test func perplexityOfUniformDistribution() {
        let vocabSize = 10
        let logits = [Float](repeating: 0, count: vocabSize)
        let targetID = 3

        let negativeLogLikelihood = Perplexity.negLogLikelihood(logits: logits, targetId: targetID)
        let perplexity = exp(negativeLogLikelihood)

        #expect(abs(perplexity - Float(vocabSize)) < 0.01)
    }

    @Test func perplexityOfPerfectPrediction() {
        var logits = [Float](repeating: -100, count: 10)
        logits[5] = 100
        let targetID = 5

        let negativeLogLikelihood = Perplexity.negLogLikelihood(logits: logits, targetId: targetID)
        let perplexity = exp(negativeLogLikelihood)

        #expect(perplexity < 1.01)
    }

    @Test func perplexityOfWrongPrediction() {
        var logits = [Float](repeating: -100, count: 10)
        logits[0] = 100
        let targetID = 5

        let negativeLogLikelihood = Perplexity.negLogLikelihood(logits: logits, targetId: targetID)
        #expect(negativeLogLikelihood > 10.0)
    }

    @Test func sequencePerplexity() {
        let vocabSize = 10
        let sequenceLength = 5
        let allLogits = [[Float]](
            repeating: [Float](repeating: 0, count: vocabSize),
            count: sequenceLength
        )
        let targetIDs = [1, 3, 5, 7, 9]

        let perplexity = Perplexity.compute(logitsPerToken: allLogits, targetIds: targetIDs)
        #expect(abs(perplexity - Float(vocabSize)) < 0.01)
    }

    @Test func perplexityNumericalStability() {
        var logits = [Float](repeating: 0, count: 100)
        logits[50] = 1000
        let targetID = 50

        let negativeLogLikelihood = Perplexity.negLogLikelihood(logits: logits, targetId: targetID)
        #expect(negativeLogLikelihood.isFinite)
        #expect(negativeLogLikelihood >= 0)
    }
}
