import Foundation
import Testing
@testable import EdgeRunnerCore

private func softmax(_ logits: [Float]) -> [Float] {
    let maxLogit = logits.max() ?? 0
    let exps = logits.map { exp($0 - maxLogit) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

@Suite("GreedySampler")
struct GreedySamplerTests {
    @Test func selectsHighestLogit() {
        let sampler = GreedySampler()
        let logits: [Float] = [1.0, 3.0, 2.0, 0.5]
        #expect(sampler.sample(logits: logits) == 1)
    }
    @Test func handlesNegativeLogits() {
        let sampler = GreedySampler()
        #expect(sampler.sample(logits: [-5.0, -1.0, -3.0, -2.0]) == 1)
    }
    @Test func handlesSingleElement() {
        #expect(GreedySampler().sample(logits: [42.0]) == 0)
    }
    @Test func tieBreaksToFirst() {
        #expect(GreedySampler().sample(logits: [5.0, 5.0, 5.0]) == 0)
    }
}

@Suite("TemperatureSampler")
struct TemperatureSamplerTests {
    @Test func temperatureZeroIsGreedy() {
        let result = TemperatureSampler(temperature: 0.0).transformLogits([1.0, 5.0, 2.0])
        let maxIndex = result.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(maxIndex == 1)
    }
    @Test func temperatureOnePreservesDistribution() {
        let logits: [Float] = [1.0, 2.0, 3.0]
        let result = TemperatureSampler(temperature: 1.0).transformLogits(logits)
        for i in 0..<3 { #expect(abs(result[i] - logits[i]) < 1e-6) }
    }
    @Test func highTemperatureFlattensDistribution() {
        let result = TemperatureSampler(temperature: 100.0).transformLogits([1.0, 10.0, 1.0])
        let probs = softmax(result)
        for p in probs { #expect(abs(p - 1.0 / 3.0) < 0.05) }
    }
    @Test func lowTemperatureSharpenDistribution() {
        let result = TemperatureSampler(temperature: 0.01).transformLogits([1.0, 3.0, 2.0])
        let probs = softmax(result)
        #expect(probs[1] > 0.99)
    }
}

@Suite("TopKSampler")
struct TopKSamplerTests {
    @Test func filtersToTopK() {
        let result = TopKSampler(k: 2).transformLogits([1.0, 5.0, 3.0, 2.0])
        #expect(result[1] == 5.0)
        #expect(result[2] == 3.0)
        #expect(result[0] == -.infinity)
        #expect(result[3] == -.infinity)
    }
    @Test func kGreaterThanVocabKeepsAll() {
        let logits: [Float] = [1.0, 2.0, 3.0]
        #expect(TopKSampler(k: 100).transformLogits(logits) == logits)
    }
    @Test func kOfOneIsGreedy() {
        let result = TopKSampler(k: 1).transformLogits([1.0, 5.0, 3.0])
        #expect(result[1] == 5.0)
        #expect(result[0] == -.infinity)
        #expect(result[2] == -.infinity)
    }
}

@Suite("TopPSampler")
struct TopPSamplerTests {
    @Test func pOfOneKeepsAll() {
        let result = TopPSampler(p: 1.0).transformLogits([1.0, 2.0, 3.0])
        for i in 0..<3 { #expect(result[i] > -.infinity) }
    }
    @Test func lowPKeepsFewTokens() {
        let result = TopPSampler(p: 0.1).transformLogits([0.0, 5.0, 0.5])
        #expect(result[1] == 5.0)
        #expect(result[0] == -.infinity)
        #expect(result[2] == -.infinity)
    }
    @Test func moderatePKeepsTopTokens() {
        let result = TopPSampler(p: 0.9).transformLogits([2.0, 2.1, 2.2, -10.0, -10.0])
        #expect(result[0] > -.infinity)
        #expect(result[1] > -.infinity)
        #expect(result[2] > -.infinity)
        #expect(result[3] == -.infinity)
        #expect(result[4] == -.infinity)
    }
}

@Suite("MinPSampler")
struct MinPSamplerTests {
    @Test func filtersTokensBelowMinProbability() {
        let result = MinPSampler(minP: 0.1).transformLogits([10.0, 1.0, 1.0, 1.0])
        #expect(result[0] == 10.0)
        #expect(result[1] == -.infinity)
        #expect(result[2] == -.infinity)
        #expect(result[3] == -.infinity)
    }
    @Test func minPZeroKeepsAll() {
        let result = MinPSampler(minP: 0.0).transformLogits([10.0, 1.0, 0.5])
        for i in 0..<3 { #expect(result[i] > -.infinity) }
    }
}

@Suite("RepetitionPenalty")
struct RepetitionPenaltyTests {
    @Test func penalizesRepeatedTokens() {
        let result = RepetitionPenalty(penalty: 1.5).apply(logits: [2.0, 3.0, 1.0, 4.0], previousTokens: [1, 3])
        #expect(result[0] == 2.0)
        #expect(abs(result[1] - 3.0 / 1.5) < 1e-6)
        #expect(result[2] == 1.0)
        #expect(abs(result[3] - 4.0 / 1.5) < 1e-6)
    }
    @Test func penalizesNegativeLogitsByMultiplying() {
        let result = RepetitionPenalty(penalty: 2.0).apply(logits: [-1.0, -2.0, 3.0], previousTokens: [0, 1])
        #expect(abs(result[0] - (-1.0 * 2.0)) < 1e-6)
        #expect(abs(result[1] - (-2.0 * 2.0)) < 1e-6)
        #expect(result[2] == 3.0)
    }
    @Test func penaltyOfOneNoChange() {
        let logits: [Float] = [2.0, 3.0, 1.0]
        let result = RepetitionPenalty(penalty: 1.0).apply(logits: logits, previousTokens: [0, 1, 2])
        for i in 0..<3 { #expect(abs(result[i] - logits[i]) < 1e-6) }
    }
    @Test func frequencyPenalty() {
        let result = RepetitionPenalty(penalty: 1.0, frequencyPenalty: 0.5).apply(logits: [2.0, 3.0, 1.0], previousTokens: [1, 1, 0])
        #expect(abs(result[1] - (3.0 - 1.0)) < 1e-6)
        #expect(abs(result[0] - (2.0 - 0.5)) < 1e-6)
        #expect(abs(result[2] - 1.0) < 1e-6)
    }
}

@Suite("SeededRandomSource")
struct SeededRandomSourceTests {
    @Test func deterministicWithSameSeed() {
        var rng1 = SeededRandomSource(seed: 42)
        var rng2 = SeededRandomSource(seed: 42)
        let values1 = (0..<10).map { _ in Float.random(in: 0...1, using: &rng1) }
        let values2 = (0..<10).map { _ in Float.random(in: 0...1, using: &rng2) }
        #expect(values1 == values2)
    }
    @Test func differentSeedsProduceDifferentValues() {
        var rng1 = SeededRandomSource(seed: 42)
        var rng2 = SeededRandomSource(seed: 99)
        #expect(Float.random(in: 0...1, using: &rng1) != Float.random(in: 0...1, using: &rng2))
    }
}

@Suite("SamplingPipeline")
struct SamplingPipelineTests {
    @Test func greedyPipelineSelectsMax() {
        #expect(SamplingPipeline(transforms: [], selector: GreedySampler()).sample(logits: [1.0, 5.0, 3.0]) == 1)
    }
    @Test func temperaturePlusSampling() {
        var rng = SeededRandomSource(seed: 12345)
        let pipeline = SamplingPipeline(
            transforms: [TemperatureSampler(temperature: 0.001)],
            selector: StochasticSampler(randomSource: &rng)
        )
        #expect(pipeline.sample(logits: [1.0, 10.0, 2.0]) == 1)
    }
    @Test func topKPlusTopPPipeline() {
        let pipeline = SamplingPipeline(
            transforms: [TopKSampler(k: 2), TopPSampler(p: 0.9)],
            selector: GreedySampler()
        )
        #expect(pipeline.sample(logits: [1.0, 5.0, 3.0, 0.5]) == 1)
    }
    @Test func fullPipelineWithRepetitionPenalty() {
        let pipeline = SamplingPipeline(
            transforms: [TemperatureSampler(temperature: 1.0), TopKSampler(k: 3)],
            selector: GreedySampler(),
            repetitionPenalty: RepetitionPenalty(penalty: 100.0)
        )
        #expect(pipeline.sample(logits: [5.0, 4.9, 4.8, 1.0], previousTokens: [0]) == 1)
    }
    @Test func defaultPipelineIsGreedy() {
        #expect(SamplingPipeline.greedy.sample(logits: [1.0, 9.0, 3.0]) == 1)
    }
    @Test func topPWithTemperature() {
        #expect(SamplingPipeline.nucleus(temperature: 0.001, topP: 0.9).sample(logits: [1.0, 10.0, 2.0, 0.5]) == 1)
    }
}
