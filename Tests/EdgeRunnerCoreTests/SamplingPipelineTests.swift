import Foundation
import Testing
@testable import EdgeRunnerCore
@testable import EdgeRunner

@Suite("SamplingConfiguration.toPipeline")
struct SamplingConfigurationPipelineTests {

    @Test func defaultConfigProducesStochasticSampling() {
        // Near-uniform logits — stochastic sampling should produce varied results
        let logits: [Float] = [1.0, 1.01, 1.02, 0.99, 0.98]
        var results = Set<Int>()
        for _ in 0..<50 {
            // Each call creates a fresh pipeline with a random seed
            let p = SamplingConfiguration().toPipeline()
            results.insert(p.sample(logits: logits))
        }
        // With near-uniform logits and stochastic sampling, we expect more than 1 distinct token
        #expect(results.count > 1, "Stochastic sampling with near-uniform logits should produce varied results")
    }

    @Test func zeroTemperatureIsGreedy() {
        let config = SamplingConfiguration(temperature: 0)
        let pipeline = config.toPipeline()
        let logits: [Float] = [1.0, 5.0, 3.0, 2.0]
        for _ in 0..<10 {
            #expect(pipeline.sample(logits: logits) == 1, "Temperature 0 should always pick argmax")
        }
    }

    @Test func repetitionPenaltyReducesRepeats() {
        let config = SamplingConfiguration(
            temperature: 0,
            repetitionPenalty: 2.0
        )
        let pipeline = config.toPipeline()
        // Token 0 has highest logit, but is in history — penalty should suppress it
        let logits: [Float] = [5.0, 4.9, 1.0, 0.5]
        let result = pipeline.sample(logits: logits, previousTokens: [0])
        #expect(result != 0, "Token 0 should be penalized and not selected")
        #expect(result == 1, "Token 1 should be selected after token 0 is penalized")
    }

    @Test func seededSamplingIsDeterministic() {
        let config = SamplingConfiguration(seed: 42)
        let pipeline1 = config.toPipeline()
        let pipeline2 = config.toPipeline()
        let logits: [Float] = [1.0, 1.01, 1.02, 0.99, 0.98, 1.03, 0.97]
        let results1 = (0..<20).map { _ in pipeline1.sample(logits: logits) }
        let results2 = (0..<20).map { _ in pipeline2.sample(logits: logits) }
        #expect(results1 == results2, "Same seed should produce identical sampling sequences")
    }
}
