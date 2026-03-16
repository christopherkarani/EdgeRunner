import Testing
import Foundation
@testable import EdgeRunnerCore

@Suite("Sampling Benchmarks")
struct SamplingBenchmarks {

    // Use 32K vocab for benchmarks (realistic for many models, fast enough for CI)
    static let vocabSize = 32_000
    // Also test at Llama 3 scale for the headline number
    static let llamaVocabSize = 128_256

    static func randomLogits(size: Int) -> [Float] {
        (0..<size).map { _ in Float.random(in: -10...10) }
    }

    // MARK: - 32K Vocab (fast, CI-friendly)

    @Test func greedySampling_32k() {
        let sampler = GreedySampler()
        let logits = Self.randomLogits(size: Self.vocabSize)

        for _ in 0..<10 { _ = sampler.sample(logits: logits) }

        let iterations = 1000
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = sampler.sample(logits: logits)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: greedy_vocab32k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 50000) // Loose bound
    }

    @Test func topKSampling_32k() {
        let sampler = TopKSampler(k: 40)
        let logits = Self.randomLogits(size: Self.vocabSize)

        for _ in 0..<5 { _ = sampler.transformLogits(logits) }

        let iterations = 100
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = sampler.transformLogits(logits)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: topk_k40_vocab32k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 500000)
    }

    @Test func topPSampling_32k() {
        let sampler = TopPSampler(p: 0.9)
        let logits = Self.randomLogits(size: Self.vocabSize)

        for _ in 0..<5 { _ = sampler.transformLogits(logits) }

        let iterations = 100
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = sampler.transformLogits(logits)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: topp_p09_vocab32k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 500000)
    }

    @Test func fullPipeline_32k() {
        let pipeline = SamplingPipeline.nucleus(temperature: 0.8, topP: 0.9, seed: 42)
        let logits = Self.randomLogits(size: Self.vocabSize)
        let prevTokens = Array(0..<100)

        for _ in 0..<5 { _ = pipeline.sample(logits: logits, previousTokens: prevTokens) }

        let iterations = 100
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = pipeline.sample(logits: logits, previousTokens: prevTokens)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: full_pipeline_vocab32k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 1_000_000)
    }

    // MARK: - 128K Vocab (headline number, fewer iterations)

    @Test func greedySampling_128k() {
        let sampler = GreedySampler()
        let logits = Self.randomLogits(size: Self.llamaVocabSize)

        for _ in 0..<3 { _ = sampler.sample(logits: logits) }

        let iterations = 100
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = sampler.sample(logits: logits)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: greedy_vocab128k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 500_000) // ~18-150ms observed depending on system load
    }

    @Test func fullPipeline_128k() {
        let pipeline = SamplingPipeline.nucleus(temperature: 0.8, topP: 0.9, seed: 42)
        let logits = Self.randomLogits(size: Self.llamaVocabSize)
        let prevTokens = Array(0..<50)

        for _ in 0..<3 { _ = pipeline.sample(logits: logits, previousTokens: prevTokens) }

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = pipeline.sample(logits: logits, previousTokens: prevTokens)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let usPerSample = seconds / Double(iterations) * 1_000_000

        print("BENCHMARK: full_pipeline_vocab128k \(String(format: "%.1f", usPerSample)) µs/sample")
        #expect(usPerSample < 1_000_000) // ~330ms observed, allow up to 1s
    }
}
