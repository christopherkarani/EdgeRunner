import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

@Suite("Speculative Generation Benchmark")
struct SpeculativeGenerationBenchmark {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let expectedModelFileSizeBytes: Int64 = 639_447_744
    static let generateCount = 64
    static let benchmarkRuns = 3
    static let bosToken = 1
    static let draftTokenCount = 2

    @Test func selfSpeculativeLookahead() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: Self.modelPath)
        let modelFileSize = attrs[.size] as? Int64 ?? -1
        #expect(modelFileSize == Self.expectedModelFileSizeBytes)

        let greedyModel = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )
        let draftModel = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )
        let verifyModel = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Warm all three models so the comparison measures steady-state generation.
        _ = try await generateGreedyTokens(model: greedyModel, tokenCount: 8)
        _ = try await generateGreedyTokens(model: draftModel, tokenCount: 8)
        _ = try await generateGreedyTokens(model: verifyModel, tokenCount: 8)

        var greedyRuns = [Double]()
        var speculativeRuns = [Double]()
        var referenceTokens: [Int] = []

        for run in 0..<Self.benchmarkRuns {
            let greedy = try await measureGreedyGeneration(model: greedyModel, tokenCount: Self.generateCount)
            let speculative = try await measureSelfSpeculativeGeneration(
                draftModel: draftModel,
                verifyModel: verifyModel,
                tokenCount: Self.generateCount
            )

            if run == 0 { referenceTokens = greedy.tokens }

            greedyRuns.append(greedy.tokensPerSecond)
            speculativeRuns.append(speculative.tokensPerSecond)

            #expect(greedy.tokens == referenceTokens)
            #expect(speculative.tokens == referenceTokens)
        }

        let greedyMedian = median(greedyRuns)
        let speculativeMedian = median(speculativeRuns)
        let deltaPct = ((speculativeMedian / greedyMedian) - 1.0) * 100.0

        print("SPEC_BENCH: greedy_generation_median \(String(format: "%.1f", greedyMedian)) tok/s")
        print("SPEC_BENCH: self_spec_generation_median \(String(format: "%.1f", speculativeMedian)) tok/s")
        print("SPEC_BENCH: delta_pct \(String(format: "%.1f", deltaPct))")

        // This is an experimental benchmark, so correctness matters more than speed.
        #expect(greedyMedian > 0)
        #expect(speculativeMedian > 0)
    }

    private struct GenerationResult {
        let tokens: [Int]
        let tokensPerSecond: Double
    }

    private struct ModelAdapter: SpeculativeModel {
        let model: LlamaLanguageModel

        func logits(for tokenIDs: [Int]) async throws -> [Float] {
            try await model.logits(for: tokenIDs)
        }
    }

    private func measureGreedyGeneration(
        model: LlamaLanguageModel,
        tokenCount: Int
    ) async throws -> GenerationResult {
        let clock = ContinuousClock()
        let start = clock.now
        let tokens = try await generateGreedyTokens(model: model, tokenCount: tokenCount)
        let elapsedMs = durationMs(from: start, to: clock.now)
        let tokPerSec = Double(tokens.count - 1) / max(elapsedMs / 1000.0, 1e-9)
        return GenerationResult(tokens: tokens, tokensPerSecond: tokPerSec)
    }

    private func measureSelfSpeculativeGeneration(
        draftModel: LlamaLanguageModel,
        verifyModel: LlamaLanguageModel,
        tokenCount: Int
    ) async throws -> GenerationResult {
        let decoder = SpeculativeDecoder(
            draftModel: ModelAdapter(model: draftModel),
            verificationModel: ModelAdapter(model: verifyModel),
            draftTokenCount: Self.draftTokenCount,
            samplingPipeline: .greedy
        )

        let clock = ContinuousClock()
        let start = clock.now
        var tokenIDs = [Self.bosToken]

        while tokenIDs.count - 1 < tokenCount {
            let result = try await decoder.decodeStep(inputTokens: tokenIDs)
            for token in result.acceptedTokens {
                tokenIDs.append(token)
                if tokenIDs.count - 1 >= tokenCount { break }
            }
            if tokenIDs.last == 151645 { break }
        }

        let elapsedMs = durationMs(from: start, to: clock.now)
        let tokPerSec = Double(tokenIDs.count - 1) / max(elapsedMs / 1000.0, 1e-9)
        return GenerationResult(tokens: tokenIDs, tokensPerSecond: tokPerSec)
    }

    private func generateGreedyTokens(
        model: LlamaLanguageModel,
        tokenCount: Int
    ) async throws -> [Int] {
        var tokenIDs = [Self.bosToken]
        while tokenIDs.count - 1 < tokenCount {
            let logits = try await model.logits(for: tokenIDs)
            tokenIDs.append(argmax(logits))
            if tokenIDs.last == 151645 { break }
        }
        return tokenIDs
    }

    private func argmax(_ logits: [Float]) -> Int {
        var bestIndex = 0
        var bestValue: Float = -.infinity
        for (index, value) in logits.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return bestIndex
    }

    private func durationMs(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: end)
        return Double(duration.components.seconds) * 1000.0
            + Double(duration.components.attoseconds) / 1e15
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        if sorted.count % 2 == 1 {
            return sorted[sorted.count / 2]
        }
        let upper = sorted.count / 2
        return (sorted[upper - 1] + sorted[upper]) / 2
    }
}
