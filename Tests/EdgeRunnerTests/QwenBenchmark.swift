import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// The primary benchmark for the autoresearch optimization loop.
///
/// Produces a single metric: `qwen_decode_throughput` (tokens/sec).
/// Writes results to `benchmarks/baseline.json` for comparison across runs.
/// Validates correctness via argmax stability (same greedy tokens every run).
@Suite("Qwen 3 0.6B Real Model Benchmark")
struct QwenBenchmark {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let baselinePath = "benchmarks/baseline.json"

    // Expected greedy output for correctness — must be stable across optimizations.
    // If an optimization changes these tokens, the optimization broke correctness.
    static let expectedGreedyTokens = [1, 1479, 21456, 96793, 15859]

    // MARK: - Primary Benchmark (Autoresearch Target)

    @Test func decodeBenchmark() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return
        }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Autoregressive decode: generate 4 tokens
        let generateCount = 4
        var tokenIDs = [1] // BOS

        // Warmup (caches dequantized weights)
        _ = try await model.logits(for: tokenIDs)

        let clock = ContinuousClock()
        let start = clock.now

        for _ in 0..<generateCount {
            let logits = try await model.logits(for: tokenIDs)

            // Correctness: logits must be finite
            let hasNaN = logits.contains(where: { !$0.isFinite })
            #expect(!hasNaN, "Logits contain NaN/Inf — optimization broke numerical stability")

            // Greedy argmax
            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (i, v) in logits.enumerated() {
                if v > maxVal { maxVal = v; maxIdx = i }
            }
            tokenIDs.append(maxIdx)
        }

        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(generateCount) / seconds
        let msPerToken = seconds / Double(generateCount) * 1000

        // Correctness guard: greedy tokens must match expected output
        #expect(
            tokenIDs == Self.expectedGreedyTokens,
            "Greedy decode changed! Got \(tokenIDs), expected \(Self.expectedGreedyTokens). Optimization broke correctness."
        )

        // Print parseable results
        print("BENCHMARK: qwen_decode_throughput \(String(format: "%.4f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: qwen_decode_latency \(String(format: "%.0f", msPerToken)) ms/token")
        print("BENCHMARK: qwen_generated_tokens \(tokenIDs)")

        // Write baseline JSON
        try writeBaseline(
            tokensPerSec: tokensPerSec,
            msPerToken: msPerToken,
            generatedTokens: tokenIDs
        )

        #expect(tokenIDs.count == generateCount + 1)
    }

    // MARK: - Supporting Benchmarks

    @Test func loadModel() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let clock = ContinuousClock()
        let start = clock.now
        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18

        print("BENCHMARK: qwen_load_time \(String(format: "%.0f", seconds * 1000)) ms")
        print("BENCHMARK: qwen_vocab_size \(model.vocabularySize)")
        #expect(model.vocabularySize > 100_000)
    }

    @Test func singleForwardPass() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let clock = ContinuousClock()
        let start = clock.now
        let logits = try await model.logits(for: [1])
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18

        print("BENCHMARK: qwen_single_forward \(String(format: "%.0f", seconds * 1000)) ms")
        print("BENCHMARK: qwen_logits_size \(logits.count)")

        #expect(logits.count == model.vocabularySize)
        #expect(logits.allSatisfy { $0.isFinite })
    }

    // MARK: - Baseline Tracking

    private func writeBaseline(
        tokensPerSec: Double,
        msPerToken: Double,
        generatedTokens: [Int]
    ) throws {
        let projectDir = ProcessInfo.processInfo.environment["PROJECT_DIR"]
            ?? FileManager.default.currentDirectoryPath
        let benchDir = URL(fileURLWithPath: projectDir).appendingPathComponent("benchmarks")

        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)

        let baselineURL = benchDir.appendingPathComponent("baseline.json")

        // Read previous baseline if exists
        var previousBest: Double? = nil
        if let data = try? Data(contentsOf: baselineURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let prev = json["best_tokens_per_sec"] as? Double {
            previousBest = prev
        }

        let isImprovement = previousBest.map { tokensPerSec > $0 } ?? true
        let improvementPct = previousBest.map { ((tokensPerSec - $0) / $0) * 100 } ?? 0

        let baseline: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "model": "Qwen3-0.6B-Q8_0",
            "tokens_per_sec": tokensPerSec,
            "ms_per_token": msPerToken,
            "generated_tokens": generatedTokens,
            "best_tokens_per_sec": isImprovement ? tokensPerSec : (previousBest ?? tokensPerSec),
            "is_improvement": isImprovement,
            "improvement_pct": improvementPct,
            "device": deviceName(),
        ]

        let data = try JSONSerialization.data(
            withJSONObject: baseline,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: baselineURL)

        if let prev = previousBest {
            let arrow = isImprovement ? "^" : "v"
            print("BASELINE: \(String(format: "%.4f", tokensPerSec)) tok/s \(arrow) (prev: \(String(format: "%.4f", prev)), \(String(format: "%+.1f", improvementPct))%)")
        } else {
            print("BASELINE: \(String(format: "%.4f", tokensPerSec)) tok/s (first run)")
        }
    }

    private func deviceName() -> String {
        "Apple Silicon"
    }
}
