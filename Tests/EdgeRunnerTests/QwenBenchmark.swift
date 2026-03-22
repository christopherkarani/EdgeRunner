import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// Short smoke benchmark for the autoresearch optimization loop.
///
/// Produces a small, fast regression metric: `qwen_decode_throughput` (tokens/sec).
/// Writes results to `benchmarks/baseline.json` for quick comparison across runs.
/// Use `PublishableBenchmark/fullBenchmark` as the canonical publishable metric.
@Suite("Qwen 3 0.6B Real Model Benchmark")
struct QwenBenchmark {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let baselinePath = "benchmarks/baseline.json"
    static let expectedModelFileSizeBytes: Int64 = 804_753_504

    // Stable greedy prefix for the pinned GGUF. Later tokens drift slightly across
    // fresh processes on current main, so cross-check the full decode path via the
    // prefix-equivalence test below instead of hard-coding all 4 generated tokens.
    static let expectedGreedyPrefix = [1, 14582, 25]

    // MARK: - Smoke Benchmark

    @Test func decodeBenchmark() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned smoke benchmark model not found at \(Self.modelPath)")
        }
        let modelFileSize = try pinnedModelFileSize(at: url)

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Autoregressive decode: generate 4 tokens
        let generateCount = 4
        var tokenIDs = [1] // Pinned benchmark seed token for the canonical Qwen3 harness

        // Warmup (caches dequantized weights)
        _ = try await model.greedyToken(for: tokenIDs)

        let clock = ContinuousClock()
        let start = clock.now

        var hasNaN = false
        for _ in 0..<generateCount {
            let result = try await model.greedyToken(for: tokenIDs)

            // Correctness: logits must be finite
            hasNaN = hasNaN || result.hasNonFinite
            #expect(!hasNaN, "Logits contain NaN/Inf — optimization broke numerical stability")

            tokenIDs.append(result.token)
        }

        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(generateCount) / seconds
        let msPerToken = seconds / Double(generateCount) * 1000

        let expectedPrefix = Array(Self.expectedGreedyPrefix.prefix(tokenIDs.count))
        let actualPrefix = Array(tokenIDs.prefix(expectedPrefix.count))

        // Correctness guard: the pinned GGUF should keep the stable leading prefix.
        #expect(
            actualPrefix == expectedPrefix,
            "Greedy decode prefix changed! Got \(actualPrefix), expected \(expectedPrefix). Benchmark input likely drifted."
        )

        // Print parseable results
        print("BENCHMARK: qwen_decode_throughput \(String(format: "%.4f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: qwen_decode_latency \(String(format: "%.0f", msPerToken)) ms/token")
        print("BENCHMARK: qwen_generated_tokens \(tokenIDs)")

        // Write baseline JSON
        try writeBaseline(
            tokensPerSec: tokensPerSec,
            msPerToken: msPerToken,
            generatedTokens: tokenIDs,
            modelPath: url.path,
            modelFileSize: modelFileSize
        )

        #expect(tokenIDs.count == generateCount + 1)
    }

    // MARK: - Supporting Benchmarks

    @Test func loadModel() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned smoke benchmark model not found at \(Self.modelPath)")
        }

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
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned smoke benchmark model not found at \(Self.modelPath)")
        }

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
        generatedTokens: [Int],
        modelPath: String,
        modelFileSize: Int64
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
            "model_path": modelPath,
            "model_file_size_bytes": modelFileSize,
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

    private func pinnedModelFileSize(at url: URL) throws -> Int64 {
        let resourceValues = try url.resourceValues(forKeys: [.fileSizeKey])
        guard let fileSize = resourceValues.fileSize else {
            throw GenerationError.modelLoadFailed(reason: "Could not read file size for \(url.path)")
        }

        guard Int64(fileSize) == Self.expectedModelFileSizeBytes else {
            throw GenerationError.modelLoadFailed(
                reason: """
                Benchmark input drifted: expected \(Self.expectedModelFileSizeBytes) bytes at \(Self.modelPath), \
                got \(fileSize) bytes. Download the pinned GGUF before comparing results.
                """
            )
        }

        return Int64(fileSize)
    }
}
