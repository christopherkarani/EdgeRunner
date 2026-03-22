import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// Publishable benchmark for EdgeRunner LLM inference.
///
/// Generates 128 tokens with proper methodology:
/// - Separates Time-to-First-Token (TTFT) from decode throughput
/// - Reports decode-only tok/s (no cache hits, no prefill inflation)
/// - Runs multiple iterations for statistical reliability
/// - Reports p50, p90, p99, mean, stddev
/// - Outputs JSON for reproducible comparison
///
/// Usage:
///   swift test -c release --filter "PublishableBenchmark/fullBenchmark"
@Suite("Publishable Inference Benchmark")
struct PublishableBenchmark {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let generateCount = 128
    static let benchmarkRuns = 5
    static let expectedModelFileSizeBytes: Int64 = 804_753_504
    static let expectedGreedyPrefix = [1, 14582, 25]
    static let expectedTokenHash = "722b9d67f78744b9"

    // MARK: - Primary Publishable Benchmark

    @Test func fullBenchmark() async throws {
        let env = ProcessInfo.processInfo.environment
        let tokenCount = try validatedPositiveInt(
            env["EDGERUNNER_BENCHMARK_TOKENS"],
            defaultValue: Self.generateCount,
            name: "EDGERUNNER_BENCHMARK_TOKENS"
        )
        let runCount = try validatedPositiveInt(
            env["EDGERUNNER_BENCHMARK_RUNS"],
            defaultValue: Self.benchmarkRuns,
            name: "EDGERUNNER_BENCHMARK_RUNS"
        )
        let contextWindow = try validatedPositiveInt(
            env["EDGERUNNER_BENCHMARK_CONTEXT"],
            defaultValue: 2048,
            name: "EDGERUNNER_BENCHMARK_CONTEXT"
        )
        guard contextWindow >= tokenCount else {
            throw GenerationError.decodingFailed("Context window must be >= token count for publishable benchmark")
        }
        let profileLMHead = isEnabled(env["EDGERUNNER_PROFILE_LMHEAD"])
        let hasDecodeOverride =
            profileLMHead
            ||
            isEnabled(env["EDGERUNNER_DECODE_FORCE_BASE"])
            || isEnabled(env["EDGERUNNER_DECODE_PREFER_METAL4"])
            || isEnabled(env["EDGERUNNER_DECODE_DISABLE_MEGA_GQA"])
            || isEnabled(env["EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD"])
            || isEnabled(env["EDGERUNNER_DECODE_DISABLE_KV_BARRIER"])
        let isCanonicalRun =
            tokenCount == Self.generateCount
            && runCount == Self.benchmarkRuns
            && contextWindow == 2048
            && !hasDecodeOverride

        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned publishable benchmark model not found at \(Self.modelPath)")
        }
        let modelFileSize = try pinnedModelFileSize(at: url)

        // Track peak memory
        let memBefore = currentRSS()

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: contextWindow)
        )

        let memAfterLoad = currentRSS()

        // Warmup: one full generation to JIT all pipelines
        var warmupTokens = [1]
        for _ in 0..<4 {
            let result = try await model.greedyToken(for: warmupTokens)
            warmupTokens.append(result.token)
        }

        // Run benchmarkRuns iterations (single model, reset via fresh prefill each run)
        var runResults: [RunResult] = []

        for run in 0..<runCount {
            // Each run starts with logits(for: [1]) which triggers prefill + KV cache reset.
            // The model's decode detector sees a fresh sequence and resets automatically.
            let result = try await measureGeneration(
                model: model,
                tokenCount: tokenCount,
                runIndex: run
            )
            runResults.append(result)
        }

        let memPeak = currentRSS()

        // Compute statistics
        let stats = computeStatistics(runs: runResults)

        let lmHeadMs = profileLMHead ? try await model.measureLMHeadLatency(samples: 5) : nil

        // Determinism check: all runs should produce identical tokens
        let referenceTokens = runResults[0].tokens
        let allDeterministic = runResults.allSatisfy { $0.tokens == referenceTokens }
        let expectedPrefix = Array(Self.expectedGreedyPrefix.prefix(referenceTokens.count))
        let actualPrefix = Array(referenceTokens.prefix(expectedPrefix.count))
        let tokenHash = tokenSequenceHash(referenceTokens)

        // Hard validation guards: invalid runs must not persist artifacts.
        guard allDeterministic else {
            throw GenerationError.decodingFailed("Non-deterministic output across runs — publishable benchmark aborted")
        }
        guard actualPrefix == expectedPrefix else {
            throw GenerationError.modelLoadFailed(
                reason: """
                Benchmark input or decode path drifted: expected greedy prefix \(expectedPrefix), \
                got \(actualPrefix). Re-pin the GGUF before comparing publishable results.
                """
            )
        }
        let isPinnedHarness =
            tokenCount == Self.generateCount
            && runCount == Self.benchmarkRuns
            && contextWindow == 2048

        if isPinnedHarness {
            guard tokenHash == Self.expectedTokenHash else {
                throw GenerationError.modelLoadFailed(
                    reason: """
                    Publishable output drifted: expected token hash \(Self.expectedTokenHash), \
                    got \(tokenHash). Re-pin the GGUF or investigate decode correctness before publishing results.
                    """
                )
            }
        }
        guard stats.decodeMedian > 0 else {
            throw GenerationError.decodingFailed("Decode throughput must be positive")
        }
        for run in runResults where run.hasNaN {
            throw GenerationError.decodingFailed("Run \(run.runIndex) produced NaN/Inf logits")
        }

        // Print and persist only after validation passes.
        printReport(
            stats: stats,
            runs: runResults,
            memLoadMB: Double(memAfterLoad - memBefore) / 1_048_576,
            memPeakMB: Double(memPeak - memBefore) / 1_048_576,
            deterministic: allDeterministic,
            tokenCount: tokenCount,
            runCount: runCount,
            lmHeadMs: lmHeadMs,
            isCanonicalRun: isCanonicalRun
        )

        try writeJSON(
            stats: stats,
            runs: runResults,
            memLoadMB: Double(memAfterLoad - memBefore) / 1_048_576,
            memPeakMB: Double(memPeak - memBefore) / 1_048_576,
            deterministic: allDeterministic,
            modelPath: url.path,
            modelFileSize: modelFileSize,
            tokenPrefix: actualPrefix,
            tokenHash: tokenHash,
            lmHeadMs: lmHeadMs,
            tokenCount: tokenCount,
            runCount: runCount,
            contextWindow: contextWindow,
            isCanonicalRun: isCanonicalRun
        )
    }

    // MARK: - Single-Run Measurement

    private func measureGeneration(
        model: LlamaLanguageModel,
        tokenCount: Int,
        runIndex: Int
    ) async throws -> RunResult {
        let clock = ContinuousClock()
        var tokenIDs = [1] // Pinned benchmark seed token for the canonical Qwen3 harness
        var hasNaN = false
        var decodeLatencies: [Double] = [] // ms per decode token

        // === PREFILL (token 1) — measures TTFT ===
        let prefillStart = clock.now
        let firstLogits = try await model.greedyToken(for: tokenIDs)
        let prefillEnd = clock.now
        let ttft = durationMs(from: prefillStart, to: prefillEnd)

        if firstLogits.hasNonFinite { hasNaN = true }
        tokenIDs.append(firstLogits.token)

        // === DECODE (tokens 2..tokenCount) — pure autoregressive ===
        let decodeStart = clock.now

        for _ in 1..<tokenCount {
            let tokenStart = clock.now
            let logits = try await model.greedyToken(for: tokenIDs)
            let tokenEnd = clock.now

            if logits.hasNonFinite { hasNaN = true }
            tokenIDs.append(logits.token)

            let latency = durationMs(from: tokenStart, to: tokenEnd)
            decodeLatencies.append(latency)

            // Early exit on EOS
            if tokenIDs.last == 151645 { break }
        }

        let decodeEnd = clock.now
        let totalDecodeMs = durationMs(from: decodeStart, to: decodeEnd)
        let decodeTokenCount = tokenIDs.count - 2 // exclude seed token and first generated token (prefill)
        let decodeTokPerSec = decodeTokenCount > 0 ? Double(decodeTokenCount) / (totalDecodeMs / 1000.0) : 0

        let totalMs = durationMs(from: prefillStart, to: decodeEnd)
        let endToEndTokPerSec = Double(tokenIDs.count - 1) / (totalMs / 1000.0) // generated tokens / total time

        return RunResult(
            runIndex: runIndex,
            tokens: tokenIDs,
            ttftMs: ttft,
            decodeTokenCount: decodeTokenCount,
            decodeTotalMs: totalDecodeMs,
            decodeTokPerSec: decodeTokPerSec,
            endToEndTokPerSec: endToEndTokPerSec,
            totalMs: totalMs,
            decodeLatencies: decodeLatencies,
            hasNaN: hasNaN
        )
    }

    // MARK: - Statistics

    private struct RunResult {
        let runIndex: Int
        let tokens: [Int]
        let ttftMs: Double
        let decodeTokenCount: Int
        let decodeTotalMs: Double
        let decodeTokPerSec: Double
        let endToEndTokPerSec: Double
        let totalMs: Double
        let decodeLatencies: [Double]
        let hasNaN: Bool
    }

    private struct BenchmarkStats {
        let decodeMedian: Double
        let decodeP90: Double
        let decodeP99: Double
        let decodeMean: Double
        let decodeStddev: Double
        let decodeMin: Double
        let decodeMax: Double
        let ttftMedian: Double
        let ttftMean: Double
        let e2eMedian: Double
        let e2eMean: Double
        // Per-token latency stats (ms)
        let tokenLatencyMedian: Double
        let tokenLatencyP90: Double
        let tokenLatencyP99: Double
    }

    private func computeStatistics(runs: [RunResult]) -> BenchmarkStats {
        let decodeThroughputs = runs.map(\.decodeTokPerSec).sorted()
        let ttfts = runs.map(\.ttftMs).sorted()
        let e2es = runs.map(\.endToEndTokPerSec).sorted()

        // Aggregate all per-token latencies across runs
        let allLatencies = runs.flatMap(\.decodeLatencies).sorted()

        return BenchmarkStats(
            decodeMedian: percentile(decodeThroughputs, 0.50),
            decodeP90: percentile(decodeThroughputs, 0.10), // p10 of throughput = p90 of latency
            decodeP99: percentile(decodeThroughputs, 0.01),
            decodeMean: decodeThroughputs.reduce(0, +) / Double(decodeThroughputs.count),
            decodeStddev: stddev(decodeThroughputs),
            decodeMin: decodeThroughputs.first ?? 0,
            decodeMax: decodeThroughputs.last ?? 0,
            ttftMedian: percentile(ttfts, 0.50),
            ttftMean: ttfts.reduce(0, +) / Double(ttfts.count),
            e2eMedian: percentile(e2es, 0.50),
            e2eMean: e2es.reduce(0, +) / Double(e2es.count),
            tokenLatencyMedian: percentile(allLatencies, 0.50),
            tokenLatencyP90: percentile(allLatencies, 0.90),
            tokenLatencyP99: percentile(allLatencies, 0.99)
        )
    }

    // MARK: - Report

    private func printReport(
        stats: BenchmarkStats,
        runs: [RunResult],
        memLoadMB: Double,
        memPeakMB: Double,
        deterministic: Bool,
        tokenCount: Int,
        runCount: Int,
        lmHeadMs: Double?,
        isCanonicalRun: Bool
    ) {
        print("")
        print("=" * 70)
        print("  EdgeRunner Inference Benchmark — Qwen 3 0.6B Q8_0")
        print("=" * 70)
        print("")
        print("  Model:           Qwen3-0.6B-Q8_0 (804,753,504-byte GGUF, 28 layers, 151K vocab)")
        print("  Device:          \(deviceDescription())")
        print("  Tokens:          \(tokenCount) per run")
        print("  Runs:            \(runCount)")
        print("  Deterministic:   \(deterministic ? "YES" : "NO")")
        if !isCanonicalRun {
            print("  Mode:            NON-CANONICAL (override-driven profiling run)")
        }
        print("")
        print("  --- Decode Throughput (tokens 2-\(tokenCount), excludes prefill) ---")
        print(String(format: "  Median:          %.1f tok/s", stats.decodeMedian))
        print(String(format: "  Mean:            %.1f tok/s (stddev: %.1f)", stats.decodeMean, stats.decodeStddev))
        print(String(format: "  Min:             %.1f tok/s", stats.decodeMin))
        print(String(format: "  Max:             %.1f tok/s", stats.decodeMax))
        print("")
        print("  --- Time to First Token (TTFT) ---")
        print(String(format: "  Median:          %.1f ms", stats.ttftMedian))
        print(String(format: "  Mean:            %.1f ms", stats.ttftMean))
        print("")
        print("  --- End-to-End (all \(tokenCount) tokens / total wall time) ---")
        print(String(format: "  Median:          %.1f tok/s", stats.e2eMedian))
        print(String(format: "  Mean:            %.1f tok/s", stats.e2eMean))
        print("")
        print("  --- Per-Token Decode Latency ---")
        print(String(format: "  Median:          %.2f ms", stats.tokenLatencyMedian))
        print(String(format: "  P90:             %.2f ms", stats.tokenLatencyP90))
        print(String(format: "  P99:             %.2f ms", stats.tokenLatencyP99))
        print("")
        if let lmHeadMs {
            print("  --- LM Head Profiling (EDGERUNNER_PROFILE_LMHEAD=1) ---")
            print(String(format: "  Avg latency:     %.2f ms (final norm + LM head only)", lmHeadMs))
            print("")
        }
        print("  --- Memory ---")
        print(String(format: "  Model load:      %.0f MB", memLoadMB))
        print(String(format: "  Peak RSS:        %.0f MB", memPeakMB))
        print("")
        print("  --- Per-Run Breakdown ---")
        for run in runs {
            print(String(format: "  Run %d: decode=%.1f tok/s  e2e=%.1f tok/s  ttft=%.1fms  tokens=%d",
                         run.runIndex, run.decodeTokPerSec, run.endToEndTokPerSec,
                         run.ttftMs, run.tokens.count))
        }
        print("")
        print("=" * 70)
        print("")
        let prefix = isCanonicalRun ? "PUBLISH" : "PROFILE"
        print("\(prefix): qwen_decode_throughput_median \(String(format: "%.1f", stats.decodeMedian)) tok/s")
        print("\(prefix): qwen_decode_throughput_max \(String(format: "%.1f", stats.decodeMax)) tok/s")
        print("\(prefix): qwen_ttft_median \(String(format: "%.1f", stats.ttftMedian)) ms")
        print("\(prefix): qwen_e2e_throughput_median \(String(format: "%.1f", stats.e2eMedian)) tok/s")
        print("\(prefix): qwen_token_hash \(tokenSequenceHash(runs[0].tokens))")
    }

    // MARK: - JSON Output

    private func writeJSON(
        stats: BenchmarkStats,
        runs: [RunResult],
        memLoadMB: Double,
        memPeakMB: Double,
        deterministic: Bool,
        modelPath: String,
        modelFileSize: Int64,
        tokenPrefix: [Int],
        tokenHash: String,
        lmHeadMs: Double?,
        tokenCount: Int,
        runCount: Int,
        contextWindow: Int,
        isCanonicalRun: Bool
    ) throws {
        let runData: [[String: Any]] = runs.map { run in
            [
                "run": run.runIndex,
                "decode_tok_per_sec": run.decodeTokPerSec,
                "e2e_tok_per_sec": run.endToEndTokPerSec,
                "ttft_ms": run.ttftMs,
                "decode_total_ms": run.decodeTotalMs,
                "decode_token_count": run.decodeTokenCount,
                "total_ms": run.totalMs,
                "token_count": run.tokens.count,
                "has_nan": run.hasNaN,
            ]
        }

        let json: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "model": "Qwen3-0.6B-Q8_0",
            "model_path": modelPath,
            "model_file_size_bytes": modelFileSize,
            "greedy_prefix": tokenPrefix,
            "token_hash": tokenHash,
            "device": deviceDescription(),
            "tokens_per_run": tokenCount,
            "num_runs": runCount,
            "context_window": contextWindow,
            "is_canonical_run": isCanonicalRun,
            "deterministic": deterministic,
            "decode_throughput": [
                "median": stats.decodeMedian,
                "mean": stats.decodeMean,
                "stddev": stats.decodeStddev,
                "min": stats.decodeMin,
                "max": stats.decodeMax,
            ],
            "ttft_ms": [
                "median": stats.ttftMedian,
                "mean": stats.ttftMean,
            ],
            "e2e_throughput": [
                "median": stats.e2eMedian,
                "mean": stats.e2eMean,
            ],
            "lm_head_avg_ms": lmHeadMs as Any,
            "per_token_latency_ms": [
                "median": stats.tokenLatencyMedian,
                "p90": stats.tokenLatencyP90,
                "p99": stats.tokenLatencyP99,
            ],
            "memory_mb": [
                "model_load": memLoadMB,
                "peak_rss": memPeakMB,
            ],
            "runs": runData,
        ]

        let projectDir = ProcessInfo.processInfo.environment["PROJECT_DIR"]
            ?? FileManager.default.currentDirectoryPath
        let benchDir = URL(fileURLWithPath: projectDir).appendingPathComponent("benchmarks")
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)

        let outputFileName = isCanonicalRun ? "publishable_benchmark.json" : "publishable_profile_benchmark.json"
        let outputURL = benchDir.appendingPathComponent(outputFileName)
        let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outputURL)
        print("Benchmark results written to: \(outputURL.path)")
    }

    // MARK: - Helpers

    private func durationMs(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: end)
        return Double(d.components.seconds) * 1000.0 + Double(d.components.attoseconds) * 1e-15
    }

    private func validatedPositiveInt(_ rawValue: String?, defaultValue: Int, name: String) throws -> Int {
        guard let rawValue, !rawValue.isEmpty else { return defaultValue }
        guard let parsed = Int(rawValue), parsed > 0 else {
            throw GenerationError.decodingFailed("Invalid \(name)=\(rawValue). Expected a positive integer.")
        }
        return parsed
    }

    private func tokenSequenceHash(_ tokens: [Int]) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for token in tokens {
            for byte in "\(token),".utf8 {
                hash ^= UInt64(byte)
                hash &*= 0x100000001b3
            }
        }
        return String(format: "%016llx", hash)
    }

    private func isEnabled(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = max(0, min(sorted.count - 1, Int(Double(sorted.count - 1) * p)))
        return sorted[idx]
    }

    private func stddev(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count - 1)
        return sqrt(variance)
    }

    private func currentRSS() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    private func deviceDescription() -> String {
        var size: Int = 0
        sysctlbyname("machdw.model", nil, &size, nil, 0)
        var modelChars = [CChar](repeating: 0, count: size)
        sysctlbyname("machdw.model", &modelChars, &size, nil, 0)
        let modelStr = size > 1 ? String(decoding: modelChars.prefix(size - 1).map { UInt8($0) }, as: UTF8.self) : ""
        return modelStr.isEmpty ? "Apple Silicon" : modelStr
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
                got \(fileSize) bytes. Download the pinned GGUF before comparing publishable results.
                """
            )
        }

        return Int64(fileSize)
    }
}

// String repetition helper
private func *(lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
}
