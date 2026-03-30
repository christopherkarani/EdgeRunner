import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// Benchmark for LiquidAI LFM2.5-1.2B-Thinking model.
///
/// Usage:
///   swift test -c release --filter "LFMBenchmark/fullBenchmark"
@Suite("LFM2.5-1.2B-Thinking Benchmark")
struct LFMBenchmark {

    static let modelPath = "/tmp/edgerunner-models/LFM2.5-1.2B-Thinking-Q8_0.gguf"
    static let generateCount = 128
    static let benchmarkRuns = 5

    // MARK: - Primary Benchmark

    @Test func fullBenchmark() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return
        }
        let modelFileSize = try modelFileSizeBytes(at: url)

        // Track peak memory
        let memBefore = currentRSS()

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let memAfterLoad = currentRSS()

        // Warmup: one full generation to JIT all pipelines
        var warmupTokens = [1]
        for _ in 0..<4 {
            let logits = try await model.logits(for: warmupTokens)
            warmupTokens.append(argmax(logits))
        }

        // Run benchmarkRuns iterations
        var runResults: [RunResult] = []

        for run in 0..<Self.benchmarkRuns {
            model.resetGenerationState(keepDecodeWarmup: true)
            let result = try await measureGeneration(
                model: model,
                tokenCount: Self.generateCount,
                runIndex: run
            )
            runResults.append(result)
        }

        let memPeak = currentRSS()

        // Compute statistics
        let stats = computeStatistics(runs: runResults)

        // Determinism check
        let referenceTokens = runResults[0].tokens
        let allDeterministic = runResults.allSatisfy { $0.tokens == referenceTokens }

        // Print human-readable report
        printReport(
            stats: stats,
            runs: runResults,
            memLoadMB: Double(memAfterLoad - memBefore) / 1_048_576,
            memPeakMB: Double(memPeak - memBefore) / 1_048_576,
            deterministic: allDeterministic,
            tokenCount: Self.generateCount,
            runCount: Self.benchmarkRuns
        )

        // Write JSON
        try writeJSON(
            stats: stats,
            runs: runResults,
            memLoadMB: Double(memAfterLoad - memBefore) / 1_048_576,
            memPeakMB: Double(memPeak - memBefore) / 1_048_576,
            deterministic: allDeterministic,
            modelPath: url.path,
            modelFileSize: modelFileSize,
            greedyPrefix: Array(referenceTokens.prefix(10))
        )

        // Assertions
        #expect(allDeterministic, "Non-deterministic output across runs")
        #expect(stats.decodeMedian > 0, "Decode throughput must be positive")
        for run in runResults {
            #expect(!run.hasNaN, "Run \(run.runIndex) produced NaN/Inf logits")
        }
    }

    // MARK: - Single-Run Measurement

    private func measureGeneration(
        model: LlamaLanguageModel,
        tokenCount: Int,
        runIndex: Int
    ) async throws -> RunResult {
        let clock = ContinuousClock()
        var tokenIDs = [1] // BOS
        var hasNaN = false
        var decodeLatencies: [Double] = []

        // === PREFILL (token 1) — measures TTFT ===
        let prefillStart = clock.now
        let firstLogits = try await model.logits(for: tokenIDs)
        let prefillEnd = clock.now
        let ttft = durationMs(from: prefillStart, to: prefillEnd)

        if firstLogits.contains(where: { !$0.isFinite }) { hasNaN = true }
        tokenIDs.append(argmax(firstLogits))

        // === DECODE (tokens 2..tokenCount) ===
        let decodeStart = clock.now

        for _ in 1..<tokenCount {
            let tokenStart = clock.now
            let logits = try await model.logits(for: tokenIDs)
            let tokenEnd = clock.now

            if logits.contains(where: { !$0.isFinite }) { hasNaN = true }
            tokenIDs.append(argmax(logits))

            let latency = durationMs(from: tokenStart, to: tokenEnd)
            decodeLatencies.append(latency)

            // Early exit on EOS (common EOS tokens)
            if tokenIDs.last == 151645 || tokenIDs.last == 2 { break }
        }

        let decodeEnd = clock.now
        let totalDecodeMs = durationMs(from: decodeStart, to: decodeEnd)
        let decodeTokenCount = tokenIDs.count - 2
        let decodeTokPerSec = decodeTokenCount > 0 ? Double(decodeTokenCount) / (totalDecodeMs / 1000.0) : 0

        let totalMs = durationMs(from: prefillStart, to: decodeEnd)
        let endToEndTokPerSec = Double(tokenIDs.count - 1) / (totalMs / 1000.0)

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
        let tokenLatencyMedian: Double
        let tokenLatencyP90: Double
        let tokenLatencyP99: Double
    }

    private func computeStatistics(runs: [RunResult]) -> BenchmarkStats {
        let decodeThroughputs = runs.map(\.decodeTokPerSec).sorted()
        let ttfts = runs.map(\.ttftMs).sorted()
        let e2es = runs.map(\.endToEndTokPerSec).sorted()
        let allLatencies = runs.flatMap(\.decodeLatencies).sorted()

        return BenchmarkStats(
            decodeMedian: percentile(decodeThroughputs, 0.50),
            decodeP90: percentile(decodeThroughputs, 0.10),
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
        runCount: Int
    ) {
        print("")
        print("=" * 70)
        print("  EdgeRunner Inference Benchmark — LFM2.5-1.2B-Thinking Q8_0")
        print("=" * 70)
        print("")
        print("  Model:           LFM2.5-1.2B-Thinking-Q8_0 (1.2 GB)")
        print("  Device:          \(deviceDescription())")
        print("  Tokens:          \(tokenCount) per run")
        print("  Runs:            \(runCount)")
        print("  Deterministic:   \(deterministic ? "YES" : "NO")")
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
        print("PUBLISH: lfm_decode_throughput_median \(String(format: "%.1f", stats.decodeMedian)) tok/s")
        print("PUBLISH: lfm_decode_throughput_max \(String(format: "%.1f", stats.decodeMax)) tok/s")
        print("PUBLISH: lfm_ttft_median \(String(format: "%.1f", stats.ttftMedian)) ms")
        print("PUBLISH: lfm_e2e_throughput_median \(String(format: "%.1f", stats.e2eMedian)) tok/s")
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
        greedyPrefix: [Int]
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
            "model": "LFM2.5-1.2B-Thinking-Q8_0",
            "model_path": modelPath,
            "model_file_size_bytes": modelFileSize,
            "greedy_prefix": greedyPrefix,
            "device": deviceDescription(),
            "tokens_per_run": Self.generateCount,
            "num_runs": Self.benchmarkRuns,
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

        let outputURL = benchDir.appendingPathComponent("lfm_benchmark.json")
        let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outputURL)
        print("Benchmark results written to: \(outputURL.path)")
    }

    // MARK: - Helpers

    private func argmax(_ array: [Float]) -> Int {
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for (i, v) in array.enumerated() {
            if v > maxVal { maxVal = v; maxIdx = i }
        }
        return maxIdx
    }

    private func durationMs(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: end)
        return Double(d.components.seconds) * 1000.0 + Double(d.components.attoseconds) * 1e-15
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
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var modelChars = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &modelChars, &size, nil, 0)
        let modelStr = size > 1 ? String(decoding: modelChars.prefix(size - 1).map { UInt8($0) }, as: UTF8.self) : ""
        return modelStr.isEmpty ? "Apple Silicon" : modelStr
    }

    private func modelFileSizeBytes(at url: URL) throws -> Int64 {
        let resourceValues = try url.resourceValues(forKeys: [.fileSizeKey])
        guard let fileSize = resourceValues.fileSize else {
            throw GenerationError.modelLoadFailed(reason: "Could not read file size for \(url.path)")
        }
        return Int64(fileSize)
    }
}

// String repetition helper
private func *(lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
}
