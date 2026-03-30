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

    static let contract = BenchmarkContract.pinned
    static let modelPath = contract.model.localPath
    static let generateCount = contract.publishable.tokenCount
    static let benchmarkRuns = contract.publishable.runCount
    static let expectedModelFileSizeBytes = contract.model.sizeBytes
    static let expectedGreedyPrefix = contract.publishable.expectedGreedyPrefix
    static let expectedTokenHash = contract.publishable.expectedTokenHash
    static let childRunEnvKey = "EDGERUNNER_PUBLISHABLE_CHILD_RUN"
    static let processIsolationEnvKey = "EDGERUNNER_BENCHMARK_PROCESS_ISOLATION"
    static let outputPathEnvKey = "EDGERUNNER_BENCHMARK_OUTPUT_PATH"
    static let megaParityRunEnvKey = "EDGERUNNER_RUN_MEGA_PARITY"
    static let megaParityTokensEnvKey = "EDGERUNNER_MEGA_PARITY_TOKENS"

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
            defaultValue: Self.contract.publishable.contextWindow,
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
            && contextWindow == Self.contract.publishable.contextWindow
            && !hasDecodeOverride

        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned publishable benchmark model not found at \(Self.modelPath)")
        }
        let modelFileSize = try pinnedModelFileSize(at: url)
        let shouldIsolateRuns =
            !isEnabled(env[Self.childRunEnvKey])
            && runCount > 1
            && !isExplicitlyDisabled(env[Self.processIsolationEnvKey])

        let benchmarkResult: BenchmarkExecution
        if shouldIsolateRuns {
            benchmarkResult = try runProcessIsolatedBenchmark(
                tokenCount: tokenCount,
                runCount: runCount,
                contextWindow: contextWindow,
                profileLMHead: profileLMHead
            )
        } else {
            benchmarkResult = try await runInProcessBenchmark(
                modelURL: url,
                tokenCount: tokenCount,
                runCount: runCount,
                contextWindow: contextWindow,
                profileLMHead: profileLMHead
            )
        }

        // Compute statistics
        let stats = computeStatistics(runs: benchmarkResult.runResults)

        // Determinism check: all runs should produce identical output fingerprints.
        let allDeterministic = benchmarkResult.deterministic
        let actualPrefix = benchmarkResult.tokenPrefix
        let expectedPrefix = Array(Self.expectedGreedyPrefix.prefix(actualPrefix.count))
        let tokenHash = benchmarkResult.tokenHash

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
            && contextWindow == Self.contract.publishable.contextWindow

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
        for run in benchmarkResult.runResults where run.hasNaN {
            throw GenerationError.decodingFailed("Run \(run.runIndex) produced NaN/Inf logits")
        }

        // Print and persist only after validation passes.
        printReport(
            stats: stats,
            runs: benchmarkResult.runResults,
            memLoadMB: benchmarkResult.memLoadMB,
            memPeakMB: benchmarkResult.memPeakMB,
            deterministic: allDeterministic,
            tokenCount: tokenCount,
            runCount: runCount,
            lmHeadMs: benchmarkResult.lmHeadMs,
            isCanonicalRun: isCanonicalRun,
            tokenHash: tokenHash
        )

        try writeJSON(
            stats: stats,
            runs: benchmarkResult.runResults,
            memLoadMB: benchmarkResult.memLoadMB,
            memPeakMB: benchmarkResult.memPeakMB,
            deterministic: allDeterministic,
            modelPath: url.path,
            modelFileSize: modelFileSize,
            tokenPrefix: actualPrefix,
            tokenHash: tokenHash,
            lmHeadMs: benchmarkResult.lmHeadMs,
            tokenCount: tokenCount,
            runCount: runCount,
            contextWindow: contextWindow,
            isCanonicalRun: isCanonicalRun
        )
    }

    @Test
    func megaKernelMatchesSafePath() async throws {
        guard isEnabled(ProcessInfo.processInfo.environment[Self.megaParityRunEnvKey]) else {
            Swift.print("SKIP: Set \(Self.megaParityRunEnvKey)=1 to run the mega-kernel parity probe.")
            return
        }

        let tokenCount = try validatedPositiveInt(
            ProcessInfo.processInfo.environment[Self.megaParityTokensEnvKey],
            defaultValue: 16,
            name: Self.megaParityTokensEnvKey
        )
        let contextWindow = Self.contract.publishable.contextWindow
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned publishable benchmark model not found at \(Self.modelPath)")
        }

        let safeModel = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: .pinnedBenchmarkConfiguration(contextWindow: contextWindow, disableMegaKernel: true)
        )
        let megaModel = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: .pinnedBenchmarkConfiguration(contextWindow: contextWindow, disableMegaKernel: false)
        )

        try await warmBenchmarkModel(safeModel)
        try await warmBenchmarkModel(megaModel)

        let safeTokens = try await greedyTokens(model: safeModel, tokenCount: tokenCount)
        let megaTokens = try await greedyTokens(
            model: megaModel,
            tokenCount: tokenCount,
            referenceTokens: safeTokens
        )

        #expect(megaTokens == safeTokens, "Mega path diverged from safe path. safe=\(safeTokens) mega=\(megaTokens)")
        let actualPrefix = Array(megaTokens.prefix(Self.expectedGreedyPrefix.count))
        let expectedPrefix = Array(Self.expectedGreedyPrefix.prefix(actualPrefix.count))
        #expect(actualPrefix == expectedPrefix, "Expected prefix \(expectedPrefix), got \(actualPrefix)")

        if tokenCount == Self.generateCount {
            let tokenHash = tokenSequenceHash(megaTokens)
            #expect(
                tokenHash == Self.expectedTokenHash,
                "Expected token hash \(Self.expectedTokenHash), got \(tokenHash)"
            )
        }
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

    private func runInProcessBenchmark(
        modelURL: URL,
        tokenCount: Int,
        runCount: Int,
        contextWindow: Int,
        profileLMHead: Bool
    ) async throws -> BenchmarkExecution {
        let memBefore = currentRSS()
        var memAfterLoad = memBefore
        var memPeak = memBefore
        var runResults: [RunResult] = []

        for run in 0..<runCount {
            let model = try await LlamaLanguageModel.load(
                from: modelURL,
                configuration: .pinnedBenchmarkConfiguration(contextWindow: contextWindow)
            )
            if run == 0 {
                memAfterLoad = currentRSS()
            }

            try await warmBenchmarkModel(model)
            let result = try await measureGeneration(
                model: model,
                tokenCount: tokenCount,
                runIndex: run
            )
            runResults.append(result)
            memPeak = max(memPeak, currentRSS())
        }

        let referenceTokens = runResults[0].tokens
        let tokenPrefix = Array(referenceTokens.prefix(Self.expectedGreedyPrefix.count))
        let tokenHash = tokenSequenceHash(referenceTokens)
        let lmHeadMs: Double?
        if profileLMHead {
            let profilingModel = try await LlamaLanguageModel.load(
                from: modelURL,
                configuration: .pinnedBenchmarkConfiguration(contextWindow: contextWindow)
            )
            lmHeadMs = try await profilingModel.measureLMHeadLatency(samples: 5)
        } else {
            lmHeadMs = nil
        }

        return BenchmarkExecution(
            runResults: runResults,
            memLoadMB: Double(memAfterLoad - memBefore) / 1_048_576,
            memPeakMB: Double(memPeak - memBefore) / 1_048_576,
            deterministic: runResults.allSatisfy { $0.tokens == referenceTokens },
            tokenPrefix: tokenPrefix,
            tokenHash: tokenHash,
            lmHeadMs: lmHeadMs
        )
    }

    private func warmBenchmarkModel(_ model: LlamaLanguageModel) async throws {
        var warmupTokens = [1]
        for _ in 0..<4 {
            let result = try await model.greedyToken(for: warmupTokens)
            #expect(!result.hasNonFinite, "Benchmark warmup produced NaN/Inf logits")
            warmupTokens.append(result.token)
        }
        model.resetGenerationState(keepDecodeWarmup: true)
    }

    private func greedyTokens(
        model: LlamaLanguageModel,
        tokenCount: Int,
        referenceTokens: [Int]? = nil
    ) async throws -> [Int] {
        var tokenIDs = [1]
        let referenceTokens = referenceTokens ?? []

        let first = try await model.greedyToken(for: tokenIDs)
        guard !first.hasNonFinite else {
            throw GenerationError.decodingFailed("Step 0 produced NaN/Inf logits")
        }
        tokenIDs.append(first.token)
        if !referenceTokens.isEmpty {
            let safeToken = referenceTokens[1]
            guard first.token == safeToken else {
                throw GenerationError.decodingFailed(
                    "Mega path diverged at step 0. safeToken=\(safeToken) megaToken=\(first.token) prefix=\(tokenIDs)"
                )
            }
        }

        for stepIndex in 1..<tokenCount {
            let result = try await model.greedyToken(for: tokenIDs)
            guard !result.hasNonFinite else {
                throw GenerationError.decodingFailed("Step \(stepIndex) produced NaN/Inf logits")
            }
            tokenIDs.append(result.token)
            if !referenceTokens.isEmpty {
                let safeToken = referenceTokens[stepIndex + 1]
                guard result.token == safeToken else {
                    throw GenerationError.decodingFailed(
                        "Mega path diverged at step \(stepIndex). safeToken=\(safeToken) megaToken=\(result.token) prefix=\(tokenIDs)"
                    )
                }
            }
        }

        return tokenIDs
    }

    private func runProcessIsolatedBenchmark(
        tokenCount: Int,
        runCount: Int,
        contextWindow: Int,
        profileLMHead: Bool
    ) throws -> BenchmarkExecution {
        let fileManager = FileManager.default
        let projectDir = ProcessInfo.processInfo.environment["PROJECT_DIR"] ?? fileManager.currentDirectoryPath
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent(
            "edgerunner-publishable-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempDir) }

        var runResults: [RunResult] = []
        var memLoadMB: Double = 0
        var memPeakMB: Double = 0
        var referenceHash: String?
        var referencePrefix: [Int]?
        var childLMHeadValues: [Double] = []
        var deterministic = true

        for run in 0..<runCount {
            let reportURL = tempDir.appendingPathComponent("child-run-\(run).json")
            let report = try runIsolatedChildProcess(
                projectDir: projectDir,
                reportURL: reportURL,
                tokenCount: tokenCount,
                contextWindow: contextWindow,
                profileLMHead: profileLMHead
            )

            guard let persistedRun = report.runs.first else {
                throw GenerationError.decodingFailed("Publishable child run \(run) produced no per-run data")
            }

            let syntheticTokens = Array(repeating: 0, count: persistedRun.tokenCount)
            runResults.append(
                RunResult(
                    runIndex: run,
                    tokens: syntheticTokens,
                    ttftMs: persistedRun.ttftMs,
                    decodeTokenCount: persistedRun.decodeTokenCount,
                    decodeTotalMs: persistedRun.decodeTotalMs,
                    decodeTokPerSec: persistedRun.decodeTokPerSec,
                    endToEndTokPerSec: persistedRun.e2eTokPerSec,
                    totalMs: persistedRun.totalMs,
                    decodeLatencies: persistedRun.decodeLatencies,
                    hasNaN: persistedRun.hasNaN
                )
            )

            if run == 0 {
                memLoadMB = report.memoryMB.modelLoad
            }
            memPeakMB = max(memPeakMB, report.memoryMB.peakRSS)
            if let lmHead = report.lmHeadAvgMs {
                childLMHeadValues.append(lmHead)
            }
            if report.deterministic == false {
                deterministic = false
            }

            if let referenceHash, let referencePrefix {
                if report.tokenHash != referenceHash || report.greedyPrefix != referencePrefix {
                    deterministic = false
                }
            } else {
                referenceHash = report.tokenHash
                referencePrefix = report.greedyPrefix
            }
        }

        guard let referenceHash, let referencePrefix else {
            throw GenerationError.decodingFailed("Publishable benchmark produced no child results")
        }

        let lmHeadMs = childLMHeadValues.isEmpty
            ? nil
            : childLMHeadValues.reduce(0, +) / Double(childLMHeadValues.count)

        return BenchmarkExecution(
            runResults: runResults,
            memLoadMB: memLoadMB,
            memPeakMB: memPeakMB,
            deterministic: deterministic,
            tokenPrefix: referencePrefix,
            tokenHash: referenceHash,
            lmHeadMs: lmHeadMs
        )
    }

    private func runIsolatedChildProcess(
        projectDir: String,
        reportURL: URL,
        tokenCount: Int,
        contextWindow: Int,
        profileLMHead: Bool
    ) throws -> PersistedBenchmarkReport {
        let fileManager = FileManager.default
        let stdoutURL = reportURL.deletingPathExtension().appendingPathExtension("stdout.log")
        let stderrURL = reportURL.deletingPathExtension().appendingPathExtension("stderr.log")
        fileManager.createFile(atPath: stdoutURL.path, contents: nil)
        fileManager.createFile(atPath: stderrURL.path, contents: nil)
        defer {
            try? fileManager.removeItem(at: stdoutURL)
            try? fileManager.removeItem(at: stderrURL)
        }

        let testingHelperURL = try swiftPMTestingHelperURL()
        let testBinaryURL = try currentTestBinaryURL()
        let testBundleURL = try currentTestBundleURL(fallbackBinaryURL: testBinaryURL)

        let process = Process()
        process.currentDirectoryURL = URL(fileURLWithPath: projectDir)
        process.executableURL = testingHelperURL
        process.arguments = [
            "--test-bundle-path", testBundleURL.path,
            "-c", "release",
            "--filter", "PublishableBenchmark/fullBenchmark",
            testBinaryURL.path,
            "--testing-library", "swift-testing",
        ]

        var childEnv = ProcessInfo.processInfo.environment
        childEnv["PROJECT_DIR"] = projectDir
        childEnv["EDGERUNNER_BENCHMARK_TOKENS"] = String(tokenCount)
        childEnv["EDGERUNNER_BENCHMARK_RUNS"] = "1"
        childEnv["EDGERUNNER_BENCHMARK_CONTEXT"] = String(contextWindow)
        childEnv[Self.childRunEnvKey] = "1"
        childEnv[Self.processIsolationEnvKey] = "0"
        childEnv[Self.outputPathEnvKey] = reportURL.path
        if profileLMHead {
            childEnv["EDGERUNNER_PROFILE_LMHEAD"] = "1"
        } else {
            childEnv.removeValue(forKey: "EDGERUNNER_PROFILE_LMHEAD")
        }
        process.environment = childEnv

        process.standardOutput = try FileHandle(forWritingTo: stdoutURL)
        process.standardError = try FileHandle(forWritingTo: stderrURL)

        try process.run()
        process.waitUntilExit()

        let stdout = (try? String(contentsOf: stdoutURL, encoding: .utf8)) ?? ""
        let stderr = (try? String(contentsOf: stderrURL, encoding: .utf8)) ?? ""

        guard process.terminationStatus == 0 else {
            throw GenerationError.decodingFailed(
                """
                Publishable child benchmark failed with status \(process.terminationStatus).
                stdout:
                \(tailLines(stdout))
                stderr:
                \(tailLines(stderr))
                """
            )
        }

        let data = try Data(contentsOf: reportURL)
        return try JSONDecoder().decode(PersistedBenchmarkReport.self, from: data)
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

    private struct BenchmarkExecution {
        let runResults: [RunResult]
        let memLoadMB: Double
        let memPeakMB: Double
        let deterministic: Bool
        let tokenPrefix: [Int]
        let tokenHash: String
        let lmHeadMs: Double?
    }

    private struct PersistedBenchmarkReport: Decodable {
        struct PersistedRun: Decodable {
            let run: Int
            let decodeTokPerSec: Double
            let e2eTokPerSec: Double
            let ttftMs: Double
            let decodeTotalMs: Double
            let decodeTokenCount: Int
            let totalMs: Double
            let tokenCount: Int
            let decodeLatencies: [Double]
            let hasNaN: Bool

            private enum CodingKeys: String, CodingKey {
                case run
                case decodeTokPerSec = "decode_tok_per_sec"
                case e2eTokPerSec = "e2e_tok_per_sec"
                case ttftMs = "ttft_ms"
                case decodeTotalMs = "decode_total_ms"
                case decodeTokenCount = "decode_token_count"
                case totalMs = "total_ms"
                case tokenCount = "token_count"
                case decodeLatencies = "decode_latencies_ms"
                case hasNaN = "has_nan"
            }
        }

        struct Memory: Decodable {
            let modelLoad: Double
            let peakRSS: Double

            private enum CodingKeys: String, CodingKey {
                case modelLoad = "model_load"
                case peakRSS = "peak_rss"
            }
        }

        let greedyPrefix: [Int]
        let tokenHash: String
        let deterministic: Bool
        let lmHeadAvgMs: Double?
        let memoryMB: Memory
        let runs: [PersistedRun]

        private enum CodingKeys: String, CodingKey {
            case greedyPrefix = "greedy_prefix"
            case tokenHash = "token_hash"
            case deterministic
            case lmHeadAvgMs = "lm_head_avg_ms"
            case memoryMB = "memory_mb"
            case runs
        }
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
        isCanonicalRun: Bool,
        tokenHash: String
    ) {
        print("")
        print("=" * 70)
        print("  EdgeRunner Inference Benchmark — \(Self.contract.model.name)")
        print("=" * 70)
        print("")
        print("  Model:           \(Self.contract.model.name) (\(Self.expectedModelFileSizeBytes)-byte GGUF, 28 layers, 151K vocab)")
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
        print("\(prefix): qwen_token_hash \(tokenHash)")
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
                "decode_latencies_ms": run.decodeLatencies,
                "has_nan": run.hasNaN,
            ]
        }

        let json: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "model": Self.contract.model.name,
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

        let outputURL: URL
        if let overridePath = ProcessInfo.processInfo.environment[Self.outputPathEnvKey], !overridePath.isEmpty {
            if overridePath.hasPrefix("/") {
                outputURL = URL(fileURLWithPath: overridePath)
            } else {
                outputURL = URL(fileURLWithPath: projectDir).appendingPathComponent(overridePath)
            }
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        } else {
            let outputFileName = isCanonicalRun ? "publishable_benchmark.json" : "publishable_profile_benchmark.json"
            outputURL = benchDir.appendingPathComponent(outputFileName)
        }
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

    private func isExplicitlyDisabled(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.lowercased() {
        case "0", "false", "no", "off":
            return true
        default:
            return false
        }
    }

    private func tailLines(_ output: String, maxLines: Int = 40) -> String {
        let lines = output.split(whereSeparator: \.isNewline)
        guard lines.count > maxLines else { return output }
        return lines.suffix(maxLines).joined(separator: "\n")
    }

    private func swiftPMTestingHelperURL() throws -> URL {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["--find", "swift-test"]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let swiftTestPath = String(
            data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        )?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        guard process.terminationStatus == 0, !swiftTestPath.isEmpty else {
            throw GenerationError.decodingFailed(
                "Could not locate swift-test via xcrun for process-isolated benchmark child. \(tailLines(stderr))"
            )
        }

        let toolchainUsrURL = URL(fileURLWithPath: swiftTestPath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let helperURL = toolchainUsrURL.appendingPathComponent("libexec/swift/pm/swiftpm-testing-helper")
        guard FileManager.default.isExecutableFile(atPath: helperURL.path) else {
            throw GenerationError.decodingFailed(
                "swiftpm-testing-helper is not executable at \(helperURL.path)"
            )
        }

        return helperURL
    }

    private func currentTestBinaryURL() throws -> URL {
        let arguments = CommandLine.arguments
        if let testingLibraryIndex = arguments.firstIndex(of: "--testing-library"),
           testingLibraryIndex > 0 {
            return URL(fileURLWithPath: arguments[testingLibraryIndex - 1])
        }

        if let binaryArgument = arguments.last(where: { !$0.hasPrefix("-") }) {
            return URL(fileURLWithPath: binaryArgument)
        }

        throw GenerationError.decodingFailed("Could not resolve current test binary path from command-line arguments.")
    }

    private func currentTestBundleURL(fallbackBinaryURL: URL) throws -> URL {
        let arguments = CommandLine.arguments
        if let bundleIndex = arguments.firstIndex(of: "--test-bundle-path"),
           bundleIndex + 1 < arguments.count {
            return URL(fileURLWithPath: arguments[bundleIndex + 1])
        }

        return fallbackBinaryURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
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
