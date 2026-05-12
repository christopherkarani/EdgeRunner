import Foundation
import Testing
@testable import EdgeRunner

@Suite("TurboQuant Long Context Benchmark")
struct TurboQuantLongContextBenchmark {
    private static let runEnvKey = "EDGERUNNER_RUN_TURBOQUANT_BENCHMARK"
    private static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    @Test
    func compareTurboQuantAgainstFP16() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_PROMPT_LEN"] ?? "4096") ?? 4096
        let decodeCount = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_DECODE_TOKENS"] ?? "32") ?? 32
        let mode = ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_BENCHMARK_MODE"] ?? "both"
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let prompt = Array(repeating: 9707, count: promptLength)

        let fp16: BenchmarkResult?
        if mode == "both" || mode == "fp16" {
            fp16 = try await runCase(
                modelURL: modelURL,
                prompt: prompt,
                decodeCount: decodeCount,
                compression: .disabled
            )
        } else {
            fp16 = nil
        }
        let aggressive: BenchmarkResult?
        if mode == "both" || mode == "aggressive" {
            aggressive = try await runCase(
                modelURL: modelURL,
                prompt: prompt,
                decodeCount: decodeCount,
                compression: .turboquantV2
            )
        } else {
            aggressive = nil
        }

        print("""
        [turboquant-benchmark]
          prompt_len=\(promptLength)
          decode_tokens=\(decodeCount)
          mode=\(mode)
          fp16_decode_tok_s=\(fp16.map { String(format: "%.2f", $0.decodeTokensPerSecond) } ?? "n/a")
          turboquant_decode_tok_s=\(aggressive.map { String(format: "%.2f", $0.decodeTokensPerSecond) } ?? "n/a")
          fp16_ttft_ms=\(fp16.map { String(format: "%.2f", $0.ttftMilliseconds) } ?? "n/a")
          turboquant_ttft_ms=\(aggressive.map { String(format: "%.2f", $0.ttftMilliseconds) } ?? "n/a")
          fp16_latency_p50_ms=\(fp16.map { String(format: "%.2f", $0.latencyStats.p50) } ?? "n/a")
          fp16_latency_p90_ms=\(fp16.map { String(format: "%.2f", $0.latencyStats.p90) } ?? "n/a")
          fp16_latency_p99_ms=\(fp16.map { String(format: "%.2f", $0.latencyStats.p99) } ?? "n/a")
          turboquant_latency_p50_ms=\(aggressive.map { String(format: "%.2f", $0.latencyStats.p50) } ?? "n/a")
          turboquant_latency_p90_ms=\(aggressive.map { String(format: "%.2f", $0.latencyStats.p90) } ?? "n/a")
          turboquant_latency_p99_ms=\(aggressive.map { String(format: "%.2f", $0.latencyStats.p99) } ?? "n/a")
          fp16_token_hash=\(fp16?.tokenHash ?? "n/a")
          turboquant_token_hash=\(aggressive?.tokenHash ?? "n/a")
        """)

        if let fp16 {
            #expect(fp16.decodeTokensPerSecond > 0)
        }
        if let aggressive {
            #expect(aggressive.decodeTokensPerSecond > 0)
        }
    }

    private func runCase(
        modelURL: URL,
        prompt: [Int],
        decodeCount: Int,
        compression: KVCacheCompression
    ) async throws -> BenchmarkResult {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(prompt.count + decodeCount + 16, 8192),
                kvCacheCompression: compression
            )
        )

        let clock = ContinuousClock()
        let prefillStart = clock.now
        let logits = try await model.logits(for: prompt)
        let prefillElapsed = prefillStart.duration(to: clock.now)
        #expect(!logits.contains(where: { !$0.isFinite }))

        var tokenIDs = prompt
        var currentLogits = logits
        var generated = 0
        var generatedTokenIDs: [Int] = []
        var decodeLatenciesMilliseconds: [Double] = []
        let decodeStart = clock.now
        while generated < decodeCount {
            let tokenStart = clock.now
            var maxValue: Float = -.infinity
            var maxIndex = 0
            for (index, value) in currentLogits.enumerated() where value > maxValue {
                maxValue = value
                maxIndex = index
            }
            tokenIDs.append(maxIndex)
            generatedTokenIDs.append(maxIndex)
            currentLogits = try await model.logits(for: tokenIDs)
            let tokenEnd = clock.now
            decodeLatenciesMilliseconds.append(seconds(tokenStart.duration(to: tokenEnd)) * 1000.0)
            generated += 1
        }
        let decodeElapsed = decodeStart.duration(to: clock.now)
        let latencyStats = LatencyStats.make(decodeLatenciesMilliseconds)

        return BenchmarkResult(
            decodeTokensPerSecond: Double(decodeCount) / seconds(decodeElapsed),
            ttftMilliseconds: seconds(prefillElapsed) * 1000,
            latencyStats: latencyStats,
            tokenHash: tokenSequenceHash(generatedTokenIDs)
        )
    }

    private func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
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

    @Test
    func benchmarkSummaryComputesPercentilesAndStableTokenHash() throws {
        let stats = LatencyStats.make([10, 20, 30, 40, 50])
        #expect(stats.p50 == 30)
        #expect(stats.p90 == 50)
        #expect(stats.p99 == 50)
        #expect(tokenSequenceHash([1, 1479, 35]) == tokenSequenceHash([1, 1479, 35]))
        #expect(tokenSequenceHash([1, 1479, 35]) != tokenSequenceHash([1, 1479, 36]))
    }
}

private struct BenchmarkResult {
    let decodeTokensPerSecond: Double
    let ttftMilliseconds: Double
    let latencyStats: LatencyStats
    let tokenHash: String
}

private struct LatencyStats {
    let p50: Double
    let p90: Double
    let p99: Double

    static func make(_ values: [Double]) -> LatencyStats {
        guard !values.isEmpty else {
            return LatencyStats(p50: 0, p90: 0, p99: 0)
        }
        let sorted = values.sorted()
        return LatencyStats(
            p50: percentile(0.50, in: sorted),
            p90: percentile(0.90, in: sorted),
            p99: percentile(0.99, in: sorted)
        )
    }

    private static func percentile(_ percentile: Double, in sortedValues: [Double]) -> Double {
        let clamped = min(max(percentile, 0), 1)
        let index = Int((Double(sortedValues.count - 1) * clamped).rounded(.up))
        return sortedValues[index]
    }
}
