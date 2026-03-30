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
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let prompt = Array(repeating: 9707, count: promptLength)

        let fp16 = try await runCase(
            modelURL: modelURL,
            prompt: prompt,
            decodeCount: decodeCount,
            compression: .disabled
        )
        let aggressive = try await runCase(
            modelURL: modelURL,
            prompt: prompt,
            decodeCount: decodeCount,
            compression: .turboQuantAggressive
        )

        print("""
        [turboquant-benchmark]
          prompt_len=\(promptLength)
          decode_tokens=\(decodeCount)
          fp16_decode_tok_s=\(String(format: "%.2f", fp16.decodeTokensPerSecond))
          turboquant_decode_tok_s=\(String(format: "%.2f", aggressive.decodeTokensPerSecond))
          fp16_ttft_ms=\(String(format: "%.2f", fp16.ttftMilliseconds))
          turboquant_ttft_ms=\(String(format: "%.2f", aggressive.ttftMilliseconds))
        """)

        #expect(fp16.decodeTokensPerSecond > 0)
        #expect(aggressive.decodeTokensPerSecond > 0)
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
        let decodeStart = clock.now
        while generated < decodeCount {
            var maxValue: Float = -.infinity
            var maxIndex = 0
            for (index, value) in currentLogits.enumerated() where value > maxValue {
                maxValue = value
                maxIndex = index
            }
            tokenIDs.append(maxIndex)
            currentLogits = try await model.logits(for: tokenIDs)
            generated += 1
        }
        let decodeElapsed = decodeStart.duration(to: clock.now)

        return BenchmarkResult(
            decodeTokensPerSecond: Double(decodeCount) / seconds(decodeElapsed),
            ttftMilliseconds: seconds(prefillElapsed) * 1000
        )
    }

    private func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }
}

private struct BenchmarkResult {
    let decodeTokensPerSecond: Double
    let ttftMilliseconds: Double
}
