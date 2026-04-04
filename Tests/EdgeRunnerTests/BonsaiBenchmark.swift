import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Bonsai 1.7B Benchmark")
struct BonsaiBenchmark {
    static let modelPath = "/tmp/edgerunner-models/Bonsai-1.7B.gguf"

    @Test("Bonsai 1.7B Q1_0_g128 end-to-end benchmark")
    func bonsaiEndToEndBenchmark() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let fileSize = try FileManager.default.attributesOfItem(atPath: Self.modelPath)[.size] as? Int64 ?? 0
        let fileSizeMB = Double(fileSize) / 1024.0 / 1024.0

        // Load via GGUFLoader to dump config
        let loader = try GGUFLoader(url: URL(fileURLWithPath: Self.modelPath))
        let ggufConfig = try LlamaConfig(fromGGUFMetadata: loader.modelConfig.metadata)

        print("\n=== Bonsai-1.7B Q1_0_g128 Benchmark ===")
        print("File size: \(String(format: "%.1f", fileSizeMB)) MB")
        print("Architecture:")
        print("  embeddingDim: \(ggufConfig.embeddingDim)")
        print("  layerCount: \(ggufConfig.layerCount)")
        print("  headCount: \(ggufConfig.headCount)")
        print("  kvHeadCount: \(ggufConfig.kvHeadCount)")
        print("  headDim: \(ggufConfig.headDim)")
        print("  gqaRatio: \(ggufConfig.gqaRatio)")
        print("  intermediateDim: \(ggufConfig.intermediateDim)")
        print("  vocabSize: \(ggufConfig.vocabSize)")
        print("  ropeFreqBase: \(ggufConfig.ropeFreqBase)")
        print("  rmsNormEpsilon: \(ggufConfig.rmsNormEpsilon)")

        // Compare with Qwen3 0.6B
        let qwen3Flops = Double(6 * ggufConfig.layerCount)
            * Double(ggufConfig.embeddingDim)
            * Double(ggufConfig.embeddingDim * 4 + ggufConfig.intermediateDim * 2)
        print("\n  Estimated FLOPs/token: \(String(format: "%.1e", qwen3Flops))")
        print("  Qwen3 0.6B FLOPs/token: \(String(format: "%.1e", 6 * 28 * 1024 * (1024 * 4 + 3072 * 2)))")

        // Load model
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let prompt = "Explain quantum computing in simple terms:"
        var tokenIDs = model.tokenize(prompt)
        if let bos = model.bosTokenID, tokenIDs.first != bos {
            tokenIDs.insert(bos, at: 0)
        }
        let promptTokenIDs = tokenIDs
        print("Prompt tokens: \(promptTokenIDs.count)")

        // Warmup: generate a few tokens to prime kernels and cache
        print("\n=== Warmup (4 tokens) ===")
        var warmupTokens = Array(tokenIDs)
        for _ in 0..<4 {
            let result = try await model.greedyToken(for: warmupTokens)
            warmupTokens.append(result.token)
        }
        model.resetGenerationState(keepDecodeWarmup: true)
        print("Warmup complete. Starting benchmark...\n")

        var timings: [Double] = []

        for run in 0..<5 {
            // Reset to prompt tokens for each run
            var tokenIDs = Array(promptTokenIDs)

            let clock = ContinuousClock()
            let startTime = clock.now

            // === PREFILL (TTFT) ===
            let prefillStart = clock.now
            let firstResult = try await model.greedyToken(for: tokenIDs)
            let prefillEnd = clock.now
            let prefillDuration = prefillStart.duration(to: prefillEnd)
            let ttftMs = (Double(prefillDuration.components.seconds) + Double(prefillDuration.components.attoseconds) * 1e-18) * 1000.0

            tokenIDs.append(firstResult.token)
            var generatedCount = 1

            // === DECODE (up to 128 tokens total) ===
            for _ in 1..<128 {
                let tokenStart = clock.now
                let result = try await model.greedyToken(for: tokenIDs)
                let tokenEnd = clock.now

                tokenIDs.append(result.token)
                generatedCount += 1

                // Early exit on EOS (try common EOS tokens)
                if result.token == 151645 || result.token == 2 || result.token == 0 {
                    break
                }
            }

            let totalDuration = startTime.duration(to: clock.now)
            let totalSeconds = Double(totalDuration.components.seconds)
                + Double(totalDuration.components.attoseconds) * 1e-18

            // Decode throughput (excluding prefill time)
            let decodeSeconds = totalSeconds - (ttftMs / 1000.0)
            let decodeTokens = generatedCount - 1 // exclude first token (prefill)
            let decodeTokPerSec = decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0

            timings.append(decodeTokPerSec)

            print("Run \(run): decode=\(String(format: "%.1f", decodeTokPerSec)) tok/s  tokens=\(generatedCount)  time=\(String(format: "%.3f", totalSeconds))s  ttft=\(String(format: "%.1f", ttftMs))ms")

            // Reset KV cache between runs
            model.resetGenerationState(keepDecodeWarmup: true)
        }

        // Compute statistics
        let sortedTimings = timings.sorted()
        let median = sortedTimings[sortedTimings.count / 2]
        let mean = timings.reduce(0, +) / Double(timings.count)
        let maxTok = timings.max() ?? 0
        let minTok = timings.min() ?? 0

        print("\n=== Results ===")
        print("Median decode: \(String(format: "%.1f", median)) tok/s")
        print("Mean decode: \(String(format: "%.1f", mean)) tok/s")
        print("Max decode: \(String(format: "%.1f", maxTok)) tok/s")
        print("Min decode: \(String(format: "%.1f", minTok)) tok/s")

        #expect(Bool(true), "Bonsai-1.7B Q1_0_g128 benchmark completed: \(String(format: "%.1f", median)) tok/s median decode")
    }
}
