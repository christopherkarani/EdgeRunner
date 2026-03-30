import Foundation
import Testing
@testable import EdgeRunner

/// Long story generation benchmark to test KV cache performance over extended turns.
/// Generates a 1000+ token story and tracks decode performance.
@Suite("Long Story Generation - KV Cache Stress Test")
struct LongStoryGenerationTest {
    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let targetTokenCount = 1000
    static let contextWindowSize = 2048

    @Test func generateLongStory() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(Self.modelPath)")
        }

        print("\n=== Loading Qwen 3 0.6B ===")
        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )
        print("Model loaded. Vocab size: \(model.vocabularySize)")

        // Story prompt - something that should generate long creative output
        let storyPrompt = "Write a long, detailed story about a space explorer who discovers an ancient alien civilization on a distant moon. Include dialogue, descriptions, and plot twists:"
        var tokenIDs = model.tokenize(storyPrompt)
        print("Prompt tokens: \(tokenIDs.count)")

        var generatedTokens: [Int] = []
        var timestamps: [Double] = []
        var perfMeasurements: [(tokenIndex: Int, tokensPerSec: Double, msPerToken: Double)] = []

        let clock = ContinuousClock()
        var lastCheckpoint = clock.now
        let checkpointInterval = 100 // Measure every 100 tokens

        print("\n=== Starting Generation (target: \(Self.targetTokenCount) tokens) ===\n")

        for i in 0..<Self.targetTokenCount {
            let tokenStart = clock.now

            let result = try await model.greedyToken(for: tokenIDs)

            let tokenEnd = clock.now
            let tokenDuration = tokenStart.duration(to: tokenEnd)
            let tokenSeconds = Double(tokenDuration.components.seconds) + Double(tokenDuration.components.attoseconds) * 1e-18
            let tokensPerSec = 1.0 / tokenSeconds
            let msPerToken = tokenSeconds * 1000

            generatedTokens.append(result.token)
            tokenIDs.append(result.token)

            timestamps.append(tokenSeconds)

            // Checkpoint every N tokens
            if (i + 1) % checkpointInterval == 0 {
                let checkpointDuration = lastCheckpoint.duration(to: tokenEnd)
                let checkpointSeconds = Double(checkpointDuration.components.seconds) + Double(checkpointDuration.components.attoseconds) * 1e-18
                let checkpointTokensPerSec = Double(checkpointInterval) / checkpointSeconds
                let checkpointMsPerToken = checkpointSeconds / Double(checkpointInterval) * 1000

                perfMeasurements.append((
                    tokenIndex: i + 1,
                    tokensPerSec: checkpointTokensPerSec,
                    msPerToken: checkpointMsPerToken
                ))

                print("[\(i + 1)/\(Self.targetTokenCount)] " +
                      "Speed: \(String(format: "%.2f", checkpointTokensPerSec)) tok/s, " +
                      "Latency: \(String(format: "%.2f", checkpointMsPerToken)) ms/tok")

                lastCheckpoint = tokenEnd
            }
        }

        // Calculate overall statistics
        let totalTime = timestamps.reduce(0, +)
        let avgTokensPerSec = Double(Self.targetTokenCount) / totalTime
        let avgMsPerToken = totalTime / Double(Self.targetTokenCount) * 1000

        // Check for performance degradation
        let firstHalf = perfMeasurements.prefix(perfMeasurements.count / 2)
        let secondHalf = perfMeasurements.suffix(perfMeasurements.count / 2)

        let firstHalfAvg = firstHalf.map { $0.tokensPerSec }.reduce(0, +) / Double(firstHalf.count)
        let secondHalfAvg = secondHalf.map { $0.tokensPerSec }.reduce(0, +) / Double(secondHalf.count)

        let degradation = ((firstHalfAvg - secondHalfAvg) / firstHalfAvg) * 100

        print("\n=== GENERATION COMPLETE ===")
        print("Total tokens generated: \(generatedTokens.count)")
        print("Overall average: \(String(format: "%.2f", avgTokensPerSec)) tok/s (\(String(format: "%.2f", avgMsPerToken)) ms/tok)")
        print("\n=== PERFORMANCE BREAKDOWN ===")
        print("First half avg: \(String(format: "%.2f", firstHalfAvg)) tok/s")
        print("Second half avg: \(String(format: "%.2f", secondHalfAvg)) tok/s")

        if degradation > 10 {
            print("⚠️ WARNING: Performance degraded by \(String(format: "%.1f", degradation))%")
            print("This suggests a potential KV cache issue!")
        } else if degradation > 0 {
            print("ℹ️ Performance change: -\(String(format: "%.1f", degradation))% (within normal variance)")
        } else {
            print("✅ Performance improved by \(String(format: "%.1f", abs(degradation)))%")
        }

        // Detokenize and show sample
        let fullText = model.detokenize(tokenIDs)
        print("\n=== GENERATED STORY (first 500 chars) ===")
        print(String(fullText.prefix(500)) + "...")

        // Save detailed report
        let report = generateReport(
            prompt: storyPrompt,
            promptTokens: model.tokenize(storyPrompt).count,
            generatedTokens: generatedTokens.count,
            overallTokensPerSec: avgTokensPerSec,
            overallMsPerToken: avgMsPerToken,
            checkpoints: perfMeasurements,
            degradation: degradation,
            sample: String(fullText.prefix(1000))
        )

        let reportPath = "benchmarks/long_story_report.json"
        try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
        print("\n📊 Detailed report saved to: \(reportPath)")

        // Assertions to catch KV cache issues
        #expect(degradation < 50, "Performance degradation exceeds 50% - possible KV cache issue")
    }

    private func generateReport(
        prompt: String,
        promptTokens: Int,
        generatedTokens: Int,
        overallTokensPerSec: Double,
        overallMsPerToken: Double,
        checkpoints: [(tokenIndex: Int, tokensPerSec: Double, msPerToken: Double)],
        degradation: Double,
        sample: String
    ) -> String {
        let formatter = ISO8601DateFormatter()
        let timestamp = formatter.string(from: Date())

        var checkpointJSON: [String] = []
        for cp in checkpoints {
            checkpointJSON.append("""
                {
                    "tokenIndex": \(cp.tokenIndex),
                    "tokensPerSec": \(String(format: "%.4f", cp.tokensPerSec)),
                    "msPerToken": \(String(format: "%.4f", cp.msPerToken))
                }
            """)
        }

        return """
        {
            "timestamp": "\(timestamp)",
            "model": "Qwen3-0.6B-Q8_0",
            "prompt": "\(prompt.replacingOccurrences(of: "\"", with: "\\\""))",
            "promptTokens": \(promptTokens),
            "generatedTokens": \(generatedTokens),
            "contextWindowSize": \(Self.contextWindowSize),
            "performance": {
                "overallTokensPerSec": \(String(format: "%.4f", overallTokensPerSec)),
                "overallMsPerToken": \(String(format: "%.4f", overallMsPerToken)),
                "degradationPercent": \(String(format: "%.2f", degradation))
            },
            "checkpoints": [\(checkpointJSON.joined(separator: ",\n"))],
            "sample": "\(sample.replacingOccurrences(of: "\"", with: "\\\""))"
        }
        """
    }
}
