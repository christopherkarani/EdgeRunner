import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen TurboQuant Smoke")
struct QwenTurboQuantSmokeTest {
    private static let runEnvKey = "EDGERUNNER_RUN_TURBOQUANT_SMOKE"
    private static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    // Keep this prompt multi-token so the smoke test exercises TurboQuant prefill
    // cache writes, not only single-token decode appends.
    private static let promptTokens = [9707, 25, 220]
    private static let generateCount = 4
    private static let expectedGenerated = [16, 11, 220, 508]

    @Test
    func aggressiveTurboQuantGreedyTrace() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(
                contextWindowSize: 8192,
                kvCacheCompression: .turboquantV2
            )
        )

        var tokenIDs = Self.promptTokens
        var generated: [Int] = []
        for _ in 0..<Self.generateCount {
            let logits = try await model.logits(for: tokenIDs)
            #expect(!logits.contains(where: { !$0.isFinite }))

            var maxValue: Float = -.infinity
            var maxIndex = 0
            for (index, value) in logits.enumerated() where value > maxValue {
                maxValue = value
                maxIndex = index
            }

            generated.append(maxIndex)
            tokenIDs.append(maxIndex)
        }

        print("[edgerunner-qwen-turboquant] generated=\(generated)")
        #expect(generated.count == Self.generateCount)
        #expect(generated == Self.expectedGenerated)
    }
}
