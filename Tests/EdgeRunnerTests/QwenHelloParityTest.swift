import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen Hello Parity")
struct QwenHelloParityTest {
    private static let runEnvKey = "EDGERUNNER_RUN_HELLO_PARITY"
    private static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    private static let promptTokens = [9707]
    private static let generateCount = 8

    @Test
    func helloGreedyTrace() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )

        var tokenIDs = Self.promptTokens
        var generated: [Int] = []
        for _ in 0..<Self.generateCount {
            let logits = try await model.logits(for: tokenIDs)
            #expect(!logits.contains(where: { !$0.isFinite }))

            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (index, value) in logits.enumerated() where value > maxVal {
                maxVal = value
                maxIdx = index
            }

            generated.append(maxIdx)
            tokenIDs.append(maxIdx)
        }

        print("[edgerunner-qwen-hello] generated=\(generated)")
    }
}
