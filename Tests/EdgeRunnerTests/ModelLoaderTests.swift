import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO

@Suite("ModelLoader")
struct ModelLoaderTests {
    static let qwenModelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    @Test func loadsQwenModel() async throws {
        guard FileManager.default.fileExists(atPath: Self.qwenModelPath) else {
            print("SKIP: Model not found at \(Self.qwenModelPath)")
            return
        }

        let model = try await ModelLoader.load(
            from: URL(fileURLWithPath: Self.qwenModelPath)
        )
        #expect(model is LlamaLanguageModel)
        #expect(model.vocabularySize > 0)
    }

    @Test func throwsForInvalidPath() async {
        do {
            _ = try await ModelLoader.load(
                from: URL(fileURLWithPath: "/nonexistent/model.gguf")
            )
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Expected: file not found or load error
            #expect(error is GenerationError || error is any Error)
        }
    }
}
