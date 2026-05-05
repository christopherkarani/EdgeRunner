import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO

@Suite("ModelLoader")
struct ModelLoaderTests {
    static let qwenModelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    static let bonsaiModelPath = (NSHomeDirectory() as NSString).appendingPathComponent("edgerunner-models/Bonsai-1.7B.gguf")

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

    @Test func routesBonsaiToDedicatedBackend() async throws {
        guard FileManager.default.fileExists(atPath: Self.bonsaiModelPath) else {
            print("SKIP: Model not found at \(Self.bonsaiModelPath)")
            return
        }

        let model = try await ModelLoader.load(
            from: URL(fileURLWithPath: Self.bonsaiModelPath)
        )
        #expect(model is BonsaiLanguageModel)
        #expect(model.vocabularySize > 0)
    }

    @Test func recognizesGemma4AsDedicatedBackend() {
        let config = ModelConfig(
            architectureName: "gemma4",
            metadata: Gemma4ModelConfigTests.makeReferenceModelConfigMetadata()
        )

        #expect(Gemma4LanguageModel.supports(modelConfig: config))
    }

    @Test func throwsForInvalidPath() async {
        do {
            _ = try await ModelLoader.load(
                from: URL(fileURLWithPath: "/nonexistent/model.gguf")
            )
            #expect(Bool(false), "Should have thrown")
        } catch {
            #expect(String(describing: error).isEmpty == false)
        }
    }
}
