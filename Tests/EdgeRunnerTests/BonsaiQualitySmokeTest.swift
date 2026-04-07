import Foundation
import Testing
@testable import EdgeRunner

@Suite("Bonsai Quality Smoke")
struct BonsaiQualitySmokeTest {
    static let modelPath = (NSHomeDirectory() as NSString).appendingPathComponent("edgerunner-models/Bonsai-1.7B.gguf")

    @Test
    func capitalPromptContainsParis() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let generated = try await Self.generate(
            model: model,
            prompt: "The capital of France is",
            maxTokens: 16
        )

        print("[bonsai-capital] \(generated)")
        #expect(generated.lowercased().contains("paris"))
    }

    @Test
    func quantumPromptIsNotDegenerate() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let generated = try await Self.generate(
            model: model,
            prompt: "Explain quantum computing in simple terms:",
            maxTokens: 24
        )

        print("[bonsai-quantum] \(generated)")
        let words = generated
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
        let uniqueWords = Set(words.map { $0.lowercased() })

        #expect(words.count >= 8)
        #expect(uniqueWords.count >= max(4, words.count / 3))
    }

    private static func generate(
        model: BonsaiLanguageModel,
        prompt: String,
        maxTokens: Int
    ) async throws -> String {
        var tokenIDs = model.tokenize(prompt)
        var generated: [Int] = []

        for _ in 0..<maxTokens {
            let result = try await model.greedyToken(for: tokenIDs)
            #expect(!result.hasNonFinite)
            tokenIDs.append(result.token)
            generated.append(result.token)
            if result.token == model.eosTokenID {
                break
            }
        }

        return model.detokenize(generated)
    }
}
