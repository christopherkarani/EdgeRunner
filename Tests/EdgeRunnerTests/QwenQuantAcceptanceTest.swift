import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen GGUF quant acceptance")
struct QwenQuantAcceptanceTest {
    private static let directoryEnvKey = "EDGERUNNER_QWEN_QUANT_DIR"
    private static let maxTokens = 16
    private static let minimumGeneratedTokens = 4
    private static let prompt = """
    <|im_start|>user
    Write one short sentence about mobile inference.<|im_end|>
    <|im_start|>assistant
    """

    private struct QuantCase {
        let name: String
        let fileName: String
        let envKey: String
    }

    private static let quantCases: [QuantCase] = [
        QuantCase(name: "Q2_K", fileName: "Qwen3-0.6B-Q2_K.gguf", envKey: "EDGERUNNER_QWEN_Q2_K_MODEL"),
        QuantCase(name: "Q3_K_M", fileName: "Qwen3-0.6B-Q3_K_M.gguf", envKey: "EDGERUNNER_QWEN_Q3_K_M_MODEL"),
        QuantCase(name: "Q4_K_M", fileName: "Qwen3-0.6B-Q4_K_M.gguf", envKey: "EDGERUNNER_QWEN_Q4_K_M_MODEL"),
        QuantCase(name: "Q5_K_M", fileName: "Qwen3-0.6B-Q5_K_M.gguf", envKey: "EDGERUNNER_QWEN_Q5_K_M_MODEL"),
        QuantCase(name: "Q6_K", fileName: "Qwen3-0.6B-Q6_K.gguf", envKey: "EDGERUNNER_QWEN_Q6_K_MODEL"),
        QuantCase(name: "Q8_0", fileName: "Qwen3-0.6B-Q8_0.gguf", envKey: "EDGERUNNER_QWEN_Q8_0_MODEL"),
    ]

    @Test func selectedMobileQuantsGenerateText() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env[Self.directoryEnvKey] != nil || Self.quantCases.contains(where: { env[$0.envKey] != nil }) else {
            return
        }

        for quant in Self.quantCases {
            guard let modelURL = Self.resolveModelURL(for: quant, env: env) else {
                Issue.record("Missing \(quant.name) model. Set \(quant.envKey) or place \(quant.fileName) in \(Self.directoryEnvKey).")
                continue
            }

            let result = try await Self.runQuantAcceptance(
                modelURL: modelURL,
                prompt: Self.prompt,
                maxTokens: Self.maxTokens,
                minimumGeneratedTokens: Self.minimumGeneratedTokens
            )
            print("QWEN_QUANT_ACCEPTANCE quant=\(quant.name) tokens=\(result.generatedTokens) text=\(result.text)")
        }
    }

    private static func resolveModelURL(for quant: QuantCase, env: [String: String]) -> URL? {
        if let path = env[quant.envKey], !path.isEmpty {
            let url = URL(fileURLWithPath: path)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }

        guard let directory = env[directoryEnvKey], !directory.isEmpty else {
            return nil
        }
        let url = URL(fileURLWithPath: directory).appendingPathComponent(quant.fileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private static func runQuantAcceptance(
        modelURL: URL,
        prompt: String,
        maxTokens: Int,
        minimumGeneratedTokens: Int
    ) async throws -> (generatedTokens: Int, text: String) {
        let model = try await ModelLoader.load(
            from: modelURL,
            configuration: ModelConfiguration(contextWindowSize: 1024)
        )
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )
        var tokenIDs = model.tokenize(prompt)
        var generated: [Int] = []

        for _ in 0..<maxTokens {
            let token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            guard token != model.eosTokenID else { break }
            generated.append(token)
            tokenIDs.append(token)
        }

        let text = model.detokenize(generated).trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(generated.count >= minimumGeneratedTokens)
        #expect(!text.isEmpty)
        #expect(Set(generated).count > 1, "Generated token stream collapsed to one repeated token: \(generated)")
        #expect(text.rangeOfCharacter(from: .letters) != nil, "Generated text has no letters: \(text)")
        return (generated.count, text)
    }
}
