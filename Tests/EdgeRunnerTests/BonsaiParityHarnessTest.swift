import Foundation
import Testing
@testable import EdgeRunner

@Suite("Bonsai Decode Parity Harness")
struct BonsaiParityHarnessTest {
    struct PromptCase: Sendable {
        let label: String
        let text: String
    }

    struct ModeCase: Sendable {
        let label: String
        let overrides: LlamaDecodeOverrides
    }

    struct StepReport: Codable, Sendable {
        let stepIndex: Int
        let promptLength: Int
        let tokenID: Int
        let tokenText: String
    }

    struct ModeReport: Codable, Sendable {
        let mode: String
        let prompt: String
        let promptTokenIDs: [Int]
        let generatedTokenIDs: [Int]
        let generatedText: String
        let steps: [StepReport]
    }

    static let modelPath = (NSHomeDirectory() as NSString).appendingPathComponent("edgerunner-models/Bonsai-1.7B.gguf")
    static let outputPath = "/tmp/bonsai_parity.json"
    static let maxSteps = 8

    static let prompts = [
        PromptCase(label: "hello", text: "Hello"),
        PromptCase(label: "quantum", text: "Explain quantum computing in simple terms:"),
        PromptCase(label: "capital", text: "The capital of France is"),
    ]

    static let modes = [
        ModeCase(
            label: "default",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: false,
                disableMegaKernel: false,
                disableFusedFinalNormLMHead: false,
                preferMetal4DecodePath: false
            )
        ),
        ModeCase(
            label: "base",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: true,
                disableMegaKernel: false,
                disableFusedFinalNormLMHead: false,
                preferMetal4DecodePath: false
            )
        ),
        ModeCase(
            label: "base_no_mega",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: true,
                disableMegaKernel: true,
                disableFusedFinalNormLMHead: false,
                preferMetal4DecodePath: false
            )
        ),
    ]

    @Test
    func bonsaiParity() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Bonsai model not found at \(Self.modelPath)")
            return
        }

        var reports: [ModeReport] = []

        for mode in Self.modes {
            var configuration = ModelConfiguration(contextWindowSize: 2048)
            configuration.llamaDecodeOverrides = mode.overrides
            let model = try await BonsaiLanguageModel.load(
                from: URL(fileURLWithPath: Self.modelPath),
                configuration: configuration
            )

            for prompt in Self.prompts {
                reports.append(try await Self.trace(prompt: prompt, mode: mode.label, model: model))
            }
        }

        try Self.writeArtifact(reports)

        for prompt in Self.prompts {
            let promptReports = reports.filter { $0.prompt == prompt.label }
            #expect(promptReports.count == Self.modes.count)
            let baseline = try #require(promptReports.first(where: { $0.mode == "base_no_mega" }))
            for report in promptReports where report.mode != baseline.mode {
                #expect(
                    report.generatedTokenIDs == baseline.generatedTokenIDs,
                    "\(prompt.label) diverged for \(report.mode): \(report.generatedTokenIDs) vs \(baseline.generatedTokenIDs)"
                )
            }
        }
    }

    private static func trace(
        prompt: PromptCase,
        mode: String,
        model: BonsaiLanguageModel
    ) async throws -> ModeReport {
        var tokenIDs = model.tokenize(prompt.text)

        let promptTokenIDs = tokenIDs
        var generatedTokenIDs: [Int] = []
        var steps: [StepReport] = []

        for stepIndex in 0..<Self.maxSteps {
            let result = try await model.greedyToken(for: tokenIDs)
            #expect(!result.hasNonFinite, "Non-finite logits for \(prompt.label) in \(mode)")
            tokenIDs.append(result.token)
            generatedTokenIDs.append(result.token)
            steps.append(
                StepReport(
                    stepIndex: stepIndex,
                    promptLength: tokenIDs.count - 1,
                    tokenID: result.token,
                    tokenText: model.detokenize([result.token])
                )
            )
            if result.token == model.eosTokenID {
                break
            }
        }

        return ModeReport(
            mode: mode,
            prompt: prompt.label,
            promptTokenIDs: promptTokenIDs,
            generatedTokenIDs: generatedTokenIDs,
            generatedText: model.detokenize(generatedTokenIDs),
            steps: steps
        )
    }

    private static func writeArtifact(_ reports: [ModeReport]) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(reports).write(to: URL(fileURLWithPath: Self.outputPath))
    }
}
