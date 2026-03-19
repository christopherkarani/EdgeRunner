import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen 4B Decode Parity Harness")
struct Qwen4BParityHarnessTest {
    struct PromptCase: Sendable {
        let label: String
        let tokenIDs: [Int]
    }

    struct ModeCase: Sendable {
        let label: String
        let overrides: LlamaDecodeOverrides
    }

    struct TopCandidate: Codable, Sendable {
        let tokenID: Int
        let tokenText: String
        let logit: Float
    }

    struct TraceStepReport: Codable, Sendable {
        let stepIndex: Int
        let promptLength: Int
        let argmaxTokenID: Int
        let argmaxTokenText: String
        let topCandidates: [TopCandidate]
    }

    struct ModeTraceReport: Codable, Sendable {
        let mode: String
        let generatedTokenIDs: [Int]
        let generatedText: String
        let steps: [TraceStepReport]
    }

    struct PairwiseStepDelta: Codable, Sendable {
        let stepIndex: Int
        let lhsTokenID: Int
        let rhsTokenID: Int
        let argmaxMatches: Bool
        let maxAbsoluteLogitDelta: Float
    }

    struct PairwiseComparisonReport: Codable, Sendable {
        let lhsMode: String
        let rhsMode: String
        let firstArgmaxDivergenceStep: Int?
        let largestMaxAbsoluteLogitDelta: Float
        let steps: [PairwiseStepDelta]
    }

    struct PromptReport: Codable, Sendable {
        let prompt: String
        let promptTokenCount: Int
        let traces: [ModeTraceReport]
        let comparisons: [PairwiseComparisonReport]
    }

    struct Artifact: Codable, Sendable {
        let timestamp: String
        let modelPath: String
        let maxGeneratedSteps: Int
        let prompts: [PromptReport]
    }

    private struct RawTraceStep: Sendable {
        let stepIndex: Int
        let promptLength: Int
        let argmaxTokenID: Int
        let argmaxTokenText: String
        let topCandidates: [TopCandidate]
        let logits: [Float]
    }

    private struct RawModeTrace: Sendable {
        let mode: String
        let generatedTokenIDs: [Int]
        let generatedText: String
        let steps: [RawTraceStep]
    }

    static let runEnvKey = "EDGERUNNER_RUN_4B_PARITY"
    static let maxStepsEnvKey = "EDGERUNNER_PARITY_MAX_STEPS"
    static let promptFilterEnvKey = "EDGERUNNER_PARITY_PROMPT_FILTER"
    static let modeFilterEnvKey = "EDGERUNNER_PARITY_MODE_FILTER"
    static let outputPathEnvKey = "EDGERUNNER_PARITY_OUTPUT_PATH"

    static let modelPath = "/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf"
    static let defaultMaxSteps = 8
    static let defaultOutputPath = "/tmp/qwen_4b_parity.json"

    static let promptCases = [
        PromptCase(label: "story", tokenIDs: QwenQualityComparisonTest.storyPrompt),
        PromptCase(label: "capital_of_france", tokenIDs: CoherenceTest.completionPrompt),
        PromptCase(label: "chat_2_plus_2", tokenIDs: CoherenceTest.chatPrompt),
    ]

    static let modeCases = [
        ModeCase(
            label: "optimized_mega",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: false,
                disableMegaKernel: false,
                disableFusedFinalNormLMHead: false
            )
        ),
        ModeCase(
            label: "base_mega",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: true,
                disableMegaKernel: false,
                disableFusedFinalNormLMHead: false
            )
        ),
        ModeCase(
            label: "base_no_mega",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: true,
                disableMegaKernel: true,
                disableFusedFinalNormLMHead: false
            )
        ),
        ModeCase(
            label: "base_no_mega_no_fused_final",
            overrides: LlamaDecodeOverrides(
                forceBaseDecodePath: true,
                disableMegaKernel: true,
                disableFusedFinalNormLMHead: true
            )
        ),
    ]

    @Test
    func parityHarness() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            Swift.print("SKIP: Set \(Self.runEnvKey)=1 to run the manual 4B parity harness.")
            return
        }

        #expect(FileManager.default.fileExists(atPath: Self.modelPath), "Missing model file at \(Self.modelPath)")

        let maxSteps = Int(ProcessInfo.processInfo.environment[Self.maxStepsEnvKey] ?? "") ?? Self.defaultMaxSteps
        let selectedPrompts = Self.filteredPrompts(using: ProcessInfo.processInfo.environment[Self.promptFilterEnvKey])
        let selectedModes = Self.filteredModes(using: ProcessInfo.processInfo.environment[Self.modeFilterEnvKey])
        #expect(selectedPrompts.count > 0, "No prompts selected for parity harness")
        #expect(selectedModes.count >= 2, "Parity harness needs at least two decode modes")

        let vocabulary = try CoherenceTest.loadVocabulary(from: Self.modelPath)
        var tracesByPrompt: [String: [RawModeTrace]] = [:]

        for mode in selectedModes {
            Swift.print("PARITY: loading 4B model for mode=\(mode.label)")
            var configuration = ModelConfiguration(contextWindowSize: 2048)
            configuration.llamaDecodeOverrides = mode.overrides

            let model = try await LlamaLanguageModel.load(
                from: URL(fileURLWithPath: Self.modelPath),
                configuration: configuration
            )

            for prompt in selectedPrompts {
                let trace = try await Self.generateTrace(
                    mode: mode.label,
                    model: model,
                    prompt: prompt,
                    maxSteps: maxSteps,
                    vocabulary: vocabulary
                )
                tracesByPrompt[prompt.label, default: []].append(trace)
            }
        }

        let promptReports = selectedPrompts.map { prompt -> PromptReport in
            let traces = tracesByPrompt[prompt.label] ?? []
            let comparisons = Self.makePairwiseComparisons(for: traces)
            return PromptReport(
                prompt: prompt.label,
                promptTokenCount: prompt.tokenIDs.count,
                traces: traces.map(Self.report(from:)),
                comparisons: comparisons
            )
        }

        let artifact = Artifact(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            modelPath: Self.modelPath,
            maxGeneratedSteps: maxSteps,
            prompts: promptReports
        )

        let outputPath = ProcessInfo.processInfo.environment[Self.outputPathEnvKey] ?? Self.defaultOutputPath
        try Self.writeArtifact(artifact, to: outputPath)
        Self.printSummary(promptReports, outputPath: outputPath)
    }

    private static func filteredPrompts(using filter: String?) -> [PromptCase] {
        guard let filter, !filter.isEmpty else { return promptCases }
        let normalized = filter.lowercased()
        return promptCases.filter { $0.label.lowercased().contains(normalized) }
    }

    private static func filteredModes(using filter: String?) -> [ModeCase] {
        guard let filter, !filter.isEmpty else { return modeCases }
        let normalized = filter.lowercased()
        return modeCases.filter { $0.label.lowercased().contains(normalized) }
    }

    private static func generateTrace(
        mode: String,
        model: LlamaLanguageModel,
        prompt: PromptCase,
        maxSteps: Int,
        vocabulary: [String]
    ) async throws -> RawModeTrace {
        var tokenIDs = prompt.tokenIDs
        var generatedTokenIDs: [Int] = []
        var steps: [RawTraceStep] = []

        for stepIndex in 0..<maxSteps {
            let logits = try await model.logits(for: tokenIDs)
            #expect(!logits.contains(where: { !$0.isFinite }), "NaN/Inf logits in mode trace for \(prompt.label)")

            let topCandidates = Self.topCandidates(from: logits, vocabulary: vocabulary, count: 5)
            let argmax = topCandidates[0]
            generatedTokenIDs.append(argmax.tokenID)
            steps.append(
                RawTraceStep(
                    stepIndex: stepIndex,
                    promptLength: tokenIDs.count,
                    argmaxTokenID: argmax.tokenID,
                    argmaxTokenText: argmax.tokenText,
                    topCandidates: topCandidates,
                    logits: logits
                )
            )

            tokenIDs.append(argmax.tokenID)
            if argmax.tokenID == model.eosTokenID {
                break
            }
        }

        return RawModeTrace(
            mode: mode,
            generatedTokenIDs: generatedTokenIDs,
            generatedText: CoherenceTest.detokenize(generatedTokenIDs, vocabulary: vocabulary),
            steps: steps
        )
    }

    private static func topCandidates(
        from logits: [Float],
        vocabulary: [String],
        count: Int
    ) -> [TopCandidate] {
        var top: [(id: Int, logit: Float)] = []
        top.reserveCapacity(count)

        for (tokenID, logit) in logits.enumerated() {
            if top.count < count {
                top.append((tokenID, logit))
                top.sort { $0.logit > $1.logit }
                continue
            }

            if logit <= top[top.count - 1].logit {
                continue
            }

            top[top.count - 1] = (tokenID, logit)
            top.sort { $0.logit > $1.logit }
        }

        return top.map { candidate in
            TopCandidate(
                tokenID: candidate.id,
                tokenText: tokenText(for: candidate.id, vocabulary: vocabulary),
                logit: candidate.logit
            )
        }
    }

    private static func tokenText(for tokenID: Int, vocabulary: [String]) -> String {
        guard tokenID >= 0 && tokenID < vocabulary.count else { return "<invalid>" }
        return CoherenceTest.detokenize([tokenID], vocabulary: vocabulary)
    }

    private static func report(from rawTrace: RawModeTrace) -> ModeTraceReport {
        ModeTraceReport(
            mode: rawTrace.mode,
            generatedTokenIDs: rawTrace.generatedTokenIDs,
            generatedText: rawTrace.generatedText,
            steps: rawTrace.steps.map { step in
                TraceStepReport(
                    stepIndex: step.stepIndex,
                    promptLength: step.promptLength,
                    argmaxTokenID: step.argmaxTokenID,
                    argmaxTokenText: step.argmaxTokenText,
                    topCandidates: step.topCandidates
                )
            }
        )
    }

    private static func makePairwiseComparisons(for traces: [RawModeTrace]) -> [PairwiseComparisonReport] {
        var reports: [PairwiseComparisonReport] = []
        for lhsIndex in 0..<traces.count {
            for rhsIndex in (lhsIndex + 1)..<traces.count {
                let lhs = traces[lhsIndex]
                let rhs = traces[rhsIndex]
                reports.append(compare(lhs: lhs, rhs: rhs))
            }
        }
        return reports
    }

    private static func compare(lhs: RawModeTrace, rhs: RawModeTrace) -> PairwiseComparisonReport {
        let stepCount = min(lhs.steps.count, rhs.steps.count)
        var firstArgmaxDivergenceStep: Int?
        var largestDelta: Float = 0
        var stepReports: [PairwiseStepDelta] = []

        for stepIndex in 0..<stepCount {
            let lhsStep = lhs.steps[stepIndex]
            let rhsStep = rhs.steps[stepIndex]
            let maxAbsoluteLogitDelta = zip(lhsStep.logits, rhsStep.logits).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            largestDelta = max(largestDelta, maxAbsoluteLogitDelta)

            let argmaxMatches = lhsStep.argmaxTokenID == rhsStep.argmaxTokenID
            if !argmaxMatches && firstArgmaxDivergenceStep == nil {
                firstArgmaxDivergenceStep = stepIndex
            }

            stepReports.append(
                PairwiseStepDelta(
                    stepIndex: stepIndex,
                    lhsTokenID: lhsStep.argmaxTokenID,
                    rhsTokenID: rhsStep.argmaxTokenID,
                    argmaxMatches: argmaxMatches,
                    maxAbsoluteLogitDelta: maxAbsoluteLogitDelta
                )
            )
        }

        return PairwiseComparisonReport(
            lhsMode: lhs.mode,
            rhsMode: rhs.mode,
            firstArgmaxDivergenceStep: firstArgmaxDivergenceStep,
            largestMaxAbsoluteLogitDelta: largestDelta,
            steps: stepReports
        )
    }

    private static func writeArtifact(_ artifact: Artifact, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
    }

    private static func printSummary(_ reports: [PromptReport], outputPath: String) {
        Swift.print("")
        Swift.print(String(repeating: "=", count: 72))
        Swift.print("  QWEN 4B PARITY SUMMARY")
        Swift.print(String(repeating: "=", count: 72))
        for report in reports {
            Swift.print("Prompt: \(report.prompt) (\(report.promptTokenCount) tokens)")
            for comparison in report.comparisons {
                let divergence = comparison.firstArgmaxDivergenceStep.map(String.init) ?? "none"
                Swift.print(
                    "  \(comparison.lhsMode) vs \(comparison.rhsMode): first divergence=\(divergence), max |logit delta|=\(String(format: "%.4f", comparison.largestMaxAbsoluteLogitDelta))"
                )
            }
        }
        Swift.print("Artifact: \(outputPath)")
        Swift.print(String(repeating: "=", count: 72))
        Swift.print("")
    }
}
