import Foundation
import Testing
@testable import EdgeRunner

@Suite("TurboQuant Quality Harness")
struct TurboQuantQualityHarnessTest {
    private static let runEnvKey = "EDGERUNNER_RUN_TURBOQUANT_QUALITY"
    private static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    private static let presetEnvKey = "EDGERUNNER_TURBOQUANT_QUALITY_PRESET"

    @Test
    func compareAggressiveTurboQuantAgainstFP16GreedyTrace() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_QUALITY_PROMPT_LEN"] ?? "4096") ?? 4096
        let decodeCount = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_QUALITY_DECODE_TOKENS"] ?? "8") ?? 8
        let presetName = ProcessInfo.processInfo.environment[Self.presetEnvKey] ?? "aggressive"
        guard presetName == "balanced" || presetName == "aggressive" else {
            Issue.record("Unknown TurboQuant quality preset: \(presetName)")
            return
        }
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let prompt = Array(repeating: 9707, count: promptLength)

        let fp16Trace = try await runTrace(
            modelURL: modelURL,
            prompt: prompt,
            decodeCount: decodeCount,
            compression: .disabled
        )
        let turboTrace = try await runTrace(
            modelURL: modelURL,
            prompt: prompt,
            decodeCount: decodeCount,
            compression: .turboquantV2
        )

        let report = Self.compare(lhs: fp16Trace, rhs: turboTrace)
        print("""
        [turboquant-quality]
          prompt_len=\(promptLength)
          decode_tokens=\(decodeCount)
          preset=\(presetName)
          first_divergence_step=\(report.firstArgmaxDivergenceStep.map(String.init) ?? "none")
          max_abs_logit_delta=\(String(format: "%.4f", report.largestMaxAbsoluteLogitDelta))
          fp16_generated=\(fp16Trace.steps.map { $0.argmaxTokenID })
          turboquant_generated=\(turboTrace.steps.map { $0.argmaxTokenID })
        """)

        #expect(fp16Trace.steps.count == decodeCount)
        #expect(turboTrace.steps.count == decodeCount)
        #expect(!fp16Trace.steps.contains { $0.logits.contains(where: { !$0.isFinite }) })
        #expect(!turboTrace.steps.contains { $0.logits.contains(where: { !$0.isFinite }) })
    }

    private func runTrace(
        modelURL: URL,
        prompt: [Int],
        decodeCount: Int,
        compression: KVCacheCompression
    ) async throws -> ModeTrace {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(prompt.count + decodeCount + 16, 8192),
                kvCacheCompression: compression
            )
        )

        var tokenIDs = prompt
        var steps: [TraceStep] = []
        for _ in 0..<decodeCount {
            let logits = try await model.logits(for: tokenIDs)
            var maxValue: Float = -.infinity
            var maxIndex = 0
            for (index, value) in logits.enumerated() where value > maxValue {
                maxValue = value
                maxIndex = index
            }
            steps.append(TraceStep(argmaxTokenID: maxIndex, logits: logits))
            tokenIDs.append(maxIndex)
        }

        let mode = compression == .disabled ? "fp16" : "turboquant_aggressive"
        return ModeTrace(mode: mode, steps: steps)
    }

    private static func compare(lhs: ModeTrace, rhs: ModeTrace) -> ComparisonReport {
        let stepCount = min(lhs.steps.count, rhs.steps.count)
        var firstArgmaxDivergenceStep: Int?
        var largestDelta: Float = 0

        for stepIndex in 0..<stepCount {
            let lhsStep = lhs.steps[stepIndex]
            let rhsStep = rhs.steps[stepIndex]
            let maxAbsoluteLogitDelta = zip(lhsStep.logits, rhsStep.logits).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            largestDelta = max(largestDelta, maxAbsoluteLogitDelta)
            if lhsStep.argmaxTokenID != rhsStep.argmaxTokenID, firstArgmaxDivergenceStep == nil {
                firstArgmaxDivergenceStep = stepIndex
            }
        }

        return ComparisonReport(
            firstArgmaxDivergenceStep: firstArgmaxDivergenceStep,
            largestMaxAbsoluteLogitDelta: largestDelta
        )
    }
}

private struct ModeTrace {
    let mode: String
    let steps: [TraceStep]
}

private struct TraceStep {
    let argmaxTokenID: Int
    let logits: [Float]
}

private struct ComparisonReport {
    let firstArgmaxDivergenceStep: Int?
    let largestMaxAbsoluteLogitDelta: Float
}
