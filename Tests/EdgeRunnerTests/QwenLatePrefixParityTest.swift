import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen Late Prefix Parity")
struct QwenLatePrefixParityTest {
    private static let runEnvKey = "EDGERUNNER_RUN_LATE_PREFIX_PARITY"
    private static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
    private static let prefix = [9707, 21806, 11, 358, 2776, 14589, 369, 279]
    private static let topK = 8

    private struct PathReport: Codable {
        let label: String
        let nextToken: Int
        let topTokens: [Int]
        let topLogits: [Float]
    }

    @Test
    func latePrefixPrefillVsIncrementalDecode() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let baseConfiguration = ModelConfiguration(contextWindowSize: 512)

        let prefillModel = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: baseConfiguration
        )
        let prefillLogits = try await prefillModel.logits(for: Self.prefix)
        let incrementalModel = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: baseConfiguration
        )
        for end in 1..<Self.prefix.count {
            _ = try await incrementalModel.logits(for: Array(Self.prefix[0..<end]))
        }
        let incrementalLogits = try await incrementalModel.logits(for: Self.prefix)

        var baseDecodeConfiguration = ModelConfiguration(contextWindowSize: 512)
        baseDecodeConfiguration.llamaDecodeOverrides = LlamaDecodeOverrides(forceBaseDecodePath: true)
        let baseDecodeModel = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: baseDecodeConfiguration
        )
        for end in 1..<Self.prefix.count {
            _ = try await baseDecodeModel.logits(for: Array(Self.prefix[0..<end]))
        }
        let baseDecodeLogits = try await baseDecodeModel.logits(for: Self.prefix)

        let reports = [
            Self.makeReport(label: "prefill", logits: prefillLogits),
            Self.makeReport(label: "incremental-default", logits: incrementalLogits),
            Self.makeReport(label: "incremental-base-decode", logits: baseDecodeLogits),
        ]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(reports), as: UTF8.self))
    }

    private static func makeReport(label: String, logits: [Float]) -> PathReport {
        let ranked = logits.enumerated().sorted { lhs, rhs in
            if lhs.element == rhs.element {
                return lhs.offset < rhs.offset
            }
            return lhs.element > rhs.element
        }
        let top = Array(ranked.prefix(Self.topK))
        return PathReport(
            label: label,
            nextToken: top[0].offset,
            topTokens: top.map(\.offset),
            topLogits: top.map(\.element)
        )
    }
}
