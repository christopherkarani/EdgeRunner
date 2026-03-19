import Foundation
import Testing
@testable import EdgeRunner

@Suite("Qwen Quality Comparison")
struct QwenQualityComparisonTest {
    struct ModelCase: Sendable {
        let label: String
        let path: String
    }

    struct StorySample: Sendable {
        let model: ModelCase
        let generatedTokenCount: Int
        let generatedWordCount: Int
        let loadSeconds: Double
        let decodeSeconds: Double
        let text: String
    }

    static let runEnvKey = "EDGERUNNER_RUN_QUALITY_COMPARISON"
    static let maxTokensEnvKey = "EDGERUNNER_QUALITY_MAX_TOKENS"
    static let modelFilterEnvKey = "EDGERUNNER_QUALITY_MODEL_FILTER"
    static let recoveryCheckEnvKey = "EDGERUNNER_RUN_4B_RECOVERY_CHECK"
    static let defaultGenerateCount = 384
    static let modelCases = [
        ModelCase(label: "Qwen3 0.6B Q8_0", path: "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"),
        ModelCase(label: "Qwen3 1.7B Q8_0", path: "/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf"),
        ModelCase(label: "Qwen3 4B Q8_0", path: "/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf"),
    ]

    static let storyPromptText = """
    Write a literary short story about a lighthouse keeper who discovers the sea can remember names.

    Story:
    """

    // Pre-tokenized with `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`
    static let storyPrompt = [
        7985, 264, 31365, 2805, 3364, 911, 264, 326, 57909, 53416,
        879, 51014, 279, 9396, 646, 6099, 5036, 382, 17938, 510,
    ]

    @Test
    func storyComparison() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            Swift.print("SKIP: Set \(Self.runEnvKey)=1 to run the manual long-form comparison harness.")
            return
        }

        let generateCount = Int(ProcessInfo.processInfo.environment[Self.maxTokensEnvKey] ?? "") ?? Self.defaultGenerateCount
        let requestedModelFilter = ProcessInfo.processInfo.environment[Self.modelFilterEnvKey]
        let selectedModels = Self.modelCases(matching: requestedModelFilter)
        #expect(!selectedModels.isEmpty, "No model matches \(requestedModelFilter ?? "<nil>")")
        Swift.print("")
        Swift.print(String(repeating: "=", count: 72))
        Swift.print("  QWEN3 LONG-FORM QUALITY COMPARISON")
        Swift.print(String(repeating: "=", count: 72))
        Swift.print("  Prompt: \(Self.storyPromptText.debugDescription)")
        Swift.print("  Prompt token count: \(Self.storyPrompt.count)")
        Swift.print("  Max generated tokens per model: \(generateCount)")
        if let requestedModelFilter, !requestedModelFilter.isEmpty {
            Swift.print("  Model filter: \(requestedModelFilter)")
        }
        Swift.print(String(repeating: "=", count: 72))
        Swift.print("")

        for modelCase in selectedModels {
            #expect(FileManager.default.fileExists(atPath: modelCase.path), "Missing model file at \(modelCase.path)")
            let sample = try await Self.generateStory(for: modelCase, generateCount: generateCount)
            Self.printSample(sample)
        }
    }

    @Test
    func recovered4BStoryPrefix() async throws {
        guard ProcessInfo.processInfo.environment[Self.recoveryCheckEnvKey] == "1" else {
            Swift.print("SKIP: Set \(Self.recoveryCheckEnvKey)=1 to run the recovered 4B story regression.")
            return
        }

        let modelCase = try #require(Self.modelCases.first(where: { $0.label.contains("4B") }))
        let sample = try await Self.generateStory(for: modelCase, generateCount: 64)

        #expect(
            sample.text.hasPrefix("The lighthouse keeper, Elias, had been alone for years."),
            "Recovered 4B path should start with the coherent story prefix"
        )
    }

    private static func modelCases(matching filter: String?) -> [ModelCase] {
        guard let filter, !filter.isEmpty else { return Self.modelCases }
        let normalized = filter.lowercased()
        return Self.modelCases.filter { modelCase in
            modelCase.label.lowercased().contains(normalized) || modelCase.path.lowercased().contains(normalized)
        }
    }

    private static func generateStory(
        for modelCase: ModelCase,
        generateCount: Int
    ) async throws -> StorySample {
        let modelURL = URL(fileURLWithPath: modelCase.path)
        let vocab = try CoherenceTest.loadVocabulary(from: modelCase.path)
        #expect(vocab.count == 151936, "Unexpected vocabulary size for \(modelCase.label)")

        let loadClock = ContinuousClock()
        let loadStart = loadClock.now
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )
        let loadElapsed = loadStart.duration(to: loadClock.now)

        var tokenIDs = Self.storyPrompt
        let decodeClock = ContinuousClock()
        let decodeStart = decodeClock.now
        for _ in 0..<generateCount {
            let logits = try await model.logits(for: tokenIDs)
            #expect(!logits.contains(where: { !$0.isFinite }), "NaN/Inf logits while decoding \(modelCase.label)")

            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (index, value) in logits.enumerated() {
                if value > maxVal {
                    maxVal = value
                    maxIdx = index
                }
            }

            tokenIDs.append(maxIdx)
            if maxIdx == model.eosTokenID {
                break
            }
        }
        let decodeElapsed = decodeStart.duration(to: decodeClock.now)

        let generatedIDs = Array(tokenIDs.dropFirst(Self.storyPrompt.count))
        let generatedText = CoherenceTest.detokenize(generatedIDs, vocabulary: vocab)
        let generatedWordCount = generatedText.split(whereSeparator: \.isWhitespace).count

        return StorySample(
            model: modelCase,
            generatedTokenCount: generatedIDs.count,
            generatedWordCount: generatedWordCount,
            loadSeconds: Self.seconds(loadElapsed),
            decodeSeconds: Self.seconds(decodeElapsed),
            text: generatedText
        )
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }

    private static func printSample(_ sample: StorySample) {
        let tokPerSec = sample.decodeSeconds > 0 ? Double(sample.generatedTokenCount) / sample.decodeSeconds : 0
        Swift.print(String(repeating: "-", count: 72))
        Swift.print("  MODEL: \(sample.model.label)")
        Swift.print("  PATH:  \(sample.model.path)")
        Swift.print("  LOAD:  \(String(format: "%.2f", sample.loadSeconds)) s")
        Swift.print("  DECODE: \(String(format: "%.2f", sample.decodeSeconds)) s")
        Swift.print("  THROUGHPUT: \(String(format: "%.1f", tokPerSec)) tok/s")
        Swift.print("  GENERATED TOKENS: \(sample.generatedTokenCount)")
        Swift.print("  GENERATED WORDS:  \(sample.generatedWordCount)")
        Swift.print(String(repeating: "-", count: 72))
        Swift.print(sample.text)
        Swift.print("")
    }
}
