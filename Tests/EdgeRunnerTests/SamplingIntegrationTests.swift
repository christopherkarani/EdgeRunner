import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

@Suite("Sampling Integration")
struct SamplingIntegrationTests {
    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    private func shouldRun() -> Bool {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return false
        }
        return true
    }

    @Test func greedyProducesDeterministicOutput() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("Hello")
        let greedy = SamplingConfiguration(temperature: 0)

        var run1 = prompt
        var run2 = prompt
        for _ in 0..<10 {
            run1.append(try await model.nextToken(for: run1, sampling: greedy))
            run2.append(try await model.nextToken(for: run2, sampling: greedy))
        }
        #expect(run1 == run2, "Greedy sampling should be deterministic")
    }

    @Test func temperatureSamplingProducesVariedOutput() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("Once upon")
        let sampling = SamplingConfiguration(temperature: 1.0, topK: 50, topP: 0.95)

        var outputs = Set<[Int]>()
        for _ in 0..<3 {
            var tokens = prompt
            for _ in 0..<5 {
                tokens.append(try await model.nextToken(for: tokens, sampling: sampling))
            }
            outputs.insert(Array(tokens.dropFirst(prompt.count)))
        }
        #expect(outputs.count > 1, "Temperature sampling should produce varied output across runs")
    }

    @Test func samplingWithDifferentTemperaturesProducesDifferentOutput() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("The capital of")
        let greedy = SamplingConfiguration(temperature: 0)
        let creative = SamplingConfiguration(temperature: 1.5, topK: 50, topP: 0.95)

        var greedyTokens = prompt
        var creativeTokens = prompt
        for _ in 0..<10 {
            greedyTokens.append(try await model.nextToken(for: greedyTokens, sampling: greedy))
            creativeTokens.append(try await model.nextToken(for: creativeTokens, sampling: creative))
        }
        // High temperature should diverge from greedy at some point
        let greedyGen = Array(greedyTokens.dropFirst(prompt.count))
        let creativeGen = Array(creativeTokens.dropFirst(prompt.count))
        #expect(greedyGen != creativeGen, "Different temperatures should produce different output")
    }

    @Test func coherentStoryWithTemperature() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let prompt = "<|im_start|>user\nWrite a short story about a cat who discovers snow for the first time.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        var tokenIDs = model.tokenize(prompt)
        let promptLen = tokenIDs.count
        let sampling = SamplingConfiguration(temperature: 0.7, topK: 40, topP: 0.9, seed: 123)

        let start = ContinuousClock.now
        for _ in 0..<200 {
            let next = try await model.nextToken(for: tokenIDs, sampling: sampling)
            tokenIDs.append(next)
            if next == model.eosTokenID { break }
        }
        let elapsed = ContinuousClock.now - start
        let generatedCount = tokenIDs.count - promptLen
        let ms = Double(elapsed.components.attoseconds) / 1e15
        let tokPerSec = Double(generatedCount) / (ms / 1000.0)

        let text = model.detokenize(Array(tokenIDs.dropFirst(promptLen)))

        print("\n" + String(repeating: "=", count: 60))
        print("  COHERENT STORY WITH TEMPERATURE SAMPLING")
        print("  temp=0.7, topK=40, topP=0.9, seed=123")
        print(String(repeating: "=", count: 60))
        print("  Tokens: \(generatedCount), Speed: \(String(format: "%.1f", tokPerSec)) tok/s")
        print(String(repeating: "-", count: 60))
        print(text)
        print(String(repeating: "=", count: 60))

        #expect(generatedCount > 30, "Should generate substantial output")
        #expect(!text.isEmpty, "Output should not be empty")
    }
}
