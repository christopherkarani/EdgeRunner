import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

@Suite("Gemma 4 downloaded benchmark")
struct Gemma4DownloadedBenchmark {
    private static let modelPathEnvironmentKey = "EDGERUNNER_GEMMA4_BENCHMARK_MODEL"
    private static let maxTokens = 16
    private static let contextWindowSize = 4096
    private static let warmupRuns = 1
    private static let medianRuns = 5

    @Test func runDownloadedGGUFThroughEdgeRunner() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )
        let prompt = promptText(for: model)
        var tokenIDs = model.tokenize(prompt)
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        let start = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?
        var generatedTokens = 0
        var generatedTokenIDs: [Int] = []

        for _ in 0..<Self.maxTokens {
            let token: Int
            if generatedTokens == 0, let logitsModel = model as? any LogitsModel {
                let logits = try await logitsModel.logits(for: tokenIDs)
                let top = logits.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(8)
                    .map { (id: $0.offset, value: $0.element, text: model.detokenize([$0.offset])) }
                print("GEMMA4_BENCHMARK first_top_logits=\(top)")
                token = top.first?.id ?? 0
            } else {
                token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            }
            let now = ProcessInfo.processInfo.systemUptime
            if token == model.eosTokenID {
                break
            }
            firstTokenTime = firstTokenTime ?? now
            tokenIDs.append(token)
            generatedTokenIDs.append(token)
            generatedTokens += 1
        }

        let finish = ProcessInfo.processInfo.systemUptime
        let ttft = firstTokenTime.map { $0 - start } ?? 0
        let decodeDuration = firstTokenTime.map { finish - $0 } ?? 0
        let decodeTokens = max(0, generatedTokens - 1)
        let tokensPerSecond = decodeDuration > 0 ? Double(decodeTokens) / decodeDuration : 0
        let generatedText = model.detokenize(generatedTokenIDs)

        print("GEMMA4_BENCHMARK model=\(url.lastPathComponent)")
        print("GEMMA4_BENCHMARK generated_tokens=\(generatedTokens)")
        print("GEMMA4_BENCHMARK generated_token_ids=\(generatedTokenIDs)")
        print("GEMMA4_BENCHMARK generated_text=\(generatedText)")
        print("GEMMA4_BENCHMARK ttft_seconds=\(String(format: "%.6f", ttft))")
        print("GEMMA4_BENCHMARK decode_tok_s=\(String(format: "%.3f", tokensPerSecond))")

        #expect(generatedTokens > 0)
        #expect(!generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!generatedText.contains("\u{0000}"))
        #expect(Set(generatedTokenIDs).count > 1)
    }

    @Test func runDownloadedGGUFMedianBenchmark() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )

        var results: [(ttft: Double, decodeTokS: Double, generatedTokens: Int)] = []
        results.reserveCapacity(Self.medianRuns)
        for warmup in 0..<Self.warmupRuns {
            let result = try await runGeneration(model: model)
            print(
                "GEMMA4_MEDIAN_BENCHMARK warmup=\(warmup) generated_tokens=\(result.generatedTokens) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttft)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS))"
            )
        }
        for _ in 0..<Self.medianRuns {
            results.append(try await runGeneration(model: model))
        }

        for (index, result) in results.enumerated() {
            print(
                "GEMMA4_MEDIAN_BENCHMARK run=\(index) generated_tokens=\(result.generatedTokens) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttft)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS))"
            )
        }

        let decodeValues = results.map(\.decodeTokS).sorted()
        let ttftValues = results.map(\.ttft).sorted()
        let medianDecode = decodeValues[decodeValues.count / 2]
        let medianTTFT = ttftValues[ttftValues.count / 2]
        let bestDecode = decodeValues.last ?? 0
        let minDecode = decodeValues.first ?? 0

        print("GEMMA4_MEDIAN_BENCHMARK model=\(url.lastPathComponent)")
        print("GEMMA4_MEDIAN_BENCHMARK runs=\(Self.medianRuns)")
        print("GEMMA4_MEDIAN_BENCHMARK median_decode_tok_s=\(String(format: "%.3f", medianDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK best_decode_tok_s=\(String(format: "%.3f", bestDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK min_decode_tok_s=\(String(format: "%.3f", minDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK median_ttft_seconds=\(String(format: "%.6f", medianTTFT))")
    }

    private func runGeneration(
        model: any EdgeRunnerLanguageModel
    ) async throws -> (ttft: Double, decodeTokS: Double, generatedTokens: Int) {
        let prompt = promptText(for: model)
        var tokenIDs = model.tokenize(prompt)
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        let start = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?
        var generatedTokens = 0

        for _ in 0..<Self.maxTokens {
            let token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let now = ProcessInfo.processInfo.systemUptime
            if token == model.eosTokenID {
                break
            }
            firstTokenTime = firstTokenTime ?? now
            tokenIDs.append(token)
            generatedTokens += 1
        }

        let finish = ProcessInfo.processInfo.systemUptime
        let ttft = firstTokenTime.map { $0 - start } ?? 0
        let decodeDuration = firstTokenTime.map { finish - $0 } ?? 0
        let decodeTokens = max(0, generatedTokens - 1)
        let tokensPerSecond = decodeDuration > 0 ? Double(decodeTokens) / decodeDuration : 0
        return (ttft, tokensPerSecond, generatedTokens)
    }

    private func promptText(for model: any EdgeRunnerLanguageModel) -> String {
        let userPrompt = "Write one short sentence about fast local inference."
        return model.applyChatTemplate(
            messages: [EdgeRunnerCore.ChatMessage(role: "user", content: userPrompt)],
            addGenerationPrompt: true
        ) ?? userPrompt
    }
}
