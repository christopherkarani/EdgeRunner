import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

@Suite("Generation Benchmarks")
struct GenerationBenchmarks {

    // Tiny config: GPT2's LM head is CPU-computed, so keep small for CI.
    // Real Llama benchmarks will come when LM head moves to Metal.
    static let benchConfig = GPT2Config(
        vocabSize: 64,
        maxSeqLen: 32,
        numLayers: 2,
        numHeads: 2,
        hiddenDim: 64,
        layerNormEps: 1e-5
    )

    @Test func prefillThroughput() async throws {
        let model = try GPT2Model(config: Self.benchConfig)
        let promptTokens = Array(0..<16)

        // Warmup
        _ = try await model.forward(promptTokens)

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await model.forward(promptTokens)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(promptTokens.count * iterations) / seconds
        let latencyMs = seconds / Double(iterations) * 1000

        print("BENCHMARK: prefill_throughput_tiny \(String(format: "%.0f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: prefill_latency_16tok \(String(format: "%.1f", latencyMs)) ms")
        #expect(tokensPerSec > 1)
    }

    @Test func decodeThroughput() async throws {
        let model = try GPT2Model(config: Self.benchConfig)
        let generateCount = 16
        var tokenIDs = [0]

        _ = try await model.forward(tokenIDs)

        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<generateCount {
            let logits = try await model.forward(tokenIDs)
            let vocabSlice = Array(logits.suffix(Self.benchConfig.vocabSize))
            let nextToken = vocabSlice.enumerated().max(by: { $0.element < $1.element })!.offset
            tokenIDs.append(min(nextToken, Self.benchConfig.vocabSize - 1))
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(generateCount) / seconds
        let msPerToken = seconds / Double(generateCount) * 1000

        print("BENCHMARK: decode_throughput_tiny \(String(format: "%.1f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: decode_latency_per_token \(String(format: "%.1f", msPerToken)) ms/token")
        #expect(tokensPerSec > 0.1)
    }

    @Test func timeToFirstToken() async throws {
        let model = try GPT2Model(config: Self.benchConfig)
        let promptTokens = Array(0..<8)

        _ = try await model.forward(promptTokens)

        let iterations = 5
        var ttftValues: [Double] = []

        for _ in 0..<iterations {
            let clock = ContinuousClock()
            let start = clock.now
            let logits = try await model.forward(promptTokens)
            let vocabSlice = Array(logits.suffix(Self.benchConfig.vocabSize))
            _ = vocabSlice.enumerated().max(by: { $0.element < $1.element })!.offset
            let elapsed = start.duration(to: clock.now)
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
            ttftValues.append(seconds * 1000)
        }

        let avgTTFT = ttftValues.reduce(0, +) / Double(ttftValues.count)
        print("BENCHMARK: time_to_first_token_tiny \(String(format: "%.1f", avgTTFT)) ms")
        #expect(avgTTFT < 120000)
    }

    @Test func generationSessionOverhead() async throws {
        let config = Self.benchConfig
        let model = try GPT2Model(config: config)
        let mockModel = BenchmarkMockModel(config: config, realModel: model)

        let session = GenerationSession(
            model: mockModel,
            samplingPipeline: .greedy,
            maxTokens: 8
        )

        // Warmup
        _ = try await session.generate(prompt: "")

        let iterations = 3
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await session.generate(prompt: "")
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerGeneration = seconds / Double(iterations) * 1000

        print("BENCHMARK: generation_session_8tok \(String(format: "%.1f", msPerGeneration)) ms")
        #expect(msPerGeneration < 120000)
    }
}

// Mock model wrapping GPT2Model for GenerationSession benchmarks
private struct BenchmarkMockModel: EdgeRunnerLanguageModel {
    static let modelIdentifier = "bench-gpt2"
    let config: GPT2Config
    let realModel: GPT2Model

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> BenchmarkMockModel {
        fatalError("Not used in benchmarks")
    }

    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let effective = tokenIDs.suffix(config.maxSeqLen).map { min(max($0, 0), config.vocabSize - 1) }
        guard !effective.isEmpty else { return 0 }
        let logits = try await realModel.forward(Array(effective))
        let vocabSlice = Array(logits.suffix(config.vocabSize))
        guard let best = vocabSlice.enumerated().max(by: { $0.element < $1.element }) else { return 0 }
        let token = best.offset
        return token == config.vocabSize - 1 ? 0 : token
    }

    func tokenize(_ text: String) -> [Int] { [0] }
    func detokenize(_ ids: [Int]) -> String { ids.map { "\($0)" }.joined(separator: " ") }
    var eosTokenID: Int { config.vocabSize - 1 }
    var bosTokenID: Int? { 0 }
    var vocabularySize: Int { config.vocabSize }
}
