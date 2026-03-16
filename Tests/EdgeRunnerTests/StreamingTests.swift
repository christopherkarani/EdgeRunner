import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

private struct StreamingMockModel: EdgeRunnerLanguageModel {
    static let modelIdentifier = "streaming-mock"
    let tokenSequence: [Int]
    let vocab: [String]

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> StreamingMockModel {
        StreamingMockModel(tokenSequence: [0, 1, 2, 3], vocab: ["Hello", " ", "world", "!"])
    }

    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let step = tokenIDs.count
        var logits = [Float](repeating: -100.0, count: vocab.count + 1)
        if step < tokenSequence.count {
            logits[tokenSequence[step]] = 10.0
        } else {
            logits[vocab.count] = 10.0
        }
        // Greedy argmax
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for (i, v) in logits.enumerated() {
            if v > maxVal { maxVal = v; maxIdx = i }
        }
        return maxIdx
    }

    func tokenize(_ text: String) -> [Int] { [] }
    func detokenize(_ ids: [Int]) -> String {
        ids.compactMap { id in id < vocab.count ? vocab[id] : nil }.joined()
    }
    var eosTokenID: Int { vocab.count }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocab.count + 1 }
}

/// Thread-safe collector for token IDs in callback tests.
/// Callbacks are called serially from the generation loop so access is safe.
private final class TokenCollector: @unchecked Sendable {
    private(set) var ids = [Int]()
    func append(_ id: Int) { ids.append(id) }
}

@Suite("TokenStream")
struct TokenStreamTests {
    @Test func basicStreaming() async throws {
        let model = StreamingMockModel(tokenSequence: [0, 1, 2, 3], vocab: ["Hello", " ", "world", "!"])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 10)
        var tokens = [String]()
        for try await token in session.stream(prompt: "") { tokens.append(token) }
        #expect(tokens == ["Hello", " ", "world", "!"])
    }

    @Test func streamStopsAtEOS() async throws {
        let model = StreamingMockModel(tokenSequence: [0, 1], vocab: ["Hi", "!"])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 100)
        var count = 0
        for try await _ in session.stream(prompt: "") { count += 1 }
        #expect(count == 2)
    }

    @Test func streamRespectsMaxTokens() async throws {
        let model = StreamingMockModel(tokenSequence: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], vocab: ["a"])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 3)
        var count = 0
        for try await _ in session.stream(prompt: "") { count += 1 }
        #expect(count == 3)
    }

    @Test func streamCancellation() async throws {
        let model = StreamingMockModel(tokenSequence: (0..<100).map { _ in 0 }, vocab: ["token"])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 100)
        var count = 0
        for try await _ in session.stream(prompt: "") {
            count += 1
            if count >= 5 { break }
        }
        #expect(count == 5)
    }

    @Test func tokenCallbackHook() async throws {
        let model = StreamingMockModel(tokenSequence: [0, 1, 2], vocab: ["a", "b", "c"])
        let collector = TokenCollector()
        let session = GenerationSession(
            model: model, samplingPipeline: .greedy, maxTokens: 10,
            onToken: { tokenID, _ in collector.append(tokenID) }
        )
        for try await _ in session.stream(prompt: "") {}
        #expect(collector.ids == [0, 1, 2])
    }

    @Test func collectFullResponse() async throws {
        let model = StreamingMockModel(tokenSequence: [0, 1, 2, 3], vocab: ["Hello", " ", "world", "!"])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 10)
        let response = try await session.generate(prompt: "")
        #expect(response == "Hello world!")
    }

    @Test func generateWithStructuredOutput() async throws {
        let jsonTokens = ["{", "\"", "n", "\"", ":", "1", "}"]
        let model = StreamingMockModel(tokenSequence: Array(0..<jsonTokens.count), vocab: jsonTokens)
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 20)
        let response = try await session.generate(prompt: "")
        #expect(response.contains("{"))
        #expect(response.contains("}"))
    }
}

@Suite("GenerationSession")
struct GenerationSessionTests {
    @Test func sessionMetadata() {
        let model = StreamingMockModel(tokenSequence: [], vocab: [])
        let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 512)
        #expect(session.maxTokens == 512)
    }

    @Test func sessionWithCustomSampling() async throws {
        let model = StreamingMockModel(tokenSequence: [0], vocab: ["test"])
        let pipeline = SamplingPipeline(transforms: [TemperatureSampler(temperature: 0.001)], selector: GreedySampler())
        let session = GenerationSession(model: model, samplingPipeline: pipeline, maxTokens: 5)
        let response = try await session.generate(prompt: "")
        #expect(response == "test")
    }
}
