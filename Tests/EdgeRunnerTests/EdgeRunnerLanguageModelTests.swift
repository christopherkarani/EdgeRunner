import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

struct MockLanguageModel: LogitsModel {
    static let modelIdentifier = "mock-test-v1"
    let vocabulary: [String] = ["Hello", " world", "!", "<eos>"]
    let fixedTokenIDs: [Int]

    init(fixedTokenIDs: [Int] = [0, 1, 2, 3]) {
        self.fixedTokenIDs = fixedTokenIDs
    }

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockLanguageModel {
        return MockLanguageModel()
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        var result = [Float](repeating: -100.0, count: vocabulary.count)
        let nextIndex = min(tokenIDs.count, fixedTokenIDs.count - 1)
        result[fixedTokenIDs[nextIndex]] = 10.0
        return result
    }

    func tokenize(_ text: String) -> [Int] {
        var ids: [Int] = []
        var remaining = text
        for (i, word) in vocabulary.enumerated() {
            while remaining.hasPrefix(word) {
                ids.append(i)
                remaining.removeFirst(word.count)
            }
        }
        return ids
    }

    func detokenize(_ ids: [Int]) -> String {
        ids.map { id in
            guard id >= 0, id < vocabulary.count else { return "" }
            return vocabulary[id]
        }.joined()
    }

    var eosTokenID: Int { 3 }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocabulary.count }
}

@Suite("EdgeRunnerLanguageModel Protocol")
struct EdgeRunnerLanguageModelProtocolTests {

    @Test func protocolConformance() async throws {
        let model = MockLanguageModel()
        #expect(MockLanguageModel.modelIdentifier == "mock-test-v1")
        #expect(model.vocabularySize == 4)
        #expect(model.eosTokenID == 3)
    }

    @Test func loadFromURL() async throws {
        let url = URL(fileURLWithPath: "/tmp/fake-model.gguf")
        let config = ModelConfiguration()
        let model = try await MockLanguageModel.load(from: url, configuration: config)
        #expect(model.vocabularySize == 4)
    }

    @Test func logitsReturnCorrectSize() async throws {
        let model = MockLanguageModel()
        let logits = try await model.logits(for: [0])
        #expect(logits.count == model.vocabularySize)
    }

    @Test func logitsHighestAtExpectedToken() async throws {
        let model = MockLanguageModel(fixedTokenIDs: [0, 1, 2, 3])
        let logits = try await model.logits(for: [0])
        let maxIndex = logits.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(maxIndex == 1)
    }

    @Test func tokenizeRoundTrip() {
        let model = MockLanguageModel()
        let text = "Hello world!"
        let ids = model.tokenize(text)
        let decoded = model.detokenize(ids)
        #expect(decoded == text)
    }

    @Test func modelConfigurationDefaults() {
        let config = ModelConfiguration()
        #expect(config.maxTokens == 2048)
        #expect(config.contextWindowSize == 4096)
    }

    @Test func modelConfigurationCustom() {
        let config = ModelConfiguration(
            maxTokens: 512,
            contextWindowSize: 8192
        )
        #expect(config.maxTokens == 512)
        #expect(config.contextWindowSize == 8192)
    }

    @Test func generationErrorDescriptions() {
        let error1 = GenerationError.modelLoadFailed(reason: "File not found")
        let error2 = GenerationError.contextWindowExceeded(requested: 5000, maximum: 4096)
        let error3 = GenerationError.cancelled
        #expect("\(error1)".contains("File not found"))
        #expect("\(error2)".contains("5000"))
        #expect("\(error3)".contains("cancelled"))
    }

    @Test func samplingConfigurationDefaults() {
        let config = SamplingConfiguration()
        #expect(config.temperature == 1.0)
        #expect(config.topK == 40)
        #expect(config.topP == 0.9)
        #expect(config.repetitionPenalty == 1.0)
        #expect(config.seed == nil)
    }

    @Test func nextTokenDefaultImplementation() async throws {
        let model = MockLanguageModel(fixedTokenIDs: [0, 1, 2, 3])
        let token = try await model.nextToken(for: [0], sampling: SamplingConfiguration())
        #expect(token == 1) // greedy picks argmax from logits
    }
}
