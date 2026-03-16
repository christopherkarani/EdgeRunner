import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Mock Backends

struct MockFoundationModelsBackend: EdgeRunnerLanguageModel {
    static let modelIdentifier = "foundation-mock-v1"
    let vocabulary: [String] = ["The", " answer", " is", " 42", "<eos>"]

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockFoundationModelsBackend {
        return MockFoundationModelsBackend()
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

    var eosTokenID: Int { 4 }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocabulary.count }

    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let nextIndex = min(tokenIDs.count, vocabulary.count - 1)
        return nextIndex
    }
}

struct MockLocalBackend: LogitsModel, LocalModelBackend {
    static let modelIdentifier = "local-mock-v1"
    static let supportedFormat = "gguf"
    let vocabulary: [String] = ["Hello", " world", "!", "<eos>"]

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockLocalBackend {
        return MockLocalBackend()
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        var result = [Float](repeating: -100.0, count: vocabulary.count)
        let nextIndex = min(tokenIDs.count, vocabulary.count - 1)
        result[nextIndex] = 10.0
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

    func estimatedMemoryUsage() -> Int { 1024 * 1024 * 100 } // 100MB
}

// MARK: - BackendFactory Tests

@Suite("BackendFactory")
struct BackendFactoryTests {

    @Test func registerAndRetrieveBackend() {
        let registry = BackendRegistry()
        registry.register(MockLocalBackend.self, for: "gguf")
        let backend = registry.backend(for: "gguf")
        #expect(backend != nil)
    }

    @Test func retrieveUnregisteredBackendReturnsNil() {
        let registry = BackendRegistry()
        let backend = registry.backend(for: "nonexistent")
        #expect(backend == nil)
    }

    @Test func availableBackendsEmpty() {
        let registry = BackendRegistry()
        #expect(registry.availableBackends.isEmpty)
    }

    @Test func availableBackendsAfterRegistration() {
        let registry = BackendRegistry()
        registry.register(MockLocalBackend.self, for: "gguf")
        registry.register(MockFoundationModelsBackend.self, for: "foundation")
        #expect(registry.availableBackends.count == 2)
        #expect(registry.availableBackends.contains("gguf"))
        #expect(registry.availableBackends.contains("foundation"))
    }

    @Test func loadFromRegisteredBackend() async throws {
        let registry = BackendRegistry()
        registry.register(MockLocalBackend.self, for: "gguf")
        let url = URL(fileURLWithPath: "/tmp/model.gguf")
        let model = try await registry.load(from: url, format: "gguf")
        #expect(model.vocabularySize == 4)
    }

    @Test func loadFromUnregisteredBackendThrows() async {
        let registry = BackendRegistry()
        let url = URL(fileURLWithPath: "/tmp/model.unknown")
        await #expect(throws: GenerationError.self) {
            try await registry.load(from: url, format: "unknown")
        }
    }

    @Test func registerOverwritesExisting() {
        let registry = BackendRegistry()
        registry.register(MockLocalBackend.self, for: "gguf")
        registry.register(MockFoundationModelsBackend.self, for: "gguf")
        #expect(registry.availableBackends.count == 1)
    }

    @Test func loadWithCustomConfiguration() async throws {
        let registry = BackendRegistry()
        registry.register(MockFoundationModelsBackend.self, for: "foundation")
        let url = URL(fileURLWithPath: "/tmp/model")
        let config = ModelConfiguration(maxTokens: 512)
        let model = try await registry.load(from: url, format: "foundation", configuration: config)
        #expect(model.vocabularySize == 5)
    }
}

// MARK: - EdgeRunnerLocalBackend Tests

@Suite("EdgeRunnerLocalBackend")
struct EdgeRunnerLocalBackendTests {

    @Test func localBackendConformsToLogitsModel() async throws {
        let model = MockLocalBackend()
        let logits = try await model.logits(for: [0])
        #expect(logits.count == model.vocabularySize)
    }

    @Test func localBackendSupportedFormat() {
        #expect(MockLocalBackend.supportedFormat == "gguf")
    }

    @Test func localBackendModelIdentifier() {
        #expect(MockLocalBackend.modelIdentifier == "local-mock-v1")
    }

    @Test func localBackendEstimatedMemory() {
        let model = MockLocalBackend()
        #expect(model.estimatedMemoryUsage() == 1024 * 1024 * 100)
    }

    @Test func localBackendTokenization() {
        let model = MockLocalBackend()
        let ids = model.tokenize("Hello world!")
        let decoded = model.detokenize(ids)
        #expect(decoded == "Hello world!")
    }

    @Test func localBackendNextToken() async throws {
        let model = MockLocalBackend()
        let token = try await model.nextToken(for: [0], sampling: SamplingConfiguration())
        #expect(token == 1) // greedy picks argmax from logits
    }

    @Test func localBackendLoad() async throws {
        let url = URL(fileURLWithPath: "/tmp/model.gguf")
        let model = try await MockLocalBackend.load(from: url, configuration: ModelConfiguration())
        #expect(model.eosTokenID == 3)
    }
}

// MARK: - FoundationModelsBackend Tests

@Suite("FoundationModelsBackend")
struct FoundationModelsBackendTests {

    @Test func foundationModelsAvailability() {
        // On macOS 26+ with FoundationModels this would be true; otherwise false
        let available = FoundationModelsAvailability.isAvailable
        #expect(type(of: available) == Bool.self)
    }

    @Test func foundationModelsBackendConformance() {
        let model = MockFoundationModelsBackend()
        #expect(MockFoundationModelsBackend.modelIdentifier == "foundation-mock-v1")
        #expect(model.vocabularySize == 5)
        #expect(model.eosTokenID == 4)
    }

    @Test func foundationModelsBackendLoad() async throws {
        let url = URL(fileURLWithPath: "/tmp/model")
        let model = try await MockFoundationModelsBackend.load(from: url, configuration: ModelConfiguration())
        #expect(model.vocabularySize == 5)
    }

    @Test func foundationModelsBackendTokenization() {
        let model = MockFoundationModelsBackend()
        let ids = model.tokenize("The answer is 42")
        let decoded = model.detokenize(ids)
        #expect(decoded == "The answer is 42")
    }

    @Test func foundationModelsBackendNextToken() async throws {
        let model = MockFoundationModelsBackend()
        let token = try await model.nextToken(for: [0], sampling: SamplingConfiguration())
        #expect(token == 1)
    }

    @Test func foundationModelsNotLogitsModel() {
        // MockFoundationModelsBackend conforms to EdgeRunnerLanguageModel but NOT LogitsModel
        let model = MockFoundationModelsBackend()
        #expect(!(model is any LogitsModel))
    }
}
