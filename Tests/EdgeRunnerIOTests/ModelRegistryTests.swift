import Foundation
import Testing
@testable import EdgeRunnerIO

struct StubModel: LoadableModel, Sendable {
    var parameterNames: [String] { ["stub.weight"] }
    var weightsLoaded = false

    mutating func loadWeights(from map: WeightMap) throws {
        guard map["stub.weight"] != nil else {
            throw ModelLoadError.loadFailed(description: "Missing stub.weight")
        }
        weightsLoaded = true
    }
}

struct StubArchitectureFactory: ArchitectureFactory, Sendable {
    let architectureName: String = "stub"

    func create(config: ModelConfig) throws -> any LoadableModel {
        StubModel()
    }
}

@Suite("Model Registry Tests")
struct ModelRegistryTests: Sendable {
    @Test("Register and retrieve architecture factory")
    func registerAndRetrieve() {
        let registry = ModelRegistry()
        registry.register(StubArchitectureFactory())

        let retrieved = registry.factory(for: "stub")
        #expect(retrieved != nil)
        #expect(retrieved?.architectureName == "stub")
    }

    @Test("Unknown architecture returns nil")
    func unknownArchitecture() {
        let registry = ModelRegistry()
        #expect(registry.factory(for: "nonexistent") == nil)
    }

    @Test("Default registry includes Llama")
    func defaultRegistryHasLlama() {
        #expect(ModelRegistry.default.factory(for: "llama") != nil)
    }

    @Test("Overwrite existing registration")
    func overwrite() {
        let registry = ModelRegistry()
        registry.register(StubArchitectureFactory())
        registry.register(StubArchitectureFactory())
        #expect(registry.factory(for: "stub") != nil)
    }
}
