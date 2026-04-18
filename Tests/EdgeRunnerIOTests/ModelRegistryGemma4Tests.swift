import Foundation
import Testing
@testable import EdgeRunnerIO

@Suite("ModelRegistry Gemma 4")
struct ModelRegistryGemma4Tests: Sendable {
    @Test("Default registry resolves gemma4 architecture to Gemma4ArchitectureFactory")
    func defaultRegistryHandlesGemma4() {
        let registry = ModelRegistry.default
        let factory = registry.factory(for: "gemma4")
        #expect(factory != nil)
        #expect(factory is Gemma4ArchitectureFactory)
        #expect(factory?.architectureName == "gemma4")
    }

    @Test("Default registry continues to resolve llama alongside gemma4")
    func defaultRegistryStillHandlesLlama() {
        let registry = ModelRegistry.default
        #expect(registry.factory(for: "llama") is LlamaArchitectureFactory)
        #expect(registry.factory(for: "gemma4") is Gemma4ArchitectureFactory)
    }

    @Test("Gemma4 factory create throws not-yet-implemented until forward pass lands")
    func createThrowsNotYetImplemented() {
        let factory = Gemma4ArchitectureFactory()
        let config = ModelConfig(architectureName: "gemma4", metadata: [:])
        #expect(throws: ModelLoadError.notYetImplemented(architecture: "gemma4")) {
            _ = try factory.create(config: config)
        }
    }
}
