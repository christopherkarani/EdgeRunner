import Foundation

/// Architecture factory for Google's Gemma 4 (E4B) family.
///
/// Wave 1 and Wave 2 of the Gemma 4 roadmap delivered the configuration,
/// weight-binding, chat-template, and Metal kernel scaffolding. The forward
/// pass (`Gemma4LanguageModel`) is scheduled for Tasks 16-19 of the roadmap,
/// so ``create(config:)`` intentionally throws until that work lands. Wiring
/// the factory into ``ModelRegistry/default`` here lets ``EdgeRunnerModel``
/// route GGUF files whose `general.architecture == "gemma4"` through the
/// dedicated pipeline instead of the llama fallback.
public struct Gemma4ArchitectureFactory: ArchitectureFactory, Sendable {
    public let architectureName = "gemma4"

    public init() {}

    public func create(config: ModelConfig) throws -> any LoadableModel {
        throw ModelLoadError.loadFailed(
            description: "Gemma 4 forward pass is under construction; "
                + "Gemma4LanguageModel lands in Tasks 16-19 of the roadmap."
        )
    }
}
