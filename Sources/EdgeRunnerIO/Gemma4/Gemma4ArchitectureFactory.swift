import Foundation

/// Architecture factory for Google's Gemma 4 (E4B) family.
///
/// The configuration, weight-binding, chat-template, and Metal kernel
/// scaffolding are in place, but the forward pass (`Gemma4LanguageModel`)
/// has not yet been implemented, so ``create(config:)`` throws
/// ``ModelLoadError/notYetImplemented(architecture:)`` until that work
/// lands. Wiring the factory into ``ModelRegistry/default`` here lets
/// ``EdgeRunnerModel`` route GGUF files whose `general.architecture ==
/// "gemma4"` through the dedicated pipeline instead of the llama fallback.
public struct Gemma4ArchitectureFactory: ArchitectureFactory, Sendable {
    public let architectureName = "gemma4"

    public init() {}

    public func create(config: ModelConfig) throws -> any LoadableModel {
        throw ModelLoadError.notYetImplemented(architecture: "gemma4")
    }
}
