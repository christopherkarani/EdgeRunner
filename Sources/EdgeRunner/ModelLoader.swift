import Foundation
import EdgeRunnerCore
import EdgeRunnerIO

/// Loads GGUF models with automatic architecture detection.
///
/// Reads `general.architecture` from GGUF metadata and dispatches
/// to the appropriate model implementation.
///
/// ```swift
/// let model = try await ModelLoader.load(
///     from: modelURL,
///     configuration: ModelConfiguration()
/// )
/// // Automatically detects Llama, Qwen, Gemma, etc.
/// ```
public enum ModelLoader: Sendable {
    /// Architectures that use the standard Llama transformer pattern
    /// (RMSNorm + RoPE + GQA + SwiGLU FFN).
    private static let llamaCompatibleArchitectures: Set<String> = [
        "llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3",
        "phi3", "mistral", "starcoder", "starcoder2",
        "internlm2", "yi", "deepseek", "deepseek2",
        "command-r", "falcon",
    ]

    /// Load a model from a GGUF file with automatic architecture detection.
    ///
    /// - Parameters:
    ///   - url: Path to the GGUF model file.
    ///   - configuration: Model configuration options.
    /// - Returns: A loaded model ready for inference.
    /// - Throws: `GenerationError.modelLoadFailed` if the architecture is unsupported.
    public static func load(
        from url: URL,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws -> any EdgeRunnerLanguageModel {
        let loader = try GGUFLoader(url: url)
        let architecture = loader.modelConfig.architectureName.lowercased()

        if llamaCompatibleArchitectures.contains(architecture) {
            return try await LlamaLanguageModel.load(from: url, configuration: configuration)
        }

        throw GenerationError.modelLoadFailed(
            reason: "Unsupported model architecture: '\(architecture)'. "
                + "Supported: \(llamaCompatibleArchitectures.sorted().joined(separator: ", "))"
        )
    }
}
