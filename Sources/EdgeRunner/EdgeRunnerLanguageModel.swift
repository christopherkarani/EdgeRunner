import Foundation
import EdgeRunnerCore

/// A protocol that defines the interface for language models in EdgeRunner.
///
/// Conforming types provide methods for loading models, tokenizing text,
/// generating tokens, and streaming responses. All implementations are
/// thread-safe (`Sendable`) and designed for concurrent usage.
///
/// ## Example Usage
///
/// ```swift
/// let model = try await LlamaLanguageModel.load(
///     from: modelURL,
///     configuration: ModelConfiguration()
/// )
///
/// // Simple generation
/// var tokens = model.tokenize("Hello, world!")
/// for _ in 0..<50 {
///     let next = try await model.nextToken(for: tokens, sampling: SamplingConfiguration())
///     tokens.append(next)
/// }
/// let text = model.detokenize(tokens)
/// ```
public protocol EdgeRunnerLanguageModel: Sendable {
    /// A unique identifier for this model type (e.g., "llama", "gpt2").
    static var modelIdentifier: String { get }
    
    /// Loads a model from a file URL.
    ///
    /// - Parameters:
    ///   - url: The file URL pointing to the model file (e.g., `.gguf` format).
    ///   - configuration: Configuration options for model loading and behavior.
    /// - Returns: A loaded model instance ready for inference.
    /// - Throws: `GenerationError` if loading fails.
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self
    
    /// Converts text into an array of token IDs.
    ///
    /// - Parameter text: The input text to tokenize.
    /// - Returns: An array of integer token IDs.
    func tokenize(_ text: String) -> [Int]
    
    /// Converts an array of token IDs back into text.
    ///
    /// - Parameter ids: The token IDs to convert.
    /// - Returns: The decoded text string.
    func detokenize(_ ids: [Int]) -> String
    
    /// The end-of-sequence token ID.
    var eosTokenID: Int { get }
    
    /// The beginning-of-sequence token ID, if applicable.
    var bosTokenID: Int? { get }
    
    /// The total number of tokens in the model's vocabulary.
    var vocabularySize: Int { get }

    /// Applies a chat template to format messages into a prompt string.
    ///
    /// Implementations should use the model's native chat template format
    /// (e.g., ChatML, Llama-style) to produce a formatted prompt.
    ///
    /// - Parameters:
    ///   - messages: The chat messages to format.
    ///   - addGenerationPrompt: Whether to append the generation prompt suffix
    ///     (e.g., `<|im_start|>assistant\n`). Defaults to `true`.
    /// - Returns: The formatted prompt string, or `nil` if chat templates
    ///   are not supported by this model.
    func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool
    ) -> String?

    /// Generates the next token given a sequence of token IDs.
    ///
    /// - Parameters:
    ///   - tokenIDs: The input sequence of token IDs.
    ///   - sampling: Configuration for sampling strategy (temperature, top-p, etc.).
    /// - Returns: The ID of the generated next token.
    /// - Throws: `GenerationError` if generation fails.
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
    
    /// Streams generated text as an asynchronous sequence of string chunks.
    ///
    /// - Parameter prompt: The input prompt text.
    /// - Returns: An `AsyncThrowingStream` that yields generated text chunks.
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}

/// Sub-protocol for local Metal-accelerated models that expose raw logits.
/// Foundation Models backends do NOT conform to this.
public protocol LogitsModel: EdgeRunnerLanguageModel {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}

// Default implementation: LogitsModel gets nextToken via logits + sampling pipeline
extension LogitsModel {
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let logitsArray = try await logits(for: tokenIDs)
        let pipeline = sampling.toPipeline()
        return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
    }
}

// Default applyChatTemplate implementation — returns nil (no template support)
extension EdgeRunnerLanguageModel {
    public func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String? { nil }
}

// Default stream implementation — satisfies protocol requirement
extension EdgeRunnerLanguageModel {
    public func stream(_ prompt: String) -> AsyncThrowingStream<String, Error> {
        stream(prompt, sampling: SamplingConfiguration())
    }

    /// Streams generated text using the specified sampling configuration.
    ///
    /// - Parameters:
    ///   - prompt: The input prompt text.
    ///   - sampling: Configuration for sampling strategy (temperature, top-p, etc.).
    /// - Returns: An `AsyncThrowingStream` that yields generated text chunks.
    public func stream(
        _ prompt: String,
        sampling: SamplingConfiguration
    ) -> AsyncThrowingStream<String, Error> {
        let model = self
        let sampling = sampling
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    for _ in 0..<2048 {
                        try Task.checkCancellation()
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: sampling
                        )
                        if tokenID == model.eosTokenID { break }
                        tokenIDs.append(tokenID)
                        let text = model.detokenize([tokenID])
                        continuation.yield(text)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: GenerationError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
