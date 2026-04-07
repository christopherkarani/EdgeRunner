import Foundation
import EdgeRunnerCore
import EdgeRunnerIO
import EdgeRunnerMetal

/// A simple entry point for on-device LLM inference.
///
/// ```swift
/// let runner = try await EdgeRunner(modelPath: "Qwen3-0.6B-Q8_0.gguf")
///
/// for try await token in runner.stream("Hello") {
///     print(token, terminator: "")
/// }
///
/// // Or one-shot
/// let text = try await runner.generate("Hello", maxTokens: 100)
/// ```
public actor EdgeRunner {
    private let model: any EdgeRunnerLanguageModel

    /// Load a model from a file path.
    ///
    /// - Parameters:
    ///   - modelPath: Path to a GGUF model file.
    ///   - configuration: Optional model configuration.
    /// - Throws: `GenerationError.modelLoadFailed` if the model can't be loaded.
    public init(
        modelPath: String,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws {
        self.model = try await ModelLoader.load(
            from: URL(fileURLWithPath: modelPath),
            configuration: configuration
        )
    }

    /// Load a model from a URL.
    ///
    /// - Parameters:
    ///   - url: URL to a GGUF model file.
    ///   - configuration: Optional model configuration.
    /// - Throws: `GenerationError.modelLoadFailed` if the model can't be loaded.
    public init(
        from url: URL,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws {
        self.model = try await ModelLoader.load(from: url, configuration: configuration)
    }

    /// Stream generated text token by token.
    ///
    /// ```swift
    /// for try await token in runner.stream("Once upon a time") {
    ///     print(token, terminator: "")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - prompt: The input text to generate from.
    ///   - maxTokens: Maximum tokens to generate (default: 2048).
    ///   - sampling: Sampling configuration. Temperature 0 = greedy.
    /// - Returns: An `AsyncThrowingStream` of generated text chunks.
    public func stream(
        _ prompt: String,
        maxTokens: Int = 2048,
        sampling: SamplingConfiguration = SamplingConfiguration()
    ) -> AsyncThrowingStream<String, Error> {
        let model = self.model
        let sampling = sampling
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    for _ in 0..<maxTokens {
                        try Task.checkCancellation()
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: sampling
                        )
                        if tokenID == model.eosTokenID { break }
                        tokenIDs.append(tokenID)
                        continuation.yield(model.detokenize([tokenID]))
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

    /// Generate a complete response in one call.
    ///
    /// ```swift
    /// let text = try await runner.generate("What is Swift?", maxTokens: 100)
    /// ```
    ///
    /// - Parameters:
    ///   - prompt: The input text to generate from.
    ///   - maxTokens: Maximum tokens to generate (default: 2048).
    ///   - sampling: Sampling configuration. Temperature 0 = greedy.
    /// - Returns: The full generated text.
    public func generate(
        _ prompt: String,
        maxTokens: Int = 2048,
        sampling: SamplingConfiguration = SamplingConfiguration()
    ) async throws -> String {
        var result = ""
        for try await token in stream(prompt, maxTokens: maxTokens, sampling: sampling) {
            result += token
        }
        return result
    }
}
