import Foundation
import EdgeRunnerCore

/// Manages a single text generation session with streaming output.
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    private let model: Model
    private let samplingPipeline: SamplingPipeline
    public let maxTokens: Int
    private let onToken: (@Sendable (Int, String) -> Void)?

    public init(
        model: Model,
        samplingPipeline: SamplingPipeline = .greedy,
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    ) {
        self.model = model
        self.samplingPipeline = samplingPipeline
        self.maxTokens = maxTokens
        self.onToken = onToken
    }

    /// Stream generated tokens one at a time.
    public func stream(prompt: String) -> AsyncThrowingStream<String, Error> {
        let model = self.model
        let maxTokens = self.maxTokens
        let onToken = self.onToken

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    if let bos = model.bosTokenID, tokenIDs.first != bos {
                        tokenIDs.insert(bos, at: 0)
                    }

                    var generatedCount = 0

                    for _ in 0..<maxTokens {
                        try Task.checkCancellation()

                        // Use nextToken — universal path for both FM and local models
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: SamplingConfiguration()
                        )

                        if tokenID == model.eosTokenID {
                            break
                        }

                        tokenIDs.append(tokenID)
                        generatedCount += 1
                        let text = model.detokenize([tokenID])
                        onToken?(tokenID, text)
                        continuation.yield(text)
                    }

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: GenerationError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Generate a complete response (non-streaming).
    public func generate(prompt: String) async throws -> String {
        var result = ""
        let stream = self.stream(prompt: prompt)
        for try await token in stream {
            result += token
        }
        return result
    }
}
