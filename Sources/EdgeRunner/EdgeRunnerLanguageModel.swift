import Foundation
import EdgeRunnerCore

/// Protocol that all EdgeRunner language models must conform to.
public protocol EdgeRunnerLanguageModel: Sendable {
    static var modelIdentifier: String { get }
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self
    func tokenize(_ text: String) -> [Int]
    func detokenize(_ ids: [Int]) -> String
    var eosTokenID: Int { get }
    var bosTokenID: Int? { get }
    var vocabularySize: Int { get }
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}

/// Sub-protocol for local Metal-accelerated models that expose raw logits.
/// Foundation Models backends do NOT conform to this.
public protocol LogitsModel: EdgeRunnerLanguageModel {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}

// Default implementation: LogitsModel gets nextToken via logits + greedy sampling
extension LogitsModel {
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let logitsArray = try await logits(for: tokenIDs)
        // Greedy: pick argmax
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for (i, v) in logitsArray.enumerated() {
            if v > maxVal {
                maxVal = v
                maxIdx = i
            }
        }
        return maxIdx
    }
}

// Default stream implementation
extension EdgeRunnerLanguageModel {
    public func stream(_ prompt: String) -> AsyncThrowingStream<String, Error> {
        let model = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    for _ in 0..<2048 {
                        try Task.checkCancellation()
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: SamplingConfiguration()
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
