import Foundation

public struct RepetitionPenalty: Sendable {
    public let penalty: Float
    public let frequencyPenalty: Float
    public init(penalty: Float = 1.0, frequencyPenalty: Float = 0.0) {
        precondition(penalty >= 1.0, "Repetition penalty must be >= 1.0")
        precondition(frequencyPenalty >= 0, "Frequency penalty must be non-negative")
        self.penalty = penalty
        self.frequencyPenalty = frequencyPenalty
    }
    public func apply(logits: [Float], previousTokens: [Int]) -> [Float] {
        guard !previousTokens.isEmpty else { return logits }
        var counts = [Int: Int]()
        for token in previousTokens { counts[token, default: 0] += 1 }
        var result = logits
        for (tokenID, count) in counts {
            guard tokenID >= 0, tokenID < result.count else { continue }
            if penalty != 1.0 {
                if result[tokenID] > 0 { result[tokenID] /= penalty }
                else { result[tokenID] *= penalty }
            }
            if frequencyPenalty > 0 {
                result[tokenID] -= frequencyPenalty * Float(count)
            }
        }
        return result
    }
}
