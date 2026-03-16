import Foundation

/// Perplexity computation utilities for language model evaluation.
public enum Perplexity: Sendable {
    /// Compute negative log-likelihood for a single token prediction.
    public static func negLogLikelihood(logits: [Float], targetId: Int) -> Float {
        precondition(
            targetId >= 0 && targetId < logits.count,
            "targetId \(targetId) out of range [0, \(logits.count))"
        )

        let maxLogit = logits.max() ?? 0
        let shiftedLogits = logits.map { $0 - maxLogit }
        let logSumExp = log(shiftedLogits.map { exp($0) }.reduce(0, +))
        let logProbability = shiftedLogits[targetId] - logSumExp
        return -logProbability
    }

    /// Compute perplexity over a sequence of predictions.
    public static func compute(logitsPerToken: [[Float]], targetIds: [Int]) -> Float {
        precondition(
            logitsPerToken.count == targetIds.count,
            "Logits count \(logitsPerToken.count) != targets count \(targetIds.count)"
        )
        precondition(!targetIds.isEmpty, "Cannot compute perplexity on empty sequence")

        var totalNegativeLogLikelihood: Float = 0
        for (logits, targetID) in zip(logitsPerToken, targetIds) {
            totalNegativeLogLikelihood += negLogLikelihood(logits: logits, targetId: targetID)
        }

        let averageNegativeLogLikelihood = totalNegativeLogLikelihood / Float(targetIds.count)
        return exp(averageNegativeLogLikelihood)
    }
}
