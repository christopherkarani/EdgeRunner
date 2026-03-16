import Foundation

public struct MinPSampler: LogitsTransform, Sendable {
    public let minP: Float
    public init(minP: Float) {
        precondition(minP >= 0 && minP <= 1.0, "minP must be in [0, 1]")
        self.minP = minP
    }
    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard minP > 0 else { return logits }
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }
        let maxProb = probs.max() ?? 0
        let threshold = minP * maxProb
        var result = logits
        for i in 0..<result.count {
            if probs[i] < threshold { result[i] = -.infinity }
        }
        return result
    }
}
