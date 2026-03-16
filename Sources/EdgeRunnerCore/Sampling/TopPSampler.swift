import Foundation

public struct TopPSampler: LogitsTransform, Sendable {
    public let p: Float
    public init(p: Float) {
        precondition(p > 0 && p <= 1.0, "p must be in (0, 1]")
        self.p = p
    }
    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard p < 1.0 else { return logits }
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }
        let sortedIndices = probs.enumerated().sorted { $0.element > $1.element }.map(\.offset)
        var cumulative: Float = 0
        var keepSet = Set<Int>()
        for index in sortedIndices {
            keepSet.insert(index)
            cumulative += probs[index]
            if cumulative >= p { break }
        }
        var result = logits
        for i in 0..<result.count {
            if !keepSet.contains(i) { result[i] = -.infinity }
        }
        return result
    }
}
