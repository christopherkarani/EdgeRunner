import Foundation

public struct TopKSampler: LogitsTransform, Sendable {
    public let k: Int
    public init(k: Int) {
        precondition(k >= 1, "k must be at least 1")
        self.k = k
    }
    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard k < logits.count else { return logits }
        let sorted = logits.sorted(by: >)
        let threshold = sorted[k - 1]
        var result = [Float](repeating: -.infinity, count: logits.count)
        var kept = 0
        for (i, logit) in logits.enumerated() {
            if logit >= threshold && kept < k {
                result[i] = logit
                kept += 1
            }
        }
        return result
    }
}
