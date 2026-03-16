import Foundation

public struct GreedySampler: TokenSelector, Sendable {
    public init() {}
    public func sample(logits: [Float]) -> Int {
        var maxValue: Float = -.infinity
        var maxIndex = 0
        for (index, value) in logits.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }
        return maxIndex
    }
}
