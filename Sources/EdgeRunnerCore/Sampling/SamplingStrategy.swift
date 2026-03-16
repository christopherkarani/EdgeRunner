import Foundation

public protocol LogitsTransform: Sendable {
    func transformLogits(_ logits: [Float]) -> [Float]
}

public protocol TokenSelector: Sendable {
    func sample(logits: [Float]) -> Int
}
