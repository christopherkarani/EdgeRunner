import Foundation

public struct TemperatureSampler: LogitsTransform, Sendable {
    public let temperature: Float
    public init(temperature: Float) {
        precondition(temperature >= 0, "Temperature must be non-negative")
        self.temperature = temperature
    }
    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard temperature > 0, temperature != 1.0 else { return logits }
        return logits.map { $0 / temperature }
    }
}
