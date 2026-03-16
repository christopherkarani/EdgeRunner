import Foundation

/// Configuration for token sampling during generation.
public struct SamplingConfiguration: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float
    public var seed: UInt64?

    public init(
        temperature: Float = 1.0,
        topK: Int = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }
}
