import Foundation
import EdgeRunnerCore

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

extension SamplingConfiguration {
    /// Converts this configuration into a composable `SamplingPipeline`.
    ///
    /// - Temperature <= 0 produces a greedy (argmax) pipeline.
    /// - Otherwise, builds a stochastic pipeline with the configured transforms
    ///   (temperature, top-k, top-p) and optional repetition penalty.
    public func toPipeline() -> SamplingPipeline {
        // Temperature <= 0 → greedy
        if temperature <= 0 {
            let penalty: RepetitionPenalty? = repetitionPenalty > 1.0
                ? RepetitionPenalty(penalty: repetitionPenalty)
                : nil
            return SamplingPipeline(
                transforms: [],
                selector: GreedySampler(),
                repetitionPenalty: penalty
            )
        }

        var transforms: [any LogitsTransform] = []

        if temperature != 1.0 {
            transforms.append(TemperatureSampler(temperature: temperature))
        }

        if topK > 0 {
            transforms.append(TopKSampler(k: topK))
        }

        if topP < 1.0 {
            transforms.append(TopPSampler(p: topP))
        }

        // Build selector
        let selector: any TokenSelector
        if let seed {
            var rng = SeededRandomSource(seed: seed)
            selector = StochasticSampler(randomSource: &rng)
        } else {
            var rng = SeededRandomSource(seed: UInt64.random(in: 0...UInt64.max))
            selector = StochasticSampler(randomSource: &rng)
        }

        // Repetition penalty
        let penalty: RepetitionPenalty? = repetitionPenalty > 1.0
            ? RepetitionPenalty(penalty: repetitionPenalty)
            : nil

        return SamplingPipeline(
            transforms: transforms,
            selector: selector,
            repetitionPenalty: penalty
        )
    }
}
