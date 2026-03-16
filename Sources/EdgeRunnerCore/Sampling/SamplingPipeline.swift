import Foundation

/// Stochastic token selector that samples from the probability distribution.
/// Uses a class wrapper for the RNG to allow mutation through non-mutating protocol method.
public final class StochasticSampler<RNG: RandomNumberGenerator & Sendable>: TokenSelector, @unchecked Sendable {
    // @unchecked Sendable: RNG state is only mutated during sample() calls,
    // which are serialized by the caller (single-threaded generation loop).
    private var rng: RNG

    public init(randomSource: inout RNG) {
        self.rng = randomSource
    }

    public func sample(logits: [Float]) -> Int {
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        guard sumExps > 0 else {
            return logits.firstIndex(where: { $0 > -.infinity }) ?? 0
        }
        let probs = exps.map { $0 / sumExps }
        let r = Float.random(in: 0..<1, using: &rng)
        var cumulative: Float = 0
        for (i, p) in probs.enumerated() {
            cumulative += p
            if r < cumulative { return i }
        }
        return probs.count - 1
    }
}

/// Composable sampling pipeline.
public struct SamplingPipeline: Sendable {
    private let transforms: [any LogitsTransform]
    private let selector: any TokenSelector
    private let repetitionPenalty: RepetitionPenalty?

    public init(
        transforms: [any LogitsTransform],
        selector: any TokenSelector,
        repetitionPenalty: RepetitionPenalty? = nil
    ) {
        self.transforms = transforms
        self.selector = selector
        self.repetitionPenalty = repetitionPenalty
    }

    public func sample(logits: [Float], previousTokens: [Int] = []) -> Int {
        var currentLogits = logits
        if let penalty = repetitionPenalty {
            currentLogits = penalty.apply(logits: currentLogits, previousTokens: previousTokens)
        }
        for transform in transforms {
            currentLogits = transform.transformLogits(currentLogits)
        }
        return selector.sample(logits: currentLogits)
    }

    public static var greedy: SamplingPipeline {
        SamplingPipeline(transforms: [], selector: GreedySampler())
    }

    public static func nucleus(temperature: Float = 0.8, topP: Float = 0.9, seed: UInt64 = 0) -> SamplingPipeline {
        var rng = SeededRandomSource(seed: seed)
        return SamplingPipeline(
            transforms: [TemperatureSampler(temperature: temperature), TopPSampler(p: topP)],
            selector: StochasticSampler(randomSource: &rng)
        )
    }

    public static func topK(k: Int = 40, temperature: Float = 0.8, seed: UInt64 = 0) -> SamplingPipeline {
        var rng = SeededRandomSource(seed: seed)
        return SamplingPipeline(
            transforms: [TemperatureSampler(temperature: temperature), TopKSampler(k: k)],
            selector: StochasticSampler(randomSource: &rng)
        )
    }
}
