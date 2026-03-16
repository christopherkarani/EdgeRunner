import Foundation

/// GPT-2 multi-head attention with combined QKV projection.
public struct GPT2Attention: EdgeRunnerModule, Sendable {
    public typealias Input = AttentionInput
    public typealias Output = [Float]

    private let config: GPT2Config
    private let cAttn: LinearModule
    private let cProj: LinearModule

    public init(config: GPT2Config) throws {
        self.config = config
        let hiddenDim = config.hiddenDim
        let attentionScale = sqrt(2.0 / Float(hiddenDim + 3 * hiddenDim))
        let projectionScale = sqrt(2.0 / Float(hiddenDim + hiddenDim))

        self.cAttn = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: 3 * hiddenDim,
            weight: (0..<(3 * hiddenDim * hiddenDim)).map { _ in
                Float.random(in: -attentionScale...attentionScale)
            },
            bias: [Float](repeating: 0, count: 3 * hiddenDim)
        )
        self.cProj = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: hiddenDim,
            weight: (0..<(hiddenDim * hiddenDim)).map { _ in
                Float.random(in: -projectionScale...projectionScale)
            },
            bias: [Float](repeating: 0, count: hiddenDim)
        )
    }

    public init(
        config: GPT2Config,
        cAttnWeight: [Float],
        cAttnBias: [Float],
        cProjWeight: [Float],
        cProjBias: [Float]
    ) throws {
        self.config = config
        let hiddenDim = config.hiddenDim
        self.cAttn = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: 3 * hiddenDim,
            weight: cAttnWeight,
            bias: cAttnBias
        )
        self.cProj = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: hiddenDim,
            weight: cProjWeight,
            bias: cProjBias
        )
    }

    public func forward(_ input: AttentionInput) async throws -> [Float] {
        let sequenceLength = input.seqLen
        let hiddenDim = config.hiddenDim
        let headDim = config.headDim
        let numHeads = config.numHeads
        let scale = 1.0 / sqrt(Float(headDim))

        var allQKV: [[Float]] = []
        allQKV.reserveCapacity(sequenceLength)
        for tokenIndex in 0..<sequenceLength {
            let token = Array(input.hidden[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let qkv = try await cAttn.forward(token)
            allQKV.append(qkv)
        }

        var attentionOutput = [Float](repeating: 0, count: sequenceLength * hiddenDim)
        for head in 0..<numHeads {
            let qOffset = head * headDim
            let kOffset = hiddenDim + head * headDim
            let vOffset = 2 * hiddenDim + head * headDim

            for queryIndex in 0..<sequenceLength {
                let qVector = Array(allQKV[queryIndex][qOffset..<(qOffset + headDim)])

                var scores = [Float](repeating: 0, count: sequenceLength)
                for keyIndex in 0..<sequenceLength {
                    var dot: Float = 0
                    for dimension in 0..<headDim {
                        dot += qVector[dimension] * allQKV[keyIndex][kOffset + dimension]
                    }
                    scores[keyIndex] = dot * scale
                }

                for keyIndex in 0..<sequenceLength where keyIndex > queryIndex {
                    scores[keyIndex] = -.infinity
                }

                let maxScore = scores.max() ?? 0
                let exponentials = scores.map { exp($0 - maxScore) }
                let sumExp = exponentials.reduce(0, +)
                let probabilities = exponentials.map { $0 / sumExp }

                let outputOffset = queryIndex * hiddenDim + head * headDim
                for valueIndex in 0..<sequenceLength {
                    for dimension in 0..<headDim {
                        attentionOutput[outputOffset + dimension] +=
                            probabilities[valueIndex] * allQKV[valueIndex][vOffset + dimension]
                    }
                }
            }
        }

        var result: [Float] = []
        result.reserveCapacity(sequenceLength * hiddenDim)
        for tokenIndex in 0..<sequenceLength {
            let token = Array(attentionOutput[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let projected = try await cProj.forward(token)
            result.append(contentsOf: projected)
        }
        return result
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]
        for (key, value) in cAttn.parameters {
            parameters["c_attn.\(key)"] = value
        }
        for (key, value) in cProj.parameters {
            parameters["c_proj.\(key)"] = value
        }
        return parameters
    }
}
