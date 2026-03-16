import Foundation

/// Input to the multi-head attention module.
public struct AttentionInput: Sendable {
    public let hidden: [Float]
    public let seqLen: Int
    public let startPos: Int

    public init(hidden: [Float], seqLen: Int, startPos: Int) {
        self.hidden = hidden
        self.seqLen = seqLen
        self.startPos = startPos
    }
}

/// CPU reference multi-head attention with GQA support.
public struct MultiHeadAttention: EdgeRunnerModule, Sendable {
    public typealias Input = AttentionInput
    public typealias Output = [Float]

    private let config: TransformerConfig
    private let wq: LinearModule
    private let wk: LinearModule
    private let wv: LinearModule
    private let wo: LinearModule

    public init(config: TransformerConfig, zeroWeights: Bool = false) throws {
        self.config = config

        let hiddenDim = config.hiddenDim
        let kvDim = config.numKVHeads * config.headDim

        if zeroWeights {
            self.wq = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: hiddenDim,
                weight: [Float](repeating: 0, count: hiddenDim * hiddenDim),
                bias: nil
            )
            self.wk = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: kvDim,
                weight: [Float](repeating: 0, count: kvDim * hiddenDim),
                bias: nil
            )
            self.wv = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: kvDim,
                weight: [Float](repeating: 0, count: kvDim * hiddenDim),
                bias: nil
            )
            self.wo = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: hiddenDim,
                weight: [Float](repeating: 0, count: hiddenDim * hiddenDim),
                bias: nil
            )
        } else {
            let qScale = sqrt(2.0 / Float(hiddenDim + hiddenDim))
            let kvScale = sqrt(2.0 / Float(hiddenDim + kvDim))
            self.wq = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: hiddenDim,
                weight: (0..<(hiddenDim * hiddenDim)).map { _ in Float.random(in: -qScale...qScale) },
                bias: nil
            )
            self.wk = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: kvDim,
                weight: (0..<(kvDim * hiddenDim)).map { _ in Float.random(in: -kvScale...kvScale) },
                bias: nil
            )
            self.wv = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: kvDim,
                weight: (0..<(kvDim * hiddenDim)).map { _ in Float.random(in: -kvScale...kvScale) },
                bias: nil
            )
            self.wo = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: hiddenDim,
                weight: (0..<(hiddenDim * hiddenDim)).map { _ in Float.random(in: -qScale...qScale) },
                bias: nil
            )
        }
    }

    public func forward(_ input: AttentionInput) async throws -> [Float] {
        let sequenceLength = input.seqLen
        let hiddenDim = config.hiddenDim
        let headDim = config.headDim
        let numHeads = config.numHeads
        let numKVHeads = config.numKVHeads
        let kvGroupSize = config.kvGroupSize

        var allQ: [Float] = []
        var allK: [Float] = []
        var allV: [Float] = []
        allQ.reserveCapacity(sequenceLength * hiddenDim)
        allK.reserveCapacity(sequenceLength * numKVHeads * headDim)
        allV.reserveCapacity(sequenceLength * numKVHeads * headDim)

        for tokenIndex in 0..<sequenceLength {
            let token = Array(input.hidden[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let q = try await wq.forward(token)
            let k = try await wk.forward(token)
            let v = try await wv.forward(token)
            allQ.append(contentsOf: q)
            allK.append(contentsOf: k)
            allV.append(contentsOf: v)
        }

        var attentionOutput = [Float](repeating: 0, count: sequenceLength * hiddenDim)
        let scale = 1.0 / sqrt(Float(headDim))

        for head in 0..<numHeads {
            let kvHead = head / kvGroupSize

            for queryIndex in 0..<sequenceLength {
                let qOffset = queryIndex * hiddenDim + head * headDim
                let qVector = Array(allQ[qOffset..<(qOffset + headDim)])

                var scores = [Float](repeating: 0, count: sequenceLength)
                for keyIndex in 0..<sequenceLength {
                    let kOffset = keyIndex * (numKVHeads * headDim) + kvHead * headDim
                    var dot: Float = 0
                    for dimension in 0..<headDim {
                        dot += qVector[dimension] * allK[kOffset + dimension]
                    }
                    scores[keyIndex] = dot * scale
                }

                for keyIndex in 0..<sequenceLength where keyIndex > queryIndex {
                    scores[keyIndex] = -.infinity
                }

                let maxScore = scores.max() ?? 0
                let expScores = scores.map { exp($0 - maxScore) }
                let sumExp = expScores.reduce(0, +)
                let probabilities = expScores.map { $0 / sumExp }

                let outputOffset = queryIndex * hiddenDim + head * headDim
                for valueIndex in 0..<sequenceLength {
                    let vOffset = valueIndex * (numKVHeads * headDim) + kvHead * headDim
                    for dimension in 0..<headDim {
                        attentionOutput[outputOffset + dimension] +=
                            probabilities[valueIndex] * allV[vOffset + dimension]
                    }
                }
            }
        }

        var result: [Float] = []
        result.reserveCapacity(sequenceLength * hiddenDim)
        for tokenIndex in 0..<sequenceLength {
            let token = Array(attentionOutput[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let projected = try await wo.forward(token)
            result.append(contentsOf: projected)
        }
        return result
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]
        for (key, value) in wq.parameters {
            parameters["wq.\(key)"] = value
        }
        for (key, value) in wk.parameters {
            parameters["wk.\(key)"] = value
        }
        for (key, value) in wv.parameters {
            parameters["wv.\(key)"] = value
        }
        for (key, value) in wo.parameters {
            parameters["wo.\(key)"] = value
        }
        return parameters
    }
}
