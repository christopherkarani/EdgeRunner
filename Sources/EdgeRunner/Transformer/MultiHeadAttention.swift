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

    init(
        config: TransformerConfig,
        wq: LinearModule,
        wk: LinearModule,
        wv: LinearModule,
        wo: LinearModule
    ) {
        self.config = config
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
    }

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

        let (allQ, allK, allV) = try await projectQKV(input)

        var attentionOutput = [Float](repeating: 0, count: sequenceLength * hiddenDim)
        let scale = 1.0 / sqrt(Float(headDim))

        for head in 0..<numHeads {
            let kvHead = head / kvGroupSize
            computeHeadAttention(
                head: head, kvHead: kvHead, seqLen: sequenceLength,
                hiddenDim: hiddenDim, headDim: headDim, numKVHeads: numKVHeads,
                scale: scale, allQ: allQ, allK: allK, allV: allV,
                output: &attentionOutput
            )
        }

        return try await projectOutput(attentionOutput, seqLen: sequenceLength, hiddenDim: hiddenDim)
    }

    private func projectQKV(_ input: AttentionInput) async throws -> ([Float], [Float], [Float]) {
        let sequenceLength = input.seqLen
        let hiddenDim = config.hiddenDim
        let headDim = config.headDim
        let numKVHeads = config.numKVHeads

        var allQ: [Float] = []
        var allK: [Float] = []
        var allV: [Float] = []
        allQ.reserveCapacity(sequenceLength * hiddenDim)
        allK.reserveCapacity(sequenceLength * numKVHeads * headDim)
        allV.reserveCapacity(sequenceLength * numKVHeads * headDim)

        for tokenIndex in 0..<sequenceLength {
            let token = Array(input.hidden[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let absolutePosition = input.startPos + tokenIndex
            let q = Self.applyRoPE(
                try await wq.forward(token), headDim: headDim,
                position: absolutePosition, theta: config.ropeTheta
            )
            let k = Self.applyRoPE(
                try await wk.forward(token), headDim: headDim,
                position: absolutePosition, theta: config.ropeTheta
            )
            allQ.append(contentsOf: q)
            allK.append(contentsOf: k)
            allV.append(contentsOf: try await wv.forward(token))
        }
        return (allQ, allK, allV)
    }

    private func computeHeadAttention(
        head: Int, kvHead: Int, seqLen: Int,
        hiddenDim: Int, headDim: Int, numKVHeads: Int,
        scale: Float, allQ: [Float], allK: [Float], allV: [Float],
        output: inout [Float]
    ) {
        for queryIndex in 0..<seqLen {
            let qOffset = queryIndex * hiddenDim + head * headDim
            let qVector = Array(allQ[qOffset..<(qOffset + headDim)])

            let scores = computeAttentionScores(
                qVector: qVector, queryIndex: queryIndex, seqLen: seqLen,
                numKVHeads: numKVHeads, kvHead: kvHead, headDim: headDim, scale: scale,
                allK: allK
            )

            let outputOffset = queryIndex * hiddenDim + head * headDim
            for valueIndex in 0..<seqLen {
                let vOffset = valueIndex * (numKVHeads * headDim) + kvHead * headDim
                for dimension in 0..<headDim {
                    output[outputOffset + dimension] += scores[valueIndex] * allV[vOffset + dimension]
                }
            }
        }
    }

    private func computeAttentionScores(
        qVector: [Float], queryIndex: Int, seqLen: Int,
        numKVHeads: Int, kvHead: Int, headDim: Int, scale: Float,
        allK: [Float]
    ) -> [Float] {
        let scores = (0..<seqLen).map { keyIndex -> Float in
            let kOffset = keyIndex * (numKVHeads * headDim) + kvHead * headDim
            var dot: Float = 0
            for dimension in 0..<headDim {
                dot += qVector[dimension] * allK[kOffset + dimension]
            }
            return keyIndex > queryIndex ? -.infinity : dot * scale
        }

        let maxScore = scores.max() ?? 0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        return expScores.map { $0 / sumExp }
    }

    private func projectOutput(
        _ attentionOutput: [Float], seqLen: Int, hiddenDim: Int
    ) async throws -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(seqLen * hiddenDim)
        for tokenIndex in 0..<seqLen {
            let token = Array(attentionOutput[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            result.append(contentsOf: try await wo.forward(token))
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

    static func applyRoPE(
        _ vector: [Float],
        headDim: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(vector.count % headDim == 0, "Vector count must be divisible by headDim")

        var rotated = vector
        for headStart in stride(from: 0, to: vector.count, by: headDim) {
            let head = Array(vector[headStart..<(headStart + headDim)])
            let rotatedHead = applyRoPE(head, position: position, theta: theta)
            rotated.replaceSubrange(headStart..<(headStart + headDim), with: rotatedHead)
        }
        return rotated
    }

    static func applyRoPE(_ head: [Float], position: Int, theta: Float) -> [Float] {
        var rotated = head
        let pairCount = head.count / 2

        for pairIndex in 0..<pairCount {
            let evenIndex = pairIndex * 2
            let oddIndex = evenIndex + 1
            let exponent = Float(evenIndex) / Float(head.count)
            let inverseFrequency = pow(theta, -exponent)
            let angle = Float(position) * inverseFrequency
            let cosAngle = cos(angle)
            let sinAngle = sin(angle)
            let evenValue = head[evenIndex]
            let oddValue = head[oddIndex]
            rotated[evenIndex] = evenValue * cosAngle - oddValue * sinAngle
            rotated[oddIndex] = evenValue * sinAngle + oddValue * cosAngle
        }

        return rotated
    }
}
