import Foundation

/// Full GPT-2 model: token embedding + position embedding + blocks + final LayerNorm + LM head.
public struct GPT2Model: EdgeRunnerModule, Sendable {
    public typealias Input = [Int]
    public typealias Output = [Float]

    private let config: GPT2Config
    private let tokenEmbedding: Embedding
    private let positionEmbedding: Embedding
    private let blocks: [GPT2Block]
    private let lnFinalGamma: [Float]
    private let lnFinalBeta: [Float]

    public init(config: GPT2Config) throws {
        self.config = config
        let hiddenDim = config.hiddenDim

        let tokenScale = sqrt(2.0 / Float(config.vocabSize + hiddenDim))
        self.tokenEmbedding = Embedding(
            weight: (0..<(config.vocabSize * hiddenDim)).map { _ in
                Float.random(in: -tokenScale...tokenScale)
            },
            vocabSize: config.vocabSize,
            dim: hiddenDim
        )

        let positionScale = sqrt(2.0 / Float(config.maxSeqLen + hiddenDim))
        self.positionEmbedding = Embedding(
            weight: (0..<(config.maxSeqLen * hiddenDim)).map { _ in
                Float.random(in: -positionScale...positionScale)
            },
            vocabSize: config.maxSeqLen,
            dim: hiddenDim
        )

        var modelBlocks: [GPT2Block] = []
        modelBlocks.reserveCapacity(config.numLayers)
        for _ in 0..<config.numLayers {
            modelBlocks.append(try GPT2Block(config: config))
        }
        self.blocks = modelBlocks

        self.lnFinalGamma = [Float](repeating: 1.0, count: hiddenDim)
        self.lnFinalBeta = [Float](repeating: 0.0, count: hiddenDim)
    }

    public func forward(_ input: [Int]) async throws -> [Float] {
        let sequenceLength = input.count
        let hiddenDim = config.hiddenDim
        precondition(
            sequenceLength <= config.maxSeqLen,
            "Sequence length \(sequenceLength) exceeds max \(config.maxSeqLen)"
        )

        let tokenEmbeddings = try await tokenEmbedding.forward(input)
        let positionIDs = Array(0..<sequenceLength)
        let positionEmbeddings = try await positionEmbedding.forward(positionIDs)
        var hidden = zip(tokenEmbeddings, positionEmbeddings).map { $0 + $1 }

        for block in blocks {
            let blockOutput = try await block.forward(
                TransformerBlockInput(hidden: hidden, seqLen: sequenceLength, startPos: 0)
            )
            hidden = blockOutput.hidden
        }

        hidden = cpuLayerNorm(
            hidden,
            gamma: lnFinalGamma,
            beta: lnFinalBeta,
            rows: sequenceLength,
            cols: hiddenDim,
            eps: config.layerNormEps
        )

        let tokenEmbeddingWeight = tokenEmbedding.parameters["weight"]!.floatArray
        var logits = [Float](repeating: 0, count: sequenceLength * config.vocabSize)
        for tokenIndex in 0..<sequenceLength {
            for vocabIndex in 0..<config.vocabSize {
                var dot: Float = 0
                for dimension in 0..<hiddenDim {
                    dot += hidden[tokenIndex * hiddenDim + dimension] *
                        tokenEmbeddingWeight[vocabIndex * hiddenDim + dimension]
                }
                logits[tokenIndex * config.vocabSize + vocabIndex] = dot
            }
        }

        return logits
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]

        for (key, value) in tokenEmbedding.parameters {
            parameters["token_embedding.\(key)"] = value
        }
        for (key, value) in positionEmbedding.parameters {
            parameters["position_embedding.\(key)"] = value
        }
        for (index, block) in blocks.enumerated() {
            for (key, value) in block.parameters {
                parameters["blocks.\(index).\(key)"] = value
            }
        }
        parameters["ln_final.weight"] = ArrayTensorBox(
            data: lnFinalGamma,
            shape: [config.hiddenDim]
        )
        parameters["ln_final.bias"] = ArrayTensorBox(
            data: lnFinalBeta,
            shape: [config.hiddenDim]
        )
        return parameters
    }

    private func cpuLayerNorm(
        _ input: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        cols: Int,
        eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for rowIndex in 0..<rows {
            let offset = rowIndex * cols
            let row = Array(input[offset..<(offset + cols)])
            let mean = row.reduce(0, +) / Float(cols)
            let variance = row.reduce(0) { partial, value in
                let delta = value - mean
                return partial + delta * delta
            } / Float(cols)
            let invStd = 1.0 / sqrt(variance + eps)
            for columnIndex in 0..<cols {
                output[offset + columnIndex] =
                    (row[columnIndex] - mean) * invStd * gamma[columnIndex] + beta[columnIndex]
            }
        }
        return output
    }
}
