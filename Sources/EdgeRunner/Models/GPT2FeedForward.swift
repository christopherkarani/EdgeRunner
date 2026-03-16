import Foundation

/// GPT-2 feed-forward network: GELU activation, not SwiGLU.
public struct GPT2FeedForward: EdgeRunnerModule, Sendable {
    public typealias Input = [Float]
    public typealias Output = [Float]

    private let config: GPT2Config
    private let cFc: LinearModule
    private let cProj: LinearModule

    public init(config: GPT2Config) throws {
        self.config = config
        let hiddenDim = config.hiddenDim
        let intermediateSize = config.intermediateSize
        let fcScale = sqrt(2.0 / Float(hiddenDim + intermediateSize))
        let projectionScale = sqrt(2.0 / Float(intermediateSize + hiddenDim))

        self.cFc = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: intermediateSize,
            weight: (0..<(intermediateSize * hiddenDim)).map { _ in
                Float.random(in: -fcScale...fcScale)
            },
            bias: [Float](repeating: 0, count: intermediateSize)
        )
        self.cProj = try LinearModule(
            inFeatures: intermediateSize,
            outFeatures: hiddenDim,
            weight: (0..<(hiddenDim * intermediateSize)).map { _ in
                Float.random(in: -projectionScale...projectionScale)
            },
            bias: [Float](repeating: 0, count: hiddenDim)
        )
    }

    public init(
        config: GPT2Config,
        cFcWeight: [Float],
        cFcBias: [Float],
        cProjWeight: [Float],
        cProjBias: [Float]
    ) throws {
        self.config = config
        let hiddenDim = config.hiddenDim
        let intermediateSize = config.intermediateSize
        self.cFc = try LinearModule(
            inFeatures: hiddenDim,
            outFeatures: intermediateSize,
            weight: cFcWeight,
            bias: cFcBias
        )
        self.cProj = try LinearModule(
            inFeatures: intermediateSize,
            outFeatures: hiddenDim,
            weight: cProjWeight,
            bias: cProjBias
        )
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        let hiddenDim = config.hiddenDim
        let intermediateSize = config.intermediateSize
        let sequenceLength = input.count / hiddenDim

        var result: [Float] = []
        result.reserveCapacity(sequenceLength * hiddenDim)
        let geluConstant: Float = sqrt(2.0 / .pi)

        for tokenIndex in 0..<sequenceLength {
            let token = Array(input[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let hidden = try await cFc.forward(token)

            var activated = [Float](repeating: 0, count: intermediateSize)
            for index in 0..<intermediateSize {
                let x = hidden[index]
                activated[index] = x * 0.5 * (1.0 + tanh(geluConstant * (x + 0.044715 * x * x * x)))
            }

            let projected = try await cProj.forward(activated)
            result.append(contentsOf: projected)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]
        for (key, value) in cFc.parameters {
            parameters["c_fc.\(key)"] = value
        }
        for (key, value) in cProj.parameters {
            parameters["c_proj.\(key)"] = value
        }
        return parameters
    }
}
