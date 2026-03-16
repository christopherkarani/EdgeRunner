import Foundation

/// SwiGLU-based feed-forward network.
public struct FeedForward: EdgeRunnerModule, Sendable {
    public typealias Input = [Float]
    public typealias Output = [Float]

    private let config: TransformerConfig
    private let gate: LinearModule
    private let up: LinearModule
    private let down: LinearModule

    public init(config: TransformerConfig, zeroWeights: Bool = false) throws {
        self.config = config

        let hiddenDim = config.hiddenDim
        let intermediateSize = config.intermediateSize

        if zeroWeights {
            self.gate = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: intermediateSize,
                weight: [Float](repeating: 0, count: intermediateSize * hiddenDim),
                bias: nil
            )
            self.up = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: intermediateSize,
                weight: [Float](repeating: 0, count: intermediateSize * hiddenDim),
                bias: nil
            )
            self.down = try LinearModule(
                inFeatures: intermediateSize,
                outFeatures: hiddenDim,
                weight: [Float](repeating: 0, count: hiddenDim * intermediateSize),
                bias: nil
            )
        } else {
            let upScale = sqrt(2.0 / Float(hiddenDim + intermediateSize))
            let downScale = sqrt(2.0 / Float(intermediateSize + hiddenDim))
            self.gate = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: intermediateSize,
                weight: (0..<(intermediateSize * hiddenDim)).map { _ in Float.random(in: -upScale...upScale) },
                bias: nil
            )
            self.up = try LinearModule(
                inFeatures: hiddenDim,
                outFeatures: intermediateSize,
                weight: (0..<(intermediateSize * hiddenDim)).map { _ in Float.random(in: -upScale...upScale) },
                bias: nil
            )
            self.down = try LinearModule(
                inFeatures: intermediateSize,
                outFeatures: hiddenDim,
                weight: (0..<(hiddenDim * intermediateSize)).map { _ in Float.random(in: -downScale...downScale) },
                bias: nil
            )
        }
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        let hiddenDim = config.hiddenDim
        let intermediateSize = config.intermediateSize
        let sequenceLength = input.count / hiddenDim
        precondition(input.count == sequenceLength * hiddenDim)

        var result: [Float] = []
        result.reserveCapacity(sequenceLength * hiddenDim)

        for tokenIndex in 0..<sequenceLength {
            let token = Array(input[(tokenIndex * hiddenDim)..<((tokenIndex + 1) * hiddenDim)])
            let gateOutput = try await gate.forward(token)
            let upOutput = try await up.forward(token)

            var swiglu = [Float](repeating: 0, count: intermediateSize)
            for index in 0..<intermediateSize {
                let sigmoid = 1.0 / (1.0 + exp(-gateOutput[index]))
                swiglu[index] = gateOutput[index] * sigmoid * upOutput[index]
            }

            let downOutput = try await down.forward(swiglu)
            result.append(contentsOf: downOutput)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var parameters: [String: any TensorBox] = [:]
        for (key, value) in gate.parameters {
            parameters["gate.\(key)"] = value
        }
        for (key, value) in up.parameters {
            parameters["up.\(key)"] = value
        }
        for (key, value) in down.parameters {
            parameters["down.\(key)"] = value
        }
        return parameters
    }
}
