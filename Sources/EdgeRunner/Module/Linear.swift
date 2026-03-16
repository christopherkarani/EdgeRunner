/// A fully-connected linear layer: y = x @ W^T + b.
///
/// Weight is stored as [outFeatures x inFeatures] row-major.
public struct LinearModule: EdgeRunnerModule, Sendable {
    public typealias Input = [Float]
    public typealias Output = [Float]

    public let inFeatures: Int
    public let outFeatures: Int

    private let weight: [Float]
    private let bias: [Float]?

    public init(
        inFeatures: Int,
        outFeatures: Int,
        weight: [Float],
        bias: [Float]?
    ) throws {
        precondition(
            weight.count == outFeatures * inFeatures,
            "Weight size \(weight.count) != \(outFeatures) * \(inFeatures)"
        )
        if let bias {
            precondition(
                bias.count == outFeatures,
                "Bias size \(bias.count) != \(outFeatures)"
            )
        }
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.bias = bias
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        precondition(
            input.count == inFeatures,
            "Input size \(input.count) != inFeatures \(inFeatures)"
        )

        var output = [Float](repeating: 0, count: outFeatures)
        for outputIndex in 0..<outFeatures {
            var sum: Float = 0
            for inputIndex in 0..<inFeatures {
                sum += weight[outputIndex * inFeatures + inputIndex] * input[inputIndex]
            }
            if let bias {
                sum += bias[outputIndex]
            }
            output[outputIndex] = sum
        }
        return output
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [
            "weight": ArrayTensorBox(data: weight, shape: [outFeatures, inFeatures])
        ]
        if let bias {
            params["bias"] = ArrayTensorBox(data: bias, shape: [outFeatures])
        }
        return params
    }
}
