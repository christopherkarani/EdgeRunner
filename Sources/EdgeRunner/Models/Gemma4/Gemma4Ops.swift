import Foundation

enum Gemma4Ops {
    static func rmsNorm(
        _ input: [Float],
        weight: [Float],
        eps: Float
    ) throws -> [Float] {
        guard input.count == weight.count else {
            throw Gemma4OpsError.invalidShape(
                "RMSNorm input count \(input.count) must match weight count \(weight.count)"
            )
        }
        guard !input.isEmpty else { return [] }

        var meanSquare: Float = 0
        for value in input {
            meanSquare += value * value
        }
        meanSquare /= Float(input.count)
        let scale = 1.0 / sqrt(meanSquare + eps)

        return zip(input, weight).map { value, weight in
            value * scale * weight
        }
    }

    static func rmsNormUnscaled(
        _ input: [Float],
        eps: Float
    ) -> [Float] {
        guard !input.isEmpty else { return [] }

        var meanSquare: Float = 0
        for value in input {
            meanSquare += value * value
        }
        meanSquare /= Float(input.count)
        let scale = 1.0 / sqrt(meanSquare + eps)
        return input.map { $0 * scale }
    }

    static func geluTanh(_ value: Float) -> Float {
        if value > 10 {
            return value
        }
        if value < -10 {
            return 0
        }
        let coefficient: Float = 0.7978845608028654
        let inner = coefficient * (value + 0.044715 * value * value * value)
        return value * 0.5 * (1.0 + tanh(inner))
    }

    static func geglu(gate: [Float], up: [Float]) throws -> [Float] {
        guard gate.count == up.count else {
            throw Gemma4OpsError.invalidShape(
                "GeGLU gate count \(gate.count) must match up count \(up.count)"
            )
        }
        return zip(gate, up).map { geluTanh($0) * $1 }
    }

    static func expandSingleTokenGQAValue(
        _ value: [Float],
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int
    ) throws -> [Float] {
        guard headDim > 0, numHeads > 0, numKVHeads > 0, numHeads % numKVHeads == 0 else {
            throw Gemma4OpsError.invalidShape("GQA head dimensions are invalid")
        }
        guard value.count == numKVHeads * headDim else {
            throw Gemma4OpsError.invalidShape(
                "GQA value count \(value.count) must equal numKVHeads*headDim \(numKVHeads * headDim)"
            )
        }

        let groupSize = numHeads / numKVHeads
        var output = [Float](repeating: 0, count: numHeads * headDim)
        for head in 0..<numHeads {
            let kvHead = head / groupSize
            let source = kvHead * headDim
            let destination = head * headDim
            for dim in 0..<headDim {
                output[destination + dim] = value[source + dim]
            }
        }
        return output
    }

    static func gemv(weight: [Float], input: [Float], rows: Int, cols: Int) throws -> [Float] {
        guard weight.count == rows * cols else {
            throw Gemma4OpsError.invalidShape(
                "GEMV weight count \(weight.count) must equal rows*cols \(rows * cols)"
            )
        }
        guard input.count == cols else {
            throw Gemma4OpsError.invalidShape(
                "GEMV input count \(input.count) must equal cols \(cols)"
            )
        }

        var output = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var sum: Float = 0
            let rowBase = row * cols
            for col in 0..<cols {
                sum += weight[rowBase + col] * input[col]
            }
            output[row] = sum
        }
        return output
    }

    static func ffnResidual(
        hidden: [Float],
        inputNormWeight: [Float],
        gateWeight: [Float],
        upWeight: [Float],
        downWeight: [Float],
        postFFNNormWeight: [Float],
        intermediateSize: Int,
        eps: Float
    ) throws -> [Float] {
        let hiddenSize = hidden.count
        guard inputNormWeight.count == hiddenSize,
              postFFNNormWeight.count == hiddenSize else {
            throw Gemma4OpsError.invalidShape(
                "FFN norm weights must match hidden size \(hiddenSize)"
            )
        }

        let normed = try rmsNorm(hidden, weight: inputNormWeight, eps: eps)
        let gate = try gemv(
            weight: gateWeight,
            input: normed,
            rows: intermediateSize,
            cols: hiddenSize
        )
        let up = try gemv(
            weight: upWeight,
            input: normed,
            rows: intermediateSize,
            cols: hiddenSize
        )
        let activated = try geglu(gate: gate, up: up)
        let down = try gemv(
            weight: downWeight,
            input: activated,
            rows: hiddenSize,
            cols: intermediateSize
        )
        let postNormed = try rmsNorm(down, weight: postFFNNormWeight, eps: eps)
        return zip(hidden, postNormed).map(+)
    }
}

enum Gemma4OpsError: Error, Sendable, Equatable {
    case invalidShape(String)
}
