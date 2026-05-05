import Foundation
import Testing
@testable import EdgeRunner

@Suite("Gemma4 ops")
struct Gemma4OpsTests {
    @Test("RMSNorm uses Gemma 4 direct weight convention")
    func rmsNormUsesDirectWeight() throws {
        let input: [Float] = [1, -2, 3, -4]
        let weight: [Float] = [0.1, -0.2, 0.0, 0.5]

        let output = try Gemma4Ops.rmsNorm(input, weight: weight, eps: 1e-6)

        let meanSquare = input.reduce(Float(0)) { $0 + $1 * $1 } / Float(input.count)
        let scale = 1 / sqrt(meanSquare + 1e-6)
        let expected = zip(input, weight).map { value, normWeight in
            value * scale * normWeight
        }
        assertClose(output, expected, tolerance: 1e-6)
    }

    @Test("Unscaled RMSNorm omits affine weight")
    func rmsNormUnscaledMatchesReference() {
        let input: [Float] = [1, -2, 3, -4]

        let output = Gemma4Ops.rmsNormUnscaled(input, eps: 1e-6)

        let meanSquare = input.reduce(Float(0)) { $0 + $1 * $1 } / Float(input.count)
        let scale = 1 / sqrt(meanSquare + 1e-6)
        let expected = input.map { $0 * scale }
        assertClose(output, expected, tolerance: 1e-6)
    }

    @Test("GeGLU matches PyTorch tanh approximation")
    func gegluMatchesReference() throws {
        let gate: [Float] = [-1, 0, 0.5, 2]
        let up: [Float] = [2, 3, 4, 5]

        let output = try Gemma4Ops.geglu(gate: gate, up: up)

        let coefficient: Float = 0.7978845608028654
        let expected = zip(gate, up).map { gateValue, upValue in
            let inner = coefficient * (gateValue + 0.044715 * gateValue * gateValue * gateValue)
            let gelu = gateValue * 0.5 * (1 + tanh(inner))
            return gelu * upValue
        }
        assertClose(output, expected, tolerance: 1e-6)
    }

    @Test("GeGLU remains finite for large gates")
    func gegluLargeGatesStayFinite() throws {
        let output = try Gemma4Ops.geglu(gate: [-100, 100], up: [3, 4])
        #expect(output == [0, 400])
    }

    @Test("Single-token GQA value expansion repeats each KV head across its query group")
    func singleTokenGQAExpansionMatchesReference() throws {
        let value: [Float] = [
            1, 2, 3,
            4, 5, 6
        ]

        let output = try Gemma4Ops.expandSingleTokenGQAValue(
            value,
            headDim: 3,
            numHeads: 4,
            numKVHeads: 2
        )

        let expected: [Float] = [
            1, 2, 3,
            1, 2, 3,
            4, 5, 6,
            4, 5, 6
        ]
        assertClose(output, expected, tolerance: 0)
    }

    @Test("FFN residual slice matches explicit reference")
    func ffnResidualMatchesReference() throws {
        let hidden: [Float] = [0.5, -1.0, 1.5]
        let inputNorm: [Float] = [0.1, 0.0, -0.1]
        let postNorm: [Float] = [0.0, 0.2, -0.2]
        let intermediateSize = 4

        let gateWeight: [Float] = [
            0.2, -0.1, 0.3,
            -0.4, 0.5, 0.1,
            0.6, -0.2, 0.2,
            -0.1, 0.4, -0.3,
        ]
        let upWeight: [Float] = [
            -0.3, 0.2, 0.1,
            0.4, -0.2, 0.3,
            0.1, 0.5, -0.4,
            -0.2, 0.1, 0.6,
        ]
        let downWeight: [Float] = [
            0.3, -0.2, 0.4, 0.1,
            -0.5, 0.2, 0.1, -0.3,
            0.2, 0.3, -0.4, 0.5,
        ]

        let output = try Gemma4Ops.ffnResidual(
            hidden: hidden,
            inputNormWeight: inputNorm,
            gateWeight: gateWeight,
            upWeight: upWeight,
            downWeight: downWeight,
            postFFNNormWeight: postNorm,
            intermediateSize: intermediateSize,
            eps: 1e-6
        )

        let normed = try referenceRMSNorm(hidden, weight: inputNorm, eps: 1e-6)
        let gate = try referenceGEMV(weight: gateWeight, input: normed, rows: intermediateSize, cols: hidden.count)
        let up = try referenceGEMV(weight: upWeight, input: normed, rows: intermediateSize, cols: hidden.count)
        let activated = try Gemma4Ops.geglu(gate: gate, up: up)
        let down = try referenceGEMV(weight: downWeight, input: activated, rows: hidden.count, cols: intermediateSize)
        let postNormed = try referenceRMSNorm(down, weight: postNorm, eps: 1e-6)
        let expected = zip(hidden, postNormed).map(+)

        assertClose(output, expected, tolerance: 1e-6)
    }

    @Test("Shape errors are reported before math")
    func shapeErrorsAreReported() throws {
        #expect(throws: Gemma4OpsError.invalidShape("RMSNorm input count 2 must match weight count 1")) {
            _ = try Gemma4Ops.rmsNorm([1, 2], weight: [1], eps: 1e-6)
        }
    }

    private func referenceRMSNorm(_ input: [Float], weight: [Float], eps: Float) throws -> [Float] {
        guard input.count == weight.count else { throw Gemma4OpsError.invalidShape("bad reference shape") }
        let meanSquare = input.reduce(Float(0)) { $0 + $1 * $1 } / Float(input.count)
        let scale = 1 / sqrt(meanSquare + eps)
        return zip(input, weight).map { $0 * scale * $1 }
    }

    private func referenceGEMV(
        weight: [Float],
        input: [Float],
        rows: Int,
        cols: Int
    ) throws -> [Float] {
        guard weight.count == rows * cols, input.count == cols else {
            throw Gemma4OpsError.invalidShape("bad reference GEMV shape")
        }
        return (0..<rows).map { row in
            (0..<cols).reduce(Float(0)) { sum, col in
                sum + weight[row * cols + col] * input[col]
            }
        }
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float) {
        #expect(actual.count == expected.count)
        for index in 0..<min(actual.count, expected.count) {
            #expect(abs(actual[index] - expected[index]) <= tolerance)
        }
    }
}
