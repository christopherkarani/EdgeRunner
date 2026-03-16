import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuRoPE(
    input: [Float],
    seqLen: Int,
    numHeads: Int,
    headDim: Int,
    startPos: Int = 0,
    theta: Float = 10_000,
    scalingFactor: Float = 1
) -> [Float] {
    var output = input
    let halfDim = headDim / 2

    for seq in 0..<seqLen {
        let position = Float(seq + startPos)
        for head in 0..<numHeads {
            for pair in 0..<halfDim {
                let frequency = 1.0 / pow(theta, Float(2 * pair) / Float(headDim))
                let angle = position * (frequency / scalingFactor)
                let cosValue = cos(angle)
                let sinValue = sin(angle)

                let index0 = (seq * numHeads * headDim) + (head * headDim) + (2 * pair)
                let index1 = index0 + 1
                let x0 = input[index0]
                let x1 = input[index1]

                output[index0] = x0 * cosValue - x1 * sinValue
                output[index1] = x0 * sinValue + x1 * cosValue
            }
        }
    }

    return output
}

@Suite("RoPE Kernel")
struct RoPETests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = commandQueue
    }

    @Test func basicRoPE() async throws {
        let seqLen = 4
        let numHeads = 2
        let headDim = 8
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let expected = cpuRoPE(input: input, seqLen: seqLen, numHeads: numHeads, headDim: headDim)

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func ropeWithOffset() async throws {
        let seqLen = 1
        let numHeads = 4
        let headDim = 64
        let startPos = 42
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let expected = cpuRoPE(input: input, seqLen: seqLen, numHeads: numHeads, headDim: headDim, startPos: startPos)

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: startPos,
            theta: 10_000,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func ropePositionZeroIsIdentityLike() async throws {
        let seqLen = 1
        let numHeads = 2
        let headDim = 16
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - input[index]) < 1e-5)
        }
    }

    @Test func ropePreservesNorm() async throws {
        let seqLen = 8
        let numHeads = 4
        let headDim = 32
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            commandQueue: commandQueue
        )

        for seq in 0..<seqLen {
            for head in 0..<numHeads {
                let offset = (seq * numHeads + head) * headDim
                var inputNorm: Float = 0
                var outputNorm: Float = 0
                for dim in 0..<headDim {
                    inputNorm += input[offset + dim] * input[offset + dim]
                    outputNorm += result[offset + dim] * result[offset + dim]
                }
                #expect(abs(sqrt(inputNorm) - sqrt(outputNorm)) < 1e-4)
            }
        }
    }

    @Test func ntkAwareScaling() async throws {
        let seqLen = 4
        let numHeads = 2
        let headDim = 16
        let scalingFactor: Float = 4
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let expected = cpuRoPE(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            scalingFactor: scalingFactor
        )

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func ropeKnownValues() async throws {
        let seqLen = 2
        let numHeads = 1
        let headDim = 2
        let input: [Float] = [0, 0, 1, 0]

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            commandQueue: commandQueue
        )

        #expect(abs(result[0]) < 1e-5)
        #expect(abs(result[1]) < 1e-5)
        #expect(abs(result[2] - cos(Float(1))) < 1e-5)
        #expect(abs(result[3] - sin(Float(1))) < 1e-5)
    }

    @Test func largerDimensions() async throws {
        let seqLen = 32
        let numHeads = 8
        let headDim = 128
        let input = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuRoPE(input: input, seqLen: seqLen, numHeads: numHeads, headDim: headDim)

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10_000,
            commandQueue: commandQueue
        )

        var maxError: Float = 0
        for index in 0..<result.count {
            maxError = max(maxError, abs(result[index] - expected[index]))
        }
        #expect(maxError < 1e-4)
    }
}
