import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuNaiveAttention(
    q: [Float],
    k: [Float],
    v: [Float],
    seqLen: Int,
    headDim: Int,
    causal: Bool
) -> [Float] {
    let scale = 1.0 / sqrt(Float(headDim))
    var scores = [Float](repeating: 0, count: seqLen * seqLen)

    for qRow in 0..<seqLen {
        for kRow in 0..<seqLen {
            if causal && kRow > qRow {
                scores[qRow * seqLen + kRow] = -.greatestFiniteMagnitude
                continue
            }

            var dot: Float = 0
            for dim in 0..<headDim {
                dot += q[qRow * headDim + dim] * k[kRow * headDim + dim]
            }
            scores[qRow * seqLen + kRow] = dot * scale
        }
    }

    var attnWeights = [Float](repeating: 0, count: seqLen * seqLen)
    for row in 0..<seqLen {
        let offset = row * seqLen
        let rowValues = Array(scores[offset..<(offset + seqLen)])
        let maxVal = rowValues.max() ?? 0
        let exps = rowValues.map { Foundation.exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        for col in 0..<seqLen {
            attnWeights[offset + col] = exps[col] / sum
        }
    }

    var output = [Float](repeating: 0, count: seqLen * headDim)
    for row in 0..<seqLen {
        for dim in 0..<headDim {
            var value: Float = 0
            for col in 0..<seqLen {
                value += attnWeights[row * seqLen + col] * v[col * headDim + dim]
            }
            output[row * headDim + dim] = value
        }
    }
    return output
}

@Suite("Flash Attention")
struct FlashAttentionTests {
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

    @Test func smallNonCausal() async throws {
        let seqLen = 16
        let headDim = 32
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuNaiveAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: false)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: false,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func smallCausal() async throws {
        let seqLen = 16
        let headDim = 32
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuNaiveAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func mediumCausal() async throws {
        let seqLen = 64
        let headDim = 64
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let expected = cpuNaiveAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-3)
        }
    }

    @Test func causalFirstRowIdentical() async throws {
        let seqLen = 32
        let headDim = 16
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        for dim in 0..<headDim {
            #expect(abs(result[dim] - v[dim]) < 1e-4)
        }
    }

    @Test func longerSequence() async throws {
        let seqLen = 128
        let headDim = 32
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.2...0.2) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.2...0.2) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -0.2...0.2) }
        let expected = cpuNaiveAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        var maxError: Float = 0
        for index in 0..<result.count {
            maxError = max(maxError, abs(result[index] - expected[index]))
        }
        #expect(maxError < 1e-3)
    }

    @Test func outputBufferDimensions() async throws {
        let seqLen = 8
        let headDim = 16
        let q = [Float](repeating: 0.1, count: seqLen * headDim)
        let k = [Float](repeating: 0.1, count: seqLen * headDim)
        let v = [Float](repeating: 0.5, count: seqLen * headDim)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            causal: false,
            commandQueue: commandQueue
        )

        #expect(result.count == seqLen * headDim)
        for value in result {
            #expect(abs(value - 0.5) < 1e-3)
        }
    }
}
