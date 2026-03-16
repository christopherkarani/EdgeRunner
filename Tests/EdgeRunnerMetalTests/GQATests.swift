import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuGQA(
    q: [Float],
    k: [Float],
    v: [Float],
    seqLen: Int,
    headDim: Int,
    numHeads: Int,
    numKVHeads: Int,
    causal: Bool
) -> [Float] {
    let scale = 1.0 / sqrt(Float(headDim))
    let groupSize = numHeads / numKVHeads
    var output = [Float](repeating: 0, count: numHeads * seqLen * headDim)

    for head in 0..<numHeads {
        let kvHead = head / groupSize
        let qOffset = head * seqLen * headDim
        let kOffset = kvHead * seqLen * headDim
        let vOffset = kvHead * seqLen * headDim
        let outOffset = head * seqLen * headDim

        for qRow in 0..<seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            for kRow in 0..<seqLen {
                if causal && kRow > qRow {
                    scores[kRow] = -.greatestFiniteMagnitude
                    continue
                }

                var dot: Float = 0
                for dim in 0..<headDim {
                    dot += q[qOffset + qRow * headDim + dim] * k[kOffset + kRow * headDim + dim]
                }
                scores[kRow] = dot * scale
            }

            let maxVal = scores.max() ?? 0
            let exps = scores.map { Foundation.exp($0 - maxVal) }
            let sum = exps.reduce(0, +)
            let weights = exps.map { $0 / sum }

            for dim in 0..<headDim {
                var value: Float = 0
                for kRow in 0..<seqLen {
                    value += weights[kRow] * v[vOffset + kRow * headDim + dim]
                }
                output[outOffset + qRow * headDim + dim] = value
            }
        }
    }

    return output
}

@Suite("Grouped Query Attention")
struct GQATests {
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

    @Test func mhaBaseline1to1() async throws {
        let seqLen = 16
        let headDim = 32
        let numHeads = 4
        let numKVHeads = 4
        let q = (0..<(numHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: true)

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func gqa1to4() async throws {
        let seqLen = 16
        let headDim = 32
        let numHeads = 8
        let numKVHeads = 2
        let q = (0..<(numHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: true)

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func gqa1to8() async throws {
        let seqLen = 16
        let headDim = 64
        let numHeads = 8
        let numKVHeads = 1
        let q = (0..<(numHeads * seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let k = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let v = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let expected = cpuGQA(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: true)

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-3)
        }
    }

    @Test func gqaGroupsShareKV() async throws {
        let seqLen = 8
        let headDim = 16
        let numHeads = 4
        let numKVHeads = 2
        let q = (0..<(numHeads * seqLen * headDim)).map { Float($0 % 7) * 0.1 }
        let k = (0..<(numKVHeads * seqLen * headDim)).map { Float($0 % 5) * 0.1 }
        let v = (0..<(numKVHeads * seqLen * headDim)).map { Float($0 % 3) * 0.1 }

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: false,
            commandQueue: commandQueue
        )

        #expect(result.count == numHeads * seqLen * headDim)
        let head0 = Array(result[0..<(seqLen * headDim)])
        let head1 = Array(result[(seqLen * headDim)..<(2 * seqLen * headDim)])
        #expect(head0 != head1)
    }

    @Test func gqaNonCausal() async throws {
        let seqLen = 16
        let headDim = 32
        let numHeads = 4
        let numKVHeads = 2
        let q = (0..<(numHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(numKVHeads * seqLen * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: false)

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: false,
            commandQueue: commandQueue
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }
}
