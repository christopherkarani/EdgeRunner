import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference GQA implementation using [S, H, D] memory layout.
/// Q/O: [seqLen, numHeads, headDim], K/V: [seqLen, numKVHeads, headDim]
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
    let qStride = numHeads * headDim      // stride between seq positions in Q/O
    let kvStride = numKVHeads * headDim    // stride between seq positions in K/V
    var output = [Float](repeating: 0, count: seqLen * numHeads * headDim)

    for head in 0..<numHeads {
        let kvHead = head / groupSize

        for qRow in 0..<seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            for kRow in 0..<seqLen {
                if causal && kRow > qRow {
                    scores[kRow] = -.greatestFiniteMagnitude
                    continue
                }

                var dot: Float = 0
                let qBase = qRow * qStride + head * headDim
                let kBase = kRow * kvStride + kvHead * headDim
                for dim in 0..<headDim {
                    dot += q[qBase + dim] * k[kBase + dim]
                }
                scores[kRow] = dot * scale
            }

            let maxVal = scores.max() ?? 0
            let exps = scores.map { Foundation.exp($0 - maxVal) }
            let sum = exps.reduce(0, +)
            let weights = exps.map { $0 / sum }

            let oBase = qRow * qStride + head * headDim
            for dim in 0..<headDim {
                var value: Float = 0
                for kRow in 0..<seqLen {
                    let vBase = kRow * kvStride + kvHead * headDim
                    value += weights[kRow] * v[vBase + dim]
                }
                output[oBase + dim] = value
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
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
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
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
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
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.3...0.3) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.3...0.3) }
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
        let q = (0..<(seqLen * numHeads * headDim)).map { Float($0 % 7) * 0.1 }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { Float($0 % 5) * 0.1 }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { Float($0 % 3) * 0.1 }

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

        #expect(result.count == seqLen * numHeads * headDim)
        // In [S,H,D] layout, extract head 0 and head 1 data for all positions
        // Head 0: positions result[s * numHeads * headDim + 0 * headDim ..< s * numHeads * headDim + 1 * headDim]
        // Head 1: positions result[s * numHeads * headDim + 1 * headDim ..< s * numHeads * headDim + 2 * headDim]
        var head0 = [Float]()
        var head1 = [Float]()
        for s in 0..<seqLen {
            let base = s * numHeads * headDim
            head0.append(contentsOf: result[base..<(base + headDim)])
            head1.append(contentsOf: result[(base + headDim)..<(base + 2 * headDim)])
        }
        #expect(head0 != head1)
    }

    @Test func gqaNonCausal() async throws {
        let seqLen = 16
        let headDim = 32
        let numHeads = 4
        let numKVHeads = 2
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.5...0.5) }
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

    @Test func gqaF16KVMatchesCPUReference() async throws {
        let seqLen = 16
        let headDim = 128
        let numHeads = 8
        let numKVHeads = 2
        let groupSize = numHeads / numKVHeads
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let expected = cpuGQA(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: true
        )

        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let kHalf = k.map(Float16.init)
        let vHalf = v.map(Float16.init)
        let kBuffer = device.makeBuffer(bytes: kHalf, length: kHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: vHalf, length: vHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: UInt32(seqLen),
            qOffset: 0
        )

        let kernel = try GQAKernel(device: device)
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(kernel.pipelineF16KV)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)
        let qBlockCount = (seqLen + GQAKernel.blockSize - 1) / GQAKernel.blockSize
        encoder.dispatchThreadgroups(
            MTLSize(width: qBlockCount, height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        let result = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().assumingMemoryBound(to: Float.self),
                count: expected.count
            )
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 2e-3)
        }
    }

    @Test func promptFlashGQAMatchesCPUReference() async throws {
        let seqLen = 8
        let headDim = 128
        let numHeads = 4
        let numKVHeads = 2
        let groupSize = numHeads / numKVHeads
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -0.25...0.25) }
        let expected = cpuGQA(
            q: q,
            k: k,
            v: v,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: true
        )

        let registry = try KernelRegistry(device: device)
        let pipeline = try registry.pipeline(for: "flash_attention_gqa_simd_f32")
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: k, length: k.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: v, length: v.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: 1,
            kvBlockSize: 0,
            qBlockSize: 0,
            kvSeqLen: UInt32(seqLen),
            qOffset: 0
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: numHeads, depth: seqLen),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        let result = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().assumingMemoryBound(to: Float.self),
                count: expected.count
            )
        )

        for index in 0..<result.count {
            #expect(abs(result[index] - expected[index]) < 2e-3)
        }
    }

}
