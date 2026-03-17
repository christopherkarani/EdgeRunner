import Metal
import EdgeRunnerSharedTypes

public final class GQAKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    /// Must match the stack array sizes in GQA.metal (scores[16], probs[16]).
    private static let blockSize = 16

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "gqa_attention_f32")
    }

    public func execute(
        q: [Float],
        k: [Float],
        v: [Float],
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        causal: Bool,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        precondition(q.count == numHeads * seqLen * headDim)
        precondition(k.count == numKVHeads * seqLen * headDim)
        precondition(v.count == numKVHeads * seqLen * headDim)
        precondition(headDim <= 128, "headDim must be <= 128")

        let groupSize = numHeads / numKVHeads
        let outputCount = numHeads * seqLen * headDim

        let qBuffer = device.makeBuffer(
            bytes: q,
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let kBuffer = device.makeBuffer(
            bytes: k,
            length: k.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let vBuffer = device.makeBuffer(
            bytes: v,
            length: v.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.blockSize),
            qBlockSize: UInt32(Self.blockSize)
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GQAError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)

        let qBlockCount = (seqLen + Self.blockSize - 1) / Self.blockSize
        let gridSize = MTLSize(width: qBlockCount, height: numHeads, depth: 1)
        let threadgroupSize = MTLSize(width: Self.blockSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    /// Encode a GQA attention dispatch into an existing command buffer without committing.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        qBuffer: MTLBuffer, kBuffer: MTLBuffer, vBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        seqLen: Int, headDim: Int,
        numHeads: Int, numKVHeads: Int,
        causal: Bool
    ) throws {
        let groupSize = numHeads / numKVHeads

        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.blockSize),
            qBlockSize: UInt32(Self.blockSize)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GQAError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)

        let qBlockCount = (seqLen + Self.blockSize - 1) / Self.blockSize
        let gridSize = MTLSize(width: qBlockCount, height: numHeads, depth: 1)
        let threadgroupSize = MTLSize(width: Self.blockSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}

public enum GQAError: Error, Sendable {
    case encodingFailed
}
