import Metal
import EdgeRunnerSharedTypes

public final class RoPEKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "rope_f32")
    }

    public func execute(
        input: [Float],
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(input.count == seqLen * numHeads * headDim)

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERRoPEParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            startPos: UInt32(startPos),
            theta: theta,
            scalingFactor: scalingFactor
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RoPEError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)

        let halfDim = headDim / 2
        let gridSize = MTLSize(width: halfDim, height: numHeads, depth: seqLen)
        let threadgroupSize = MTLSize(
            width: min(halfDim, pipelineF32.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: pointer, count: input.count))
    }

    public func applyToQK(
        q: [Float],
        k: [Float],
        seqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1,
        commandQueue: MTLCommandQueue
    ) async throws -> ([Float], [Float]) {
        let rotatedQ = try await execute(
            input: q,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )
        let rotatedK = try await execute(
            input: k,
            seqLen: seqLen,
            numHeads: numKVHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )
        return (rotatedQ, rotatedK)
    }
}

public enum RoPEError: Error, Sendable {
    case encodingFailed
}
