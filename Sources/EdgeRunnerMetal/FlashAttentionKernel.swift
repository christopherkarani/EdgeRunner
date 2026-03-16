import Metal
import EdgeRunnerSharedTypes

public final class FlashAttentionKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    public static let qBlockSize = 16
    public static let kvBlockSize = 16

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "flash_attention_f32")
    }

    public func execute(
        q: [Float],
        k: [Float],
        v: [Float],
        seqLen: Int,
        headDim: Int,
        causal: Bool,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(q.count == seqLen * headDim)
        precondition(k.count == seqLen * headDim)
        precondition(v.count == seqLen * headDim)
        precondition(headDim <= 128, "headDim must be <= 128")

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
            length: seqLen * headDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERFlashAttentionParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.kvBlockSize),
            qBlockSize: UInt32(Self.qBlockSize)
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FlashAttentionError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERFlashAttentionParams>.stride, index: 4)

        let qBlockCount = (seqLen + Self.qBlockSize - 1) / Self.qBlockSize
        let gridSize = MTLSize(width: qBlockCount, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.qBlockSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
        return Array(UnsafeBufferPointer(start: pointer, count: seqLen * headDim))
    }
}

public enum FlashAttentionError: Error, Sendable {
    case encodingFailed
}
