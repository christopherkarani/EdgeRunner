import Metal
import EdgeRunnerSharedTypes

public struct RMSNormKernel: Sendable {
    private let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "rmsnorm_f32")
    }

    public func execute(
        input: [Float],
        weight: [Float],
        rows: Int,
        cols: Int,
        eps: Float = 1e-5,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols)
        precondition(weight.count == cols)

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let weightBuffer = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERRMSNormParams(rows: UInt32(rows), cols: UInt32(cols), eps: eps)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RMSNormError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)

        let gridSize = MTLSize(width: rows, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(rows, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
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
}

public enum RMSNormError: Error, Sendable {
    case encodingFailed
}
