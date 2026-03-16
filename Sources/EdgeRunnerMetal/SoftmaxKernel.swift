import Metal
import EdgeRunnerSharedTypes

public final class SoftmaxKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    private static let threadsPerRow = 256

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "softmax_f32")
    }

    public func execute(
        input: [Float],
        rows: Int,
        cols: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols, "Input count must match rows * cols")

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERSoftmaxParams(rows: UInt32(rows), cols: UInt32(cols))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SoftmaxError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERSoftmaxParams>.stride, index: 2)

        let gridSize = MTLSize(width: rows, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        return Array(UnsafeBufferPointer(start: pointer, count: rows * cols))
    }
}

public enum SoftmaxError: Error, Sendable {
    case encodingFailed
}
