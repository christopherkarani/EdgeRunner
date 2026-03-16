import Metal
import EdgeRunnerSharedTypes

public struct LayerNormKernel: Sendable {
    private let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "layernorm_f32")
    }

    public func execute(
        input: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        cols: Int,
        eps: Float = 1e-5,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols)
        precondition(gamma.count == cols)
        precondition(beta.count == cols)

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let gammaBuffer = device.makeBuffer(
            bytes: gamma,
            length: gamma.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let betaBuffer = device.makeBuffer(
            bytes: beta,
            length: beta.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERLayerNormParams(rows: UInt32(rows), cols: UInt32(cols), eps: eps)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LayerNormError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gammaBuffer, offset: 0, index: 1)
        encoder.setBuffer(betaBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERLayerNormParams>.stride, index: 4)

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

public enum LayerNormError: Error, Sendable {
    case encodingFailed
}
