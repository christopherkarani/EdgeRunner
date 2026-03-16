import Metal
import EdgeRunnerSharedTypes

public struct ActivationKernels: Sendable {
    private let sigmoidPipeline: MTLComputePipelineState
    private let geluPipeline: MTLComputePipelineState
    private let swigluPipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.sigmoidPipeline = try registry.pipeline(for: "sigmoid_f32")
        self.geluPipeline = try registry.pipeline(for: "gelu_f32")
        self.swigluPipeline = try registry.pipeline(for: "swiglu_f32")
    }

    public func sigmoid(input: [Float], commandQueue: MTLCommandQueue) async throws -> [Float] {
        try await executeElementwise(pipeline: sigmoidPipeline, input: input, commandQueue: commandQueue)
    }

    public func gelu(input: [Float], commandQueue: MTLCommandQueue) async throws -> [Float] {
        try await executeElementwise(pipeline: geluPipeline, input: input, commandQueue: commandQueue)
    }

    public func swiglu(gate: [Float], up: [Float], commandQueue: MTLCommandQueue) async throws -> [Float] {
        precondition(gate.count == up.count)

        let gateBuffer = device.makeBuffer(
            bytes: gate,
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let upBuffer = device.makeBuffer(
            bytes: up,
            length: up.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        var params = ERActivationParams(count: UInt32(gate.count))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ActivationError.encodingFailed
        }

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERActivationParams>.stride, index: 3)

        let gridSize = MTLSize(width: gate.count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(gate.count, swigluPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: gate.count)
        return Array(UnsafeBufferPointer(start: pointer, count: gate.count))
    }

    private func executeElementwise(
        pipeline: MTLComputePipelineState,
        input: [Float],
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        var params = ERActivationParams(count: UInt32(input.count))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ActivationError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERActivationParams>.stride, index: 2)

        let gridSize = MTLSize(width: input.count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(input.count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
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

public enum ActivationError: Error, Sendable {
    case encodingFailed
}
