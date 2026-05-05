import Metal

private struct PLEGateParams {
    var count: UInt32
}

public struct PLEGateKernel: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "ple_gate_gelu_mul_f32")
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        gateBuffer: MTLBuffer,
        pleBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        count: Int,
        gateBufferOffset: Int = 0,
        pleBufferOffset: Int = 0,
        outputBufferOffset: Int = 0
    ) throws {
        guard count >= 0,
              gateBufferOffset >= 0,
              pleBufferOffset >= 0,
              outputBufferOffset >= 0,
              gateBuffer.length >= gateBufferOffset + count * MemoryLayout<Float>.stride,
              pleBuffer.length >= pleBufferOffset + count * MemoryLayout<Float>.stride,
              outputBuffer.length >= outputBufferOffset + count * MemoryLayout<Float>.stride else {
            throw PLEGateKernelError.invalidShape
        }
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLEGateKernelError.encodingFailed
        }

        var params = PLEGateParams(count: UInt32(count))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: gateBufferOffset, index: 0)
        encoder.setBuffer(pleBuffer, offset: pleBufferOffset, index: 1)
        encoder.setBuffer(outputBuffer, offset: outputBufferOffset, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<PLEGateParams>.stride, index: 3)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(
            width: min(count, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    public func run(gate: [Float], ple: [Float]) throws -> [Float] {
        guard gate.count == ple.count else {
            throw PLEGateKernelError.invalidShape
        }
        let count = gate.count
        guard count > 0 else { return [] }
        guard let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let gateBuffer = device.makeBuffer(
                bytes: gate,
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let pleBuffer = device.makeBuffer(
                bytes: ple,
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ) else {
            throw PLEGateKernelError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            pleBuffer: pleBuffer,
            outputBuffer: outputBuffer,
            count: count
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

public enum PLEGateKernelError: Error, Sendable {
    case encodingFailed
    case invalidShape
}
