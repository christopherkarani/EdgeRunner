import Metal
import EdgeRunnerSharedTypes

/// Fused GeGLU activation kernel for Gemma 4 (E4B).
///
/// Computes `y[i] = gelu_tanh(gate[i]) * up[i]` where
/// `gelu_tanh(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
/// This is the PyTorch `approximate="tanh"` GELU variant used by Gemma's GeGLU MLP.
public struct GeGLUKernel: Sendable {
    /// Parameters struct mirroring `GeGLUParams` in `Shaders/GeGLU.metal`.
    /// Declared locally because the shared-types header exposes `ERActivationParams`
    /// with the same layout but a different name; we avoid cross-kernel coupling here.
    private struct Params {
        var count: UInt32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw GeGLUKernelError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "gelu_tanh_mul_f32")
    }

    /// Synchronous run for tests and one-shot evaluation.
    /// - Parameters:
    ///   - gate: gate projection values
    ///   - up: up projection values (must match `gate.count`)
    /// - Returns: element-wise `gelu_tanh(gate) * up`
    public func run(gate: [Float], up: [Float]) throws -> [Float] {
        precondition(gate.count == up.count, "GeGLU: gate and up length mismatch")
        let count = gate.count
        guard count > 0 else { return [] }

        let stride = MemoryLayout<Float>.stride
        guard
            let gateBuffer = device.makeBuffer(bytes: gate, length: count * stride, options: .storageModeShared),
            let upBuffer = device.makeBuffer(bytes: up, length: count * stride, options: .storageModeShared),
            let outputBuffer = device.makeBuffer(length: count * stride, options: .storageModeShared)
        else {
            throw GeGLUKernelError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GeGLUKernelError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
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

    /// Encode GeGLU into an existing command buffer so callers can fuse it
    /// after gate/up projection dispatches without an intermediate CPU readback.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        count: Int
    ) throws {
        guard count >= 0,
              gateBuffer.length >= count * MemoryLayout<Float>.stride,
              upBuffer.length >= count * MemoryLayout<Float>.stride,
              outputBuffer.length >= count * MemoryLayout<Float>.stride else {
            throw GeGLUKernelError.bufferAllocationFailed
        }
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GeGLUKernelError.encodingFailed
        }

        var params = Params(count: UInt32(count))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupWidth = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}

public enum GeGLUKernelError: Error, Sendable {
    case commandQueueCreationFailed
    case bufferAllocationFailed
    case encodingFailed
}
