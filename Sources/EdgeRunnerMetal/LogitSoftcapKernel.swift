import Metal

/// In-place logit softcap kernel: `logits[i] = tanh(logits[i] / cap) * cap`.
///
/// Used by Gemma-style models that apply a final-logit softcap (e.g. Gemma 4 uses cap = 30.0)
/// on the output logits before sampling.
public struct LogitSoftcapKernel: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "logit_softcap_f32")
    }

    /// Encode the softcap dispatch into an existing command buffer (no commit).
    ///
    /// The kernel is in-place: `logits` is both read and written.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        logits: MTLBuffer,
        cap: Float,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LogitSoftcapError.encodingFailed
        }

        var capValue = cap
        var countValue = UInt32(count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBytes(&capValue, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(
            width: min(count, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1, depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Convenience: copy `logits` to the GPU, dispatch the in-place softcap, read back.
    public func run(logits: [Float], cap: Float) throws -> [Float] {
        let byteCount = logits.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: logits,
            length: byteCount,
            options: .storageModeShared
        ) else {
            throw LogitSoftcapError.encodingFailed
        }
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw LogitSoftcapError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            logits: buffer,
            cap: cap,
            count: logits.count
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: logits.count)
        return Array(UnsafeBufferPointer(start: pointer, count: logits.count))
    }
}

public enum LogitSoftcapError: Error, Sendable {
    case encodingFailed
}
