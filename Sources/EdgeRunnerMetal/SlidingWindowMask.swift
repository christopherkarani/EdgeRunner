import Metal

/// Generates additive sliding-window causal attention masks on-device.
///
/// Used by Gemma 4's interleaved attention pattern: sliding-window
/// (SWA, window=512) for most layers and global full-causal attention
/// for others. Global mode is expressed by passing `window >= seqLen`.
///
/// The produced mask is a row-major `[seqLen, seqLen]` `Float` array where
/// `0.0` means "attend" and `-inf` means "mask". Intended to be added to
/// raw attention logits prior to softmax.
public struct SlidingWindowMask: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "sliding_causal_mask_f32")
    }

    /// Build an additive SWA causal mask of shape `[seqLen, seqLen]`.
    ///
    /// - Parameters:
    ///   - seqLen: Number of query/key positions. Must be > 0.
    ///   - window: Sliding-window size. Pass `>= seqLen` for global causal.
    /// - Returns: Row-major `[seqLen * seqLen]` additive mask.
    public func build(seqLen: Int, window: Int) throws -> [Float] {
        precondition(seqLen > 0, "seqLen must be positive")
        precondition(window > 0, "window must be positive")

        guard let commandQueue = device.makeCommandQueue() else {
            throw SlidingWindowMaskError.commandQueueCreationFailed
        }
        return try build(seqLen: seqLen, window: window, commandQueue: commandQueue)
    }

    /// Build an additive SWA causal mask using a caller-provided command queue.
    public func build(seqLen: Int, window: Int, commandQueue: MTLCommandQueue) throws -> [Float] {
        precondition(seqLen > 0, "seqLen must be positive")
        precondition(window > 0, "window must be positive")

        let elementCount = seqLen * seqLen
        guard let outputBuffer = device.makeBuffer(
            length: elementCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw SlidingWindowMaskError.bufferAllocationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SlidingWindowMaskError.encodingFailed
        }

        try encode(
            encoder: encoder,
            outputBuffer: outputBuffer,
            seqLen: seqLen,
            window: window
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        return Array(UnsafeBufferPointer(start: pointer, count: elementCount))
    }

    /// Encode the SWA mask dispatch into an existing compute encoder.
    /// Caller is responsible for `endEncoding`, committing, and awaiting.
    public func encode(
        encoder: MTLComputeCommandEncoder,
        outputBuffer: MTLBuffer,
        seqLen: Int,
        window: Int
    ) throws {
        var params = SlidingMaskParams(
            seqLen: UInt32(seqLen),
            window: UInt32(window)
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<SlidingMaskParams>.stride, index: 1)

        let gridSize = MTLSize(width: seqLen, height: seqLen, depth: 1)
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let tileWidth = min(16, seqLen)
        let tileHeight = min(16, seqLen)
        let tgWidth = min(tileWidth, maxThreads)
        let tgHeight = min(tileHeight, max(1, maxThreads / tgWidth))
        let threadgroupSize = MTLSize(
            width: tgWidth,
            height: tgHeight,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

public enum SlidingWindowMaskError: Error, Sendable {
    case commandQueueCreationFailed
    case bufferAllocationFailed
    case encodingFailed
}

/// Shader parameter layout. Mirrors `ERSlidingMaskParams` defined in
/// `Sources/EdgeRunnerMetal/Shaders/SlidingCausalMask.metal`.
private struct SlidingMaskParams {
    let seqLen: UInt32
    let window: UInt32
}
