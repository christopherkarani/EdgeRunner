import Metal

/// Parameters mirror the `PLESideChannelParams` struct declared in `Shaders/PLE.metal`.
private struct PLESideChannelParams {
    var hidden: UInt32
    var batchSeq: UInt32
    var rmsEps: Float
}

/// PLE side-channel finalize kernel for Gemma-style Per-Layer Embeddings.
///
/// Computes `hidden += RMSNorm(projection, postNormWeight)` with Gemma 4's
/// direct RMSNorm weight convention.
public struct PLESideChannelKernel: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "ple_side_channel_finalize")
    }

    /// Encode side-channel finalization into an existing command buffer.
    ///
    /// The caller owns buffer allocation, command-buffer commit, and synchronization.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        hiddenBuffer: MTLBuffer,
        projectionBuffer: MTLBuffer,
        postNormWeightBuffer: MTLBuffer,
        hiddenSize: Int,
        batchSeq: Int,
        rmsEps: Float = 1e-6
    ) throws {
        guard hiddenSize > 0, batchSeq >= 0 else {
            throw PLESideChannelError.invalidShape("hiddenSize must be positive; batchSeq must be non-negative")
        }

        let outputCount = hiddenSize * batchSeq
        guard hiddenBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLESideChannelError.invalidShape("hiddenBuffer is too small for \(outputCount) Float values")
        }
        guard projectionBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLESideChannelError.invalidShape("projectionBuffer is too small for \(outputCount) Float values")
        }
        guard postNormWeightBuffer.length >= hiddenSize * MemoryLayout<Float>.stride else {
            throw PLESideChannelError.invalidShape("postNormWeightBuffer is too small for \(hiddenSize) Float values")
        }
        guard outputCount > 0 else {
            return
        }

        var params = PLESideChannelParams(
            hidden: UInt32(hiddenSize),
            batchSeq: UInt32(batchSeq),
            rmsEps: rmsEps
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLESideChannelError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(hiddenBuffer, offset: 0, index: 0)
        encoder.setBuffer(projectionBuffer, offset: 0, index: 1)
        encoder.setBuffer(postNormWeightBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<PLESideChannelParams>.stride, index: 3)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: batchSeq, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroupSize
        )
        encoder.endEncoding()
    }

    /// Test convenience: copy inputs to the GPU, finalize in place, and read back hidden.
    public func run(
        hidden: [Float],
        projection: [Float],
        postNormWeight: [Float],
        hiddenSize: Int,
        batchSeq: Int,
        rmsEps: Float = 1e-6
    ) throws -> [Float] {
        let outputCount = hiddenSize * batchSeq
        guard hidden.count == outputCount else {
            throw PLESideChannelError.invalidShape(
                "hidden must have batchSeq*hiddenSize = \(outputCount) elements, got \(hidden.count)"
            )
        }
        guard projection.count == outputCount else {
            throw PLESideChannelError.invalidShape(
                "projection must have batchSeq*hiddenSize = \(outputCount) elements, got \(projection.count)"
            )
        }
        guard postNormWeight.count == hiddenSize else {
            throw PLESideChannelError.invalidShape(
                "postNormWeight must have hiddenSize = \(hiddenSize) elements, got \(postNormWeight.count)"
            )
        }
        guard outputCount > 0 else {
            return []
        }

        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let hiddenBuffer = device.makeBuffer(
                bytes: hidden,
                length: hidden.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let projectionBuffer = device.makeBuffer(
                bytes: projection,
                length: projection.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let postNormWeightBuffer = device.makeBuffer(
                bytes: postNormWeight,
                length: postNormWeight.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              )
        else {
            throw PLESideChannelError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            hiddenBuffer: hiddenBuffer,
            projectionBuffer: projectionBuffer,
            postNormWeightBuffer: postNormWeightBuffer,
            hiddenSize: hiddenSize,
            batchSeq: batchSeq,
            rmsEps: rmsEps
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }
}

public enum PLESideChannelError: Error, Sendable {
    case encodingFailed
    case invalidShape(String)
}
