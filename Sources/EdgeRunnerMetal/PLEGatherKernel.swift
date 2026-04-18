import Metal

/// Parameters mirror the `PLEGatherParams` struct declared in `Shaders/PLE.metal`.
/// Defined in Swift (rather than EdgeRunnerSharedTypes) because this kernel has
/// no other Swift/C consumers — matches the simpler kernels landed recently.
private struct PLEGatherParams {
    var perLayerDim: UInt32
    var numLayers: UInt32
    var numTokens: UInt32
    var rowStrideBytes: UInt32
}

/// Single-row Q8_0 gather kernel for Gemma-style Per-Layer Embeddings (PLE).
///
/// For each token in the batch and each layer `ℓ ∈ 0..<L`, gathers the slice
/// `per_layer_token_embd[tok_id, ℓ·P : (ℓ+1)·P]`, dequantizes Q8_0 (32-element
/// blocks = 2-byte f16 scale + 32 int8 quants = 34 bytes), and scales by
/// `√P`.
///
/// Output layout: `[numTokens, L, P]` as `Float`.
public struct PLEGatherKernel: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "ple_gather_q8_0")
    }

    /// Test-only convenience — synchronous, creates its own command queue.
    ///
    /// The integration path should use an `encode(...)` variant that takes
    /// pre-allocated `MTLBuffer`s and a `MTLCommandBuffer`.
    public func run(
        q8Table: [UInt8],
        tokens: [Int32],
        perLayerDim: Int,
        numLayers: Int
    ) throws -> [Float] {
        let totalElemsPerRow = perLayerDim * numLayers
        guard totalElemsPerRow % 32 == 0 else {
            throw PLEGatherError.invalidShape(
                "perLayerDim (\(perLayerDim)) * numLayers (\(numLayers)) = \(totalElemsPerRow) must be a multiple of 32 for Q8_0"
            )
        }
        let rowStrideBytes = (totalElemsPerRow / 32) * 34
        guard q8Table.count % rowStrideBytes == 0 else {
            throw PLEGatherError.invalidShape(
                "q8Table size (\(q8Table.count)) is not a multiple of per-token row stride (\(rowStrideBytes))"
            )
        }
        let numTokens = tokens.count
        guard numTokens > 0 else {
            return []
        }

        guard let commandQueue = device.makeCommandQueue() else {
            throw PLEGatherError.encodingFailed
        }

        guard let tableBuffer = device.makeBuffer(
            bytes: q8Table,
            length: q8Table.count,
            options: .storageModeShared
        ) else {
            throw PLEGatherError.encodingFailed
        }
        guard let tokenBuffer = device.makeBuffer(
            bytes: tokens,
            length: tokens.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else {
            throw PLEGatherError.encodingFailed
        }
        let outputCount = numTokens * totalElemsPerRow
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEGatherError.encodingFailed
        }

        var params = PLEGatherParams(
            perLayerDim: UInt32(perLayerDim),
            numLayers: UInt32(numLayers),
            numTokens: UInt32(numTokens),
            rowStrideBytes: UInt32(rowStrideBytes)
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLEGatherError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tableBuffer, offset: 0, index: 0)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<PLEGatherParams>.stride, index: 3)

        let gridSize = MTLSize(width: totalElemsPerRow, height: numTokens, depth: 1)
        let tgWidth = min(totalElemsPerRow, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }
}

public enum PLEGatherError: Error, Sendable {
    case encodingFailed
    case invalidShape(String)
}
