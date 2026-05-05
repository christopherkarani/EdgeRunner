import Metal

/// Parameters mirror the `PLEGatherParams` struct declared in `Shaders/PLE.metal`.
/// Defined in Swift (rather than EdgeRunnerSharedTypes) because this kernel has
/// no other Swift/C consumers — matches the simpler kernels landed recently.
private struct PLEGatherParams {
    var perLayerDim: UInt32
    var numLayers: UInt32
    var numTokens: UInt32
    var rowStrideBytes: UInt32
    var tableByteOffset: UInt64
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
    private let q6KPipeline: MTLComputePipelineState
    private let q6KBlockedPipeline: MTLComputePipelineState
    private let device: MTLDevice
    private static let q8BlockBytes = 34
    private static let q8WeightsPerBlock = 32
    private static let q6KBlockBytes = 210
    private static let q6KWeightsPerBlock = 256

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "ple_gather_q8_0")
        self.q6KPipeline = try registry.pipeline(for: "ple_gather_q6_k")
        self.q6KBlockedPipeline = try registry.pipeline(for: "ple_gather_q6_k_blocked")
    }

    /// Encode PLE row gathering into an existing command buffer.
    ///
    /// The caller owns buffer allocation, command-buffer commit, and synchronization.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        q8TableBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        perLayerDim: Int,
        numLayers: Int,
        numTokens: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int = 0
    ) throws {
        try encodeGather(
            commandBuffer: commandBuffer,
            tableBuffer: q8TableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            pipeline: pipeline,
            perLayerDim: perLayerDim,
            numLayers: numLayers,
            numTokens: numTokens,
            rowStrideBytes: rowStrideBytes,
            tableByteOffset: tableByteOffset,
            weightsPerBlock: Self.q8WeightsPerBlock,
            blockByteCount: Self.q8BlockBytes,
            quantName: "Q8_0"
        )
    }

    /// Encode Q6_K PLE row gathering into an existing command buffer.
    public func encodeQ6K(
        commandBuffer: MTLCommandBuffer,
        q6KTableBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        perLayerDim: Int,
        numLayers: Int,
        numTokens: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int = 0
    ) throws {
        try encodeQ6KBlocked(
            commandBuffer: commandBuffer,
            tableBuffer: q6KTableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            perLayerDim: perLayerDim,
            numLayers: numLayers,
            numTokens: numTokens,
            rowStrideBytes: rowStrideBytes,
            tableByteOffset: tableByteOffset
        )
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
        let rowStrideBytes = (totalElemsPerRow / Self.q8WeightsPerBlock) * Self.q8BlockBytes
        guard q8Table.count % rowStrideBytes == 0 else {
            throw PLEGatherError.invalidShape(
                "q8Table size (\(q8Table.count)) is not a multiple of per-token row stride (\(rowStrideBytes))"
            )
        }
        let vocabSize = q8Table.count / rowStrideBytes
        for (i, tokenId) in tokens.enumerated() {
            guard tokenId >= 0 else {
                throw PLEGatherError.invalidShape(
                    "negative token id at index \(i): \(tokenId)"
                )
            }
            guard Int(tokenId) < vocabSize else {
                throw PLEGatherError.invalidShape(
                    "token id \(tokenId) at index \(i) exceeds vocab size \(vocabSize)"
                )
            }
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

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PLEGatherError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            q8TableBuffer: tableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            perLayerDim: perLayerDim,
            numLayers: numLayers,
            numTokens: numTokens,
            rowStrideBytes: rowStrideBytes
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    /// Test-only convenience for Q6_K PLE gather.
    public func runQ6K(
        q6KTable: [UInt8],
        tokens: [Int32],
        perLayerDim: Int,
        numLayers: Int
    ) throws -> [Float] {
        let totalElemsPerRow = perLayerDim * numLayers
        guard totalElemsPerRow % Self.q6KWeightsPerBlock == 0 else {
            throw PLEGatherError.invalidShape(
                "perLayerDim (\(perLayerDim)) * numLayers (\(numLayers)) = \(totalElemsPerRow) must be a multiple of 256 for Q6_K"
            )
        }
        let rowStrideBytes = (totalElemsPerRow / Self.q6KWeightsPerBlock) * Self.q6KBlockBytes
        guard q6KTable.count % rowStrideBytes == 0 else {
            throw PLEGatherError.invalidShape(
                "q6KTable size (\(q6KTable.count)) is not a multiple of per-token row stride (\(rowStrideBytes))"
            )
        }
        let vocabSize = q6KTable.count / rowStrideBytes
        for (i, tokenId) in tokens.enumerated() {
            guard tokenId >= 0 else {
                throw PLEGatherError.invalidShape("negative token id at index \(i): \(tokenId)")
            }
            guard Int(tokenId) < vocabSize else {
                throw PLEGatherError.invalidShape(
                    "token id \(tokenId) at index \(i) exceeds vocab size \(vocabSize)"
                )
            }
        }
        let numTokens = tokens.count
        guard numTokens > 0 else {
            return []
        }

        guard let commandQueue = device.makeCommandQueue(),
              let tableBuffer = device.makeBuffer(bytes: q6KTable, length: q6KTable.count, options: .storageModeShared),
              let tokenBuffer = device.makeBuffer(
                bytes: tokens,
                length: tokens.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
              )
        else {
            throw PLEGatherError.encodingFailed
        }
        let outputCount = numTokens * totalElemsPerRow
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEGatherError.encodingFailed
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PLEGatherError.encodingFailed
        }

        try encodeQ6K(
            commandBuffer: commandBuffer,
            q6KTableBuffer: tableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            perLayerDim: perLayerDim,
            numLayers: numLayers,
            numTokens: numTokens,
            rowStrideBytes: rowStrideBytes
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    private func encodeGather(
        commandBuffer: MTLCommandBuffer,
        tableBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        pipeline: MTLComputePipelineState,
        perLayerDim: Int,
        numLayers: Int,
        numTokens: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int,
        weightsPerBlock: Int,
        blockByteCount: Int,
        quantName: String
    ) throws {
        guard perLayerDim > 0, numLayers > 0, numTokens >= 0, tableByteOffset >= 0 else {
            throw PLEGatherError.invalidShape("perLayerDim and numLayers must be positive; numTokens and tableByteOffset must be non-negative")
        }

        let totalElemsPerRow = perLayerDim * numLayers
        guard totalElemsPerRow % weightsPerBlock == 0 else {
            throw PLEGatherError.invalidShape(
                "perLayerDim (\(perLayerDim)) * numLayers (\(numLayers)) = \(totalElemsPerRow) must be a multiple of \(weightsPerBlock) for \(quantName)"
            )
        }

        let minimumRowStride = (totalElemsPerRow / weightsPerBlock) * blockByteCount
        guard rowStrideBytes >= minimumRowStride else {
            throw PLEGatherError.invalidShape(
                "rowStrideBytes (\(rowStrideBytes)) must be at least the packed \(quantName) row size (\(minimumRowStride))"
            )
        }

        let outputCount = numTokens * totalElemsPerRow
        guard tokenBuffer.length >= numTokens * MemoryLayout<Int32>.stride else {
            throw PLEGatherError.invalidShape("tokenBuffer is too small for \(numTokens) Int32 token ids")
        }
        guard outputBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLEGatherError.invalidShape("outputBuffer is too small for \(outputCount) Float values")
        }
        guard tableBuffer.length >= tableByteOffset + rowStrideBytes || numTokens == 0 else {
            throw PLEGatherError.invalidShape("tableBuffer is smaller than one PLE table row")
        }
        guard outputCount > 0 else {
            return
        }

        var params = PLEGatherParams(
            perLayerDim: UInt32(perLayerDim),
            numLayers: UInt32(numLayers),
            numTokens: UInt32(numTokens),
            rowStrideBytes: UInt32(rowStrideBytes),
            tableByteOffset: UInt64(tableByteOffset)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
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
    }

    private func encodeQ6KBlocked(
        commandBuffer: MTLCommandBuffer,
        tableBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        perLayerDim: Int,
        numLayers: Int,
        numTokens: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int
    ) throws {
        guard perLayerDim > 0, numLayers > 0, numTokens >= 0, tableByteOffset >= 0 else {
            throw PLEGatherError.invalidShape("perLayerDim and numLayers must be positive; numTokens and tableByteOffset must be non-negative")
        }

        let totalElemsPerRow = perLayerDim * numLayers
        guard totalElemsPerRow % Self.q6KWeightsPerBlock == 0 else {
            throw PLEGatherError.invalidShape(
                "perLayerDim (\(perLayerDim)) * numLayers (\(numLayers)) = \(totalElemsPerRow) must be a multiple of \(Self.q6KWeightsPerBlock) for Q6_K"
            )
        }

        let blocksPerRow = totalElemsPerRow / Self.q6KWeightsPerBlock
        let minimumRowStride = blocksPerRow * Self.q6KBlockBytes
        guard rowStrideBytes >= minimumRowStride else {
            throw PLEGatherError.invalidShape(
                "rowStrideBytes (\(rowStrideBytes)) must be at least the packed Q6_K row size (\(minimumRowStride))"
            )
        }

        let outputCount = numTokens * totalElemsPerRow
        guard tokenBuffer.length >= numTokens * MemoryLayout<Int32>.stride else {
            throw PLEGatherError.invalidShape("tokenBuffer is too small for \(numTokens) Int32 token ids")
        }
        guard outputBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLEGatherError.invalidShape("outputBuffer is too small for \(outputCount) Float values")
        }
        guard tableBuffer.length >= tableByteOffset + rowStrideBytes || numTokens == 0 else {
            throw PLEGatherError.invalidShape("tableBuffer is smaller than one PLE table row")
        }
        guard outputCount > 0 else {
            return
        }

        var params = PLEGatherParams(
            perLayerDim: UInt32(perLayerDim),
            numLayers: UInt32(numLayers),
            numTokens: UInt32(numTokens),
            rowStrideBytes: UInt32(rowStrideBytes),
            tableByteOffset: UInt64(tableByteOffset)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLEGatherError.encodingFailed
        }

        encoder.setComputePipelineState(q6KBlockedPipeline)
        encoder.setBuffer(tableBuffer, offset: 0, index: 0)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<PLEGatherParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: blocksPerRow, height: numTokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.q6KWeightsPerBlock, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}

public enum PLEGatherError: Error, Sendable {
    case encodingFailed
    case invalidShape(String)
}
