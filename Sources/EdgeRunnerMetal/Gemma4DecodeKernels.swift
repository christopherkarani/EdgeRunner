import Foundation
import Metal

private struct ERGemma4RMSNormParams {
    var rows: UInt32
    var cols: UInt32
    var eps: Float
}

private struct ERGemma4ResidualRMSNormParams {
    var count: UInt32
    var eps: Float
}

private struct ERGemma4EmbeddingParams {
    var rowWidth: UInt32
    var tokenCount: UInt32
    var rowStrideBytes: UInt32
    var tableByteOffset: UInt64
    var scale: Float
}

private struct ERGemma4DecodeGQAParams {
    var numHeads: UInt32
    var numKVHeads: UInt32
    var groupSize: UInt32
    var headDim: UInt32
    var kvStart: UInt32
    var kvCount: UInt32
    var kvCapacity: UInt32
    var scale: Float
}

public struct Gemma4DecodeKernels: Sendable {
    private static let useFastWindowedGQA =
        ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_FAST_GQA"] != "0"

    private let device: MTLDevice
    private let rmsNormPipeline: MTLComputePipelineState
    private let residualRMSNormPipeline: MTLComputePipelineState
    private let storeF32ToF16Pipeline: MTLComputePipelineState
    private let mulScalarPipeline: MTLComputePipelineState
    private let q6KEmbeddingGatherPipeline: MTLComputePipelineState
    private let decodeGQAF16KVWindowedPipeline: MTLComputePipelineState
    private let decodeGQAF16KVWindowedFastPipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.rmsNormPipeline = try registry.pipeline(for: "gemma4_rmsnorm_f32")
        self.residualRMSNormPipeline = try registry.pipeline(for: "gemma4_residual_rmsnorm_add_f32")
        self.storeF32ToF16Pipeline = try registry.pipeline(for: "gemma4_store_f32_to_f16")
        self.mulScalarPipeline = try registry.pipeline(for: "gemma4_mul_scalar_f32")
        self.q6KEmbeddingGatherPipeline = try registry.pipeline(for: "gemma4_gather_token_embedding_q6_k")
        self.decodeGQAF16KVWindowedPipeline = try registry.pipeline(for: "gemma4_decode_gqa_f16kv_windowed")
        self.decodeGQAF16KVWindowedFastPipeline = try registry.pipeline(for: "gemma4_decode_gqa_f16kv_windowed_fast")
    }

    public func runRMSNorm(
        input: [Float],
        weight: [Float],
        rows: Int,
        cols: Int,
        eps: Float,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols)
        precondition(weight.count == cols)

        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBuffer = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            outputBuffer: outputBuffer,
            rows: rows,
            cols: cols,
            eps: eps
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: pointer, count: input.count))
    }

    public func encodeRMSNorm(
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        rows: Int,
        cols: Int,
        eps: Float
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var params = ERGemma4RMSNormParams(rows: UInt32(rows), cols: UInt32(cols), eps: eps)
        encoder.setComputePipelineState(rmsNormPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGemma4RMSNormParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func runResidualRMSNormAdd(
        residual: [Float],
        input: [Float],
        weight: [Float],
        eps: Float,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(residual.count == input.count)
        precondition(weight.count == input.count)
        guard let residualBuffer = device.makeBuffer(
            bytes: residual,
            length: residual.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBuffer = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeResidualRMSNormAdd(
            commandBuffer: commandBuffer,
            residualBuffer: residualBuffer,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            outputBuffer: outputBuffer,
            count: input.count,
            eps: eps
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: pointer, count: input.count))
    }

    public func encodeResidualRMSNormAdd(
        commandBuffer: MTLCommandBuffer,
        residualBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        count: Int,
        eps: Float
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var params = ERGemma4ResidualRMSNormParams(count: UInt32(count), eps: eps)
        encoder.setComputePipelineState(residualRMSNormPipeline)
        encoder.setBuffer(residualBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGemma4ResidualRMSNormParams>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func runStoreF32ToF16(values: [Float], outputOffset: Int = 0) async throws -> [Float16] {
        guard let inputBuffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: outputOffset + values.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let commandQueue = device.makeCommandQueue(),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeStoreF32ToF16(
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            outputOffset: outputOffset,
            count: values.count
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let capacity = outputOffset / MemoryLayout<Float16>.stride + values.count
        let pointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: capacity)
        let start = outputOffset / MemoryLayout<Float16>.stride
        return Array(UnsafeBufferPointer(start: pointer.advanced(by: start), count: values.count))
    }

    public func encodeStoreF32ToF16(
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var countValue = UInt32(count)
        encoder.setComputePipelineState(storeF32ToF16Pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: outputOffset, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, storeF32ToF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func runMulScalar(values: [Float], scale: Float) throws -> [Float] {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandQueue = device.makeCommandQueue(),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeMulScalar(
            commandBuffer: commandBuffer,
            valuesBuffer: buffer,
            scale: scale,
            count: values.count
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
        return Array(UnsafeBufferPointer(start: pointer, count: values.count))
    }

    public func encodeMulScalar(
        commandBuffer: MTLCommandBuffer,
        valuesBuffer: MTLBuffer,
        scale: Float,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var scaleValue = scale
        var countValue = UInt32(count)
        encoder.setComputePipelineState(mulScalarPipeline)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 0)
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, mulScalarPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func runGatherQ6KTokenEmbedding(
        table: [UInt8],
        tokenIDs: [Int],
        rowWidth: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int,
        scale: Float,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        guard let tableBuffer = device.makeBuffer(
            bytes: table,
            length: table.count,
            options: .storageModeShared
        ) else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        return try await runGatherQ6KTokenEmbedding(
            tableBuffer: tableBuffer,
            tokenIDs: tokenIDs,
            rowWidth: rowWidth,
            rowStrideBytes: rowStrideBytes,
            tableByteOffset: tableByteOffset,
            scale: scale,
            commandQueue: commandQueue
        )
    }

    public func runGatherQ6KTokenEmbedding(
        tableBuffer: MTLBuffer,
        tokenIDs: [Int],
        rowWidth: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int,
        scale: Float,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let tokenIDs32 = try tokenIDs.map { tokenID -> Int32 in
            guard let value = Int32(exactly: tokenID) else {
                throw Gemma4DecodeKernelError.encodingFailed
            }
            return value
        }
        guard
        let tokenBuffer = device.makeBuffer(
            bytes: tokenIDs32,
            length: tokenIDs32.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: tokenIDs.count * rowWidth * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeGatherQ6KTokenEmbedding(
            commandBuffer: commandBuffer,
            tableBuffer: tableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            tokenCount: tokenIDs.count,
            rowWidth: rowWidth,
            rowStrideBytes: rowStrideBytes,
            tableByteOffset: tableByteOffset,
            scale: scale
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let count = tokenIDs.count * rowWidth
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    public func encodeGatherQ6KTokenEmbedding(
        commandBuffer: MTLCommandBuffer,
        tableBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        tokenCount: Int,
        rowWidth: Int,
        rowStrideBytes: Int,
        tableByteOffset: Int,
        scale: Float
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var params = ERGemma4EmbeddingParams(
            rowWidth: UInt32(rowWidth),
            tokenCount: UInt32(tokenCount),
            rowStrideBytes: UInt32(rowStrideBytes),
            tableByteOffset: UInt64(tableByteOffset),
            scale: scale
        )
        encoder.setComputePipelineState(q6KEmbeddingGatherPipeline)
        encoder.setBuffer(tableBuffer, offset: 0, index: 0)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGemma4EmbeddingParams>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: rowWidth, height: tokenCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(rowWidth, q6KEmbeddingGatherPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func runDecodeGQAF16KVWindowed(
        q: [Float],
        k: [Float16],
        v: [Float16],
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float? = nil,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(q.count == numHeads * headDim)
        precondition(k.count == kvCapacity * numKVHeads * headDim)
        precondition(v.count == kvCapacity * numKVHeads * headDim)
        guard let qBuffer = device.makeBuffer(
            bytes: q,
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let kBuffer = device.makeBuffer(
            bytes: k,
            length: k.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let vBuffer = device.makeBuffer(
            bytes: v,
            length: v.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeDecodeGQAF16KVWindowedBestAvailable(
            commandBuffer: commandBuffer,
            qBuffer: qBuffer,
            keyCacheBuffer: kBuffer,
            valueCacheBuffer: vBuffer,
            outputBuffer: outputBuffer,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            kvCapacity: kvCapacity,
            attentionScale: attentionScale
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: q.count)
        return Array(UnsafeBufferPointer(start: pointer, count: q.count))
    }

    public func runDecodeGQAF16KVWindowed(
        q: [Float],
        keyCacheBuffer: MTLBuffer,
        valueCacheBuffer: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float? = nil,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(q.count == numHeads * headDim)
        guard let qBuffer = device.makeBuffer(
            bytes: q,
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }

        try encodeDecodeGQAF16KVWindowedBestAvailable(
            commandBuffer: commandBuffer,
            qBuffer: qBuffer,
            keyCacheBuffer: keyCacheBuffer,
            valueCacheBuffer: valueCacheBuffer,
            outputBuffer: outputBuffer,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            kvCapacity: kvCapacity,
            attentionScale: attentionScale
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: q.count)
        return Array(UnsafeBufferPointer(start: pointer, count: q.count))
    }

    public func encodeDecodeGQAF16KVWindowed(
        commandBuffer: MTLCommandBuffer,
        qBuffer: MTLBuffer,
        keyCacheBuffer: MTLBuffer,
        valueCacheBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float? = nil
    ) throws {
        precondition(numHeads % numKVHeads == 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var params = ERGemma4DecodeGQAParams(
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(numHeads / numKVHeads),
            headDim: UInt32(headDim),
            kvStart: UInt32(kvStart),
            kvCount: UInt32(kvCount),
            kvCapacity: UInt32(kvCapacity),
            scale: attentionScale ?? (1.0 / sqrt(Float(headDim)))
        )
        encoder.setComputePipelineState(decodeGQAF16KVWindowedPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyCacheBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueCacheBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGemma4DecodeGQAParams>.stride, index: 4)
        let outputCount = numHeads * headDim
        encoder.dispatchThreads(
            MTLSize(width: outputCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(outputCount, decodeGQAF16KVWindowedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeDecodeGQAF16KVWindowedBestAvailable(
        commandBuffer: MTLCommandBuffer,
        qBuffer: MTLBuffer,
        keyCacheBuffer: MTLBuffer,
        valueCacheBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float? = nil
    ) throws {
        try validateCacheBuffers(
            keyCacheBuffer: keyCacheBuffer,
            valueCacheBuffer: valueCacheBuffer,
            outputBuffer: outputBuffer,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvCapacity: kvCapacity
        )
        if Self.useFastWindowedGQA,
           headDim <= 512,
           kvCount <= 512,
           decodeGQAF16KVWindowedFastPipeline.maxTotalThreadsPerThreadgroup >= 512 {
            try encodeDecodeGQAF16KVWindowedFast(
                commandBuffer: commandBuffer,
                qBuffer: qBuffer,
                keyCacheBuffer: keyCacheBuffer,
                valueCacheBuffer: valueCacheBuffer,
                outputBuffer: outputBuffer,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                kvStart: kvStart,
                kvCount: kvCount,
                kvCapacity: kvCapacity,
                attentionScale: attentionScale
            )
            return
        }

        try encodeDecodeGQAF16KVWindowed(
            commandBuffer: commandBuffer,
            qBuffer: qBuffer,
            keyCacheBuffer: keyCacheBuffer,
            valueCacheBuffer: valueCacheBuffer,
            outputBuffer: outputBuffer,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            kvCapacity: kvCapacity,
            attentionScale: attentionScale
        )
    }

    private func validateCacheBuffers(
        keyCacheBuffer: MTLBuffer,
        valueCacheBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvCapacity: Int
    ) throws {
        let expectedCacheBytes = kvCapacity * numKVHeads * headDim * MemoryLayout<Float16>.stride
        let expectedOutputBytes = numHeads * headDim * MemoryLayout<Float>.stride
        guard keyCacheBuffer.length >= expectedCacheBytes,
              valueCacheBuffer.length >= expectedCacheBytes,
              outputBuffer.length >= expectedOutputBytes else {
            throw Gemma4DecodeKernelError.invalidBufferShape
        }
    }

    public func encodeDecodeGQAF16KVWindowedFast(
        commandBuffer: MTLCommandBuffer,
        qBuffer: MTLBuffer,
        keyCacheBuffer: MTLBuffer,
        valueCacheBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float? = nil
    ) throws {
        precondition(numHeads % numKVHeads == 0)
        guard headDim <= 512,
              kvCount <= 512,
              decodeGQAF16KVWindowedFastPipeline.maxTotalThreadsPerThreadgroup >= 512 else {
            try encodeDecodeGQAF16KVWindowed(
                commandBuffer: commandBuffer,
                qBuffer: qBuffer,
                keyCacheBuffer: keyCacheBuffer,
                valueCacheBuffer: valueCacheBuffer,
                outputBuffer: outputBuffer,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                kvStart: kvStart,
                kvCount: kvCount,
                kvCapacity: kvCapacity,
                attentionScale: attentionScale
            )
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Gemma4DecodeKernelError.encodingFailed
        }
        var params = ERGemma4DecodeGQAParams(
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(numHeads / numKVHeads),
            headDim: UInt32(headDim),
            kvStart: UInt32(kvStart),
            kvCount: UInt32(kvCount),
            kvCapacity: UInt32(kvCapacity),
            scale: attentionScale ?? (1.0 / sqrt(Float(headDim)))
        )
        encoder.setComputePipelineState(decodeGQAF16KVWindowedFastPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyCacheBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueCacheBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGemma4DecodeGQAParams>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}

public enum Gemma4DecodeKernelError: Error, Sendable {
    case encodingFailed
    case invalidBufferShape
}
