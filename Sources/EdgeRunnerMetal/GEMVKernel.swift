import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the row-parallel GEMV Metal kernel.
/// Optimized for autoregressive decoding: y[M] = A[M,K] * x[K].
public final class GEMVKernel: Sendable {
    private struct KQuantGEMVParams {
        var rows: UInt32
        var cols: UInt32
        var blocksPerRow: UInt32
    }

    private struct Q4KTripleGEMVParams {
        var rowsA: UInt32
        var rowsB: UInt32
        var rowsC: UInt32
        var cols: UInt32
        var blocksPerRow: UInt32
    }

    public let pipelineF32: MTLComputePipelineState // exposed for fused pipeline encoding
    private let pipelineF16: MTLComputePipelineState
    private let pipelineBF16F32: MTLComputePipelineState
    private let pipelineBF16F32Batched: MTLComputePipelineState
    private let pipelineQ4KGEMV: MTLComputePipelineState
    private let pipelineQ4KGEMVBatched: MTLComputePipelineState
    private let pipelineQ4KPackedGEMV: MTLComputePipelineState
    private let pipelineQ4KPackedFourRowGEMV: MTLComputePipelineState
    private let pipelineQ4KLlamaStyleGEMV: MTLComputePipelineState
    private let pipelineQ4KLlamaStyleDualGEMV: MTLComputePipelineState
    private let pipelineQ4K2RowGEMV: MTLComputePipelineState
    private let pipelineQ4KDualGEMV: MTLComputePipelineState
    private let pipelineQ4KDualGeGLUGEMV: MTLComputePipelineState
    private let pipelineQ4KTripleGEMV: MTLComputePipelineState
    private let pipelineQ4KTriplePackedGEMV: MTLComputePipelineState
    private let pipelineQ6KGEMV: MTLComputePipelineState
    private let pipelineQ6KPackedGEMV: MTLComputePipelineState
    private let pipelineQ6KPackedTop1GEMV: MTLComputePipelineState
    private let pipelineQ6KTop1Reduce: MTLComputePipelineState
    private let device: MTLDevice

    /// Expose pipeline for external fused command buffer encoding
    public var f32Pipeline: MTLComputePipelineState { pipelineF32 }
    public var bf16F32Pipeline: MTLComputePipelineState { pipelineBF16F32 }

    private static let threadsPerRow = 256
    private static let packedThreadsPerRow = 128
    private static let q6PackedThreadsPerRow = 64

    public static func q6KTop1PartialCount(rows: Int) -> Int {
        (rows + 3) / 4
    }

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "gemv_f32")
        self.pipelineF16 = try registry.pipeline(for: "gemv_f16")
        self.pipelineBF16F32 = try registry.pipeline(for: "gemv_bf16_f32")
        self.pipelineBF16F32Batched = try registry.pipeline(for: "gemv_bf16_f32_batched")
        self.pipelineQ4KGEMV = try registry.pipeline(for: "q4_k_gemv_f32")
        self.pipelineQ4KGEMVBatched = try registry.pipeline(for: "q4_k_gemv_f32_batched")
        self.pipelineQ4KPackedGEMV = try registry.pipeline(for: "q4_k_gemv_packed_f32")
        self.pipelineQ4KPackedFourRowGEMV = try registry.pipeline(for: "q4_k_gemv_packed_4row_f32")
        self.pipelineQ4KLlamaStyleGEMV = try registry.pipeline(for: "q4_k_gemv_llama_style_f32")
        self.pipelineQ4KLlamaStyleDualGEMV = try registry.pipeline(for: "q4_k_gemv_llama_style_dual_f32")
        self.pipelineQ4K2RowGEMV = try registry.pipeline(for: "q4_k_gemv_2row_f32")
        self.pipelineQ4KDualGEMV = try registry.pipeline(for: "q4_k_gemv_dual_f32")
        self.pipelineQ4KDualGeGLUGEMV = try registry.pipeline(for: "q4_k_gemv_dual_geglu_f32")
        self.pipelineQ4KTripleGEMV = try registry.pipeline(for: "q4_k_gemv_three_f32")
        self.pipelineQ4KTriplePackedGEMV = try registry.pipeline(for: "q4_k_gemv_three_packed_f32")
        self.pipelineQ6KGEMV = try registry.pipeline(for: "q6_k_gemv_f32")
        self.pipelineQ6KPackedGEMV = try registry.pipeline(for: "q6_k_gemv_packed_f32")
        self.pipelineQ6KPackedTop1GEMV = try registry.pipeline(for: "q6_k_gemv_packed_4row_top1_f32")
        self.pipelineQ6KTop1Reduce = try registry.pipeline(for: "q6_k_top1_reduce")
    }

    /// Execute Float32 GEMV: y[M] = A[M,K] * x[K].
    public func execute(
        a: [Float], x: [Float],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateInputShape(matrixCount: a.count, vectorCount: x.count, M: M, K: K)

        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMVParams(
            M: UInt32(M), K: UInt32(K), lda: UInt32(K)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufX, offset: 0, index: 1)
        encoder.setBuffer(bufY, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)

        // One threadgroup per row, each with threadsPerRow threads
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Execute Float32 GEMV with pre-allocated weight buffer: y[M] = A[M,K] * x[K].
    /// Avoids re-creating the weight MTLBuffer on every call.
    public func executeWithWeightBuffer(
        weightBuffer: MTLBuffer,
        x: [Float],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        guard x.count == K else {
            throw GEMVError.invalidVectorShape
        }

        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMVParams(
            M: UInt32(M), K: UInt32(K), lda: UInt32(K)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(bufX, offset: 0, index: 1)
        encoder.setBuffer(bufY, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)

        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Execute N independent GEMV operations in a single command buffer.
    /// y_i[M_i] = weightBuffers[i][M_i, K] * x[K] for i in 0..<N.
    /// Reduces N GPU synchronization round-trips to 1.
    public func executeBatchedWithWeightBuffers(
        weightBuffers: [MTLBuffer],
        x: [Float],
        Ms: [Int],
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [[Float]] {
        guard x.count == K else { throw GEMVError.invalidVectorShape }
        let n = weightBuffers.count
        guard n == Ms.count, n > 0 else { throw GEMVError.encodingFailed }

        // Shared input buffer
        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Output buffers
        var outputBuffers = [MTLBuffer]()
        outputBuffers.reserveCapacity(n)
        for i in 0..<n {
            outputBuffers.append(device.makeBuffer(
                length: Ms[i] * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )!)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        // Encode all N dispatches into a single command buffer
        for i in 0..<n {
            guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
                throw GEMVError.encodingFailed
            }
            var params = ERGEMVParams(M: UInt32(Ms[i]), K: UInt32(K), lda: UInt32(K))
            encoder.setComputePipelineState(pipelineF32)
            encoder.setBuffer(weightBuffers[i], offset: 0, index: 0)
            encoder.setBuffer(bufX, offset: 0, index: 1)
            encoder.setBuffer(outputBuffers[i], offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
            let gridSize = MTLSize(width: Ms[i], height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error { throw error }

        // Read back results
        var results = [[Float]]()
        results.reserveCapacity(n)
        for i in 0..<n {
            let ptr = outputBuffers[i].contents().bindMemory(to: Float.self, capacity: Ms[i])
            results.append(Array(UnsafeBufferPointer(start: ptr, count: Ms[i])))
        }
        return results
    }

    /// Encode a GEMV dispatch into an existing command buffer (no commit/await).
    /// Caller manages the command buffer lifecycle.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int, K: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }
        var params = ERGEMVParams(M: UInt32(M), K: UInt32(K), lda: UInt32(K))
        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Execute Float16 GEMV: y[M] = A[M,K] * x[K].
    public func executeF16(
        a: [Float16], x: [Float16],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float16] {
        try validateInputShape(matrixCount: a.count, vectorCount: x.count, M: M, K: K)

        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMVParams(
            M: UInt32(M), K: UInt32(K), lda: UInt32(K)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF16)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufX, offset: 0, index: 1)
        encoder.setBuffer(bufY, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)

        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float16.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Encode a GEMV dispatch with Float16 weights and Float16 input/output.
    public func encodeF16Weights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        guard weightBuffer.length >= M * K * MemoryLayout<Float16>.stride,
              inputBuffer.length >= K * MemoryLayout<Float16>.stride,
              outputBuffer.length >= M * MemoryLayout<Float16>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = ERGEMVParams(M: UInt32(M), K: UInt32(K), lda: UInt32(K))
        encoder.setComputePipelineState(pipelineF16)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Execute GEMV with BF16 weights and Float32 input/output: y[M] = A[M,K] * x[K].
    public func executeBF16Weights(
        a: [UInt16], x: [Float],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateInputShape(matrixCount: a.count, vectorCount: x.count, M: M, K: K)

        guard let bufA = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared
        ),
        let bufX = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        try encodeBF16Weights(
            commandBuffer: cmdBuf,
            weightBuffer: bufA,
            inputBuffer: bufX,
            outputBuffer: bufY,
            M: M,
            K: K
        )

        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Execute GEMV with a pre-allocated BF16 weight buffer and Float32 input/output.
    public func executeBF16WeightsWithWeightBuffer(
        weightBuffer: MTLBuffer,
        x: [Float],
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        guard x.count == K else {
            throw GEMVError.invalidVectorShape
        }

        guard let bufX = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        try encodeBF16Weights(
            commandBuffer: cmdBuf,
            weightBuffer: weightBuffer,
            inputBuffer: bufX,
            outputBuffer: bufY,
            M: M,
            K: K
        )

        cmdBuf.commit()
        await cmdBuf.completed()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Encode a GEMV dispatch with BF16 weights and Float32 input/output.
    public func encodeBF16Weights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }
        var params = ERGEMVParams(M: UInt32(M), K: UInt32(K), lda: UInt32(K))
        encoder.setComputePipelineState(pipelineBF16F32)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    public func executeBatchedBF16Weights(
        a: [UInt16],
        x: [Float],
        batchSeq: Int,
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateInputShape(matrixCount: a.count, vectorCount: K, M: M, K: K)
        guard batchSeq > 0, x.count == batchSeq * K else {
            throw GEMVError.invalidVectorShape
        }
        guard let weightBuffer = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: batchSeq * M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        try encodeBatchedBF16Weights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            batchSeq: batchSeq,
            M: M,
            K: K
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let count = batchSeq * M
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    public func encodeBatchedBF16Weights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        batchSeq: Int,
        M: Int,
        K: Int
    ) throws {
        guard batchSeq > 0 else { throw GEMVError.invalidVectorShape }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }
        var params = ERGEMVParams(M: UInt32(M), K: UInt32(K), lda: UInt32(K))
        encoder.setComputePipelineState(pipelineBF16F32Batched)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)
        let gridSize = MTLSize(width: M, height: batchSeq, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    public func executeQ4KWeights(
        rawWeights: [UInt8],
        x: [Float],
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateKQuantInputShape(
            rawByteCount: rawWeights.count,
            vectorCount: x.count,
            M: M,
            K: K,
            blockByteCount: 144
        )
        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }
        return try await executeQ4KWeightsWithWeightBuffer(
            weightBuffer: weightBuffer,
            x: x,
            M: M,
            K: K,
            commandQueue: commandQueue
        )
    }

    public func executeQ6KWeights(
        rawWeights: [UInt8],
        x: [Float],
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateKQuantInputShape(
            rawByteCount: rawWeights.count,
            vectorCount: x.count,
            M: M,
            K: K,
            blockByteCount: 210
        )
        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }
        return try await executeQ6KWeightsWithWeightBuffer(
            weightBuffer: weightBuffer,
            x: x,
            M: M,
            K: K,
            commandQueue: commandQueue
        )
    }

    public func executeQ4KWeightsWithWeightBuffer(
        weightBuffer: MTLBuffer,
        x: [Float],
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try await executeKQuantWeightsWithWeightBuffer(
            weightBuffer: weightBuffer,
            x: x,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KGEMV,
            commandQueue: commandQueue
        )
    }

    public func executeBatchedQ4KWeights(
        rawWeights: [UInt8],
        x: [Float],
        batchSeq: Int,
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateKQuantInputShape(
            rawByteCount: rawWeights.count,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )
        guard batchSeq > 0, x.count == batchSeq * K else {
            throw GEMVError.invalidVectorShape
        }
        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: batchSeq * M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        try encodeBatchedQ4KWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            batchSeq: batchSeq,
            M: M,
            K: K
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let count = batchSeq * M
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    public func encodeBatchedQ4KWeights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        batchSeq: Int,
        M: Int,
        K: Int
    ) throws {
        try encodeBatchedKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            batchSeq: batchSeq,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KGEMVBatched,
            threadsPerThreadgroup: Self.threadsPerRow
        )
    }

    public func encodeQ4KWeights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KGEMV
        )
    }

    public func encodeQ4KWeightsPacked(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KPackedGEMV,
            threadsPerThreadgroup: Self.packedThreadsPerRow
        )
    }

    public func encodeQ4KWeightsPackedFourRows(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KPackedFourRowGEMV,
            threadgroupCount: MTLSize(width: (M + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: Self.packedThreadsPerRow
        )
    }

    public func encodeQ4KWeightsLlamaStyle(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 144,
            pipeline: pipelineQ4KLlamaStyleGEMV,
            threadgroupCount: MTLSize(width: (M + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: 64
        )
    }

    public func encodeQ4KWeightsLlamaStyleDual(
        commandBuffer: MTLCommandBuffer,
        weightBufferA: MTLBuffer,
        weightBufferB: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBufferA: MTLBuffer,
        outputBufferB: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBufferA.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferB.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBufferA.length >= M * MemoryLayout<Float>.stride,
              outputBufferB.length >= M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipelineQ4KLlamaStyleDualGEMV)
        encoder.setBuffer(weightBufferA, offset: 0, index: 0)
        encoder.setBuffer(weightBufferB, offset: 0, index: 1)
        encoder.setBuffer(inputBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBufferA, offset: 0, index: 3)
        encoder.setBuffer(outputBufferB, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (M + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeQ4KWeightsTwoRows(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBuffer.length >= M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipelineQ4K2RowGEMV)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: (M + 1) / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeQ4KWeightsDual(
        commandBuffer: MTLCommandBuffer,
        weightBufferA: MTLBuffer,
        weightBufferB: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBufferA: MTLBuffer,
        outputBufferB: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBufferA.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferB.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBufferA.length >= M * MemoryLayout<Float>.stride,
              outputBufferB.length >= M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipelineQ4KDualGEMV)
        encoder.setBuffer(weightBufferA, offset: 0, index: 0)
        encoder.setBuffer(weightBufferB, offset: 0, index: 1)
        encoder.setBuffer(inputBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBufferA, offset: 0, index: 3)
        encoder.setBuffer(outputBufferB, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: M, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeQ4KWeightsDualGeGLU(
        commandBuffer: MTLCommandBuffer,
        gateWeightBuffer: MTLBuffer,
        upWeightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: gateWeightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: upWeightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBuffer.length >= M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipelineQ4KDualGeGLUGEMV)
        encoder.setBuffer(gateWeightBuffer, offset: 0, index: 0)
        encoder.setBuffer(upWeightBuffer, offset: 0, index: 1)
        encoder.setBuffer(inputBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: M, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeQ4KWeightsTriple(
        commandBuffer: MTLCommandBuffer,
        weightBufferA: MTLBuffer,
        weightBufferB: MTLBuffer,
        weightBufferC: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBufferA: MTLBuffer,
        outputBufferB: MTLBuffer,
        outputBufferC: MTLBuffer,
        rowsA: Int,
        rowsB: Int,
        rowsC: Int,
        K: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBufferA.length,
            vectorCount: K,
            M: rowsA,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferB.length,
            vectorCount: K,
            M: rowsB,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferC.length,
            vectorCount: K,
            M: rowsC,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBufferA.length >= rowsA * MemoryLayout<Float>.stride,
              outputBufferB.length >= rowsB * MemoryLayout<Float>.stride,
              outputBufferC.length >= rowsC * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = Q4KTripleGEMVParams(
            rowsA: UInt32(rowsA),
            rowsB: UInt32(rowsB),
            rowsC: UInt32(rowsC),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipelineQ4KTripleGEMV)
        encoder.setBuffer(weightBufferA, offset: 0, index: 0)
        encoder.setBuffer(weightBufferB, offset: 0, index: 1)
        encoder.setBuffer(weightBufferC, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBufferA, offset: 0, index: 4)
        encoder.setBuffer(outputBufferB, offset: 0, index: 5)
        encoder.setBuffer(outputBufferC, offset: 0, index: 6)
        encoder.setBytes(&params, length: MemoryLayout<Q4KTripleGEMVParams>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: rowsA + rowsB + rowsC, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func encodeQ4KWeightsTriplePacked(
        commandBuffer: MTLCommandBuffer,
        weightBufferA: MTLBuffer,
        weightBufferB: MTLBuffer,
        weightBufferC: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBufferA: MTLBuffer,
        outputBufferB: MTLBuffer,
        outputBufferC: MTLBuffer,
        rowsA: Int,
        rowsB: Int,
        rowsC: Int,
        K: Int
    ) throws {
        try encodeQ4KWeightsTriple(
            commandBuffer: commandBuffer,
            weightBufferA: weightBufferA,
            weightBufferB: weightBufferB,
            weightBufferC: weightBufferC,
            inputBuffer: inputBuffer,
            outputBufferA: outputBufferA,
            outputBufferB: outputBufferB,
            outputBufferC: outputBufferC,
            rowsA: rowsA,
            rowsB: rowsB,
            rowsC: rowsC,
            K: K,
            pipeline: pipelineQ4KTriplePackedGEMV,
            threadsPerThreadgroup: Self.packedThreadsPerRow
        )
    }

    private func encodeQ4KWeightsTriple(
        commandBuffer: MTLCommandBuffer,
        weightBufferA: MTLBuffer,
        weightBufferB: MTLBuffer,
        weightBufferC: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBufferA: MTLBuffer,
        outputBufferB: MTLBuffer,
        outputBufferC: MTLBuffer,
        rowsA: Int,
        rowsB: Int,
        rowsC: Int,
        K: Int,
        pipeline: MTLComputePipelineState,
        threadsPerThreadgroup: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBufferA.length,
            vectorCount: K,
            M: rowsA,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferB.length,
            vectorCount: K,
            M: rowsB,
            K: K,
            blockByteCount: 144
        )
        try validateKQuantInputShape(
            rawByteCount: weightBufferC.length,
            vectorCount: K,
            M: rowsC,
            K: K,
            blockByteCount: 144
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBufferA.length >= rowsA * MemoryLayout<Float>.stride,
              outputBufferB.length >= rowsB * MemoryLayout<Float>.stride,
              outputBufferC.length >= rowsC * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = Q4KTripleGEMVParams(
            rowsA: UInt32(rowsA),
            rowsB: UInt32(rowsB),
            rowsC: UInt32(rowsC),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weightBufferA, offset: 0, index: 0)
        encoder.setBuffer(weightBufferB, offset: 0, index: 1)
        encoder.setBuffer(weightBufferC, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBufferA, offset: 0, index: 4)
        encoder.setBuffer(outputBufferB, offset: 0, index: 5)
        encoder.setBuffer(outputBufferC, offset: 0, index: 6)
        encoder.setBytes(&params, length: MemoryLayout<Q4KTripleGEMVParams>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: rowsA + rowsB + rowsC, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    public func executeQ6KWeightsWithWeightBuffer(
        weightBuffer: MTLBuffer,
        x: [Float],
        M: Int,
        K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try await executeKQuantWeightsWithWeightBuffer(
            weightBuffer: weightBuffer,
            x: x,
            M: M,
            K: K,
            blockByteCount: 210,
            pipeline: pipelineQ6KGEMV,
            commandQueue: commandQueue
        )
    }

    public func encodeQ6KWeights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 210,
            pipeline: pipelineQ6KGEMV
        )
    }

    public func encodeQ6KWeightsPacked(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: 210,
            pipeline: pipelineQ6KPackedGEMV,
            threadsPerThreadgroup: Self.q6PackedThreadsPerRow
        )
    }

    public func encodeQ6KWeightsPackedTop1(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        partialValuesBuffer: MTLBuffer,
        partialIndicesBuffer: MTLBuffer,
        outputIndexBuffer: MTLBuffer,
        M: Int,
        K: Int
    ) throws {
        guard M.isMultiple(of: 4) else {
            throw GEMVError.invalidMatrixShape
        }
        try validateKQuantInputShape(
            rawByteCount: weightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: 210
        )
        let partialCount = M / 4
        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              partialValuesBuffer.length >= partialCount * MemoryLayout<Float>.stride,
              partialIndicesBuffer.length >= partialCount * MemoryLayout<UInt32>.stride,
              outputIndexBuffer.length >= MemoryLayout<UInt32>.stride,
              let gemvEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        gemvEncoder.setComputePipelineState(pipelineQ6KPackedTop1GEMV)
        gemvEncoder.setBuffer(weightBuffer, offset: 0, index: 0)
        gemvEncoder.setBuffer(inputBuffer, offset: 0, index: 1)
        gemvEncoder.setBuffer(partialValuesBuffer, offset: 0, index: 2)
        gemvEncoder.setBuffer(partialIndicesBuffer, offset: 0, index: 3)
        gemvEncoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 4)
        gemvEncoder.dispatchThreadgroups(
            MTLSize(width: partialCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.q6PackedThreadsPerRow, height: 1, depth: 1)
        )
        gemvEncoder.endEncoding()

        guard let reduceEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }
        var partialCountParam = UInt32(partialCount)
        reduceEncoder.setComputePipelineState(pipelineQ6KTop1Reduce)
        reduceEncoder.setBuffer(partialValuesBuffer, offset: 0, index: 0)
        reduceEncoder.setBuffer(partialIndicesBuffer, offset: 0, index: 1)
        reduceEncoder.setBuffer(outputIndexBuffer, offset: 0, index: 2)
        reduceEncoder.setBytes(&partialCountParam, length: MemoryLayout<UInt32>.stride, index: 3)
        reduceEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        reduceEncoder.endEncoding()
    }

    private func validateInputShape(matrixCount: Int, vectorCount: Int, M: Int, K: Int) throws {
        guard matrixCount == M * K else {
            throw GEMVError.invalidMatrixShape
        }
        guard vectorCount == K else {
            throw GEMVError.invalidVectorShape
        }
    }

    private func encodeKQuantWeights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        M: Int,
        K: Int,
        blockByteCount: Int,
        pipeline: MTLComputePipelineState,
        threadgroupCount: MTLSize? = nil,
        threadsPerThreadgroup: Int = GEMVKernel.threadsPerRow
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: blockByteCount
        )

        guard inputBuffer.length >= K * MemoryLayout<Float>.stride,
              outputBuffer.length >= M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            threadgroupCount ?? MTLSize(width: M, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeBatchedKQuantWeights(
        commandBuffer: MTLCommandBuffer,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        batchSeq: Int,
        M: Int,
        K: Int,
        blockByteCount: Int,
        pipeline: MTLComputePipelineState,
        threadsPerThreadgroup: Int
    ) throws {
        try validateKQuantInputShape(
            rawByteCount: weightBuffer.length,
            vectorCount: K,
            M: M,
            K: K,
            blockByteCount: blockByteCount
        )

        guard batchSeq > 0,
              inputBuffer.length >= batchSeq * K * MemoryLayout<Float>.stride,
              outputBuffer.length >= batchSeq * M * MemoryLayout<Float>.stride,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        var params = KQuantGEMVParams(
            rows: UInt32(M),
            cols: UInt32(K),
            blocksPerRow: UInt32(K / 256)
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<KQuantGEMVParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: M, height: batchSeq, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func executeKQuantWeightsWithWeightBuffer(
        weightBuffer: MTLBuffer,
        x: [Float],
        M: Int,
        K: Int,
        blockByteCount: Int,
        pipeline: MTLComputePipelineState,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try validateKQuantInputShape(
            rawByteCount: weightBuffer.length,
            vectorCount: x.count,
            M: M,
            K: K,
            blockByteCount: blockByteCount
        )

        guard let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        try encodeKQuantWeights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K,
            blockByteCount: blockByteCount,
            pipeline: pipeline
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    private func validateKQuantInputShape(
        rawByteCount: Int,
        vectorCount: Int,
        M: Int,
        K: Int,
        blockByteCount: Int
    ) throws {
        guard vectorCount == K else {
            throw GEMVError.invalidVectorShape
        }
        guard K > 0, K % 256 == 0 else {
            throw GEMVError.invalidVectorShape
        }
        let expectedBytes = M * (K / 256) * blockByteCount
        guard rawByteCount >= expectedBytes else {
            throw GEMVError.invalidMatrixShape
        }
    }
}

public enum GEMVError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
    case invalidMatrixShape
    case invalidVectorShape
}
