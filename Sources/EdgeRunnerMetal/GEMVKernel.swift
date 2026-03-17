import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the row-parallel GEMV Metal kernel.
/// Optimized for autoregressive decoding: y[M] = A[M,K] * x[K].
public final class GEMVKernel: Sendable {
    public let pipelineF32: MTLComputePipelineState // exposed for fused pipeline encoding
    private let pipelineF16: MTLComputePipelineState
    private let device: MTLDevice

    /// Expose pipeline for external fused command buffer encoding
    public var f32Pipeline: MTLComputePipelineState { pipelineF32 }

    private static let threadsPerRow = 256

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "gemv_f32")
        self.pipelineF16 = try registry.pipeline(for: "gemv_f16")
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

    private func validateInputShape(matrixCount: Int, vectorCount: Int, M: Int, K: Int) throws {
        guard matrixCount == M * K else {
            throw GEMVError.invalidMatrixShape
        }
        guard vectorCount == K else {
            throw GEMVError.invalidVectorShape
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
