import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the row-parallel GEMV Metal kernel.
/// Optimized for autoregressive decoding: y[M] = A[M,K] * x[K].
public final class GEMVKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let pipelineF16: MTLComputePipelineState
    private let device: MTLDevice

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

    /// Execute Float16 GEMV: y[M] = A[M,K] * x[K].
    public func executeF16(
        a: [Float16], x: [Float16],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float16] {
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
}

public enum GEMVError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
