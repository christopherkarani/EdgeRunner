import Foundation
import Metal
import EdgeRunnerMetal

public final class DequantQ4_0Kernel: Sendable {
    private let device: MTLDevice
    private let dequantPipeline: MTLComputePipelineState
    private let gemvPipeline: MTLComputePipelineState

    private static let blockByteCount = 18
    private static let weightsPerBlock = 32

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.dequantPipeline = try registry.pipeline(for: "dequant_q4_0")
        self.gemvPipeline = try registry.pipeline(for: "dequant_q4_0_gemv")
    }

    public func dequantise(
        blockData: [UInt8],
        blockCount: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let expectedByteCount = blockCount * Self.blockByteCount
        guard blockData.count == expectedByteCount else {
            throw DequantKernelError.invalidBlockDataCount(
                expected: expectedByteCount,
                actual: blockData.count
            )
        }

        let inputBuffer = try makeInputBuffer(bytes: blockData)
        let outputCount = blockCount * Self.weightsPerBlock
        let outputBuffer = try makeOutputBuffer(elementCount: outputCount)

        var params = DequantParams(blockCount: UInt32(blockCount), outputOffset: 0)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw DequantKernelError.encodingFailed
        }

        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<DequantParams>.stride, index: 2)

        let threadWidth = max(
            1,
            min(blockCount, dequantPipeline.maxTotalThreadsPerThreadgroup)
        )
        encoder.dispatchThreads(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    public func fusedDequantGEMV(
        quantisedRows: [UInt8],
        x: [Float],
        rows: Int,
        cols: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        guard cols % Self.weightsPerBlock == 0 else {
            throw DequantKernelError.invalidMatrixShape
        }
        guard x.count == cols else {
            throw DequantKernelError.invalidVectorShape
        }

        let blocksPerRow = cols / Self.weightsPerBlock
        let expectedByteCount = rows * blocksPerRow * Self.blockByteCount
        guard quantisedRows.count == expectedByteCount else {
            throw DequantKernelError.invalidBlockDataCount(
                expected: expectedByteCount,
                actual: quantisedRows.count
            )
        }

        let weightBuffer = try makeInputBuffer(bytes: quantisedRows)
        let xBuffer = try makeFloatInputBuffer(values: x)
        let yBuffer = try makeOutputBuffer(elementCount: rows)

        var params = DequantGEMVParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            blocksPerRow: UInt32(blocksPerRow)
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw DequantKernelError.encodingFailed
        }

        encoder.setComputePipelineState(gemvPipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<DequantGEMVParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    private func makeInputBuffer(bytes: [UInt8]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: bytes,
            length: bytes.count * MemoryLayout<UInt8>.stride,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(byteCount: bytes.count)
        }
        return buffer
    }

    private func makeFloatInputBuffer(values: [Float]) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: byteCount,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(byteCount: byteCount)
        }
        return buffer
    }

    private func makeOutputBuffer(elementCount: Int) throws -> MTLBuffer {
        let byteCount = elementCount * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw DequantKernelError.allocationFailed(byteCount: byteCount)
        }
        return buffer
    }
}
