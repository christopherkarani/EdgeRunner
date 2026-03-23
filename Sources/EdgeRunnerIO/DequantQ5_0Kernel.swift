import Foundation
import Metal
import EdgeRunnerMetal

public final class DequantQ5_0Kernel: Sendable {
    private let device: MTLDevice
    private let dequantPipeline: MTLComputePipelineState

    private static let blockByteCount = 22
    private static let weightsPerBlock = 32

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.dequantPipeline = try registry.pipeline(for: "dequant_q5_0")
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

        guard let inputBuffer = device.makeBuffer(
            bytes: blockData,
            length: blockData.count * MemoryLayout<UInt8>.stride,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(byteCount: blockData.count)
        }

        let outputCount = blockCount * Self.weightsPerBlock
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(
                byteCount: outputCount * MemoryLayout<Float>.stride
            )
        }

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
        let threadWidth = max(1, min(blockCount, dequantPipeline.maxTotalThreadsPerThreadgroup))
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
}
