import Foundation
import Metal
import EdgeRunnerMetal

public final class DequantQ3KKernel: Sendable {
    private let device: MTLDevice
    private let dequantPipeline: MTLComputePipelineState

    private static let blockByteCount = 110
    private static let weightsPerBlock = 256

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.dequantPipeline = try registry.pipeline(for: "dequant_q3_k")
    }

    public func dequantise(
        blockData: [UInt8],
        superBlockCount: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let expectedByteCount = superBlockCount * Self.blockByteCount
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

        let outputCount = superBlockCount * Self.weightsPerBlock
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(
                byteCount: outputCount * MemoryLayout<Float>.stride
            )
        }

        var params = DequantQ4KParams(superBlockCount: UInt32(superBlockCount), outputOffset: 0)
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw DequantKernelError.encodingFailed
        }

        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<DequantQ4KParams>.stride, index: 2)
        let threadWidth = max(1, min(superBlockCount, dequantPipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(
            MTLSize(width: superBlockCount, height: 1, depth: 1),
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
