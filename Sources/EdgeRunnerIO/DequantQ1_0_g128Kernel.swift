import Foundation
import Metal
import EdgeRunnerMetal

public final class DequantQ1_0_g128Kernel: Sendable {
    private struct Q1DequantParams {
        var blockCount: UInt32
        var outputOffset: UInt32
        var scaleByteOffset: UInt32
        var bitDataOffset: UInt32
        var bitOrderMSBFirst: UInt32
        var oneBitIsNegative: UInt32
    }

    private let device: MTLDevice
    private let dequantPipeline: MTLComputePipelineState
    private let gemvPipeline: MTLComputePipelineState

    private static let blockByteCount = 18
    private static let weightsPerBlock = 128

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.dequantPipeline = try registry.pipeline(for: "dequant_q1_0_g128")
        self.gemvPipeline = try registry.pipeline(for: "dequant_q1_0_g128_gemv")
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

        let layout = ProcessInfo.processInfo.environment["EDGERUNNER_Q1_G128_LAYOUT"] ?? "scale0_lsb"
        let useScaleAtTail = layout.contains("scale16")
        let bitDataOffset: UInt32 = useScaleAtTail ? 0 : 2
        let useMSBBitOrder = layout.contains("msb")
        let oneBitIsNegative = layout.contains("oneNeg")
        var params = Q1DequantParams(
            blockCount: UInt32(blockCount),
            outputOffset: 0,
            scaleByteOffset: useScaleAtTail ? 16 : 0,
            bitDataOffset: bitDataOffset,
            bitOrderMSBFirst: useMSBBitOrder ? 1 : 0,
            oneBitIsNegative: oneBitIsNegative ? 1 : 0
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw DequantKernelError.encodingFailed
        }

        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<Q1DequantParams>.stride, index: 2)
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

    /// Fused Q1_0_g128 matrix-vector multiply.
    /// Computes `output = quantisedWeights @ input` without materializing float weights.
    public func gemv(
        quantisedWeights: MTLBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        cols: Int,
        commandQueue: MTLCommandQueue
    ) async throws {
        guard rows > 0, cols > 0, cols % Self.weightsPerBlock == 0 else {
            throw DequantKernelError.invalidMatrixShape
        }
        // Keep fused GEMV contract explicit: the current kernel is fixed to
        // scale@0, bits@+2, LSB-first, and 1=>+scale.
        let layout = ProcessInfo.processInfo.environment["EDGERUNNER_Q1_G128_LAYOUT"] ?? "scale0_lsb"
        guard layout == "scale0_lsb" else {
            throw DequantKernelError.invalidMatrixShape
        }

        let blocksPerRow = cols / Self.weightsPerBlock
        let expectedWeightBytes = rows * blocksPerRow * Self.blockByteCount
        let expectedInputBytes = cols * MemoryLayout<Float>.stride
        let expectedOutputBytes = rows * MemoryLayout<Float>.stride
        guard quantisedWeights.length >= expectedWeightBytes else {
            throw DequantKernelError.invalidBlockDataCount(
                expected: expectedWeightBytes,
                actual: quantisedWeights.length
            )
        }
        guard input.length >= expectedInputBytes, output.length >= expectedOutputBytes else {
            throw DequantKernelError.invalidVectorShape
        }

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
        encoder.setBuffer(quantisedWeights, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<DequantGEMVParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }
    }
}
