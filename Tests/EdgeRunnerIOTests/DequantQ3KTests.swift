import Metal
import Testing
@testable import EdgeRunnerIO

private let q3KBlockByteCount = 110
private let q3KWeightsPerBlock = 256

private func packQ3KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == q3KWeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q3KBlockByteCount)

    // Compute master scale d as float16
    let absMax = values.map { abs($0) }.max() ?? 0
    let d: Float = absMax > 0 ? absMax / (31.0 * 3.0) : 0
    let dF16 = Float16(d)
    let dBits = dF16.bitPattern.littleEndian
    withUnsafeBytes(of: dBits) { pointer in
        block[108] = pointer[0]
        block[109] = pointer[1]
    }
    let dReconstructed = Float(dF16)

    // Compute 16 sub-block scales (signed, -32..31)
    var subScales = [Int](repeating: 0, count: 16)
    for sub in 0..<16 {
        let subValues = Array(values[(sub * 16)..<((sub + 1) * 16)])
        let subAbsMax = subValues.map { abs($0) }.max() ?? 0
        if dReconstructed > 0 && subAbsMax > 0 {
            // scale such that value / (d * scale) fits in -4..3 (q3 range after -4 offset)
            // q3 = round(value / (d * scale)) + 4, clamped 0..7
            // max |value| = d * |scale| * 3 (since q3 - 4 ranges -4..3, max magnitude is 4 but we target 3)
            let idealScale = subAbsMax / (dReconstructed * 3.0)
            subScales[sub] = min(31, max(-32, Int(round(idealScale))))
            if subScales[sub] == 0 && subAbsMax > 0 {
                subScales[sub] = 1
            }
        }
    }

    // Pack 16 scales into 12 bytes at offset 96
    // Bytes 0-3: lower 4 bits of scales 0-7 (nibble-packed)
    // Bytes 4-7: lower 4 bits of scales 8-15
    // Bytes 8-11: upper 2 bits of all 16 scales
    for i in 0..<16 {
        // Convert signed scale to 6-bit unsigned (0..63) with +32 offset
        let raw6 = UInt8(clamping: subScales[i] + 32)
        let lower4 = raw6 & 0x0F
        let upper2 = (raw6 >> 4) & 0x03

        // Pack lower 4 bits
        let byteIdx = i / 2
        if i % 2 == 0 {
            block[96 + byteIdx] = (block[96 + byteIdx] & 0xF0) | lower4
        } else {
            block[96 + byteIdx] = (block[96 + byteIdx] & 0x0F) | (lower4 << 4)
        }

        // Pack upper 2 bits
        let upperByteIdx = 8 + i / 4
        let shift = (i % 4) * 2
        block[96 + upperByteIdx] |= upper2 << shift
    }

    // Quantize each value and pack qs (offset 32, 64 bytes) and hmask (offset 0, 32 bytes)
    for i in 0..<256 {
        let sub = i / 16
        let scale = Float(subScales[sub])
        var q3: Int
        if dReconstructed > 0 && scale != 0 {
            q3 = Int(round(values[i] / (dReconstructed * scale))) + 4
        } else {
            q3 = 4 // zero value maps to q3=4
        }
        q3 = min(7, max(0, q3))

        let lower2 = UInt8(q3 & 0x03)
        let highBit = UInt8((q3 >> 2) & 0x01)

        // Pack lower 2 bits into qs (offset 32)
        let qsByteIdx = 32 + i / 4
        let qsShift = (i % 4) * 2
        block[qsByteIdx] |= lower2 << qsShift

        // Pack high bit into hmask (offset 0)
        let hmaskByteIdx = i / 8
        let hmaskShift = i % 8
        block[hmaskByteIdx] |= highBit << hmaskShift
    }

    return block
}

private func dequantQ3KBlock(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q3KBlockByteCount)

    // Read d from offset 108 (float16)
    let dBits = UInt16(blockData[108]) | (UInt16(blockData[109]) << 8)
    let d = Float(Float16(bitPattern: dBits))

    // Unpack 16 scales from 12 bytes at offset 96
    var scaleValues = [Float](repeating: 0, count: 16)
    for i in 0..<16 {
        let byteIdx = i / 2
        let lower4 = (blockData[96 + byteIdx] >> ((i % 2) * 4)) & 0x0F
        let upper2 = (blockData[96 + 8 + i / 4] >> ((i % 4) * 2)) & 0x03
        let raw6 = Int(lower4) | (Int(upper2) << 4)
        let signedScale = raw6 - 32
        scaleValues[i] = d * Float(signedScale)
    }

    // Dequantize 256 weights
    var result = [Float](repeating: 0, count: q3KWeightsPerBlock)
    for i in 0..<256 {
        let sub = i / 16

        // Lower 2 bits from qs (offset 32)
        let qsByte = blockData[32 + i / 4]
        let lower2 = (qsByte >> ((i % 4) * 2)) & 0x03

        // High bit from hmask (offset 0)
        let hmaskByte = blockData[i / 8]
        let highBit = (hmaskByte >> (i % 8)) & 1

        let q3 = Int(lower2) | (Int(highBit) << 2)
        result[i] = scaleValues[sub] * Float(q3 - 4)
    }
    return result
}

@Suite("Q3_K Dequantisation Kernel")
struct DequantQ3KTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue()
        else {
            throw IOMetalTestError.noMetal
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    @Test func singleSuperBlockDequant() async throws {
        let values = (0..<q3KWeightsPerBlock).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ3KBlock(values: values)
        let expected = dequantQ3KBlock(blockData: blockData)

        let kernel = try DequantQ3KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q3KWeightsPerBlock)
        for index in 0..<q3KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-2)
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for blockIndex in 0..<superBlockCount {
            let values = (0..<q3KWeightsPerBlock).map { index in
                Float(((blockIndex * 17) + index) % 29 - 14) * 0.14
            }
            let blockData = packQ3KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ3KBlock(blockData: blockData))
        }

        let kernel = try DequantQ3KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * q3KWeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-2)
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: q3KWeightsPerBlock)
        let blockData = packQ3KBlock(values: values)
        let expected = dequantQ3KBlock(blockData: blockData)

        let kernel = try DequantQ3KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q3KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q3KWeightsPerBlock).map { index in
            Float(index % 23) / 11.0 - 1.0
        }
        let blockData = packQ3KBlock(values: values)
        let dequantised = dequantQ3KBlock(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q3KWeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }

        let meanAbsoluteError = totalError / Float(q3KWeightsPerBlock)
        #expect(meanAbsoluteError < 0.2)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
