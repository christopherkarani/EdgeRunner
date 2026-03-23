import Metal
import Testing
@testable import EdgeRunnerIO

private let q5KBlockByteCount = 176
private let q5KWeightsPerBlock = 256

private func packQ5KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == q5KWeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q5KBlockByteCount)

    var subScales = [Float](repeating: 0, count: 8)
    var subMins = [Float](repeating: 0, count: 8)

    for subBlock in 0..<8 {
        let subValues = Array(values[(subBlock * 32)..<((subBlock + 1) * 32)])
        let maxValue = subValues.max() ?? 0
        let minValue = subValues.min() ?? 0
        if maxValue - minValue > 0 {
            subScales[subBlock] = (maxValue - minValue) / 31.0
            subMins[subBlock] = max(0, -minValue)
        } else {
            subScales[subBlock] = 0
            subMins[subBlock] = max(0, -minValue)
        }
    }

    let maxScale = subScales.max() ?? 0
    let maxMin = subMins.max() ?? 0
    let d: Float = maxScale > 0 ? maxScale / 63.0 : 0
    let dmin: Float = maxMin > 0 ? maxMin / 63.0 : 0

    let dBits = Float16(d).bitPattern.littleEndian
    let dminBits = Float16(dmin).bitPattern.littleEndian
    withUnsafeBytes(of: dBits) { pointer in
        block[0] = pointer[0]
        block[1] = pointer[1]
    }
    withUnsafeBytes(of: dminBits) { pointer in
        block[2] = pointer[0]
        block[3] = pointer[1]
    }

    var quantisedScales = [UInt8](repeating: 0, count: 8)
    var quantisedMins = [UInt8](repeating: 0, count: 8)
    for subBlock in 0..<8 {
        quantisedScales[subBlock] = d > 0 ? UInt8(min(63, round(subScales[subBlock] / d))) : 0
        quantisedMins[subBlock] = dmin > 0 ? UInt8(min(63, round(subMins[subBlock] / dmin))) : 0
    }

    // Pack scales and mins into 12 bytes at offset 4 (identical to Q4_K_M)
    for subBlock in 0..<4 {
        block[4 + subBlock] = (quantisedScales[subBlock] & 0x3F)
            | ((quantisedScales[subBlock + 4] & 0x03) << 6)
        block[8 + subBlock] = (quantisedMins[subBlock] & 0x3F)
            | ((quantisedMins[subBlock + 4] & 0x03) << 6)
        block[12 + subBlock] = ((quantisedScales[subBlock + 4] >> 2) & 0x0F)
            | (((quantisedMins[subBlock + 4] >> 2) & 0x0F) << 4)
    }

    // Quantise values and pack into qs (offset 48, 128 bytes) and qh (offset 16, 32 bytes)
    let dActual = Float(Float16(bitPattern: dBits))
    let dminActual = Float(Float16(bitPattern: dminBits))

    for subBlock in 0..<8 {
        let scale = dActual * Float(quantisedScales[subBlock])
        let minValue = dminActual * Float(quantisedMins[subBlock])
        for index in 0..<32 {
            let valueIndex = subBlock * 32 + index
            let quantised: Int
            if scale > 0 {
                quantised = min(31, max(0, Int(round((values[valueIndex] + minValue) / scale))))
            } else {
                quantised = 0
            }

            let lower4 = UInt8(quantised & 0x0F)
            let bit5 = UInt8((quantised >> 4) & 0x01)

            // Pack lower 4 bits into qs at offset 48 (nibble-packed, 128 bytes)
            let qsByteIndex = 48 + valueIndex / 2
            if valueIndex.isMultiple(of: 2) {
                block[qsByteIndex] = (block[qsByteIndex] & 0xF0) | lower4
            } else {
                block[qsByteIndex] = (block[qsByteIndex] & 0x0F) | (lower4 << 4)
            }

            // Pack 5th bit into qh at offset 16 (bit-packed, 32 bytes)
            let qhByteIndex = 16 + valueIndex / 8
            let qhBitIndex = valueIndex % 8
            if bit5 != 0 {
                block[qhByteIndex] |= (1 << qhBitIndex)
            }
        }
    }

    return block
}

private func dequantQ5KBlock(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q5KBlockByteCount)

    let dBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let dminBits = UInt16(blockData[2]) | (UInt16(blockData[3]) << 8)
    let d = Float(Float16(bitPattern: dBits))
    let dmin = Float(Float16(bitPattern: dminBits))

    var scales = [UInt8](repeating: 0, count: 8)
    var mins = [UInt8](repeating: 0, count: 8)
    for subBlock in 0..<4 {
        scales[subBlock] = blockData[4 + subBlock] & 0x3F
        scales[subBlock + 4] = ((blockData[4 + subBlock] >> 6) & 0x03)
            | ((blockData[12 + subBlock] & 0x0F) << 2)
        mins[subBlock] = blockData[8 + subBlock] & 0x3F
        mins[subBlock + 4] = ((blockData[8 + subBlock] >> 6) & 0x03)
            | (((blockData[12 + subBlock] >> 4) & 0x0F) << 2)
    }

    var result = [Float](repeating: 0, count: q5KWeightsPerBlock)
    for subBlock in 0..<8 {
        let scale = d * Float(scales[subBlock])
        let minValue = dmin * Float(mins[subBlock])
        for index in 0..<32 {
            let globalIdx = subBlock * 32 + index

            // Lower 4 bits from qs at offset 48
            let qsByteIndex = 48 + globalIdx / 2
            let lower4: UInt8
            if globalIdx.isMultiple(of: 2) {
                lower4 = blockData[qsByteIndex] & 0x0F
            } else {
                lower4 = (blockData[qsByteIndex] >> 4) & 0x0F
            }

            // 5th bit from qh at offset 16
            let qhByteIndex = 16 + globalIdx / 8
            let qhBitIndex = globalIdx % 8
            let bit5 = (blockData[qhByteIndex] >> qhBitIndex) & 1

            let q5 = UInt8(lower4) | (bit5 << 4)
            result[globalIdx] = (scale * Float(q5)) - minValue
        }
    }
    return result
}

@Suite("Q5_K Dequantisation Kernel")
struct DequantQ5KTests {
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
        let values = (0..<q5KWeightsPerBlock).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ5KBlock(values: values)
        let expected = dequantQ5KBlock(blockData: blockData)

        let kernel = try DequantQ5KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q5KWeightsPerBlock)
        for index in 0..<q5KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-2)
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for blockIndex in 0..<superBlockCount {
            let values = (0..<q5KWeightsPerBlock).map { index in
                Float(((blockIndex * 17) + index) % 29 - 14) * 0.14
            }
            let blockData = packQ5KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ5KBlock(blockData: blockData))
        }

        let kernel = try DequantQ5KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * q5KWeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-2)
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: q5KWeightsPerBlock)
        let blockData = packQ5KBlock(values: values)
        let expected = dequantQ5KBlock(blockData: blockData)

        let kernel = try DequantQ5KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q5KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q5KWeightsPerBlock).map { index in
            Float(index % 23) / 11.0 - 1.0
        }
        let blockData = packQ5KBlock(values: values)
        let dequantised = dequantQ5KBlock(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q5KWeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }

        let meanAbsoluteError = totalError / Float(q5KWeightsPerBlock)
        #expect(meanAbsoluteError < 0.15)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
