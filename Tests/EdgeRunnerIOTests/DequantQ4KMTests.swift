import Metal
import Testing
@testable import EdgeRunnerIO

private let q4KBlockByteCount = 144
private let q4KWeightsPerBlock = 256

private func packQ4KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == q4KWeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q4KBlockByteCount)

    var subScales = [Float](repeating: 0, count: 8)
    var subMins = [Float](repeating: 0, count: 8)

    for subBlock in 0..<8 {
        let subValues = Array(values[(subBlock * 32)..<((subBlock + 1) * 32)])
        let maxValue = subValues.max() ?? 0
        let minValue = subValues.min() ?? 0
        if maxValue - minValue > 0 {
            subScales[subBlock] = (maxValue - minValue) / 15.0
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

    for subBlock in 0..<4 {
        block[4 + subBlock] = (quantisedScales[subBlock] & 0x3F)
            | (((quantisedScales[subBlock + 4] >> 4) & 0x03) << 6)
        block[8 + subBlock] = (quantisedMins[subBlock] & 0x3F)
            | (((quantisedMins[subBlock + 4] >> 4) & 0x03) << 6)
        block[12 + subBlock] = (quantisedScales[subBlock + 4] & 0x0F)
            | ((quantisedMins[subBlock + 4] & 0x0F) << 4)
    }

    for subBlock in 0..<8 {
        let scale = Float(Float16(bitPattern: dBits)) * Float(quantisedScales[subBlock])
        let minValue = Float(Float16(bitPattern: dminBits)) * Float(quantisedMins[subBlock])
        for index in 0..<32 {
            let valueIndex = subBlock * 32 + index
            let quantised: Int
            if scale > 0 {
                quantised = min(15, max(0, Int(round((values[valueIndex] + minValue) / scale))))
            } else {
                quantised = 0
            }

            let byteIndex = 16 + (subBlock / 2) * 32 + index
            if subBlock.isMultiple(of: 2) {
                block[byteIndex] = (block[byteIndex] & 0xF0) | UInt8(quantised & 0x0F)
            } else {
                block[byteIndex] = (block[byteIndex] & 0x0F) | UInt8((quantised & 0x0F) << 4)
            }
        }
    }

    return block
}

private func dequantQ4KBlock(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q4KBlockByteCount)

    let dBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let dminBits = UInt16(blockData[2]) | (UInt16(blockData[3]) << 8)
    let d = Float(Float16(bitPattern: dBits))
    let dmin = Float(Float16(bitPattern: dminBits))

    var scales = [UInt8](repeating: 0, count: 8)
    var mins = [UInt8](repeating: 0, count: 8)
    for subBlock in 0..<4 {
        scales[subBlock] = blockData[4 + subBlock] & 0x3F
        scales[subBlock + 4] = (blockData[12 + subBlock] & 0x0F)
            | ((blockData[4 + subBlock] >> 6) << 4)
        mins[subBlock] = blockData[8 + subBlock] & 0x3F
        mins[subBlock + 4] = ((blockData[12 + subBlock] >> 4) & 0x0F)
            | ((blockData[8 + subBlock] >> 6) << 4)
    }

    var result = [Float](repeating: 0, count: q4KWeightsPerBlock)
    for subBlock in 0..<8 {
        let scale = d * Float(scales[subBlock])
        let minValue = dmin * Float(mins[subBlock])
        for index in 0..<32 {
            let byteIndex = 16 + (subBlock / 2) * 32 + index
            let nibble: UInt8
            if subBlock.isMultiple(of: 2) {
                nibble = blockData[byteIndex] & 0x0F
            } else {
                nibble = (blockData[byteIndex] >> 4) & 0x0F
            }
            result[(subBlock * 32) + index] = (scale * Float(nibble)) - minValue
        }
    }
    return result
}

@Suite("Q4_K_M Dequantisation Kernel")
struct DequantQ4KMTests {
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
        let values = (0..<q4KWeightsPerBlock).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ4KBlock(values: values)
        let expected = dequantQ4KBlock(blockData: blockData)

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q4KWeightsPerBlock)
        for index in 0..<q4KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-2)
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for blockIndex in 0..<superBlockCount {
            let values = (0..<q4KWeightsPerBlock).map { index in
                Float(((blockIndex * 17) + index) % 29 - 14) * 0.14
            }
            let blockData = packQ4KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ4KBlock(blockData: blockData))
        }

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * q4KWeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-2)
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: q4KWeightsPerBlock)
        let blockData = packQ4KBlock(values: values)
        let expected = dequantQ4KBlock(blockData: blockData)

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q4KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q4KWeightsPerBlock).map { index in
            Float(index % 23) / 11.0 - 1.0
        }
        let blockData = packQ4KBlock(values: values)
        let dequantised = dequantQ4KBlock(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q4KWeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }

        let meanAbsoluteError = totalError / Float(q4KWeightsPerBlock)
        #expect(meanAbsoluteError < 0.15)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
