import Metal
import Testing
@testable import EdgeRunnerIO

private let q2KBlockByteCount = 84
private let q2KWeightsPerBlock = 256

private func packQ2KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == q2KWeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q2KBlockByteCount)

    // 16 sub-blocks of 16 values each
    var subScales = [Float](repeating: 0, count: 16)
    var subMins = [Float](repeating: 0, count: 16)

    for sub in 0..<16 {
        let subValues = Array(values[(sub * 16)..<((sub + 1) * 16)])
        let maxValue = subValues.max() ?? 0
        let minValue = subValues.min() ?? 0
        if maxValue - minValue > 0 {
            subScales[sub] = (maxValue - minValue) / 3.0  // 2-bit: 0..3
            subMins[sub] = max(0, -minValue)
        } else {
            subScales[sub] = 0
            subMins[sub] = max(0, -minValue)
        }
    }

    let maxScale = subScales.max() ?? 0
    let maxMin = subMins.max() ?? 0
    // sc and m are 4-bit (0..15)
    let d: Float = maxScale > 0 ? maxScale / 15.0 : 0
    let dmin: Float = maxMin > 0 ? maxMin / 15.0 : 0

    // d at offset 80, dmin at offset 82 (float16)
    let dBits = Float16(d).bitPattern.littleEndian
    let dminBits = Float16(dmin).bitPattern.littleEndian
    withUnsafeBytes(of: dBits) { pointer in
        block[80] = pointer[0]
        block[81] = pointer[1]
    }
    withUnsafeBytes(of: dminBits) { pointer in
        block[82] = pointer[0]
        block[83] = pointer[1]
    }

    // Quantise sub-block scales and mins to 4-bit
    var quantisedScales = [UInt8](repeating: 0, count: 16)
    var quantisedMins = [UInt8](repeating: 0, count: 16)
    for sub in 0..<16 {
        quantisedScales[sub] = d > 0 ? UInt8(min(15, round(subScales[sub] / d))) : 0
        quantisedMins[sub] = dmin > 0 ? UInt8(min(15, round(subMins[sub] / dmin))) : 0
    }

    // Pack scales[16] at offset 0: each byte = (sc & 0xF) | (m << 4)
    for sub in 0..<16 {
        block[sub] = (quantisedScales[sub] & 0x0F) | (quantisedMins[sub] << 4)
    }

    // Reconstruct effective d and dmin for quantisation
    let dEff = Float(Float16(bitPattern: dBits))
    let dminEff = Float(Float16(bitPattern: dminBits))

    // Quantise each value to 2 bits and pack into qs[64] at offset 16
    for sub in 0..<16 {
        let sc = Float(quantisedScales[sub])
        let m = Float(quantisedMins[sub])
        let scale = dEff * sc
        let minValue = dminEff * m

        for index in 0..<16 {
            let valueIndex = sub * 16 + index
            let quantised: Int
            if scale > 0 {
                quantised = min(3, max(0, Int(round((values[valueIndex] + minValue) / scale))))
            } else {
                quantised = 0
            }

            let i = valueIndex
            let byteIndex = 16 + i / 4
            let shift = (i % 4) * 2
            block[byteIndex] = block[byteIndex] | UInt8((quantised & 0x03) << shift)
        }
    }

    return block
}

private func dequantQ2KBlock(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q2KBlockByteCount)

    // d at offset 80, dmin at offset 82
    let dBits = UInt16(blockData[80]) | (UInt16(blockData[81]) << 8)
    let dminBits = UInt16(blockData[82]) | (UInt16(blockData[83]) << 8)
    let d = Float(Float16(bitPattern: dBits))
    let dmin = Float(Float16(bitPattern: dminBits))

    var result = [Float](repeating: 0, count: q2KWeightsPerBlock)
    for i in 0..<q2KWeightsPerBlock {
        let sub = i / 16
        let scaleByte = blockData[sub]  // scales at offset 0
        let sc = Float(scaleByte & 0x0F)
        let m = Float(scaleByte >> 4)

        // 2-bit quant from qs (offset 16)
        let qsByte = blockData[16 + i / 4]
        let q2 = Float((qsByte >> ((i % 4) * 2)) & 0x03)

        result[i] = d * sc * q2 - dmin * m
    }
    return result
}

@Suite("Q2_K Dequantisation Kernel")
struct DequantQ2KTests {
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
        let values = (0..<q2KWeightsPerBlock).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ2KBlock(values: values)
        let expected = dequantQ2KBlock(blockData: blockData)

        let kernel = try DequantQ2KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q2KWeightsPerBlock)
        for index in 0..<q2KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-2)
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for blockIndex in 0..<superBlockCount {
            let values = (0..<q2KWeightsPerBlock).map { index in
                Float(((blockIndex * 17) + index) % 29 - 14) * 0.14
            }
            let blockData = packQ2KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ2KBlock(blockData: blockData))
        }

        let kernel = try DequantQ2KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * q2KWeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-2)
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: q2KWeightsPerBlock)
        let blockData = packQ2KBlock(values: values)
        let expected = dequantQ2KBlock(blockData: blockData)

        let kernel = try DequantQ2KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q2KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q2KWeightsPerBlock).map { index in
            Float(index % 23) / 11.0 - 1.0
        }
        let blockData = packQ2KBlock(values: values)
        let dequantised = dequantQ2KBlock(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q2KWeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }

        let meanAbsoluteError = totalError / Float(q2KWeightsPerBlock)
        #expect(meanAbsoluteError < 0.3)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
