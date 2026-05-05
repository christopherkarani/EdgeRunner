import Metal
import Testing
@testable import EdgeRunnerIO

private let q6KBlockByteCount = 210
private let q6KWeightsPerBlock = 256

private func packQ6KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == q6KWeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q6KBlockByteCount)

    let absMax = values.map { abs($0) }.max() ?? 0
    let d: Float = absMax > 0 ? absMax / (63.0 * 31.0) : 0
    let dF16 = Float16(d)
    let dFloat = Float(dF16)

    // Compute per-sub-block scales and quantised values
    var quantised = [Int](repeating: 0, count: q6KWeightsPerBlock)
    var scaleBytes = [Int8](repeating: 0, count: 16)

    for sub in 0..<16 {
        let base = sub * 16
        let subValues = Array(values[base..<(base + 16)])
        let subAbsMax = subValues.map { abs($0) }.max() ?? 0

        if dFloat > 0 && subAbsMax > 0 {
            let idealScale = subAbsMax / (dFloat * 31.0)
            let clampedScale = min(127, max(-128, Int(idealScale.rounded())))
            scaleBytes[sub] = Int8(clamping: clampedScale)
        } else {
            scaleBytes[sub] = 0
        }

        let scaleFloat = dFloat * Float(scaleBytes[sub])
        for i in 0..<16 {
            let idx = base + i
            if abs(scaleFloat) > 0 {
                let raw = Int((values[idx] / scaleFloat).rounded()) + 32
                quantised[idx] = min(63, max(0, raw))
            } else {
                quantised[idx] = 32
            }
        }
    }

    for halfBlock in 0..<2 {
        let outBase = halfBlock * 128
        let qlBase = halfBlock * 64
        let qhBase = 128 + halfBlock * 32
        for lane in 0..<32 {
            let q1 = UInt8(quantised[outBase + lane])
            let q2 = UInt8(quantised[outBase + 32 + lane])
            let q3 = UInt8(quantised[outBase + 64 + lane])
            let q4 = UInt8(quantised[outBase + 96 + lane])

            block[qlBase + lane] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
            block[qlBase + 32 + lane] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
            block[qhBase + lane] = ((q1 >> 4) & 0x03)
                | (((q2 >> 4) & 0x03) << 2)
                | (((q3 >> 4) & 0x03) << 4)
                | (((q4 >> 4) & 0x03) << 6)
        }
    }

    // Pack scales: 16 signed int8 values at offset 192
    for sub in 0..<16 {
        block[192 + sub] = UInt8(bitPattern: scaleBytes[sub])
    }

    // Pack d as float16 at offset 208
    let dBits = dF16.bitPattern.littleEndian
    withUnsafeBytes(of: dBits) { pointer in
        block[208] = pointer[0]
        block[209] = pointer[1]
    }

    return block
}

private func dequantQ6KBlock(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q6KBlockByteCount)

    // Read d from offset 208 (float16)
    let dBits = UInt16(blockData[208]) | (UInt16(blockData[209]) << 8)
    let d = Float(Float16(bitPattern: dBits))

    // Read scales from offset 192 (16 signed int8 values)
    var scales = [Int8](repeating: 0, count: 16)
    for sub in 0..<16 {
        scales[sub] = Int8(bitPattern: blockData[192 + sub])
    }

    var result = [Float](repeating: 0, count: q6KWeightsPerBlock)
    for halfBlock in 0..<2 {
        let outBase = halfBlock * 128
        let qlBase = halfBlock * 64
        let qhBase = 128 + halfBlock * 32
        let scaleBase = halfBlock * 8
        for lane in 0..<32 {
            let scaleOffset = lane / 16
            let q1 = Int((blockData[qlBase + lane] & 0x0F) | (((blockData[qhBase + lane] >> 0) & 0x03) << 4)) - 32
            let q2 = Int((blockData[qlBase + 32 + lane] & 0x0F) | (((blockData[qhBase + lane] >> 2) & 0x03) << 4)) - 32
            let q3 = Int((blockData[qlBase + lane] >> 4) | (((blockData[qhBase + lane] >> 4) & 0x03) << 4)) - 32
            let q4 = Int((blockData[qlBase + 32 + lane] >> 4) | (((blockData[qhBase + lane] >> 6) & 0x03) << 4)) - 32
            result[outBase + lane] = d * Float(scales[scaleBase + scaleOffset + 0]) * Float(q1)
            result[outBase + 32 + lane] = d * Float(scales[scaleBase + scaleOffset + 2]) * Float(q2)
            result[outBase + 64 + lane] = d * Float(scales[scaleBase + scaleOffset + 4]) * Float(q3)
            result[outBase + 96 + lane] = d * Float(scales[scaleBase + scaleOffset + 6]) * Float(q4)
        }
    }

    return result
}

@Suite("Q6_K Dequantisation Kernel")
struct DequantQ6KTests {
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
        let values = (0..<q6KWeightsPerBlock).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ6KBlock(values: values)
        let expected = dequantQ6KBlock(blockData: blockData)

        let kernel = try DequantQ6KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q6KWeightsPerBlock)
        for index in 0..<q6KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-2)
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for blockIndex in 0..<superBlockCount {
            let values = (0..<q6KWeightsPerBlock).map { index in
                Float(((blockIndex * 17) + index) % 29 - 14) * 0.14
            }
            let blockData = packQ6KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ6KBlock(blockData: blockData))
        }

        let kernel = try DequantQ6KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * q6KWeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-2)
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: q6KWeightsPerBlock)
        let blockData = packQ6KBlock(values: values)
        let expected = dequantQ6KBlock(blockData: blockData)

        let kernel = try DequantQ6KKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q6KWeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q6KWeightsPerBlock).map { index in
            Float(index % 23) / 11.0 - 1.0
        }
        let blockData = packQ6KBlock(values: values)
        let dequantised = dequantQ6KBlock(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q6KWeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }

        let meanAbsoluteError = totalError / Float(q6KWeightsPerBlock)
        #expect(meanAbsoluteError < 0.15)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
