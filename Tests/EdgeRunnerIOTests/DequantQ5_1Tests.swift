import Metal
import Testing
@testable import EdgeRunnerIO

private let q5_1BlockByteCount = 24
private let q5_1WeightsPerBlock = 32

private func packQ5_1Block(values: [Float]) -> [UInt8] {
    precondition(values.count == q5_1WeightsPerBlock)
    var block = [UInt8](repeating: 0, count: q5_1BlockByteCount)

    let minValue = values.min() ?? 0
    let maxValue = values.max() ?? 0
    let range = maxValue - minValue
    let d: Float = range > 0 ? range / 31.0 : 0
    let m: Float = minValue

    let dBits = Float16(d).bitPattern.littleEndian
    let mBits = Float16(m).bitPattern.littleEndian
    withUnsafeBytes(of: dBits) { ptr in
        block[0] = ptr[0]
        block[1] = ptr[1]
    }
    withUnsafeBytes(of: mBits) { ptr in
        block[2] = ptr[0]
        block[3] = ptr[1]
    }

    for i in 0..<q5_1WeightsPerBlock {
        let quantised: Int
        if d > 0 {
            quantised = min(31, max(0, Int(round((values[i] - m) / d))))
        } else {
            quantised = 0
        }

        let lower4 = UInt8(quantised & 0x0F)
        let bit5 = UInt8((quantised >> 4) & 1)

        // Pack lower 4 bits into qs at offset 8
        if i % 2 == 0 {
            block[8 + i / 2] |= lower4
        } else {
            block[8 + i / 2] |= (lower4 << 4)
        }

        // Pack 5th bit into qh at offset 4
        block[4 + i / 8] |= (bit5 << (i % 8))
    }

    return block
}

private func dequantQ5_1Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q5_1BlockByteCount)
    let dBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let mBits = UInt16(blockData[2]) | (UInt16(blockData[3]) << 8)
    let d = Float(Float16(bitPattern: dBits))
    let m = Float(Float16(bitPattern: mBits))

    var result = [Float](repeating: 0, count: q5_1WeightsPerBlock)
    for i in 0..<q5_1WeightsPerBlock {
        let qsByte = blockData[8 + i / 2]
        let lower4: UInt8 = (i % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F)

        let qhByte = blockData[4 + i / 8]
        let bit5 = (qhByte >> (i % 8)) & 1

        let q5 = Int(lower4) | (Int(bit5) << 4)
        result[i] = d * Float(q5) + m
    }
    return result
}

@Suite("Q5_1 Dequantisation Kernel")
struct DequantQ5_1Tests {
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

    @Test func singleBlockDequant() async throws {
        let values = (0..<q5_1WeightsPerBlock).map { Float($0) * 0.2 - 3.2 }
        let blockData = packQ5_1Block(values: values)
        let expected = dequantQ5_1Block(blockData: blockData)

        let kernel = try DequantQ5_1Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == q5_1WeightsPerBlock)
        for index in 0..<q5_1WeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-3)
        }
    }

    @Test func multipleBlocksDequant() async throws {
        let blockCount = 24
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for block in 0..<blockCount {
            let values = (0..<q5_1WeightsPerBlock).map { index in
                Float(((block * 31) + index) % 23 - 11) * 0.18 + 0.5
            }
            let blockData = packQ5_1Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ5_1Block(blockData: blockData))
        }

        let kernel = try DequantQ5_1Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            blockCount: blockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == blockCount * q5_1WeightsPerBlock)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-3)
        }
    }

    @Test func zeroScaleBlock() async throws {
        let values = [Float](repeating: -2.5, count: q5_1WeightsPerBlock)
        let blockData = packQ5_1Block(values: values)
        let expected = dequantQ5_1Block(blockData: blockData)

        let kernel = try DequantQ5_1Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<q5_1WeightsPerBlock {
            #expect(abs(result[index] - expected[index]) < 1e-6)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<q5_1WeightsPerBlock).map { index in
            Float(index % 17) / 16.0 - 0.5
        }
        let blockData = packQ5_1Block(values: values)
        let dequantised = dequantQ5_1Block(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<q5_1WeightsPerBlock {
            totalError += abs(values[index] - dequantised[index])
        }
        let mae = totalError / Float(q5_1WeightsPerBlock)
        #expect(mae < 1e-2)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
