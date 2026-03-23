import Metal
import Testing
@testable import EdgeRunnerIO

private let q5_0BlockByteCount = 22
private let q5_0WeightsPerBlock = 32

private func packQ5_0Block(values: [Float]) -> [UInt8] {
    precondition(values.count == 32)
    let absMax = values.map { abs($0) }.max() ?? 0
    let scale = absMax > 0 ? absMax / 16.0 : 0
    var blockData = [UInt8](repeating: 0, count: q5_0BlockByteCount)

    let f16ScaleBits = Float16(scale).bitPattern.littleEndian
    withUnsafeBytes(of: f16ScaleBits) { pointer in
        blockData[0] = pointer[0]
        blockData[1] = pointer[1]
    }

    // qh at offset 2..5 (4 bytes), qs at offset 6..21 (16 bytes)
    for i in 0..<32 {
        let q5: Int
        if scale == 0 {
            q5 = 16
        } else {
            q5 = min(31, max(0, Int(round(values[i] / scale)) + 16))
        }

        let lower4 = UInt8(q5 & 0x0F)
        let bit5 = UInt8((q5 >> 4) & 1)

        // Pack lower 4 bits into qs (nibble-packed, offset 6)
        if i % 2 == 0 {
            blockData[6 + i / 2] |= lower4
        } else {
            blockData[6 + i / 2] |= (lower4 << 4)
        }

        // Pack 5th bit into qh (bit-packed, offset 2)
        blockData[2 + i / 8] |= (bit5 << (i % 8))
    }

    return blockData
}

private func dequantQ5_0Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == q5_0BlockByteCount)
    let scaleBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let scale = Float(Float16(bitPattern: scaleBits))
    var result = [Float](repeating: 0, count: q5_0WeightsPerBlock)

    for i in 0..<32 {
        // Lower 4 bits from qs (offset 6)
        let qsByte = blockData[6 + i / 2]
        let lower4: UInt8 = (i % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F)

        // 5th bit from qh (offset 2)
        let qhByte = blockData[2 + i / 8]
        let bit5 = (qhByte >> (i % 8)) & 1

        let q5 = Int(lower4) | (Int(bit5) << 4)
        result[i] = scale * Float(q5 - 16)
    }

    return result
}

@Suite("Q5_0 Dequantisation Kernel")
struct DequantQ5_0Tests {
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
        let values = (0..<32).map { Float($0) * 0.2 - 3.2 }
        let blockData = packQ5_0Block(values: values)
        let expected = dequantQ5_0Block(blockData: blockData)

        let kernel = try DequantQ5_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == 32)
        for index in 0..<32 {
            #expect(abs(result[index] - expected[index]) < 1e-3)
        }
    }

    @Test func multipleBlocksDequant() async throws {
        let blockCount = 24
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for block in 0..<blockCount {
            let values = (0..<32).map { index in
                Float(((block * 31) + index) % 23 - 11) * 0.18
            }
            let blockData = packQ5_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ5_0Block(blockData: blockData))
        }

        let kernel = try DequantQ5_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            blockCount: blockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == blockCount * 32)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-3)
        }
    }

    @Test func zeroScaleBlock() async throws {
        let values = [Float](repeating: 0, count: 32)
        let blockData = packQ5_0Block(values: values)
        let expected = dequantQ5_0Block(blockData: blockData)

        let kernel = try DequantQ5_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<32 {
            #expect(abs(result[index] - expected[index]) < 1e-6)
        }
    }

    @Test func quantisationErrorWithinTolerance() {
        let values = (0..<32).map { index in
            Float(index % 17) / 16.0 - 0.5
        }
        let blockData = packQ5_0Block(values: values)
        let dequantised = dequantQ5_0Block(blockData: blockData)

        var totalError: Float = 0
        for index in 0..<32 {
            totalError += abs(values[index] - dequantised[index])
        }
        let mae = totalError / 32.0
        #expect(mae < 1e-2)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
