import Metal
import Testing
@testable import EdgeRunnerIO

private func packQ4_0Block(values: [Float]) -> (blockData: [UInt8], scale: Float) {
    precondition(values.count == 32)
    let absMax = values.map { abs($0) }.max() ?? 0
    let scale = absMax == 0 ? 0 : absMax / 7.0
    var blockData = [UInt8](repeating: 0, count: 18)

    let f16ScaleBits = Float16(scale).bitPattern.littleEndian
    withUnsafeBytes(of: f16ScaleBits) { pointer in
        blockData[0] = pointer[0]
        blockData[1] = pointer[1]
    }

    for index in 0..<16 {
        let q0: Int
        let q1: Int
        if scale == 0 {
            q0 = 8
            q1 = 8
        } else {
            q0 = min(15, max(0, Int(round(values[index] / scale)) + 8))
            q1 = min(15, max(0, Int(round(values[index + 16] / scale)) + 8))
        }
        blockData[2 + index] = UInt8(q0) | (UInt8(q1) << 4)
    }

    return (blockData, scale)
}

private func dequantQ4_0Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == 18)
    let scaleBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let scale = Float(Float16(bitPattern: scaleBits))
    var result = [Float](repeating: 0, count: 32)

    for index in 0..<16 {
        let packed = blockData[2 + index]
        let low = Int(packed & 0x0F)
        let high = Int(packed >> 4)
        result[index] = scale * Float(low - 8)
        result[index + 16] = scale * Float(high - 8)
    }

    return result
}

@Suite("Q4_0 Dequantisation Kernel")
struct DequantQ4_0Tests {
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
        let values = (0..<32).map { Float($0) - 15.5 }
        let (blockData, _) = packQ4_0Block(values: values)
        let expected = dequantQ4_0Block(blockData: blockData)

        let kernel = try DequantQ4_0Kernel(device: device)
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
        let blockCount = 16
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for block in 0..<blockCount {
            let values = (0..<32).map { index in
                Float(block * 32 + index) * 0.1 - Float(blockCount) * 1.6
            }
            let (blockData, _) = packQ4_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ4_0Block(blockData: blockData))
        }

        let kernel = try DequantQ4_0Kernel(device: device)
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
        let (blockData, _) = packQ4_0Block(values: values)
        let expected = dequantQ4_0Block(blockData: blockData)

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<32 {
            #expect(abs(result[index] - expected[index]) < 1e-6)
        }
    }

    @Test func fusedDequantGEMVSingleRow() async throws {
        let cols = 64
        let rows = 1
        let blockCount = cols / 32

        var allBlockData: [UInt8] = []
        var allDequantised: [Float] = []
        for block in 0..<blockCount {
            let values = (0..<32).map { index in
                Float((block * 32) + index) / 31.0 - 0.5
            }
            let (blockData, _) = packQ4_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allDequantised.append(contentsOf: dequantQ4_0Block(blockData: blockData))
        }

        let x = (0..<cols).map { index in
            Float(index % 7) * 0.125 - 0.35
        }

        var expected: Float = 0
        for index in 0..<cols {
            expected += allDequantised[index] * x[index]
        }

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.fusedDequantGEMV(
            quantisedRows: allBlockData,
            x: x,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        #expect(result.count == 1)
        #expect(abs(result[0] - expected) < 0.05)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
