import Metal
import Testing
@testable import EdgeRunnerIO

private func packQ8_0Block(values: [Float]) -> (blockData: [UInt8], scale: Float) {
    precondition(values.count == 32)
    let absMax = values.map { abs($0) }.max() ?? 0
    let scale = absMax == 0 ? 0 : absMax / 127.0
    var blockData = [UInt8](repeating: 0, count: 34)

    let f16ScaleBits = Float16(scale).bitPattern.littleEndian
    withUnsafeBytes(of: f16ScaleBits) { pointer in
        blockData[0] = pointer[0]
        blockData[1] = pointer[1]
    }

    for index in 0..<32 {
        let quantised = scale == 0 ? 0 : min(127, max(-128, Int(round(values[index] / scale))))
        blockData[2 + index] = UInt8(bitPattern: Int8(quantised))
    }

    return (blockData, scale)
}

private func dequantQ8_0Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == 34)
    let scaleBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let scale = Float(Float16(bitPattern: scaleBits))

    return (0..<32).map { index in
        let quantised = Int8(bitPattern: blockData[2 + index])
        return scale * Float(quantised)
    }
}

@Suite("Q8_0 Dequantisation Kernel")
struct DequantQ8_0Tests {
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
        let values = (0..<32).map { Float($0) * 0.1 - 1.6 }
        let (blockData, _) = packQ8_0Block(values: values)
        let expected = dequantQ8_0Block(blockData: blockData)

        let kernel = try DequantQ8_0Kernel(device: device)
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
        let blockCount = 32
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for block in 0..<blockCount {
            let values = (0..<32).map { index in
                Float(((block * 31) + index) % 19 - 9) * 0.22
            }
            let (blockData, _) = packQ8_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ8_0Block(blockData: blockData))
        }

        let kernel = try DequantQ8_0Kernel(device: device)
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
        let (blockData, _) = packQ8_0Block(values: values)
        let expected = dequantQ8_0Block(blockData: blockData)

        let kernel = try DequantQ8_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<32 {
            #expect(abs(result[index] - expected[index]) < 1e-6)
        }
    }

    @Test func higherPrecisionThanQ4() {
        let values = (0..<32).map { index in
            Float(index % 13) / 12.0 - 0.45
        }
        let (q8Data, _) = packQ8_0Block(values: values)
        let q8Dequant = dequantQ8_0Block(blockData: q8Data)

        var q8Error: Float = 0
        for index in 0..<32 {
            q8Error += abs(values[index] - q8Dequant[index])
        }

        let mae = q8Error / 32.0
        #expect(mae < 0.02)
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}
