import Metal
import Testing
@testable import EdgeRunnerIO

private func packQ1_0_g128Block(values: [Float]) -> (blockData: [UInt8], scale: Float) {
    precondition(values.count == 128)
    let absMax = values.map { abs($0) }.max() ?? 0
    let scale = absMax == 0 ? 0 : absMax
    var blockData = [UInt8](repeating: 0, count: 18)

    // Pack scale as little-endian FP16
    let f16ScaleBits = Float16(scale).bitPattern.littleEndian
    withUnsafeBytes(of: f16ScaleBits) { pointer in
        blockData[0] = pointer[0]
        blockData[1] = pointer[1]
    }

    // Pack 128 bits into 16 bytes (LSB first bit order, 1 => +scale)
    for byteIndex in 0..<16 {
        var byteValue: UInt8 = 0
        for bitIndex in 0..<8 {
            let weightIndex = byteIndex * 8 + bitIndex
            if weightIndex < values.count {
                // 1 bit represents +scale, 0 bit represents -scale
                let positive = values[weightIndex] >= 0
                if positive {
                    byteValue |= (1 << bitIndex)
                }
            }
        }
        blockData[2 + byteIndex] = byteValue
    }

    return (blockData, scale)
}

private func dequantQ1_0_g128Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == 18)
    let scaleBits = UInt16(blockData[0]) | (UInt16(blockData[1]) << 8)
    let scale = Float(Float16(bitPattern: scaleBits))
    var result = [Float](repeating: 0, count: 128)

    // Unpack 128 bits from 16 bytes (LSB first bit order, 1 => +scale)
    for byteIndex in 0..<16 {
        let bits = blockData[2 + byteIndex]
        for bitIndex in 0..<8 {
            let weightIndex = byteIndex * 8 + bitIndex
            let bit = (bits >> bitIndex) & 1
            // 1 bit represents +scale, 0 bit represents -scale
            result[weightIndex] = bit == 1 ? scale : -scale
        }
    }

    return result
}

@Suite("Q1_0_g128 Dequantisation Kernel")
struct DequantQ1_0_g128Tests {
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
        let values = (0..<128).map { Float($0) - 63.5 }
        let (blockData, _) = packQ1_0_g128Block(values: values)
        let expected = dequantQ1_0_g128Block(blockData: blockData)

        let kernel = try DequantQ1_0_g128Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == 128)
        for index in 0..<128 {
            #expect(abs(result[index] - expected[index]) < 1e-3)
        }
    }

    @Test func multipleBlocksDequant() async throws {
        let blockCount = 16
        var allBlockData: [UInt8] = []
        var allExpected: [Float] = []

        for block in 0..<blockCount {
            let values = (0..<128).map { index in
                Float(block * 128 + index) * 0.01 - Float(blockCount) * 0.64
            }
            let (blockData, _) = packQ1_0_g128Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ1_0_g128Block(blockData: blockData))
        }

        let kernel = try DequantQ1_0_g128Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            blockCount: blockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == blockCount * 128)
        for index in result.indices {
            #expect(abs(result[index] - allExpected[index]) < 1e-3)
        }
    }

    @Test func zeroScaleBlock() async throws {
        let values = [Float](repeating: 0, count: 128)
        let (blockData, _) = packQ1_0_g128Block(values: values)
        let expected = dequantQ1_0_g128Block(blockData: blockData)

        let kernel = try DequantQ1_0_g128Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for index in 0..<128 {
            #expect(abs(result[index] - expected[index]) < 1e-6)
        }
    }

    @Test func fusedDequantGEMVSingleRow() async throws {
        let cols = 256
        let rows = 1
        let blockCount = cols / 128

        var allBlockData: [UInt8] = []
        var allDequantised: [Float] = []
        for block in 0..<blockCount {
            let values = (0..<128).map { index in
                Float((block * 128) + index) / 127.0 - 0.5
            }
            let (blockData, _) = packQ1_0_g128Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allDequantised.append(contentsOf: dequantQ1_0_g128Block(blockData: blockData))
        }

        let x = (0..<cols).map { index in
            Float(index % 7) * 0.125 - 0.35
        }

        var expected: Float = 0
        for index in 0..<cols {
            expected += allDequantised[index] * x[index]
        }

        let kernel = try DequantQ1_0_g128Kernel(device: device)
        let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared)!
        try await kernel.gemv(
            quantisedWeights: device.makeBuffer(bytes: allBlockData, length: allBlockData.count, options: .storageModeShared)!,
            input: device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared)!,
            output: outputBuffer,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        let result = outputBuffer.contents().load(as: Float.self)
        #expect(abs(result - expected) < 1e-2, "GEMV result mismatch: expected \(expected), got \(result)")
    }

    @Test func dequantGEMVMultipleRows() async throws {
        let cols = 2048
        let rows = 2048
        let blockCountPerRow = cols / 128
        let totalBlocks = rows * blockCountPerRow

        // Create random-ish weights
        var allBlockData: [UInt8] = []
        for block in 0..<totalBlocks {
            let values = (0..<128).map { index in
                Float((block * 128 + index) % 100) / 100.0 - 0.5
            }
            let (blockData, _) = packQ1_0_g128Block(values: values)
            allBlockData.append(contentsOf: blockData)
        }

        // Create input vector
        let x = (0..<cols).map { index in
            Float(index % 13) * 0.1 - 0.6
        }

        // Dequantize all weights and compute expected output in software
        var allDequantised: [Float] = []
        for block in 0..<totalBlocks {
            let bs = block * 18
            let blockData = Array(allBlockData[bs..<bs+18])
            allDequantised.append(contentsOf: dequantQ1_0_g128Block(blockData: blockData))
        }

        var expected = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            for c in 0..<cols {
                expected[r] += allDequantised[r * cols + c] * x[c]
            }
        }

        let kernel = try DequantQ1_0_g128Kernel(device: device)
        let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared)!
        try await kernel.gemv(
            quantisedWeights: device.makeBuffer(bytes: allBlockData, length: allBlockData.count, options: .storageModeShared)!,
            input: device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared)!,
            output: outputBuffer,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        let resultPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        var maxError: Float = 0
        var maxErrorRow = 0
        for r in 0..<rows {
            let err = abs(resultPtr[r] - expected[r])
            if err > maxError {
                maxError = err
                maxErrorRow = r
            }
        }

        #expect(maxError < 1.0, "Max GEMV error: \(maxError) at row \(maxErrorRow), expected \(expected[maxErrorRow]), got \(resultPtr[maxErrorRow])")
    }
}

private enum IOMetalTestError: Error {
    case noMetal
}