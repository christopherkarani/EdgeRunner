import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference matvec: y[M] = A[M,K] * x[K]
private func cpuGemv(a: [Float], x: [Float], M: Int, K: Int) -> [Float] {
    var y = [Float](repeating: 0, count: M)
    for i in 0..<M {
        var sum: Float = 0
        for j in 0..<K {
            sum += a[i * K + j] * x[j]
        }
        y[i] = sum
    }
    return y
}

/// CPU reference matvec Float16
private func cpuGemvF16(a: [Float16], x: [Float16], M: Int, K: Int) -> [Float16] {
    var y = [Float16](repeating: 0, count: M)
    for i in 0..<M {
        var sum: Float = 0
        for j in 0..<K {
            sum += Float(a[i * K + j]) * Float(x[j])
        }
        y[i] = Float16(sum)
    }
    return y
}

private func bf16Bits(_ value: Float) -> UInt16 {
    UInt16(value.bitPattern >> 16)
}

private func cpuGeluTanh(_ value: Float) -> Float {
    if value > 10 {
        return value
    }
    if value < -10 {
        return 0
    }
    let c: Float = 0.7978845608028654
    let inner = c * (value + 0.044715 * value * value * value)
    return value * 0.5 * (1 + tanh(inner))
}

private func writeF16(_ value: Float, into bytes: inout [UInt8], at offset: Int) {
    let bits = Float16(value).bitPattern
    bytes[offset] = UInt8(bits & 0xFF)
    bytes[offset + 1] = UInt8(bits >> 8)
}

private func readF16(_ bytes: [UInt8], at offset: Int) -> Float {
    let bits = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    return Float(Float16(bitPattern: bits))
}

private func cpuGemvQ4K(rawWeights: [UInt8], x: [Float], M: Int, K: Int) -> [Float] {
    let blockBytes = 144
    let blocksPerRow = K / 256
    var output = [Float](repeating: 0, count: M)
    for row in 0..<M {
        var sum: Float = 0
        let rowBase = row * blocksPerRow * blockBytes
        for blockIndex in 0..<blocksPerRow {
            let block = rowBase + blockIndex * blockBytes
            let d = readF16(rawWeights, at: block)
            let dmin = readF16(rawWeights, at: block + 2)
            for subBlock in 0..<8 {
                let scaleByte = rawWeights[block + 4 + (subBlock & 3)]
                let minByte = rawWeights[block + 8 + (subBlock & 3)]
                let highBits = rawWeights[block + 12 + (subBlock & 3)]
                let scale: Float
                let minValue: Float
                if subBlock < 4 {
                    scale = d * Float(scaleByte & 0x3F)
                    minValue = dmin * Float(minByte & 0x3F)
                } else {
                    scale = d * Float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4))
                    minValue = dmin * Float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4))
                }
                for index in 0..<32 {
                    let byteIndex = block + 16 + (subBlock / 2) * 32 + index
                    let packed = rawWeights[byteIndex]
                    let nibble = subBlock.isMultiple(of: 2) ? (packed & 0x0F) : ((packed >> 4) & 0x0F)
                    let col = blockIndex * 256 + subBlock * 32 + index
                    sum += (scale * Float(nibble) - minValue) * x[col]
                }
            }
        }
        output[row] = sum
    }
    return output
}

private func cpuGemvQ6K(rawWeights: [UInt8], x: [Float], M: Int, K: Int) -> [Float] {
    let blockBytes = 210
    let blocksPerRow = K / 256
    var output = [Float](repeating: 0, count: M)
    for row in 0..<M {
        var sum: Float = 0
        let rowBase = row * blocksPerRow * blockBytes
        for blockIndex in 0..<blocksPerRow {
            let block = rowBase + blockIndex * blockBytes
            let d = readF16(rawWeights, at: block + 208)
            for halfBlock in 0..<2 {
                let outBase = halfBlock * 128
                let qlBase = block + halfBlock * 64
                let qhBase = block + 128 + halfBlock * 32
                let scaleBase = block + 192 + halfBlock * 8
                for lane in 0..<32 {
                    let scaleOffset = lane / 16
                    let q1 = Int((rawWeights[qlBase + lane] & 0x0F) | (((rawWeights[qhBase + lane] >> 0) & 0x03) << 4)) - 32
                    let q2 = Int((rawWeights[qlBase + 32 + lane] & 0x0F) | (((rawWeights[qhBase + lane] >> 2) & 0x03) << 4)) - 32
                    let q3 = Int((rawWeights[qlBase + lane] >> 4) | (((rawWeights[qhBase + lane] >> 4) & 0x03) << 4)) - 32
                    let q4 = Int((rawWeights[qlBase + 32 + lane] >> 4) | (((rawWeights[qhBase + lane] >> 6) & 0x03) << 4)) - 32
                    let s1 = Int8(bitPattern: rawWeights[scaleBase + scaleOffset + 0])
                    let s2 = Int8(bitPattern: rawWeights[scaleBase + scaleOffset + 2])
                    let s3 = Int8(bitPattern: rawWeights[scaleBase + scaleOffset + 4])
                    let s4 = Int8(bitPattern: rawWeights[scaleBase + scaleOffset + 6])
                    sum += d * Float(s1) * Float(q1) * x[blockIndex * 256 + outBase + lane]
                    sum += d * Float(s2) * Float(q2) * x[blockIndex * 256 + outBase + 32 + lane]
                    sum += d * Float(s3) * Float(q3) * x[blockIndex * 256 + outBase + 64 + lane]
                    sum += d * Float(s4) * Float(q4) * x[blockIndex * 256 + outBase + 96 + lane]
                }
            }
        }
        output[row] = sum
    }
    return output
}

private func makeQ4KWeights(rows: Int, cols: Int) -> [UInt8] {
    let blocks = rows * cols / 256
    var bytes = [UInt8](repeating: 0, count: blocks * 144)
    for blockIndex in 0..<blocks {
        let base = blockIndex * 144
        writeF16(0.03125 + Float(blockIndex % 3) * 0.0078125, into: &bytes, at: base)
        writeF16(0.00390625, into: &bytes, at: base + 2)
        for i in 0..<12 {
            bytes[base + 4 + i] = UInt8((blockIndex * 17 + i * 13) & 0xFF)
        }
        for i in 0..<128 {
            bytes[base + 16 + i] = UInt8((blockIndex * 31 + i * 7) & 0xFF)
        }
    }
    return bytes
}

private func makeQ6KWeights(rows: Int, cols: Int) -> [UInt8] {
    let blocks = rows * cols / 256
    var bytes = [UInt8](repeating: 0, count: blocks * 210)
    for blockIndex in 0..<blocks {
        let base = blockIndex * 210
        for i in 0..<128 {
            bytes[base + i] = UInt8((blockIndex * 11 + i * 5) & 0xFF)
        }
        for i in 0..<64 {
            bytes[base + 128 + i] = UInt8((blockIndex * 19 + i * 3) & 0xFF)
        }
        for i in 0..<16 {
            bytes[base + 192 + i] = UInt8(bitPattern: Int8((i % 7) - 3))
        }
        writeF16(0.015625 + Float(blockIndex % 5) * 0.001953125, into: &bytes, at: base + 208)
    }
    return bytes
}

@Suite("GEMV Kernel")
struct GEMVTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func smallGemvFloat32() async throws {
        let M = 64, K = 128
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func largeGemvFloat32() async throws {
        let M = 4096, K = 4096
        let a = (0..<M*K).map { _ in Float.random(in: -0.1...0.1) }
        let x = (0..<K).map { _ in Float.random(in: -0.1...0.1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func nonAlignedDimensions() async throws {
        let M = 37, K = 73
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func identityGemv() async throws {
        let N = 64
        var identity = [Float](repeating: 0, count: N * N)
        for i in 0..<N { identity[i * N + i] = 1.0 }
        let x = (0..<N).map { _ in Float.random(in: -1...1) }

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: identity, x: x, M: N, K: N,
            commandQueue: commandQueue
        )

        for i in 0..<N {
            #expect(abs(result[i] - x[i]) < 1e-5,
                    "Identity gemv failed at [\(i)]")
        }
    }

    @Test func gemvFloat16() async throws {
        let M = 64, K = 128
        let a = (0..<M*K).map { _ in Float16.random(in: -1...1) }
        let x = (0..<K).map { _ in Float16.random(in: -1...1) }
        let expected = cpuGemvF16(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result: [Float16] = try await kernel.executeF16(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(Float(result[i]) - Float(expected[i])) < 1e-2,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func gemvBFloat16WeightsFloatInput() async throws {
        let M = 17, K = 39
        let a = (0..<M*K).map { Float(($0 % 23) - 11) / 16.0 }
        let x = (0..<K).map { Float(($0 % 13) - 6) / 9.0 }
        let bf16A = a.map(bf16Bits)
        let expected = cpuGemv(
            a: bf16A.map { Float(bitPattern: UInt32($0) << 16) },
            x: x,
            M: M,
            K: K
        )

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.executeBF16Weights(
            a: bf16A,
            x: x,
            M: M,
            K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func gemvQ4KWeightsMatchCPUReference() async throws {
        let M = 9, K = 512
        let rawWeights = makeQ4KWeights(rows: M, cols: K)
        let x = (0..<K).map { Float(($0 % 17) - 8) / 19.0 }
        let expected = cpuGemvQ4K(rawWeights: rawWeights, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.executeQ4KWeights(
            rawWeights: rawWeights,
            x: x,
            M: M,
            K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func gemvQ6KWeightsMatchCPUReference() async throws {
        let M = 7, K = 512
        let rawWeights = makeQ6KWeights(rows: M, cols: K)
        let x = (0..<K).map { Float(($0 % 23) - 11) / 29.0 }
        let expected = cpuGemvQ6K(rawWeights: rawWeights, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.executeQ6KWeights(
            rawWeights: rawWeights,
            x: x,
            M: M,
            K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func encodedKQuantGemvsShareOneCommandBuffer() async throws {
        let q4Rows = 5
        let q6Rows = 4
        let K = 512
        let q4Weights = makeQ4KWeights(rows: q4Rows, cols: K)
        let q6Weights = makeQ6KWeights(rows: q6Rows, cols: K)
        let x = (0..<K).map { Float(($0 % 19) - 9) / 23.0 }
        let expectedQ4 = cpuGemvQ4K(rawWeights: q4Weights, x: x, M: q4Rows, K: K)
        let expectedQ6 = cpuGemvQ6K(rawWeights: q6Weights, x: x, M: q6Rows, K: K)

        guard let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let q4Buffer = device.makeBuffer(
            bytes: q4Weights,
            length: q4Weights.count,
            options: .storageModeShared
        ),
        let q6Buffer = device.makeBuffer(
            bytes: q6Weights,
            length: q6Weights.count,
            options: .storageModeShared
        ),
        let q4Output = device.makeBuffer(
            length: q4Rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let q6Output = device.makeBuffer(
            length: q6Rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeights(
            commandBuffer: commandBuffer,
            weightBuffer: q4Buffer,
            inputBuffer: inputBuffer,
            outputBuffer: q4Output,
            M: q4Rows,
            K: K
        )
        try kernel.encodeQ6KWeights(
            commandBuffer: commandBuffer,
            weightBuffer: q6Buffer,
            inputBuffer: inputBuffer,
            outputBuffer: q6Output,
            M: q6Rows,
            K: K
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let q4Pointer = q4Output.contents().bindMemory(to: Float.self, capacity: q4Rows)
        let q6Pointer = q6Output.contents().bindMemory(to: Float.self, capacity: q6Rows)
        let q4Result = Array(UnsafeBufferPointer(start: q4Pointer, count: q4Rows))
        let q6Result = Array(UnsafeBufferPointer(start: q6Pointer, count: q6Rows))

        for index in 0..<q4Rows {
            #expect(abs(q4Result[index] - expectedQ4[index]) < 1e-4)
        }
        for index in 0..<q6Rows {
            #expect(abs(q6Result[index] - expectedQ6[index]) < 1e-4)
        }
    }

    @Test func dualQ4KGemvMatchesSeparateCPUReferences() async throws {
        let rows = 7
        let K = 512
        let weightsA = makeQ4KWeights(rows: rows, cols: K)
        var weightsB = makeQ4KWeights(rows: rows, cols: K)
        for blockStart in stride(from: 0, to: weightsB.count, by: 144) {
            for offset in 16..<144 {
                weightsB[blockStart + offset] ^= 0x11
            }
        }
        let x = (0..<K).map { Float(($0 % 23) - 11) / 29.0 }
        let expectedA = cpuGemvQ4K(rawWeights: weightsA, x: x, M: rows, K: K)
        let expectedB = cpuGemvQ4K(rawWeights: weightsB, x: x, M: rows, K: K)

        guard let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBufferA = device.makeBuffer(
            bytes: weightsA,
            length: weightsA.count,
            options: .storageModeShared
        ),
        let weightBufferB = device.makeBuffer(
            bytes: weightsB,
            length: weightsB.count,
            options: .storageModeShared
        ),
        let outputBufferA = device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBufferB = device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsDual(
            commandBuffer: commandBuffer,
            weightBufferA: weightBufferA,
            weightBufferB: weightBufferB,
            inputBuffer: inputBuffer,
            outputBufferA: outputBufferA,
            outputBufferB: outputBufferB,
            M: rows,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointerA = outputBufferA.contents().bindMemory(to: Float.self, capacity: rows)
        let pointerB = outputBufferB.contents().bindMemory(to: Float.self, capacity: rows)
        let actualA = Array(UnsafeBufferPointer(start: pointerA, count: rows))
        let actualB = Array(UnsafeBufferPointer(start: pointerB, count: rows))

        for index in 0..<rows {
            #expect(abs(actualA[index] - expectedA[index]) < 1e-4)
            #expect(abs(actualB[index] - expectedB[index]) < 1e-4)
        }
    }

    @Test func dualQ4KGeGLUMatchesSeparateCPUReference() async throws {
        let rows = 7
        let K = 512
        let gateWeights = makeQ4KWeights(rows: rows, cols: K)
        var upWeights = makeQ4KWeights(rows: rows, cols: K)
        for blockStart in stride(from: 0, to: upWeights.count, by: 144) {
            for offset in 16..<144 {
                upWeights[blockStart + offset] ^= 0x11
            }
        }
        let x = (0..<K).map { Float(($0 % 23) - 11) / 29.0 }
        let gate = cpuGemvQ4K(rawWeights: gateWeights, x: x, M: rows, K: K)
        let up = cpuGemvQ4K(rawWeights: upWeights, x: x, M: rows, K: K)
        let expected = zip(gate, up).map { gateValue, upValue in
            cpuGeluTanh(gateValue) * upValue
        }

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let gateBuffer = device.makeBuffer(bytes: gateWeights, length: gateWeights.count, options: .storageModeShared),
              let upBuffer = device.makeBuffer(bytes: upWeights, length: upWeights.count, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsDualGeGLU(
            commandBuffer: commandBuffer,
            gateWeightBuffer: gateBuffer,
            upWeightBuffer: upBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: rows,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let actual = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: rows),
            count: rows
        ))
        for index in 0..<rows {
            #expect(abs(actual[index] - expected[index]) < 5e-3)
        }
    }

    @Test func twoRowQ4KGemvMatchesCPUReference() async throws {
        let rows = 7
        let K = 512
        let weights = makeQ4KWeights(rows: rows, cols: K)
        let x = (0..<K).map { Float(($0 % 19) - 9) / 23.0 }
        let expected = cpuGemvQ4K(rawWeights: weights, x: x, M: rows, K: K)

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsTwoRows(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: rows,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let actual = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: rows),
            count: rows
        ))
        for index in 0..<rows {
            #expect(abs(actual[index] - expected[index]) < 1e-4)
        }
    }

    @Test func tripleQ4KGemvMatchesSeparateCPUReferences() async throws {
        let rowsA = 6
        let rowsB = 3
        let rowsC = 4
        let K = 512
        let weightsA = makeQ4KWeights(rows: rowsA, cols: K)
        var weightsB = makeQ4KWeights(rows: rowsB, cols: K)
        var weightsC = makeQ4KWeights(rows: rowsC, cols: K)
        for blockStart in stride(from: 0, to: weightsB.count, by: 144) {
            for offset in 16..<144 {
                weightsB[blockStart + offset] ^= 0x11
            }
        }
        for blockStart in stride(from: 0, to: weightsC.count, by: 144) {
            for offset in 16..<144 {
                weightsC[blockStart + offset] ^= 0x22
            }
        }
        let x = (0..<K).map { Float(($0 % 31) - 15) / 41.0 }
        let expectedA = cpuGemvQ4K(rawWeights: weightsA, x: x, M: rowsA, K: K)
        let expectedB = cpuGemvQ4K(rawWeights: weightsB, x: x, M: rowsB, K: K)
        let expectedC = cpuGemvQ4K(rawWeights: weightsC, x: x, M: rowsC, K: K)

        guard let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBufferA = device.makeBuffer(bytes: weightsA, length: weightsA.count, options: .storageModeShared),
        let weightBufferB = device.makeBuffer(bytes: weightsB, length: weightsB.count, options: .storageModeShared),
        let weightBufferC = device.makeBuffer(bytes: weightsC, length: weightsC.count, options: .storageModeShared),
        let outputBufferA = device.makeBuffer(length: rowsA * MemoryLayout<Float>.stride, options: .storageModeShared),
        let outputBufferB = device.makeBuffer(length: rowsB * MemoryLayout<Float>.stride, options: .storageModeShared),
        let outputBufferC = device.makeBuffer(length: rowsC * MemoryLayout<Float>.stride, options: .storageModeShared),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsTriple(
            commandBuffer: commandBuffer,
            weightBufferA: weightBufferA,
            weightBufferB: weightBufferB,
            weightBufferC: weightBufferC,
            inputBuffer: inputBuffer,
            outputBufferA: outputBufferA,
            outputBufferB: outputBufferB,
            outputBufferC: outputBufferC,
            rowsA: rowsA,
            rowsB: rowsB,
            rowsC: rowsC,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let actualA = Array(UnsafeBufferPointer(
            start: outputBufferA.contents().bindMemory(to: Float.self, capacity: rowsA),
            count: rowsA
        ))
        let actualB = Array(UnsafeBufferPointer(
            start: outputBufferB.contents().bindMemory(to: Float.self, capacity: rowsB),
            count: rowsB
        ))
        let actualC = Array(UnsafeBufferPointer(
            start: outputBufferC.contents().bindMemory(to: Float.self, capacity: rowsC),
            count: rowsC
        ))

        for index in 0..<rowsA {
            #expect(abs(actualA[index] - expectedA[index]) < 1e-4)
        }
        for index in 0..<rowsB {
            #expect(abs(actualB[index] - expectedB[index]) < 1e-4)
        }
        for index in 0..<rowsC {
            #expect(abs(actualC[index] - expectedC[index]) < 1e-4)
        }
    }

    @Test func singleRowGemv() async throws {
        // Degenerate case: 1xK * Kx1 = scalar
        let M = 1, K = 256
        let a = (0..<K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        #expect(abs(result[0] - expected[0]) < 1e-4)
    }

    @Test func matrixShapeMismatchThrows() async throws {
        let kernel = try GEMVKernel(device: device)

        do {
            _ = try await kernel.execute(
                a: [1, 2, 3],
                x: [1, 2],
                M: 2,
                K: 2,
                commandQueue: commandQueue
            )
            Issue.record("Expected invalidMatrixShape for undersized matrix input")
        } catch let error as GEMVError {
            if case .invalidMatrixShape = error {
                return
            }
            Issue.record("Expected invalidMatrixShape, got \(error)")
        }
    }

    @Test func vectorShapeMismatchThrowsFloat16() async throws {
        let kernel = try GEMVKernel(device: device)

        do {
            _ = try await kernel.executeF16(
                a: [1, 0, 0, 1],
                x: [1],
                M: 2,
                K: 2,
                commandQueue: commandQueue
            )
            Issue.record("Expected invalidVectorShape for undersized Float16 vector input")
        } catch let error as GEMVError {
            if case .invalidVectorShape = error {
                return
            }
            Issue.record("Expected invalidVectorShape, got \(error)")
        }
    }
}
