import Foundation
import Testing
import Metal
import MetalPerformanceShaders
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

private func makePaddedF16MatrixBytes(values: [Float16], rows: Int, cols: Int, rowBytes: Int) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
    for row in 0..<rows {
        for col in 0..<cols {
            let bits = values[row * cols + col].bitPattern
            let offset = row * rowBytes + col * MemoryLayout<Float16>.stride
            bytes[offset] = UInt8(bits & 0xFF)
            bytes[offset + 1] = UInt8(bits >> 8)
        }
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

    @Test func encodeF16WeightsMatchesCPUReference() async throws {
        let M = 17, K = 39
        let a = (0..<M*K).map { Float16(Float(($0 % 23) - 11) / 16.0) }
        let x = (0..<K).map { Float16(Float(($0 % 13) - 6) / 9.0) }
        let expected = cpuGemvF16(a: a, x: x, M: M, K: K)

        guard let weightBuffer = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: M * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeF16Weights(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let result = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float16.self, capacity: M),
            count: M
        ))
        for i in 0..<M {
            #expect(abs(Float(result[i]) - Float(expected[i])) < 1e-2,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func mpsF16MatrixVectorMatchesCPUReference() async throws {
        let M = 17, K = 64
        let a = (0..<M*K).map { Float16(Float(($0 % 23) - 11) / 16.0) }
        let x = (0..<K).map { Float16(Float(($0 % 13) - 6) / 9.0) }
        let expected = cpuGemvF16(a: a, x: x, M: M, K: K)
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: K, dataType: .float16)
        let matrixBytes = makePaddedF16MatrixBytes(values: a, rows: M, cols: K, rowBytes: rowBytes)

        guard let matrixBuffer = device.makeBuffer(
            bytes: matrixBytes,
            length: matrixBytes.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: M * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        let matrixDescriptor = MPSMatrixDescriptor(
            rows: M,
            columns: K,
            rowBytes: rowBytes,
            dataType: .float16
        )
        let inputDescriptor = MPSVectorDescriptor(length: K, dataType: .float16)
        let outputDescriptor = MPSVectorDescriptor(length: M, dataType: .float16)
        let matrix = MPSMatrix(buffer: matrixBuffer, descriptor: matrixDescriptor)
        let input = MPSVector(buffer: inputBuffer, descriptor: inputDescriptor)
        let output = MPSVector(buffer: outputBuffer, descriptor: outputDescriptor)
        let kernel = MPSMatrixVectorMultiplication(
            device: device,
            transpose: false,
            rows: M,
            columns: K,
            alpha: 1.0,
            beta: 0.0
        )
        kernel.encode(commandBuffer: commandBuffer, inputMatrix: matrix, inputVector: input, resultVector: output)
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let result = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float16.self, capacity: M),
            count: M
        ))
        for i in 0..<M {
            #expect(abs(Float(result[i]) - Float(expected[i])) < 1e-1,
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

    @Test func batchedBF16WeightsMatchPerTokenReference() async throws {
        let M = 19, K = 37, batchSeq = 4
        let a = (0..<M*K).map { Float(($0 % 29) - 14) / 17.0 }
        let x = (0..<batchSeq*K).map { Float(($0 % 31) - 15) / 19.0 }
        let bf16A = a.map(bf16Bits)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.executeBatchedBF16Weights(
            a: bf16A,
            x: x,
            batchSeq: batchSeq,
            M: M,
            K: K,
            commandQueue: commandQueue
        )

        #expect(result.count == batchSeq * M)
        let bf16Floats = bf16A.map { Float(bitPattern: UInt32($0) << 16) }
        for tokenIndex in 0..<batchSeq {
            let tokenX = Array(x[(tokenIndex*K)..<((tokenIndex + 1)*K)])
            let expected = cpuGemv(a: bf16Floats, x: tokenX, M: M, K: K)
            for row in 0..<M {
                let actual = result[tokenIndex * M + row]
                #expect(abs(actual - expected[row]) < 1e-4,
                        "Mismatch at batch \(tokenIndex), row \(row): GPU=\(actual) CPU=\(expected[row])")
            }
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

    @Test func batchedQ4KWeightsMatchPerTokenReference() async throws {
        let M = 7, K = 512, batchSeq = 3
        let rawWeights = makeQ4KWeights(rows: M, cols: K)
        let x = (0..<batchSeq*K).map { Float(($0 % 37) - 18) / 23.0 }

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.executeBatchedQ4KWeights(
            rawWeights: rawWeights,
            x: x,
            batchSeq: batchSeq,
            M: M,
            K: K,
            commandQueue: commandQueue
        )

        #expect(result.count == batchSeq * M)
        for tokenIndex in 0..<batchSeq {
            let tokenX = Array(x[(tokenIndex*K)..<((tokenIndex + 1)*K)])
            let expectedGPU = try await kernel.executeQ4KWeights(
                rawWeights: rawWeights,
                x: tokenX,
                M: M,
                K: K,
                commandQueue: commandQueue
            )
            let expectedCPU = cpuGemvQ4K(rawWeights: rawWeights, x: tokenX, M: M, K: K)
            for row in 0..<M {
                let actual = result[tokenIndex * M + row]
                #expect(abs(actual - expectedGPU[row]) < 1e-5,
                        "Mismatch at batch \(tokenIndex), row \(row): batched=\(actual) singleGPU=\(expectedGPU[row])")
                #expect(abs(actual - expectedCPU[row]) < 2e-4,
                        "CPU mismatch at batch \(tokenIndex), row \(row): batched=\(actual) CPU=\(expectedCPU[row])")
            }
        }
    }

    @Test func packedQ4KGemvMatchesCPUReference() async throws {
        let M = 9, K = 512
        let rawWeights = makeQ4KWeights(rows: M, cols: K)
        let x = (0..<K).map { Float(($0 % 17) - 8) / 19.0 }
        let expected = cpuGemvQ4K(rawWeights: rawWeights, x: x, M: M, K: K)

        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsPacked(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let result = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: M),
            count: M
        ))
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

    @Test func packedQ6KGemvMatchesCPUReference() async throws {
        let M = 9, K = 768
        let rawWeights = makeQ6KWeights(rows: M, cols: K)
        let x = (0..<K).map { Float(($0 % 23) - 11) / 29.0 }
        let expected = cpuGemvQ6K(rawWeights: rawWeights, x: x, M: M, K: K)

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: rawWeights, length: rawWeights.count, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: M * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ6KWeightsPacked(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            M: M,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let result = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: M),
            count: M
        ))
        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func packedQ6KTop1MatchesCPUReference() async throws {
        let M = 12, K = 768
        let rawWeights = makeQ6KWeights(rows: M, cols: K)
        let x = (0..<K).map { Float(($0 % 23) - 11) / 29.0 }
        let expected = cpuGemvQ6K(rawWeights: rawWeights, x: x, M: M, K: K)
        let expectedToken = expected.enumerated().max { lhs, rhs in lhs.element < rhs.element }!.offset
        let partialCount = M / 4

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: rawWeights, length: rawWeights.count, options: .storageModeShared),
              let partialValues = device.makeBuffer(length: partialCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let partialIndices = device.makeBuffer(length: partialCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let tokenBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ6KWeightsPackedTop1(
            commandBuffer: commandBuffer,
            weightBuffer: weightBuffer,
            inputBuffer: inputBuffer,
            partialValuesBuffer: partialValues,
            partialIndicesBuffer: partialIndices,
            outputIndexBuffer: tokenBuffer,
            M: M,
            K: K
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let token = tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        #expect(Int(token) == expectedToken)
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

    @Test func packedFourRowQ4KGemvMatchesCPUReferenceWithTailRows() async throws {
        let rows = 7
        let K = 512
        let weights = makeQ4KWeights(rows: rows, cols: K)
        let x = (0..<K).map { Float(($0 % 29) - 14) / 37.0 }
        let expected = cpuGemvQ4K(rawWeights: weights, x: x, M: rows, K: K)

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsPackedFourRows(
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

    @Test func llamaStyleQ4KGemvMatchesCPUReference() async throws {
        let rows = 7
        let K = 512
        let weights = makeQ4KWeights(rows: rows, cols: K)
        let x = (0..<K).map { Float(($0 % 29) - 14) / 37.0 }
        let expected = cpuGemvQ4K(rawWeights: weights, x: x, M: rows, K: K)

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsLlamaStyle(
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

    @Test func dualLlamaStyleQ4KGemvMatchesSeparateCPUReferences() async throws {
        let rows = 7
        let K = 512
        let weightsA = makeQ4KWeights(rows: rows, cols: K)
        var weightsB = makeQ4KWeights(rows: rows, cols: K)
        for blockStart in stride(from: 0, to: weightsB.count, by: 144) {
            for offset in 16..<144 {
                weightsB[blockStart + offset] ^= 0x33
            }
        }
        let x = (0..<K).map { Float(($0 % 29) - 14) / 37.0 }
        let expectedA = cpuGemvQ4K(rawWeights: weightsA, x: x, M: rows, K: K)
        let expectedB = cpuGemvQ4K(rawWeights: weightsB, x: x, M: rows, K: K)

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let weightBufferA = device.makeBuffer(bytes: weightsA, length: weightsA.count, options: .storageModeShared),
              let weightBufferB = device.makeBuffer(bytes: weightsB, length: weightsB.count, options: .storageModeShared),
              let outputBufferA = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outputBufferB = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernel = try GEMVKernel(device: device)
        try kernel.encodeQ4KWeightsLlamaStyleDual(
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

        let actualA = Array(UnsafeBufferPointer(
            start: outputBufferA.contents().bindMemory(to: Float.self, capacity: rows),
            count: rows
        ))
        let actualB = Array(UnsafeBufferPointer(
            start: outputBufferB.contents().bindMemory(to: Float.self, capacity: rows),
            count: rows
        ))
        for index in 0..<rows {
            #expect(abs(actualA[index] - expectedA[index]) < 1e-4)
            #expect(abs(actualB[index] - expectedB[index]) < 1e-4)
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

    @Test func packedTripleQ4KGemvMatchesSeparateCPUReferences() async throws {
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

        guard let inputBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.stride, options: .storageModeShared),
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
        try kernel.encodeQ4KWeightsTriplePacked(
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

    @Test func gemmaQ4KShapeMicrobenchmark() async throws {
        guard ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_KQUANT_MICROBENCH"] == "1" else {
            return
        }

        let kernel = try GEMVKernel(device: device)
        try await benchmarkQ4KShape(name: "gemma_local_qkv_q4k", rows: 3072, cols: 2560, kernel: kernel)
        try await benchmarkQ4KShape(name: "gemma_local_qkv_q4k_packed", rows: 3072, cols: 2560, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_local_qkv_q4k_llama_style", rows: 3072, cols: 2560, kernel: kernel, useLlamaStyle: true)
        try await benchmarkQ4KTripleShape(
            name: "gemma_local_qkv_triple_q4k",
            rowsA: 2048,
            rowsB: 512,
            rowsC: 512,
            cols: 2560,
            kernel: kernel
        )
        try await benchmarkQ4KTripleShape(
            name: "gemma_local_qkv_triple_q4k_packed",
            rowsA: 2048,
            rowsB: 512,
            rowsC: 512,
            cols: 2560,
            kernel: kernel,
            usePacked: true
        )
        try await benchmarkDenseF16Shape(name: "gemma_local_qkv_f16", rows: 3072, cols: 2560, kernel: kernel)
        try await benchmarkMPSDenseF16Shape(name: "gemma_local_qkv_mps_f16", rows: 3072, cols: 2560)
        try await benchmarkQ4KShape(name: "gemma_global_qkv_q4k", rows: 6144, cols: 2560, kernel: kernel)
        try await benchmarkQ4KShape(name: "gemma_global_qkv_q4k_packed", rows: 6144, cols: 2560, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_global_qkv_q4k_llama_style", rows: 6144, cols: 2560, kernel: kernel, useLlamaStyle: true)
        try await benchmarkQ4KTripleShape(
            name: "gemma_global_qkv_triple_q4k",
            rowsA: 4096,
            rowsB: 1024,
            rowsC: 1024,
            cols: 2560,
            kernel: kernel
        )
        try await benchmarkQ4KTripleShape(
            name: "gemma_global_qkv_triple_q4k_packed",
            rowsA: 4096,
            rowsB: 1024,
            rowsC: 1024,
            cols: 2560,
            kernel: kernel,
            usePacked: true
        )
        try await benchmarkDenseF16Shape(name: "gemma_global_qkv_f16", rows: 6144, cols: 2560, kernel: kernel)
        try await benchmarkMPSDenseF16Shape(name: "gemma_global_qkv_mps_f16", rows: 6144, cols: 2560)
        try await benchmarkQ4KShape(name: "gemma_local_attn_out_q4k_packed", rows: 2560, cols: 2048, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_local_attn_out_q4k_llama_style", rows: 2560, cols: 2048, kernel: kernel, useLlamaStyle: true)
        try await benchmarkQ4KShape(name: "gemma_global_attn_out_q4k_packed", rows: 2560, cols: 4096, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_global_attn_out_q4k_llama_style", rows: 2560, cols: 4096, kernel: kernel, useLlamaStyle: true)
        try await benchmarkQ4KShape(name: "gemma_ffn_gate_q4k", rows: 10240, cols: 2560, kernel: kernel)
        try await benchmarkQ4KShape(name: "gemma_ffn_gate_q4k_packed", rows: 10240, cols: 2560, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_ffn_gate_q4k_packed_4row", rows: 10240, cols: 2560, kernel: kernel, usePackedFourRows: true)
        try await benchmarkQ4KShape(name: "gemma_ffn_gate_q4k_llama_style", rows: 10240, cols: 2560, kernel: kernel, useLlamaStyle: true)
        try await benchmarkQ4KDualShape(name: "gemma_ffn_gate_up_dual_q4k", rows: 10240, cols: 2560, kernel: kernel)
        try await benchmarkQ4KDualLlamaStyleShape(name: "gemma_ffn_gate_up_dual_q4k_llama_style", rows: 10240, cols: 2560, kernel: kernel)
        try await benchmarkQ4KDualGeGLUShape(name: "gemma_ffn_gate_up_geglu_q4k", rows: 10240, cols: 2560, kernel: kernel)
        try await benchmarkDenseF16Shape(name: "gemma_ffn_gate_f16", rows: 10240, cols: 2560, kernel: kernel)
        try await benchmarkMPSDenseF16Shape(name: "gemma_ffn_gate_mps_f16", rows: 10240, cols: 2560)
        try await benchmarkQ4KShape(name: "gemma_ffn_down_q4k", rows: 2560, cols: 10240, kernel: kernel)
        try await benchmarkQ4KShape(name: "gemma_ffn_down_q4k_packed", rows: 2560, cols: 10240, kernel: kernel, usePacked: true)
        try await benchmarkQ4KShape(name: "gemma_ffn_down_q4k_packed_4row", rows: 2560, cols: 10240, kernel: kernel, usePackedFourRows: true)
        try await benchmarkQ4KShape(name: "gemma_ffn_down_q4k_llama_style", rows: 2560, cols: 10240, kernel: kernel, useLlamaStyle: true)
        try await benchmarkDenseF16Shape(name: "gemma_ffn_down_f16", rows: 2560, cols: 10240, kernel: kernel)
        try await benchmarkMPSDenseF16Shape(name: "gemma_ffn_down_mps_f16", rows: 2560, cols: 10240)
        try await benchmarkQ6KShape(name: "gemma_lm_head_q6k", rows: 262144, cols: 2560, kernel: kernel)
        try await benchmarkQ6KShape(name: "gemma_lm_head_q6k_packed", rows: 262144, cols: 2560, kernel: kernel, usePacked: true)
        try await benchmarkQ6KShape(name: "gemma_lm_head_q6k_top1", rows: 262144, cols: 2560, kernel: kernel, useTop1: true)
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

    private func benchmarkQ4KShape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel,
        usePacked: Bool = false,
        usePackedFourRows: Bool = false,
        useLlamaStyle: Bool = false
    ) async throws {
        let rawWeights = makeQ4KWeights(rows: rows, cols: cols)
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if usePackedFourRows {
                        try kernel.encodeQ4KWeightsPackedFourRows(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else if useLlamaStyle {
                        try kernel.encodeQ4KWeightsLlamaStyle(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else if usePacked {
                        try kernel.encodeQ4KWeightsPacked(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else {
                        try kernel.encodeQ4KWeights(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    }
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if usePackedFourRows {
                        try kernel.encodeQ4KWeightsPackedFourRows(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else if useLlamaStyle {
                        try kernel.encodeQ4KWeightsLlamaStyle(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else if usePacked {
                        try kernel.encodeQ4KWeightsPacked(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else {
                        try kernel.encodeQ4KWeights(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    }
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(rawWeights.count)
        let ioBytes = Double((cols + rows) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkQ6KShape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel,
        usePacked: Bool = false,
        useTop1: Bool = false
    ) async throws {
        let rawWeights = makeQ6KWeights(rows: rows, cols: cols)
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        let partialCount = rows / 4
        guard let weightBuffer = device.makeBuffer(
            bytes: rawWeights,
            length: rawWeights.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let partialValues = device.makeBuffer(
            length: max(1, partialCount) * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let partialIndices = device.makeBuffer(
            length: max(1, partialCount) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ),
        let outputIndex = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if useTop1 {
                        try kernel.encodeQ6KWeightsPackedTop1(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            partialValuesBuffer: partialValues,
                            partialIndicesBuffer: partialIndices,
                            outputIndexBuffer: outputIndex,
                            M: rows,
                            K: cols
                        )
                    } else if usePacked {
                        try kernel.encodeQ6KWeightsPacked(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else {
                        try kernel.encodeQ6KWeights(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    }
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if useTop1 {
                        try kernel.encodeQ6KWeightsPackedTop1(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            partialValuesBuffer: partialValues,
                            partialIndicesBuffer: partialIndices,
                            outputIndexBuffer: outputIndex,
                            M: rows,
                            K: cols
                        )
                    } else if usePacked {
                        try kernel.encodeQ6KWeightsPacked(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    } else {
                        try kernel.encodeQ6KWeights(
                            commandBuffer: commandBuffer,
                            weightBuffer: weightBuffer,
                            inputBuffer: inputBuffer,
                            outputBuffer: outputBuffer,
                            M: rows,
                            K: cols
                        )
                    }
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(rawWeights.count)
        let ioBytes = Double((cols + rows) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkQ4KDualShape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel
    ) async throws {
        let weightsA = makeQ4KWeights(rows: rows, cols: cols)
        var weightsB = makeQ4KWeights(rows: rows, cols: cols)
        for blockStart in stride(from: 0, to: weightsB.count, by: 144) {
            for offset in 16..<144 {
                weightsB[blockStart + offset] ^= 0x11
            }
        }
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        guard let weightBufferA = device.makeBuffer(bytes: weightsA, length: weightsA.count, options: .storageModeShared),
              let weightBufferB = device.makeBuffer(bytes: weightsB, length: weightsB.count, options: .storageModeShared),
              let inputBuffer = device.makeBuffer(
                bytes: input,
                length: input.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBufferA = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outputBufferB = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: weightBufferA,
                        weightBufferB: weightBufferB,
                        inputBuffer: inputBuffer,
                        outputBufferA: outputBufferA,
                        outputBufferB: outputBufferB,
                        M: rows,
                        K: cols
                    )
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: weightBufferA,
                        weightBufferB: weightBufferB,
                        inputBuffer: inputBuffer,
                        outputBufferA: outputBufferA,
                        outputBufferB: outputBufferB,
                        M: rows,
                        K: cols
                    )
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(weightsA.count + weightsB.count)
        let ioBytes = Double((cols + rows * 2) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkQ4KDualGeGLUShape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel
    ) async throws {
        let gateWeights = makeQ4KWeights(rows: rows, cols: cols)
        var upWeights = makeQ4KWeights(rows: rows, cols: cols)
        for blockStart in stride(from: 0, to: upWeights.count, by: 144) {
            for offset in 16..<144 {
                upWeights[blockStart + offset] ^= 0x11
            }
        }
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        guard let gateBuffer = device.makeBuffer(bytes: gateWeights, length: gateWeights.count, options: .storageModeShared),
              let upBuffer = device.makeBuffer(bytes: upWeights, length: upWeights.count, options: .storageModeShared),
              let inputBuffer = device.makeBuffer(
                bytes: input,
                length: input.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsDualGeGLU(
                        commandBuffer: commandBuffer,
                        gateWeightBuffer: gateBuffer,
                        upWeightBuffer: upBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: outputBuffer,
                        M: rows,
                        K: cols
                    )
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsDualGeGLU(
                        commandBuffer: commandBuffer,
                        gateWeightBuffer: gateBuffer,
                        upWeightBuffer: upBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: outputBuffer,
                        M: rows,
                        K: cols
                    )
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(gateWeights.count + upWeights.count)
        let ioBytes = Double((cols + rows) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkQ4KDualLlamaStyleShape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel
    ) async throws {
        let weightsA = makeQ4KWeights(rows: rows, cols: cols)
        var weightsB = makeQ4KWeights(rows: rows, cols: cols)
        for blockStart in stride(from: 0, to: weightsB.count, by: 144) {
            for offset in 16..<144 {
                weightsB[blockStart + offset] ^= 0x33
            }
        }
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        guard let weightBufferA = device.makeBuffer(bytes: weightsA, length: weightsA.count, options: .storageModeShared),
              let weightBufferB = device.makeBuffer(bytes: weightsB, length: weightsB.count, options: .storageModeShared),
              let inputBuffer = device.makeBuffer(
                bytes: input,
                length: input.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBufferA = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outputBufferB = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsLlamaStyleDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: weightBufferA,
                        weightBufferB: weightBufferB,
                        inputBuffer: inputBuffer,
                        outputBufferA: outputBufferA,
                        outputBufferB: outputBufferB,
                        M: rows,
                        K: cols
                    )
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeQ4KWeightsLlamaStyleDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: weightBufferA,
                        weightBufferB: weightBufferB,
                        inputBuffer: inputBuffer,
                        outputBufferA: outputBufferA,
                        outputBufferB: outputBufferB,
                        M: rows,
                        K: cols
                    )
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(weightsA.count + weightsB.count)
        let ioBytes = Double((cols + rows * 2) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkQ4KTripleShape(
        name: String,
        rowsA: Int,
        rowsB: Int,
        rowsC: Int,
        cols: Int,
        kernel: GEMVKernel,
        usePacked: Bool = false
    ) async throws {
        let weightsA = makeQ4KWeights(rows: rowsA, cols: cols)
        var weightsB = makeQ4KWeights(rows: rowsB, cols: cols)
        var weightsC = makeQ4KWeights(rows: rowsC, cols: cols)
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
        let input = (0..<cols).map { Float(($0 % 31) - 15) / 41.0 }
        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBufferA = device.makeBuffer(bytes: weightsA, length: weightsA.count, options: .storageModeShared),
        let weightBufferB = device.makeBuffer(bytes: weightsB, length: weightsB.count, options: .storageModeShared),
        let weightBufferC = device.makeBuffer(bytes: weightsC, length: weightsC.count, options: .storageModeShared),
        let outputBufferA = device.makeBuffer(length: rowsA * MemoryLayout<Float>.stride, options: .storageModeShared),
        let outputBufferB = device.makeBuffer(length: rowsB * MemoryLayout<Float>.stride, options: .storageModeShared),
        let outputBufferC = device.makeBuffer(length: rowsC * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if usePacked {
                        try kernel.encodeQ4KWeightsTriplePacked(
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
                            K: cols
                        )
                    } else {
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
                            K: cols
                        )
                    }
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    if usePacked {
                        try kernel.encodeQ4KWeightsTriplePacked(
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
                            K: cols
                        )
                    } else {
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
                            K: cols
                        )
                    }
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(weightsA.count + weightsB.count + weightsC.count)
        let ioBytes = Double((cols + rowsA + rowsB + rowsC) * MemoryLayout<Float>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rowsA=\(rowsA) rowsB=\(rowsB) rowsC=\(rowsC) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkDenseF16Shape(
        name: String,
        rows: Int,
        cols: Int,
        kernel: GEMVKernel
    ) async throws {
        let weights = (0..<(rows * cols)).map { Float16(Float(($0 % 37) - 18) / 67.0) }
        let input = (0..<cols).map { Float16(Float(($0 % 31) - 15) / 41.0) }
        guard let weightBuffer = device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: rows * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeF16Weights(
                        commandBuffer: commandBuffer,
                        weightBuffer: weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: outputBuffer,
                        M: rows,
                        K: cols
                    )
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    try kernel.encodeF16Weights(
                        commandBuffer: commandBuffer,
                        weightBuffer: weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: outputBuffer,
                        M: rows,
                        K: cols
                    )
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(weights.count * MemoryLayout<Float16>.stride)
        let ioBytes = Double((cols + rows) * MemoryLayout<Float16>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func benchmarkMPSDenseF16Shape(
        name: String,
        rows: Int,
        cols: Int
    ) async throws {
        let weights = (0..<(rows * cols)).map { Float16(Float(($0 % 37) - 18) / 67.0) }
        let input = (0..<cols).map { Float16(Float(($0 % 31) - 15) / 41.0) }
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: cols, dataType: .float16)
        let matrixBytes = rowBytes == cols * MemoryLayout<Float16>.stride
            ? weights.withUnsafeBytes { Array($0) }
            : makePaddedF16MatrixBytes(values: weights, rows: rows, cols: cols, rowBytes: rowBytes)
        guard let matrixBuffer = device.makeBuffer(
            bytes: matrixBytes,
            length: matrixBytes.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: rows * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw GEMVError.encodingFailed
        }

        let matrixDescriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: cols,
            rowBytes: rowBytes,
            dataType: .float16
        )
        let inputDescriptor = MPSVectorDescriptor(length: cols, dataType: .float16)
        let outputDescriptor = MPSVectorDescriptor(length: rows, dataType: .float16)
        let matrix = MPSMatrix(buffer: matrixBuffer, descriptor: matrixDescriptor)
        let inputVector = MPSVector(buffer: inputBuffer, descriptor: inputDescriptor)
        let outputVector = MPSVector(buffer: outputBuffer, descriptor: outputDescriptor)
        let mpsKernel = MPSMatrixVectorMultiplication(
            device: device,
            transpose: false,
            rows: rows,
            columns: cols,
            alpha: 1.0,
            beta: 0.0
        )

        for _ in 0..<2 {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    mpsKernel.encode(
                        commandBuffer: commandBuffer,
                        inputMatrix: matrix,
                        inputVector: inputVector,
                        resultVector: outputVector
                    )
                }
            )
        }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            try await runEncodedBenchmarkIteration(
                name: name,
                encode: { commandBuffer in
                    mpsKernel.encode(
                        commandBuffer: commandBuffer,
                        inputMatrix: matrix,
                        inputVector: inputVector,
                        resultVector: outputVector
                    )
                }
            )
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerIteration = seconds / Double(iterations) * 1000
        let weightBytes = Double(matrixBytes.count)
        let ioBytes = Double((cols + rows) * MemoryLayout<Float16>.stride)
        let gbPerSecond = ((weightBytes + ioBytes) * Double(iterations)) / seconds / 1e9

        print(
            "BENCHMARK: \(name) \(String(format: "%.3f", msPerIteration)) ms/op "
            + "\(String(format: "%.1f", gbPerSecond)) GB/s rows=\(rows) cols=\(cols)"
        )
        #expect(msPerIteration > 0)
    }

    private func runEncodedBenchmarkIteration(
        name: String,
        encode: (MTLCommandBuffer) throws -> Void
    ) async throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GEMVError.encodingFailed
        }
        commandBuffer.label = name
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }
    }
}
