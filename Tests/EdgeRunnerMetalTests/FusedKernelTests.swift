import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// Correctness tests for fused Q8_0 GEMV kernels.
/// Validates that fused variants produce identical output to separate operations.
@Suite("Fused Kernel Correctness")
struct FusedKernelTests {

    static let device = MTLCreateSystemDefaultDevice()!
    static let queue = device.makeCommandQueue()!

    // MARK: - Helpers

    /// Create a Q8_0 quantized buffer from float data.
    /// Each block: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes.
    static func makeQ8Buffer(device: MTLDevice, rows: Int, cols: Int, values: [Float]) -> MTLBuffer {
        let blocksPerRow = cols / 32
        let totalBlocks = rows * blocksPerRow
        var data = [UInt8](repeating: 0, count: totalBlocks * 34)

        for row in 0..<rows {
            for block in 0..<blocksPerRow {
                let blockOffset = (row * blocksPerRow + block) * 34
                // Find max abs value for scale
                var maxAbs: Float = 0
                for i in 0..<32 {
                    let idx = row * cols + block * 32 + i
                    let val = idx < values.count ? values[idx] : 0
                    maxAbs = max(maxAbs, abs(val))
                }
                let scale = maxAbs / 127.0
                // Write f16 scale
                var f16Scale = Float16(scale)
                withUnsafeBytes(of: &f16Scale) { bytes in
                    data[blockOffset] = bytes[0]
                    data[blockOffset + 1] = bytes[1]
                }
                // Write int8 quants
                for i in 0..<32 {
                    let idx = row * cols + block * 32 + i
                    let val = idx < values.count ? values[idx] : 0
                    let quantized = scale > 0 ? Int8(clamping: Int(round(val / scale))) : 0
                    data[blockOffset + 2 + i] = UInt8(bitPattern: quantized)
                }
            }
        }
        return device.makeBuffer(bytes: data, length: data.count, options: .storageModeShared)!
    }

    /// CPU reference: RMSNorm
    static func cpuRMSNorm(_ input: [Float], weight: [Float], eps: Float) -> [Float] {
        let sumSq = input.reduce(0) { $0 + $1 * $1 }
        let scale = 1.0 / sqrt(sumSq / Float(input.count) + eps)
        return zip(input, weight).map { $0 * scale * $1 }
    }

    /// CPU reference: Q8_0 GEMV
    static func cpuQ8GEMV(weight: MTLBuffer, x: [Float], rows: Int, cols: Int) -> [Float] {
        let blocksPerRow = cols / 32
        let ptr = weight.contents().assumingMemoryBound(to: UInt8.self)
        var result = [Float](repeating: 0, count: rows)

        for row in 0..<rows {
            var sum: Float = 0
            for block in 0..<blocksPerRow {
                let offset = (row * blocksPerRow + block) * 34
                var f16Scale: Float16 = 0
                withUnsafeMutableBytes(of: &f16Scale) { bytes in
                    bytes[0] = ptr[offset]
                    bytes[1] = ptr[offset + 1]
                }
                let scale = Float(f16Scale)
                for i in 0..<32 {
                    let quant = Int8(bitPattern: ptr[offset + 2 + i])
                    sum += Float(quant) * scale * x[block * 32 + i]
                }
            }
            result[row] = sum
        }
        return result
    }

    static func silu(_ x: Float) -> Float {
        x / (1 + exp(-x))
    }

    // MARK: - Tests

    @Test("Base GEMV matches CPU reference")
    func testBaseGEMV() async throws {
        let rows = 64, cols = 128
        let x = (0..<cols).map { Float.random(in: -1...1) * Float($0 + 1) / Float(cols) }
        let w = (0..<rows * cols).map { _ in Float.random(in: -0.5...0.5) }

        let wBuf = Self.makeQ8Buffer(device: Self.device, rows: rows, cols: cols, values: w)
        let xBuf = Self.device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
        let yBuf = Self.device.makeBuffer(length: rows * 4, options: .storageModeShared)!

        let registry = try KernelRegistry(device: Self.device)
        let pso = try registry.pipeline(for: "dequant_q8_0_gemv")

        var params = (UInt32(rows), UInt32(cols), UInt32(cols / 32))
        let cmd = Self.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(wBuf, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBytes(&params, length: 12, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        await cmd.completed()

        let gpuResult = Array(UnsafeBufferPointer(
            start: yBuf.contents().assumingMemoryBound(to: Float.self), count: rows))
        let cpuResult = Self.cpuQ8GEMV(weight: wBuf, x: x, rows: rows, cols: cols)

        for i in 0..<rows {
            let relError = abs(gpuResult[i] - cpuResult[i]) / max(abs(cpuResult[i]), 1e-6)
            #expect(relError < 0.02, "Row \(i): GPU=\(gpuResult[i]) CPU=\(cpuResult[i]) relErr=\(relError)")
        }
    }

    @Test("Fused Norm+QKV matches separate Norm + QKV")
    func testFusedNormQKV() async throws {
        let dim = 128, qDim = 64, kvDim = 32
        let eps: Float = 1e-5

        let x = (0..<dim).map { _ in Float.random(in: -1...1) }
        let normW = (0..<dim).map { _ in Float.random(in: 0.5...1.5) }
        let wqData = (0..<qDim * dim).map { _ in Float.random(in: -0.3...0.3) }
        let wkData = (0..<kvDim * dim).map { _ in Float.random(in: -0.3...0.3) }
        let wvData = (0..<kvDim * dim).map { _ in Float.random(in: -0.3...0.3) }

        // CPU reference: norm then GEMV
        let normed = Self.cpuRMSNorm(x, weight: normW, eps: eps)
        let wqBuf = Self.makeQ8Buffer(device: Self.device, rows: qDim, cols: dim, values: wqData)
        let wkBuf = Self.makeQ8Buffer(device: Self.device, rows: kvDim, cols: dim, values: wkData)
        let wvBuf = Self.makeQ8Buffer(device: Self.device, rows: kvDim, cols: dim, values: wvData)
        let cpuQ = Self.cpuQ8GEMV(weight: wqBuf, x: normed, rows: qDim, cols: dim)
        let cpuK = Self.cpuQ8GEMV(weight: wkBuf, x: normed, rows: kvDim, cols: dim)
        _ = Self.cpuQ8GEMV(weight: wvBuf, x: normed, rows: kvDim, cols: dim)

        // GPU fused norm+QKV
        let xBuf = Self.device.makeBuffer(bytes: x, length: dim * 4, options: .storageModeShared)!
        let normBuf = Self.device.makeBuffer(bytes: normW, length: dim * 4, options: .storageModeShared)!
        let outQ = Self.device.makeBuffer(length: qDim * 4, options: .storageModeShared)!
        let outK = Self.device.makeBuffer(length: kvDim * 4, options: .storageModeShared)!
        let outV = Self.device.makeBuffer(length: kvDim * 2, options: .storageModeShared)!  // f16

        let registry = try KernelRegistry(device: Self.device)
        let pso = try registry.pipeline(for: "dequant_q8_0_fused_qkv")

        struct Params { var qR: UInt32; var kvR: UInt32; var c: UInt32; var bpr: UInt32; var tokenCount: UInt32; var eps: Float }
        var params = Params(qR: UInt32(qDim), kvR: UInt32(kvDim), c: UInt32(dim),
                           bpr: UInt32(dim / 32), tokenCount: 1, eps: eps)
        let totalRows = qDim + kvDim + kvDim

        let cmd = Self.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(wqBuf, offset: 0, index: 0)
        enc.setBuffer(wkBuf, offset: 0, index: 1)
        enc.setBuffer(wvBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3)
        enc.setBuffer(outQ, offset: 0, index: 4)
        enc.setBuffer(outK, offset: 0, index: 5)
        enc.setBuffer(outV, offset: 0, index: 6)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setBuffer(normBuf, offset: 0, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 1) / 2, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        await cmd.completed()

        let gpuQ = Array(UnsafeBufferPointer(
            start: outQ.contents().assumingMemoryBound(to: Float.self), count: qDim))
        let gpuK = Array(UnsafeBufferPointer(
            start: outK.contents().assumingMemoryBound(to: Float.self), count: kvDim))

        // Check Q and K (float32 outputs)
        for i in 0..<qDim {
            let relError = abs(gpuQ[i] - cpuQ[i]) / max(abs(cpuQ[i]), 1e-4)
            #expect(relError < 0.05, "Q[\(i)]: GPU=\(gpuQ[i]) CPU=\(cpuQ[i]) relErr=\(relError)")
        }
        for i in 0..<kvDim {
            let relError = abs(gpuK[i] - cpuK[i]) / max(abs(cpuK[i]), 1e-4)
            #expect(relError < 0.05, "K[\(i)]: GPU=\(gpuK[i]) CPU=\(cpuK[i]) relErr=\(relError)")
        }
    }

    @Test("Fused Norm+QKV supports batched prompt tokens")
    func testFusedNormQKVBatched() async throws {
        let tokenCount = 3
        let dim = 128, qDim = 64, kvDim = 32
        let eps: Float = 1e-5

        let x = (0..<(tokenCount * dim)).map { _ in Float.random(in: -1...1) }
        let normW = (0..<dim).map { _ in Float.random(in: 0.5...1.5) }
        let wqData = (0..<qDim * dim).map { _ in Float.random(in: -0.3...0.3) }
        let wkData = (0..<kvDim * dim).map { _ in Float.random(in: -0.3...0.3) }
        let wvData = (0..<kvDim * dim).map { _ in Float.random(in: -0.3...0.3) }

        let wqBuf = Self.makeQ8Buffer(device: Self.device, rows: qDim, cols: dim, values: wqData)
        let wkBuf = Self.makeQ8Buffer(device: Self.device, rows: kvDim, cols: dim, values: wkData)
        let wvBuf = Self.makeQ8Buffer(device: Self.device, rows: kvDim, cols: dim, values: wvData)

        var cpuQ = [Float]()
        var cpuK = [Float]()
        var cpuV = [Float]()
        cpuQ.reserveCapacity(tokenCount * qDim)
        cpuK.reserveCapacity(tokenCount * kvDim)
        cpuV.reserveCapacity(tokenCount * kvDim)
        for tokenIndex in 0..<tokenCount {
            let start = tokenIndex * dim
            let end = start + dim
            let tokenX = Array(x[start..<end])
            let normed = Self.cpuRMSNorm(tokenX, weight: normW, eps: eps)
            cpuQ += Self.cpuQ8GEMV(weight: wqBuf, x: normed, rows: qDim, cols: dim)
            cpuK += Self.cpuQ8GEMV(weight: wkBuf, x: normed, rows: kvDim, cols: dim)
            cpuV += Self.cpuQ8GEMV(weight: wvBuf, x: normed, rows: kvDim, cols: dim)
        }

        let xBuf = Self.device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
        let normBuf = Self.device.makeBuffer(bytes: normW, length: dim * 4, options: .storageModeShared)!
        let outQ = Self.device.makeBuffer(length: tokenCount * qDim * 4, options: .storageModeShared)!
        let outK = Self.device.makeBuffer(length: tokenCount * kvDim * 4, options: .storageModeShared)!
        let outV = Self.device.makeBuffer(length: tokenCount * kvDim * 2, options: .storageModeShared)!

        let registry = try KernelRegistry(device: Self.device)
        let pso = try registry.pipeline(for: "dequant_q8_0_fused_qkv")

        struct Params { var qR: UInt32; var kvR: UInt32; var c: UInt32; var bpr: UInt32; var tokenCount: UInt32; var eps: Float }
        var params = Params(qR: UInt32(qDim), kvR: UInt32(kvDim), c: UInt32(dim),
                           bpr: UInt32(dim / 32), tokenCount: UInt32(tokenCount), eps: eps)
        let totalRows = qDim + kvDim + kvDim

        let cmd = Self.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(wqBuf, offset: 0, index: 0)
        enc.setBuffer(wkBuf, offset: 0, index: 1)
        enc.setBuffer(wvBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3)
        enc.setBuffer(outQ, offset: 0, index: 4)
        enc.setBuffer(outK, offset: 0, index: 5)
        enc.setBuffer(outV, offset: 0, index: 6)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setBuffer(normBuf, offset: 0, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 1) / 2, height: tokenCount, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        await cmd.completed()

        let gpuQ = Array(UnsafeBufferPointer(
            start: outQ.contents().assumingMemoryBound(to: Float.self), count: tokenCount * qDim))
        let gpuK = Array(UnsafeBufferPointer(
            start: outK.contents().assumingMemoryBound(to: Float.self), count: tokenCount * kvDim))
        let gpuVHalf = Array(UnsafeBufferPointer(
            start: outV.contents().assumingMemoryBound(to: Float16.self), count: tokenCount * kvDim))
        let gpuV = gpuVHalf.map(Float.init)

        for i in 0..<gpuQ.count {
            let relError = abs(gpuQ[i] - cpuQ[i]) / max(abs(cpuQ[i]), 1e-4)
            #expect(relError < 0.05, "Q[\(i)]: GPU=\(gpuQ[i]) CPU=\(cpuQ[i]) relErr=\(relError)")
        }
        for i in 0..<gpuK.count {
            let relError = abs(gpuK[i] - cpuK[i]) / max(abs(cpuK[i]), 1e-4)
            #expect(relError < 0.05, "K[\(i)]: GPU=\(gpuK[i]) CPU=\(cpuK[i]) relErr=\(relError)")
        }
        for i in 0..<gpuV.count {
            let relError = abs(gpuV[i] - cpuV[i]) / max(abs(cpuV[i]), 1e-4)
            #expect(relError < 0.05, "V[\(i)]: GPU=\(gpuV[i]) CPU=\(cpuV[i]) relErr=\(relError)")
        }
    }

    @Test("Fused Gate+Up+SwiGLU supports batched prompt tokens")
    func testFusedGateUpSiluBatched() async throws {
        let tokenCount = 3
        let dim = 128
        let interDim = 96
        let eps: Float = 1e-5

        let x = (0..<(tokenCount * dim)).map { _ in Float.random(in: -1...1) }
        let normW = (0..<dim).map { _ in Float.random(in: 0.5...1.5) }
        let gateData = (0..<interDim * dim).map { _ in Float.random(in: -0.3...0.3) }
        let upData = (0..<interDim * dim).map { _ in Float.random(in: -0.3...0.3) }

        let gateBuf = Self.makeQ8Buffer(device: Self.device, rows: interDim, cols: dim, values: gateData)
        let upBuf = Self.makeQ8Buffer(device: Self.device, rows: interDim, cols: dim, values: upData)

        var cpuActivated = [Float]()
        cpuActivated.reserveCapacity(tokenCount * interDim)
        for tokenIndex in 0..<tokenCount {
            let start = tokenIndex * dim
            let end = start + dim
            let tokenX = Array(x[start..<end])
            let normed = Self.cpuRMSNorm(tokenX, weight: normW, eps: eps)
            let gate = Self.cpuQ8GEMV(weight: gateBuf, x: normed, rows: interDim, cols: dim)
            let up = Self.cpuQ8GEMV(weight: upBuf, x: normed, rows: interDim, cols: dim)
            cpuActivated += zip(gate, up).map { Self.silu($0) * $1 }
        }

        let xBuf = Self.device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
        let normBuf = Self.device.makeBuffer(bytes: normW, length: dim * 4, options: .storageModeShared)!
        let activatedBuf = Self.device.makeBuffer(length: tokenCount * interDim * 4, options: .storageModeShared)!

        let registry = try KernelRegistry(device: Self.device)
        let pso = try registry.pipeline(for: "dequant_q8_0_fused_gate_up_silu")

        struct Params { var rows: UInt32; var cols: UInt32; var bpr: UInt32; var tokenCount: UInt32; var eps: Float }
        var params = Params(
            rows: UInt32(interDim),
            cols: UInt32(dim),
            bpr: UInt32(dim / 32),
            tokenCount: UInt32(tokenCount),
            eps: eps
        )

        let cmd = Self.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(gateBuf, offset: 0, index: 0)
        enc.setBuffer(upBuf, offset: 0, index: 1)
        enc.setBuffer(xBuf, offset: 0, index: 2)
        enc.setBuffer(activatedBuf, offset: 0, index: 3)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 4)
        enc.setBuffer(normBuf, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (interDim + 1) / 2, height: tokenCount, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        await cmd.completed()

        let gpuActivated = Array(UnsafeBufferPointer(
            start: activatedBuf.contents().assumingMemoryBound(to: Float.self), count: tokenCount * interDim))
        for i in 0..<gpuActivated.count {
            let relError = abs(gpuActivated[i] - cpuActivated[i]) / max(abs(cpuActivated[i]), 1e-4)
            #expect(relError < 0.05, "activated[\(i)]: GPU=\(gpuActivated[i]) CPU=\(cpuActivated[i]) relErr=\(relError)")
        }
    }
}
