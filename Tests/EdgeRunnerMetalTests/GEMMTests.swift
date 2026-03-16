import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("GEMMKernel")
struct GEMMTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = q
        self.registry = try KernelRegistry(device: d)
    }

    // MARK: - CPU Reference

    private func cpuGEMM(a: [Float], b: [Float], M: Int, N: Int, K: Int) -> [Float] {
        var c = [Float](repeating: 0, count: M * N)
        for i in 0..<M {
            for j in 0..<N {
                var sum: Float = 0
                for k in 0..<K {
                    sum += a[i * K + k] * b[k * N + j]
                }
                c[i * N + j] = sum
            }
        }
        return c
    }

    private func dispatchGEMM_f32(a: [Float], b: [Float], M: Int, N: Int, K: Int) throws -> [Float] {
        let kernel = try GEMMKernel(registry: registry)
        let pipeline = kernel.f32Pipeline
        let bufA = device.makeBuffer(bytes: a, length: a.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: b.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let bufC = device.makeBuffer(length: M * N * MemoryLayout<Float>.size, options: .storageModeShared)!

        var params = ERGEMMParams(
            M: UInt32(M), N: UInt32(N), K: UInt32(K),
            lda: UInt32(K), ldb: UInt32(N), ldc: UInt32(N)
        )

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMMParams>.size, index: 3)

        let gridSize = MTLSize(width: N, height: M, depth: 1)
        let tgSize = MTLSize(
            width: min(32, pipeline.maxTotalThreadsPerThreadgroup),
            height: min(32, pipeline.maxTotalThreadsPerThreadgroup / min(32, pipeline.maxTotalThreadsPerThreadgroup)),
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: ptr, count: M * N))
    }

    private func dispatchGEMM_f16(a: [Float], b: [Float], M: Int, N: Int, K: Int) throws -> [Float] {
        // Convert Float32 -> Float16 for input
        let a16 = a.map { Float16($0) }
        let b16 = b.map { Float16($0) }

        let bufA = device.makeBuffer(bytes: a16, length: a16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b16, length: b16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let bufC = device.makeBuffer(length: M * N * MemoryLayout<Float16>.size, options: .storageModeShared)!

        var params = ERGEMMParams(
            M: UInt32(M), N: UInt32(N), K: UInt32(K),
            lda: UInt32(K), ldb: UInt32(N), ldc: UInt32(N)
        )

        let kernel = try GEMMKernel(registry: registry)
        let pipeline = kernel.f16Pipeline
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMMParams>.size, index: 3)

        let gridSize = MTLSize(width: N, height: M, depth: 1)
        let tgSize = MTLSize(
            width: min(32, pipeline.maxTotalThreadsPerThreadgroup),
            height: min(32, pipeline.maxTotalThreadsPerThreadgroup / min(32, pipeline.maxTotalThreadsPerThreadgroup)),
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufC.contents().bindMemory(to: Float16.self, capacity: M * N)
        let result16 = Array(UnsafeBufferPointer(start: ptr, count: M * N))
        return result16.map { Float($0) }
    }

    // MARK: - Tests

    @Test func smallSquareF32() throws {
        // 4x4 * 4x4
        let M = 4, N = 4, K = 4
        let a: [Float] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ]
        let b: [Float] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        let expected = cpuGEMM(a: a, b: b, M: M, N: N, K: K)
        let result = try dispatchGEMM_f32(a: a, b: b, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-4, "Mismatch at \(i): \(result[i]) vs \(expected[i])")
        }
    }

    @Test func rectangularF32() throws {
        // 3x5 * 5x4
        let M = 3, N = 4, K = 5
        let a: [Float] = (0..<(M * K)).map { Float($0) * 0.1 }
        let b: [Float] = (0..<(K * N)).map { Float($0) * 0.2 }
        let expected = cpuGEMM(a: a, b: b, M: M, N: N, K: K)
        let result = try dispatchGEMM_f32(a: a, b: b, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-4, "Mismatch at \(i): \(result[i]) vs \(expected[i])")
        }
    }

    @Test func largerF32() throws {
        // 64x64 * 64x64
        let M = 64, N = 64, K = 64
        let a: [Float] = (0..<(M * K)).map { _ in Float.random(in: -1...1) }
        let b: [Float] = (0..<(K * N)).map { _ in Float.random(in: -1...1) }
        let expected = cpuGEMM(a: a, b: b, M: M, N: N, K: K)
        let result = try dispatchGEMM_f32(a: a, b: b, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-3, "Mismatch at \(i): \(result[i]) vs \(expected[i])")
        }
    }

    @Test func smallSquareF16() throws {
        // 4x4 * 4x4 identity
        let M = 4, N = 4, K = 4
        let a: [Float] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ]
        let b: [Float] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        let expected = cpuGEMM(a: a, b: b, M: M, N: N, K: K)
        let result = try dispatchGEMM_f16(a: a, b: b, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-2, "F16 mismatch at \(i): \(result[i]) vs \(expected[i])")
        }
    }

    @Test func identityMatrix() throws {
        // Multiplying by identity should return the original matrix
        let M = 8, N = 8, K = 8
        let a: [Float] = (0..<(M * K)).map { Float($0 + 1) }
        var identity = [Float](repeating: 0, count: K * N)
        for i in 0..<K { identity[i * N + i] = 1.0 }
        let result = try dispatchGEMM_f32(a: a, b: identity, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - a[i]) < 1e-4, "Identity mismatch at \(i): \(result[i]) vs \(a[i])")
        }
    }

    @Test func nonMultipleOf32() throws {
        // 17x13 * 13x11 — dimensions not multiples of 32
        let M = 17, N = 11, K = 13
        let a: [Float] = (0..<(M * K)).map { _ in Float.random(in: -1...1) }
        let b: [Float] = (0..<(K * N)).map { _ in Float.random(in: -1...1) }
        let expected = cpuGEMM(a: a, b: b, M: M, N: N, K: K)
        let result = try dispatchGEMM_f32(a: a, b: b, M: M, N: N, K: K)
        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-3, "Mismatch at \(i): \(result[i]) vs \(expected[i])")
        }
    }
}
