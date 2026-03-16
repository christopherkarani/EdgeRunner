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
}
