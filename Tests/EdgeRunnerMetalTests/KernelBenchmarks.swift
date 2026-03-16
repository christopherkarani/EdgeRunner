import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("Kernel Benchmarks")
struct KernelBenchmarks {

    // MARK: - GEMV

    @Test func gemvF32_4096x4096() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try GEMVKernel(device: device)

        let M = 4096, K = 4096
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }

        // Warmup
        for _ in 0..<3 { _ = try await kernel.execute(a: a, x: x, M: M, K: K, commandQueue: queue) }

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(a: a, x: x, M: M, K: K, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let bytesAccessed = Double(M * K + K + M) * Double(MemoryLayout<Float>.stride) * Double(iterations)
        let gbPerSec = bytesAccessed / seconds / 1e9

        print("BENCHMARK: gemv_f32_4096x4096 \(String(format: "%.1f", gbPerSec)) GB/s (\(String(format: "%.2f", seconds/Double(iterations)*1000)) ms/op)")
        #expect(gbPerSec > 1.0)
    }

    @Test func gemvF32_8192x4096() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try GEMVKernel(device: device)

        let M = 8192, K = 4096
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }

        for _ in 0..<3 { _ = try await kernel.execute(a: a, x: x, M: M, K: K, commandQueue: queue) }

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(a: a, x: x, M: M, K: K, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let bytesAccessed = Double(M * K + K + M) * Double(MemoryLayout<Float>.stride) * Double(iterations)
        let gbPerSec = bytesAccessed / seconds / 1e9

        print("BENCHMARK: gemv_f32_8192x4096 \(String(format: "%.1f", gbPerSec)) GB/s (\(String(format: "%.2f", seconds/Double(iterations)*1000)) ms/op)")
        #expect(gbPerSec > 1.0)
    }

    // MARK: - Flash Attention

    @Test func flashAttention_seq512_dim128() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try FlashAttentionKernel(device: device)

        let seqLen = 512, headDim = 128
        let q = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }
        let k = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }
        let v = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }

        for _ in 0..<3 { _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true, commandQueue: queue) }

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: flash_attn_seq512_dim128 \(String(format: "%.2f", msPerOp)) ms")
        #expect(msPerOp < 1000)
    }

    @Test func flashAttention_seq2048_dim128() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try FlashAttentionKernel(device: device)

        let seqLen = 2048, headDim = 128
        let q = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }
        let k = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }
        let v = (0..<seqLen*headDim).map { _ in Float.random(in: -1...1) }

        for _ in 0..<3 { _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true, commandQueue: queue) }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: flash_attn_seq2048_dim128 \(String(format: "%.2f", msPerOp)) ms")
        #expect(msPerOp < 5000)
    }

    // MARK: - GQA

    @Test func gqa_seq1024_h32_kv8() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try GQAKernel(device: device)

        let seqLen = 1024, headDim = 128, numHeads = 32, numKVHeads = 8
        let q = (0..<seqLen*numHeads*headDim).map { _ in Float.random(in: -1...1) }
        let k = (0..<seqLen*numKVHeads*headDim).map { _ in Float.random(in: -1...1) }
        let v = (0..<seqLen*numKVHeads*headDim).map { _ in Float.random(in: -1...1) }

        for _ in 0..<3 { _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: true, commandQueue: queue) }

        let iterations = 5
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, causal: true, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: gqa_seq1024_h32_kv8 \(String(format: "%.2f", msPerOp)) ms")
        #expect(msPerOp < 5000)
    }

    // MARK: - RMSNorm

    @Test func rmsNorm_dim4096_rows512() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try RMSNormKernel(device: device)

        let rows = 512, cols = 4096
        let input = (0..<rows*cols).map { _ in Float.random(in: -1...1) }
        let weight = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }

        for _ in 0..<3 { _ = try await kernel.execute(input: input, weight: weight, rows: rows, cols: cols, commandQueue: queue) }

        let iterations = 20
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(input: input, weight: weight, rows: rows, cols: cols, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let bytesAccessed = Double(rows * cols * 2 + cols) * Double(MemoryLayout<Float>.stride) * Double(iterations)
        let gbPerSec = bytesAccessed / seconds / 1e9

        print("BENCHMARK: rmsnorm_4096x512 \(String(format: "%.1f", gbPerSec)) GB/s (\(String(format: "%.3f", seconds/Double(iterations)*1000)) ms/op)")
        #expect(gbPerSec > 0.1)
    }

    // MARK: - LayerNorm

    @Test func layerNorm_dim4096_rows512() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try LayerNormKernel(device: device)

        let rows = 512, cols = 4096
        let input = (0..<rows*cols).map { _ in Float.random(in: -1...1) }
        let gamma = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }
        let beta = (0..<cols).map { _ in Float.random(in: -0.1...0.1) }

        for _ in 0..<3 { _ = try await kernel.execute(input: input, gamma: gamma, beta: beta, rows: rows, cols: cols, commandQueue: queue) }

        let iterations = 20
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(input: input, gamma: gamma, beta: beta, rows: rows, cols: cols, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let bytesAccessed = Double(rows * cols * 2 + cols * 2) * Double(MemoryLayout<Float>.stride) * Double(iterations)
        let gbPerSec = bytesAccessed / seconds / 1e9

        print("BENCHMARK: layernorm_4096x512 \(String(format: "%.1f", gbPerSec)) GB/s (\(String(format: "%.3f", seconds/Double(iterations)*1000)) ms/op)")
        #expect(gbPerSec > 0.1)
    }

    // MARK: - RoPE

    @Test func rope_seq2048_h32_dim128() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try RoPEKernel(device: device)

        let seqLen = 2048, numHeads = 32, headDim = 128
        let input = (0..<seqLen*numHeads*headDim).map { _ in Float.random(in: -1...1) }

        for _ in 0..<3 { _ = try await kernel.execute(input: input, seqLen: seqLen, numHeads: numHeads, headDim: headDim, startPos: 0, theta: 500_000.0, commandQueue: queue) }

        let iterations = 10
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(input: input, seqLen: seqLen, numHeads: numHeads, headDim: headDim, startPos: 0, theta: 500_000.0, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: rope_seq2048_h32_dim128 \(String(format: "%.2f", msPerOp)) ms")
        #expect(msPerOp < 1000)
    }

    // MARK: - Softmax

    @Test func softmax_vocab128256() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try SoftmaxKernel(device: device)

        let rows = 1, cols = 128_256 // Llama 3 vocab size
        let input = (0..<rows*cols).map { _ in Float.random(in: -5...5) }

        for _ in 0..<3 { _ = try await kernel.execute(input: input, rows: rows, cols: cols, commandQueue: queue) }

        let iterations = 50
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.execute(input: input, rows: rows, cols: cols, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: softmax_vocab128256 \(String(format: "%.3f", msPerOp)) ms")
        #expect(msPerOp < 100)
    }

    // MARK: - Activations

    @Test func gelu_dim14336() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try ActivationKernels(device: device)

        let size = 14_336 // Llama 3 intermediate dim
        let input = (0..<size).map { _ in Float.random(in: -3...3) }

        for _ in 0..<3 { _ = try await kernel.gelu(input: input, commandQueue: queue) }

        let iterations = 50
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.gelu(input: input, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: gelu_dim14336 \(String(format: "%.3f", msPerOp)) ms")
        #expect(msPerOp < 100)
    }

    @Test func swiglu_dim14336() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let kernel = try ActivationKernels(device: device)

        let size = 14_336
        let gate = (0..<size).map { _ in Float.random(in: -3...3) }
        let up = (0..<size).map { _ in Float.random(in: -3...3) }

        for _ in 0..<3 { _ = try await kernel.swiglu(gate: gate, up: up, commandQueue: queue) }

        let iterations = 50
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<iterations {
            _ = try await kernel.swiglu(gate: gate, up: up, commandQueue: queue)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: swiglu_dim14336 \(String(format: "%.3f", msPerOp)) ms")
        #expect(msPerOp < 100)
    }
}
