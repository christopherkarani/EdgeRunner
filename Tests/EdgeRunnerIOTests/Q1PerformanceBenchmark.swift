import Metal
import Testing
import Foundation
@testable import EdgeRunnerIO

/// Performance benchmark for Q1 quantized models
struct Q1PerformanceBenchmark {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let kernel: DequantQ1_0_g128Kernel

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue()
        else {
            throw NSError(domain: "Q1PerformanceBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal not available"])
        }
        self.device = device
        self.commandQueue = commandQueue
        self.kernel = try DequantQ1_0_g128Kernel(device: device)
    }

    /// Benchmark dequantization performance
    /// - Parameters:
    ///   - blockCount: Number of Q1 blocks to process
    ///   - iterations: Number of iterations to run
    /// - Returns: Average time per iteration in milliseconds
    func benchmarkDequant(
        blockCount: Int,
        iterations: Int = 10
    ) async throws -> Double {
        // Create test data
        let totalWeights = blockCount * 128
        let testData = createRandomQ1Data(blockCount: blockCount)

        // Warmup
        for _ in 0..<3 {
            _ = try await kernel.dequantise(
                blockData: testData,
                blockCount: blockCount,
                commandQueue: commandQueue
            )
        }

        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await kernel.dequantise(
                blockData: testData,
                blockCount: blockCount,
                commandQueue: commandQueue
            )
        }
        let end = CFAbsoluteTimeGetCurrent()

        let totalTime = end - start
        let avgTimeMs = (totalTime / Double(iterations)) * 1000.0

        print("Q1 Dequant Benchmark:")
        print("  Blocks: \(blockCount)")
        print("  Weights: \(totalWeights)")
        print("  Avg Time: \(String(format: "%.2f", avgTimeMs)) ms")
        print("  Throughput: \(String(format: "%.2f", Double(totalWeights) / totalTime / 1_000_000.0)) MTok/s")

        return avgTimeMs
    }

    /// Benchmark fused GEMV performance
    /// - Parameters:
    ///   - rows: Number of output rows
    ///   - cols: Number of input columns
    ///   - iterations: Number of iterations to run
    /// - Returns: Average time per iteration in milliseconds
    func benchmarkGEMV(
        rows: Int,
        cols: Int,
        iterations: Int = 10
    ) async throws -> Double {
        // Create test data
        let blocksPerRow = cols / 128
        let totalBlocks = rows * blocksPerRow
        let quantisedWeights = createRandomQ1Data(blockCount: totalBlocks)
        let inputData = (0..<cols).map { Float($0) * 0.01 }
        let outputData = [Float](repeating: 0, count: rows)

        guard let weightBuffer = device.makeBuffer(
            bytes: quantisedWeights,
            length: quantisedWeights.count,
            options: .storageModeShared
        ),
        let inputBuffer = device.makeBuffer(
            bytes: inputData,
            length: inputData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: outputData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw DequantKernelError.allocationFailed(byteCount: 0)
        }

        // Warmup
        for _ in 0..<3 {
            try await kernel.gemv(
                quantisedWeights: weightBuffer,
                input: inputBuffer,
                output: outputBuffer,
                rows: rows,
                cols: cols,
                commandQueue: commandQueue
            )
        }

        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            try await kernel.gemv(
                quantisedWeights: weightBuffer,
                input: inputBuffer,
                output: outputBuffer,
                rows: rows,
                cols: cols,
                commandQueue: commandQueue
            )
        }
        let end = CFAbsoluteTimeGetCurrent()

        let totalTime = end - start
        let avgTimeMs = (totalTime / Double(iterations)) * 1000.0

        print("Q1 GEMV Benchmark:")
        print("  Matrix: \(rows)x\(cols)")
        print("  Blocks: \(totalBlocks)")
        print("  Avg Time: \(String(format: "%.2f", avgTimeMs)) ms")
        print("  Throughput: \(String(format: "%.2f", Double(cols) / totalTime / 1_000_000.0)) MTok/s")

        return avgTimeMs
    }

    /// Compare Q1 performance against other quantization levels
    func compareQuantizationLevels() async throws {
        print("\n=== Quantization Level Performance Comparison ===")

        let sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]

        for (rows, cols) in sizes {
            print("\nMatrix Size: \(rows)x\(cols)")
            print("----------------------------------------")

            // Q1 benchmark
            let q1Time = try await benchmarkGEMV(rows: rows, cols: cols, iterations: 5)
            print("  Q1_0_g128: \(String(format: "%.2f", q1Time)) ms")

            // Theoretical performance comparison
            let q1Weights = Double(rows * cols)
            let q4Weights = q1Weights * 4.0  // Q4 has 4 bits per weight vs 1 bit for Q1
            let q8Weights = q1Weights * 8.0  // Q8 has 8 bits per weight

            print("  Theoretical:")
            print("    Q1 memory: \(String(format: "%.1f", q1Weights / 8.0 / 1024.0 / 1024.0)) MB")
            print("    Q4 memory: \(String(format: "%.1f", q4Weights / 8.0 / 1024.0 / 1024.0)) MB")
            print("    Q8 memory: \(String(format: "%.1f", q8Weights / 8.0 / 1024.0 / 1024.0)) MB")
        }
    }

    private func createRandomQ1Data(blockCount: Int) -> [UInt8] {
        var data = [UInt8]()
        data.reserveCapacity(blockCount * 18)

        for _ in 0..<blockCount {
            // Random scale (FP16)
            let scale = Float.random(in: 0.1...1.0)
            let f16ScaleBits = Float16(scale).bitPattern.littleEndian
            withUnsafeBytes(of: f16ScaleBits) { pointer in
                data.append(pointer[0])
                data.append(pointer[1])
            }

            // Random 16 bytes of bit data
            for _ in 0..<16 {
                data.append(UInt8.random(in: 0...255))
            }
        }

        return data
    }
}

@Suite("Q1 Performance Benchmarks")
struct Q1PerformanceTests {
    @Test func runQ1Benchmarks() async throws {
        let benchmark = try Q1PerformanceBenchmark()

        // Basic dequant benchmark
        let dequantTime = try await benchmark.benchmarkDequant(blockCount: 1000, iterations: 5)
        print("Q1 dequant time for 1000 blocks: \(String(format: "%.2f", dequantTime)) ms")

        // GEMV benchmark
        let gemvTime = try await benchmark.benchmarkGEMV(rows: 1, cols: 4096, iterations: 5)
        print("Q1 GEMV time for 1x4096: \(String(format: "%.2f", gemvTime)) ms")

        // Comparison
        try await benchmark.compareQuantizationLevels()
    }
}