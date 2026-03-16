import Testing
import Foundation
import Metal

@Suite("Memory Benchmarks")
struct MemoryBenchmarks {

    @Test func quantizationMemoryEstimates() {
        // Llama 3 8B parameter count
        let params: Double = 8_000_000_000

        let fp32MB = params * 4.0 / 1_048_576.0
        let fp16MB = params * 2.0 / 1_048_576.0
        let q8MB = params * 1.0 / 1_048_576.0  // ~8 bits per weight
        let q4MB = params * 0.5 / 1_048_576.0   // ~4 bits per weight

        let fp32GB = fp32MB / 1024.0
        let fp16GB = fp16MB / 1024.0
        let q8GB = q8MB / 1024.0
        let q4GB = q4MB / 1024.0

        print("BENCHMARK: model_memory_fp32_8B \(String(format: "%.1f", fp32GB)) GB")
        print("BENCHMARK: model_memory_fp16_8B \(String(format: "%.1f", fp16GB)) GB")
        print("BENCHMARK: model_memory_q8_8B \(String(format: "%.1f", q8GB)) GB")
        print("BENCHMARK: model_memory_q4_8B \(String(format: "%.1f", q4GB)) GB")

        #expect(q4GB < 5.0)
        #expect(fp32GB > 28.0)
    }

    @Test func metalDeviceMemoryInfo() {
        let device = MTLCreateSystemDefaultDevice()!
        let maxWorkingSetGB = Double(device.recommendedMaxWorkingSetSize) / 1_073_741_824.0
        let hasUnified = device.hasUnifiedMemory

        print("BENCHMARK: device_name \(device.name)")
        print("BENCHMARK: device_max_working_set \(String(format: "%.1f", maxWorkingSetGB)) GB")
        print("BENCHMARK: device_unified_memory \(hasUnified)")

        #expect(hasUnified == true) // Apple Silicon always has unified memory
    }

    @Test func allocationThroughput() throws {
        let device = MTLCreateSystemDefaultDevice()!

        // Allocate many small buffers (simulates per-op allocation)
        let sizes = [4096, 16384, 65536, 262144, 1_048_576] // 4KB to 1MB
        for size in sizes {
            let count = 100
            let clock = ContinuousClock()
            let start = clock.now
            var buffers: [MTLBuffer] = []
            for _ in 0..<count {
                if let buf = device.makeBuffer(length: size, options: .storageModeShared) {
                    buffers.append(buf)
                }
            }
            let elapsed = start.duration(to: clock.now)
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
            let allocPerSec = Double(count) / seconds
            let sizeKB = size / 1024

            print("BENCHMARK: alloc_throughput_\(sizeKB)KB \(String(format: "%.0f", allocPerSec)) alloc/sec")
            buffers.removeAll()
        }

        #expect(true) // Just collecting data
    }
}
