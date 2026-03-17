import Testing
import Foundation
import Metal
@testable import EdgeRunnerMetal

@Suite("KV Cache Memory Benchmarks")
struct KVCacheMemoryBenchmarks {

    @Test func kvCacheMemoryByPrecision() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let maxSeqLen = 2048
        let numLayers = 32
        let numKVHeads = 8
        let headDim = 128

        // fp32
        let cacheFP32 = try KVCache(device: device, maxSeqLen: maxSeqLen, numLayers: numLayers, numKVHeads: numKVHeads, headDim: headDim, precision: .float32)
        let fp32Bytes = maxSeqLen * numKVHeads * headDim * 4 * 2 * numLayers  // keys + values
        let fp32MB = Double(fp32Bytes) / 1_048_576.0

        // fp16
        let cacheFP16 = try KVCache(device: device, maxSeqLen: maxSeqLen, numLayers: numLayers, numKVHeads: numKVHeads, headDim: headDim, precision: .float16)
        let fp16Bytes = maxSeqLen * numKVHeads * headDim * 2 * 2 * numLayers
        let fp16MB = Double(fp16Bytes) / 1_048_576.0

        print("BENCHMARK: kvcache_memory_fp32_32L_2048ctx \(String(format: "%.0f", fp32MB)) MB")
        print("BENCHMARK: kvcache_memory_fp16_32L_2048ctx \(String(format: "%.0f", fp16MB)) MB")
        print("BENCHMARK: kvcache_compression_fp16_vs_fp32 \(String(format: "%.1f", fp32MB / fp16MB))x")

        #expect(fp16MB < fp32MB)
        // Suppress unused variable warnings
        _ = cacheFP32
        _ = cacheFP16
    }
}
