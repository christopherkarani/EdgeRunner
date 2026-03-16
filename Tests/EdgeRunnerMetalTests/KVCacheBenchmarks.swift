import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("KV Cache Benchmarks")
struct KVCacheBenchmarks {
    let device: MTLDevice

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = device
    }

    @Test func appendThroughputF32() throws {
        let numLayers = 32
        let maxSeqLen = 2048
        let numKVHeads = 8
        let headDim = 128
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        let tokensPerAppend = numKVHeads * headDim
        let keys = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }
        let values = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }

        let clock = ContinuousClock()
        let totalTokens = 2048
        let start = clock.now
        for _ in 0..<totalTokens {
            for layer in 0..<numLayers {
                try cache.append(layer: layer, keys: keys, values: values)
            }
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokPerSec = Double(totalTokens) / seconds

        print("BENCHMARK: kvcache_append_f32 \(String(format: "%.0f", tokPerSec)) tokens/sec (\(numLayers) layers, \(String(format: "%.3f", seconds))s)")
        #expect(tokPerSec > 100)
    }

    @Test func appendThroughputF16() throws {
        let numLayers = 32
        let maxSeqLen = 2048
        let numKVHeads = 8
        let headDim = 128
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float16
        )

        let tokensPerAppend = numKVHeads * headDim
        let keys = (0..<tokensPerAppend).map { _ in Float16.random(in: -1...1) }
        let values = (0..<tokensPerAppend).map { _ in Float16.random(in: -1...1) }

        let clock = ContinuousClock()
        let totalTokens = 2048
        let start = clock.now
        for _ in 0..<totalTokens {
            for layer in 0..<numLayers {
                try cache.appendF16(layer: layer, keys: keys, values: values)
            }
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokPerSec = Double(totalTokens) / seconds

        print("BENCHMARK: kvcache_append_f16 \(String(format: "%.0f", tokPerSec)) tokens/sec (\(numLayers) layers, \(String(format: "%.3f", seconds))s)")
        #expect(tokPerSec > 100)
    }

    @Test func retrieveAfterFullFill() throws {
        let numLayers = 32
        let maxSeqLen = 2048
        let numKVHeads = 8
        let headDim = 128
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        let tokensPerAppend = numKVHeads * headDim
        let keys = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }
        let values = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }

        for _ in 0..<maxSeqLen {
            for layer in 0..<numLayers {
                try cache.append(layer: layer, keys: keys, values: values)
            }
        }
        #expect(cache.currentLength == maxSeqLen)

        let clock = ContinuousClock()
        let start = clock.now
        for layer in 0..<numLayers {
            let (k, v) = try cache.retrieve(layer: layer, asType: Float.self)
            #expect(k.count == maxSeqLen * tokensPerAppend)
            #expect(v.count == maxSeqLen * tokensPerAppend)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let ms = seconds * 1000.0

        print("BENCHMARK: kvcache_retrieve_full \(String(format: "%.1f", ms)) ms (\(numLayers) layers x \(maxSeqLen) tokens)")
        #expect(ms < 10000)
    }

    @Test func ringBufferWraparound() throws {
        let numLayers = 32
        let maxSeqLen = 1024
        let numKVHeads = 8
        let headDim = 128
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        let tokensPerAppend = numKVHeads * headDim
        let keys = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }
        let values = (0..<tokensPerAppend).map { _ in Float.random(in: -1...1) }

        let totalTokens = 2048
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<totalTokens {
            for layer in 0..<numLayers {
                try cache.append(layer: layer, keys: keys, values: values)
            }
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let ms = seconds * 1000.0

        #expect(cache.currentLength <= maxSeqLen)
        #expect(cache.currentLength == maxSeqLen)

        print("BENCHMARK: kvcache_wraparound \(String(format: "%.1f", ms)) ms (\(totalTokens) tokens into \(maxSeqLen) slots, currentLength=\(cache.currentLength))")
    }
}
