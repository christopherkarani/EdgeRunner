import Metal
import Testing
@testable import EdgeRunner

@Suite("Gemma4GPUKVCache")
struct Gemma4GPUKVCacheTests {
    @Test("Allocates f16 sliding rings, full global buffers, and aliases shared layers")
    func allocatesGemmaPrivateLayout() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let cache = try Gemma4GPUKVCache(device: device, config: config, maxSeqLen: 2048)

        #expect(cache.capacity(forLayer: 0) == config.slidingWindow)
        #expect(cache.capacity(forLayer: 5) == 2048)

        let f16Stride = MemoryLayout<Float16>.stride
        #expect(cache.keyBuffer(forLayer: 0).length == config.slidingWindow * 2 * 256 * f16Stride)
        #expect(cache.valueBuffer(forLayer: 0).length == config.slidingWindow * 2 * 256 * f16Stride)
        #expect(cache.keyBuffer(forLayer: 5).length == 2048 * 2 * 512 * f16Stride)
        #expect(cache.valueBuffer(forLayer: 5).length == 2048 * 2 * 512 * f16Stride)

        #expect(cache.ownsKV(layer: 22))
        #expect(!cache.ownsKV(layer: 24))
        #expect(cache.sourceLayer(for: 24) == 22)
        #expect(cache.keyBuffer(forLayer: 24) === cache.keyBuffer(forLayer: 22))
        #expect(cache.valueBuffer(forLayer: 24) === cache.valueBuffer(forLayer: 22))

        #expect(cache.sourceLayer(for: 41) == 23)
        #expect(cache.keyBuffer(forLayer: 41) === cache.keyBuffer(forLayer: 23))
        #expect(cache.valueBuffer(forLayer: 41) === cache.valueBuffer(forLayer: 23))
    }

    @Test("Computes logical attention ranges and physical write offsets")
    func rangesAndOffsets() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let cache = try Gemma4GPUKVCache(device: device, config: config, maxSeqLen: 2048)

        #expect(cache.attentionRange(layer: 0, currentPosition: 0) == Gemma4AttentionRange(start: 0, count: 1))
        #expect(cache.attentionRange(layer: 0, currentPosition: 511) == Gemma4AttentionRange(start: 0, count: 512))
        #expect(cache.attentionRange(layer: 0, currentPosition: 512) == Gemma4AttentionRange(start: 1, count: 512))
        #expect(cache.attentionRange(layer: 5, currentPosition: 512) == Gemma4AttentionRange(start: 0, count: 513))

        let slidingBytes = 2 * config.headDim * MemoryLayout<Float16>.stride
        #expect(cache.writeOffset(layer: 0, position: 0) == 0)
        #expect(cache.physicalPosition(layer: 0, logicalPosition: 0) == 0)
        #expect(cache.writeOffset(layer: 0, position: 512) == 0)
        #expect(cache.physicalPosition(layer: 0, logicalPosition: 512) == 0)
        #expect(cache.writeOffset(layer: 0, position: 513) == slidingBytes)
        #expect(cache.physicalPosition(layer: 0, logicalPosition: 513) == 1)

        let globalBytes = 2 * config.globalHeadDim * MemoryLayout<Float16>.stride
        #expect(cache.writeOffset(layer: 5, position: 0) == 0)
        #expect(cache.writeOffset(layer: 5, position: 3) == 3 * globalBytes)
        #expect(cache.physicalPosition(layer: 5, logicalPosition: 3) == 3)
    }

    @Test("Stores and reads f16 KV rows through ring offsets")
    func storeAndReadRows() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let cache = try Gemma4GPUKVCache(device: device, config: config, maxSeqLen: 2048)
        let elements = config.numKeyValueHeads * config.headDim
        let keys = (0..<elements).map { Float($0 % 17) / 10.0 }
        let values = (0..<elements).map { -Float($0 % 19) / 11.0 }

        try cache.store(layer: 0, position: 0, keys: keys, values: values)
        try cache.store(layer: 0, position: config.slidingWindow, keys: values, values: keys)

        let wrapped = try cache.read(layer: 0, position: config.slidingWindow)
        for index in 0..<elements {
            #expect(abs(Float(wrapped.keys[index]) - values[index]) < 0.001)
            #expect(abs(Float(wrapped.values[index]) - keys[index]) < 0.001)
        }
    }

    @Test("Rejects writes to KV-shared layers")
    func rejectsSharedLayerWrites() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let cache = try Gemma4GPUKVCache(device: device, config: config, maxSeqLen: 2048)
        let elements = config.numKeyValueHeads * config.headDim

        #expect(throws: Gemma4GPUKVCacheError.sharedLayerWrite(layer: 24, source: 22)) {
            try cache.store(
                layer: 24,
                position: 0,
                keys: [Float](repeating: 0, count: elements),
                values: [Float](repeating: 0, count: elements)
            )
        }
    }
}
