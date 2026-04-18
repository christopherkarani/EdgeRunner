import Testing
import Metal
@testable import EdgeRunner
@testable import EdgeRunnerMetal

@Suite("KVCache Gemma 4 layout")
struct KVCacheGemma4Tests {
    @Test("Allocates per-layer buffers with correct head dim and shares for layers 24..41")
    func kvCacheGemma4Layout() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        let cache = try KVCache.gemma4(
            device: device,
            config: config,
            maxSeqLen: 2048
        )

        // Sliding layer 0: 2 KV heads * 256 headDim * 2048 tokens * 4 bytes (float32 default)
        #expect(cache.keyBuffer(forLayer: 0).length == 2 * 256 * 2048 * 4)
        #expect(cache.valueBuffer(forLayer: 0).length == 2 * 256 * 2048 * 4)

        // Global layer 5: 2 * 512 * 2048 * 4
        #expect(cache.keyBuffer(forLayer: 5).length == 2 * 512 * 2048 * 4)
        #expect(cache.valueBuffer(forLayer: 5).length == 2 * 512 * 2048 * 4)

        // Shared layers: identity check on buffer.
        // Layer 24 is sliding; kvSourceLayer(for: 24) == 22.
        #expect(cache.keyBuffer(forLayer: 24) === cache.keyBuffer(forLayer: 22))
        #expect(cache.valueBuffer(forLayer: 24) === cache.valueBuffer(forLayer: 22))

        // Layer 29 is global; kvSourceLayer(for: 29) == 23.
        #expect(cache.keyBuffer(forLayer: 29) === cache.keyBuffer(forLayer: 23))
        #expect(cache.valueBuffer(forLayer: 29) === cache.valueBuffer(forLayer: 23))

        // Layer 41 (final, global): kvSourceLayer(for: 41) == 23.
        #expect(cache.keyBuffer(forLayer: 41) === cache.keyBuffer(forLayer: 23))
        #expect(cache.valueBuffer(forLayer: 41) === cache.valueBuffer(forLayer: 23))
    }

    @Test("Total allocated buffers — 24 unique-KV layers at correct dims")
    func uniqueBufferCountAndBytes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        let cache = try KVCache.gemma4(
            device: device,
            config: config,
            maxSeqLen: 2048
        )

        var uniqueKeyBuffers = Set<ObjectIdentifier>()
        var uniqueValueBuffers = Set<ObjectIdentifier>()
        for i in 0..<config.numHiddenLayers {
            uniqueKeyBuffers.insert(ObjectIdentifier(cache.keyBuffer(forLayer: i)))
            uniqueValueBuffers.insert(ObjectIdentifier(cache.valueBuffer(forLayer: i)))
        }
        // 42 layers - 18 shared = 24 unique owning layers.
        #expect(uniqueKeyBuffers.count == 24)
        #expect(uniqueValueBuffers.count == 24)
    }

    @Test("Sliding vs global head_dim partition matches config")
    func slidingAndGlobalHeadDimBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        let cache = try KVCache.gemma4(
            device: device,
            config: config,
            maxSeqLen: 2048
        )

        let slidingBytes = 2 * config.headDim * 2048 * 4
        let globalBytes = 2 * config.globalHeadDim * 2048 * 4

        for layer in 0..<config.numHiddenLayers {
            let expected: Int
            switch config.layerTypes[layer] {
            case .sliding:
                expected = slidingBytes
            case .global:
                expected = globalBytes
            }
            #expect(cache.keyBuffer(forLayer: layer).length == expected)
            #expect(cache.valueBuffer(forLayer: layer).length == expected)
        }
    }

    @Test("Per-layer cacheParams report correct head_dim for sliding vs global")
    func cacheParamsReportPerLayerHeadDim() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        let cache = try KVCache.gemma4(
            device: device,
            config: config,
            maxSeqLen: 2048
        )

        for layer in 0..<config.numHiddenLayers {
            let params = try cache.cacheParams(layer: layer)
            let expected: UInt32
            switch config.layerTypes[layer] {
            case .sliding:
                expected = UInt32(config.headDim)
            case .global:
                expected = UInt32(config.globalHeadDim)
            }
            #expect(params.headDim == expected)
            #expect(params.numKVHeads == UInt32(config.numKeyValueHeads))
            #expect(params.maxSeqLen == 2048)
        }
    }
}
