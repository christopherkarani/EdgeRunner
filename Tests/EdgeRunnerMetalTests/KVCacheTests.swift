import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("KV Cache Ring Buffer")
struct KVCacheTests {
    let device: MTLDevice

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = device
    }

    @Test func createCache() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 2048,
            numLayers: 32,
            numKVHeads: 8,
            headDim: 128,
            precision: .float16
        )

        #expect(cache.maxSeqLen == 2048)
        #expect(cache.currentLength == 0)
        #expect(cache.numLayers == 32)
    }

    @Test func appendAndRetrieveSingleStep() throws {
        let numKVHeads = 2
        let headDim = 4
        let cache = try KVCache(
            device: device,
            maxSeqLen: 16,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        let keys: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let values: [Float] = [9, 10, 11, 12, 13, 14, 15, 16]
        try cache.append(layer: 0, keys: keys, values: values)

        #expect(cache.currentLength == 1)

        let (retrievedKeys, retrievedValues) = try cache.retrieve(layer: 0, asType: Float.self)
        #expect(retrievedKeys.count == numKVHeads * headDim)
        #expect(retrievedValues.count == numKVHeads * headDim)
        #expect(retrievedKeys == keys)
        #expect(retrievedValues == values)
    }

    @Test func appendMultipleSteps() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        try cache.append(layer: 0, keys: [1, 2], values: [10, 20])
        try cache.append(layer: 0, keys: [3, 4], values: [30, 40])
        try cache.append(layer: 0, keys: [5, 6], values: [50, 60])

        #expect(cache.currentLength == 3)
        let (keys, values) = try cache.retrieve(layer: 0, asType: Float.self)
        #expect(keys == [1, 2, 3, 4, 5, 6] as [Float])
        #expect(values == [10, 20, 30, 40, 50, 60] as [Float])
    }

    @Test func wrapAround() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 4,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        for index in 0..<4 {
            let value = Float(index)
            try cache.append(layer: 0, keys: [value, value + 0.5], values: [value * 10, value * 10 + 5])
        }

        try cache.append(layer: 0, keys: [99, 99.5], values: [990, 995])

        #expect(cache.currentLength == 4)
        let (keys, _) = try cache.retrieve(layer: 0, asType: Float.self)
        #expect(keys[0] == 1)
        #expect(keys[1] == 1.5)
        #expect(keys[6] == 99)
        #expect(keys[7] == 99.5)
    }

    @Test func multiLayerCache() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 4,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        try cache.append(layer: 0, keys: [1, 2], values: [10, 20])
        try cache.append(layer: 1, keys: [3, 4], values: [30, 40])
        try cache.append(layer: 2, keys: [5, 6], values: [50, 60])
        try cache.append(layer: 3, keys: [7, 8], values: [70, 80])

        let (keys0, _) = try cache.retrieve(layer: 0, asType: Float.self)
        let (keys1, _) = try cache.retrieve(layer: 1, asType: Float.self)
        #expect(keys0 == [1, 2] as [Float])
        #expect(keys1 == [3, 4] as [Float])
    }

    @Test func float16Precision() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 4,
            precision: .float16
        )

        let keys: [Float16] = [1, 2, 3, 4]
        let values: [Float16] = [5, 6, 7, 8]
        try cache.appendF16(layer: 0, keys: keys, values: values)

        #expect(cache.currentLength == 1)
        let (retrievedKeys, retrievedValues) = try cache.retrieve(layer: 0, asType: Float16.self)
        #expect(retrievedKeys == keys)
        #expect(retrievedValues == values)
    }

    @Test func reset() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 16,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        try cache.append(layer: 0, keys: [1, 2], values: [3, 4])
        #expect(cache.currentLength == 1)

        cache.reset()
        #expect(cache.currentLength == 0)
    }

    @Test func metalBufferAccess() throws {
        let maxSeqLen = 16
        let numKVHeads = 2
        let headDim = 4
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 2,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        let (keyBuffer, valueBuffer) = try cache.metalBuffers(layer: 0)
        #expect(keyBuffer.length == maxSeqLen * numKVHeads * headDim * MemoryLayout<Float>.stride)
        #expect(valueBuffer.length == maxSeqLen * numKVHeads * headDim * MemoryLayout<Float>.stride)
    }

    @Test func retrieveRejectsMismatchedElementType() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 4,
            precision: .float16
        )

        try cache.appendF16(layer: 0, keys: [1, 2, 3, 4], values: [5, 6, 7, 8])

        do {
            _ = try cache.retrieve(layer: 0, asType: Float.self)
            Issue.record("Expected precisionMismatch when retrieving Float16 cache as Float")
        } catch let error as KVCacheError {
            if case .precisionMismatch = error {
                return
            }
            Issue.record("Expected precisionMismatch, got \(error)")
        }
    }

    @Test func metalBuffersRejectInvalidLayer() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 4,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        do {
            _ = try cache.metalBuffers(layer: 1)
            Issue.record("Expected invalidLayer when requesting non-existent KV cache layer")
        } catch let error as KVCacheError {
            if case .invalidLayer(1) = error {
                return
            }
            Issue.record("Expected invalidLayer(1), got \(error)")
        }
    }
}
