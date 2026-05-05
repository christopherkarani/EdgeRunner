import Metal

struct Gemma4AttentionRange: Equatable, Sendable {
    let start: Int
    let count: Int
}

struct Gemma4KVRow: Equatable, Sendable {
    let keys: [Float16]
    let values: [Float16]
}

enum Gemma4GPUKVCacheError: Error, Equatable, Sendable {
    case invalidLayer(Int)
    case invalidPosition(Int)
    case allocationFailed
    case invalidSharedLayer(layer: Int, source: Int)
    case sharedLayerWrite(layer: Int, source: Int)
    case invalidRowShape(expected: Int, gotKeys: Int, gotValues: Int)
}

final class Gemma4GPUKVCache: @unchecked Sendable {
    private let config: Gemma4ModelConfig
    private let maxSeqLen: Int
    private let keyBuffers: [MTLBuffer]
    private let valueBuffers: [MTLBuffer]
    private let capacities: [Int]
    private let sourceLayers: [Int]
    private let bytesPerToken: [Int]

    init(
        device: MTLDevice,
        config: Gemma4ModelConfig,
        maxSeqLen: Int
    ) throws {
        self.config = config
        self.maxSeqLen = maxSeqLen

        let layerCount = config.numHiddenLayers
        var sourceLayers: [Int] = []
        var capacities: [Int] = []
        var bytesPerToken: [Int] = []
        sourceLayers.reserveCapacity(layerCount)
        capacities.reserveCapacity(layerCount)
        bytesPerToken.reserveCapacity(layerCount)

        for layer in 0..<layerCount {
            let source = config.kvSourceLayer(for: layer)
            if source != layer {
                guard source < layer,
                      config.kvSourceLayer(for: source) == source,
                      config.layerTypes[source] == config.layerTypes[layer] else {
                    throw Gemma4GPUKVCacheError.invalidSharedLayer(layer: layer, source: source)
                }
            }
            sourceLayers.append(source)
            let capacity: Int
            switch config.layerTypes[layer] {
            case .sliding:
                capacity = min(config.slidingWindow, maxSeqLen)
            case .global:
                capacity = maxSeqLen
            }
            capacities.append(capacity)
            bytesPerToken.append(config.numKeyValueHeads * Self.headDim(config: config, layer: layer) * MemoryLayout<Float16>.stride)
        }

        var keyBuffers: [MTLBuffer?] = Array(repeating: nil, count: layerCount)
        var valueBuffers: [MTLBuffer?] = Array(repeating: nil, count: layerCount)
        for layer in 0..<layerCount {
            let source = sourceLayers[layer]
            if source == layer {
                let byteCount = capacities[layer] * bytesPerToken[layer]
                guard let keyBuffer = device.makeBuffer(
                    length: byteCount,
                    options: [.storageModeShared, .hazardTrackingModeUntracked]
                ),
                let valueBuffer = device.makeBuffer(
                    length: byteCount,
                    options: [.storageModeShared, .hazardTrackingModeUntracked]
                ) else {
                    throw Gemma4GPUKVCacheError.allocationFailed
                }
                keyBuffers[layer] = keyBuffer
                valueBuffers[layer] = valueBuffer
            } else {
                guard let keyBuffer = keyBuffers[source],
                      let valueBuffer = valueBuffers[source] else {
                    throw Gemma4GPUKVCacheError.invalidSharedLayer(layer: layer, source: source)
                }
                keyBuffers[layer] = keyBuffer
                valueBuffers[layer] = valueBuffer
            }
        }

        self.sourceLayers = sourceLayers
        self.capacities = capacities
        self.bytesPerToken = bytesPerToken
        self.keyBuffers = try keyBuffers.enumerated().map { layer, buffer in
            guard let buffer else { throw Gemma4GPUKVCacheError.invalidLayer(layer) }
            return buffer
        }
        self.valueBuffers = try valueBuffers.enumerated().map { layer, buffer in
            guard let buffer else { throw Gemma4GPUKVCacheError.invalidLayer(layer) }
            return buffer
        }
    }

    func ownsKV(layer: Int) -> Bool {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return sourceLayers[layer] == layer
    }

    func sourceLayer(for layer: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return sourceLayers[layer]
    }

    func headDim(forLayer layer: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return Self.headDim(config: config, layer: layer)
    }

    func capacity(forLayer layer: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return capacities[layer]
    }

    func keyBuffer(forLayer layer: Int) -> MTLBuffer {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return keyBuffers[layer]
    }

    func valueBuffer(forLayer layer: Int) -> MTLBuffer {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return valueBuffers[layer]
    }

    func elementsPerToken(forLayer layer: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        return config.numKeyValueHeads * Self.headDim(config: config, layer: layer)
    }

    func store(
        layer: Int,
        position: Int,
        keys: [Float],
        values: [Float]
    ) throws {
        guard (0..<config.numHiddenLayers).contains(layer) else {
            throw Gemma4GPUKVCacheError.invalidLayer(layer)
        }
        guard position >= 0 && position < maxSeqLen else {
            throw Gemma4GPUKVCacheError.invalidPosition(position)
        }
        let source = sourceLayers[layer]
        guard source == layer else {
            throw Gemma4GPUKVCacheError.sharedLayerWrite(layer: layer, source: source)
        }
        let expected = elementsPerToken(forLayer: layer)
        guard keys.count == expected, values.count == expected else {
            throw Gemma4GPUKVCacheError.invalidRowShape(
                expected: expected,
                gotKeys: keys.count,
                gotValues: values.count
            )
        }

        let offset = writeOffset(layer: layer, position: position)
        let keyF16 = keys.map(Float16.init)
        let valueF16 = values.map(Float16.init)
        keyF16.withUnsafeBytes { bytes in
            keyBuffers[layer].contents()
                .advanced(by: offset)
                .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        valueF16.withUnsafeBytes { bytes in
            valueBuffers[layer].contents()
                .advanced(by: offset)
                .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }

    func read(layer: Int, position: Int) throws -> Gemma4KVRow {
        guard (0..<config.numHiddenLayers).contains(layer) else {
            throw Gemma4GPUKVCacheError.invalidLayer(layer)
        }
        guard position >= 0 && position < maxSeqLen else {
            throw Gemma4GPUKVCacheError.invalidPosition(position)
        }
        let count = elementsPerToken(forLayer: layer)
        let offset = writeOffset(layer: layer, position: position)
        let keyPointer = keyBuffers[layer].contents()
            .advanced(by: offset)
            .bindMemory(to: Float16.self, capacity: count)
        let valuePointer = valueBuffers[layer].contents()
            .advanced(by: offset)
            .bindMemory(to: Float16.self, capacity: count)
        return Gemma4KVRow(
            keys: Array(UnsafeBufferPointer(start: keyPointer, count: count)),
            values: Array(UnsafeBufferPointer(start: valuePointer, count: count))
        )
    }

    func attentionRange(layer: Int, currentPosition: Int) -> Gemma4AttentionRange {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        precondition(currentPosition >= 0, "invalid Gemma position \(currentPosition)")
        switch config.layerTypes[layer] {
        case .sliding:
            let count = min(config.slidingWindow, currentPosition + 1)
            return Gemma4AttentionRange(start: currentPosition + 1 - count, count: count)
        case .global:
            return Gemma4AttentionRange(start: 0, count: currentPosition + 1)
        }
    }

    func writeOffset(layer: Int, position: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        precondition(position >= 0 && position < maxSeqLen, "invalid Gemma position \(position)")
        let physicalPosition = position % capacities[layer]
        return physicalPosition * bytesPerToken[layer]
    }

    func physicalPosition(layer: Int, logicalPosition: Int) -> Int {
        precondition((0..<config.numHiddenLayers).contains(layer), "invalid Gemma layer \(layer)")
        precondition(logicalPosition >= 0 && logicalPosition < maxSeqLen, "invalid Gemma position \(logicalPosition)")
        return logicalPosition % capacities[layer]
    }

    private static func headDim(config: Gemma4ModelConfig, layer: Int) -> Int {
        switch config.layerTypes[layer] {
        case .sliding:
            return config.headDim
        case .global:
            return config.globalHeadDim
        }
    }
}
