import Metal
import Synchronization
import EdgeRunnerSharedTypes

public final class KVCache: Sendable {
    public enum Precision: Sendable {
        case float32
        case float16
        case float8

        fileprivate var bytesPerElement: Int {
            switch self {
            case .float32:
                return MemoryLayout<Float>.stride
            case .float16:
                return MemoryLayout<Float16>.stride
            case .float8:
                return 1
            }
        }

        fileprivate var rawValue: UInt32 {
            switch self {
            case .float32: return 0
            case .float16: return 1
            case .float8: return 2
            }
        }
    }

    private struct LayerState: Sendable {
        var writePos = 0
        var totalWritten = 0
    }

    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision

    private let keyBuffers: [MetalBufferHandle]
    private let valueBuffers: [MetalBufferHandle]
    private let layerStates: Mutex<[LayerState]>

    private var elementsPerToken: Int {
        numKVHeads * headDim
    }

    private var bytesPerToken: Int {
        elementsPerToken * precision.bytesPerElement
    }

    public var currentLength: Int {
        layerStates.withLock { states in
            states.map { min($0.totalWritten, maxSeqLen) }.max() ?? 0
        }
    }

    public init(
        device: MTLDevice,
        maxSeqLen: Int,
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        precision: Precision
    ) throws {
        self.maxSeqLen = maxSeqLen
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.precision = precision
        self.layerStates = Mutex(Array(repeating: LayerState(), count: numLayers))

        let bufferLength = maxSeqLen * numKVHeads * headDim * precision.bytesPerElement
        var keyBuffers: [MetalBufferHandle] = []
        var valueBuffers: [MetalBufferHandle] = []
        keyBuffers.reserveCapacity(numLayers)
        valueBuffers.reserveCapacity(numLayers)

        for _ in 0..<numLayers {
            guard let keyBuffer = device.makeBuffer(
                length: bufferLength,
                options: [.storageModeShared, .hazardTrackingModeUntracked]
            ),
            let valueBuffer = device.makeBuffer(
                length: bufferLength,
                options: [.storageModeShared, .hazardTrackingModeUntracked]
            ) else {
                throw KVCacheError.allocationFailed
            }
            keyBuffers.append(MetalBufferHandle(keyBuffer))
            valueBuffers.append(MetalBufferHandle(valueBuffer))
        }

        self.keyBuffers = keyBuffers
        self.valueBuffers = valueBuffers
    }

    public func append(layer: Int, keys: [Float], values: [Float]) throws {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision == .float32 else { throw KVCacheError.precisionMismatch }
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)

        let writePos = nextWritePosition(for: layer)
        let byteOffset = writePos * bytesPerToken

        keys.withUnsafeBytes { bytes in
            keyBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        values.withUnsafeBytes { bytes in
            valueBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }

    public func appendF16(layer: Int, keys: [Float16], values: [Float16]) throws {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision == .float16 else { throw KVCacheError.precisionMismatch }
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)

        let writePos = nextWritePosition(for: layer)
        let byteOffset = writePos * bytesPerToken

        keys.withUnsafeBytes { bytes in
            keyBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        values.withUnsafeBytes { bytes in
            valueBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }

    public func retrieve<T>(layer: Int, asType: T.Type) throws -> ([T], [T]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else {
            return ([], [])
        }

        let totalElements = currentLen * elementsPerToken
        let keyPtr = keyBuffers[layer].contents().bindMemory(to: T.self, capacity: maxSeqLen * elementsPerToken)
        let valuePtr = valueBuffers[layer].contents().bindMemory(to: T.self, capacity: maxSeqLen * elementsPerToken)

        var keys: [T] = []
        var values: [T] = []
        keys.reserveCapacity(totalElements)
        values.reserveCapacity(totalElements)

        if state.totalWritten <= maxSeqLen {
            keys.append(contentsOf: UnsafeBufferPointer(start: keyPtr, count: totalElements))
            values.append(contentsOf: UnsafeBufferPointer(start: valuePtr, count: totalElements))
        } else {
            let firstChunkTokens = maxSeqLen - state.writePos
            let secondChunkTokens = state.writePos
            if firstChunkTokens > 0 {
                let start = state.writePos * elementsPerToken
                let count = firstChunkTokens * elementsPerToken
                keys.append(contentsOf: UnsafeBufferPointer(start: keyPtr.advanced(by: start), count: count))
                values.append(contentsOf: UnsafeBufferPointer(start: valuePtr.advanced(by: start), count: count))
            }
            if secondChunkTokens > 0 {
                let count = secondChunkTokens * elementsPerToken
                keys.append(contentsOf: UnsafeBufferPointer(start: keyPtr, count: count))
                values.append(contentsOf: UnsafeBufferPointer(start: valuePtr, count: count))
            }
        }

        return (keys, values)
    }

    public func metalBuffers(layer: Int) -> (MTLBuffer, MTLBuffer) {
        (keyBuffers[layer].rawValue, valueBuffers[layer].rawValue)
    }

    public func cacheParams(layer: Int) throws -> ERKVCacheParams {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        let state = layerStates.withLock { $0[layer] }
        return ERKVCacheParams(
            maxSeqLen: UInt32(maxSeqLen),
            currentLen: UInt32(min(state.totalWritten, maxSeqLen)),
            writePos: UInt32(state.writePos),
            numKVHeads: UInt32(numKVHeads),
            headDim: UInt32(headDim),
            precision: precision.rawValue
        )
    }

    public func reset() {
        layerStates.withLock { states in
            for index in states.indices {
                states[index] = LayerState()
            }
        }
    }

    private func nextWritePosition(for layer: Int) -> Int {
        layerStates.withLock { states in
            let writePos = states[layer].writePos
            states[layer].writePos = (states[layer].writePos + 1) % maxSeqLen
            states[layer].totalWritten += 1
            return writePos
        }
    }
}

public enum KVCacheError: Error, Sendable {
    case allocationFailed
    case invalidLayer(Int)
    case precisionMismatch
}
