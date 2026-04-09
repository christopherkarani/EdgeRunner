import Metal
import Synchronization
import EdgeRunnerSharedTypes

public struct TurboQuantMetalBuffers: @unchecked Sendable {
    public let codes: MTLBuffer
    public let residualSigns: MTLBuffer
    public let outlierMask: MTLBuffer
    public let metadata: MTLBuffer
}

public final class KVCache: Sendable {
    public enum Precision: Sendable {
        case float32
        case float16
        case float8
        case q8_0
        case turboquantV2
        case turboQuantBalanced
        case turboQuantAggressive

        fileprivate var denseBytesPerElement: Int? {
            switch self {
            case .float32:
                return MemoryLayout<Float>.stride
            case .float16:
                return MemoryLayout<Float16>.stride
            case .float8:
                return 1
            case .q8_0:
                return nil
            case .turboquantV2:
                return nil
            case .turboQuantBalanced, .turboQuantAggressive:
                return nil
            }
        }

        fileprivate var rawValue: UInt32 {
            switch self {
            case .float32: return 0
            case .float16: return 1
            case .float8: return 2
            case .q8_0: return 3
            case .turboquantV2: return 4
            case .turboQuantBalanced: return 5
            case .turboQuantAggressive: return 6
            }
        }

        fileprivate var turboQuantPresets: (key: TurboQuantPreset, value: TurboQuantPreset)? {
            switch self {
            case .turboQuantBalanced:
                return (.balanced, .balanced)
            case .turboQuantAggressive:
                return (.aggressive, .aggressive)
            default:
                return nil
            }
        }

        fileprivate var isTurboQuant: Bool {
            switch self {
            case .turboquantV2, .turboQuantBalanced, .turboQuantAggressive:
                return true
            default:
                return false
            }
        }
    }

    private struct LayerState: Sendable {
        var writePos = 0
        var totalWritten = 0
    }

    private struct TurboQuantLayerStorage: Sendable {
        let codes: MetalBufferHandle
        let residualSigns: MetalBufferHandle
        let outlierMask: MetalBufferHandle
        let metadata: MetalBufferHandle
    }

    private enum LayerStorageKind: Sendable, Equatable {
        case dense
        case q8_0
        case turboQuant(preset: TurboQuantPreset, layout: TurboQuantLayout)
    }

    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision

    private let keyBuffers: [MetalBufferHandle?]
    private let valueBuffers: [MetalBufferHandle?]
    private let turboQuantKeyBuffers: [TurboQuantLayerStorage?]
    private let turboQuantValueBuffers: [TurboQuantLayerStorage?]
    private let layerStates: Mutex<[LayerState]>
    private let keyStorageKinds: [LayerStorageKind]
    private let valueStorageKinds: [LayerStorageKind]

    private var elementsPerToken: Int {
        numKVHeads * headDim
    }

    private var denseBytesPerToken: Int? {
        guard let bytesPerElement = precision.denseBytesPerElement else { return nil }
        return elementsPerToken * bytesPerElement
    }

    private var turboQuantRows: Int {
        maxSeqLen * numKVHeads
    }

    private var q8BlocksPerRow: Int {
        headDim / Self.q8BlockElementCount
    }

    private var q8RowBytes: Int {
        q8BlocksPerRow * Self.q8BlockByteCount
    }

    private static let q8BlockElementCount = 32
    private static let q8BlockByteCount = 34

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
        if precision == .turboquantV2 {
            self.keyStorageKinds = try (0..<numLayers).map { layerIndex in
                try Self.layerStorageKind(
                    cacheType: TurboQuantV2Contract.keyCacheType(forLayer: layerIndex, layerCount: numLayers),
                    preset: TurboQuantV2Contract.keyPreset(forLayer: layerIndex, layerCount: numLayers),
                    headDim: headDim
                )
            }
            self.valueStorageKinds = try (0..<numLayers).map { layerIndex in
                try Self.layerStorageKind(
                    cacheType: TurboQuantV2Contract.valueCacheType(forLayer: layerIndex, layerCount: numLayers),
                    preset: TurboQuantV2Contract.valuePreset(forLayer: layerIndex, layerCount: numLayers),
                    headDim: headDim
                )
            }
        } else if let presets = precision.turboQuantPresets {
            let keyLayout = try TurboQuantLayout(preset: presets.key, dimension: headDim)
            let valueLayout = try TurboQuantLayout(preset: presets.value, dimension: headDim)
            self.keyStorageKinds = Array(repeating: .turboQuant(preset: presets.key, layout: keyLayout), count: numLayers)
            self.valueStorageKinds = Array(repeating: .turboQuant(preset: presets.value, layout: valueLayout), count: numLayers)
        } else if precision == .q8_0 {
            self.keyStorageKinds = Array(repeating: .q8_0, count: numLayers)
            self.valueStorageKinds = Array(repeating: .q8_0, count: numLayers)
        } else {
            self.keyStorageKinds = Array(repeating: .dense, count: numLayers)
            self.valueStorageKinds = Array(repeating: .dense, count: numLayers)
        }
        let elementsPerToken = numKVHeads * headDim
        let turboQuantRows = maxSeqLen * numKVHeads
        var keyBuffers: [MetalBufferHandle?] = []
        var valueBuffers: [MetalBufferHandle?] = []
        var turboKeyBuffers: [TurboQuantLayerStorage?] = []
        var turboValueBuffers: [TurboQuantLayerStorage?] = []
        keyBuffers.reserveCapacity(numLayers)
        valueBuffers.reserveCapacity(numLayers)
        turboKeyBuffers.reserveCapacity(numLayers)
        turboValueBuffers.reserveCapacity(numLayers)

        for layerIndex in 0..<numLayers {
            let keyStorage = try Self.allocateLayerStorage(
                device: device,
                layerStorage: keyStorageKinds[layerIndex],
                maxSeqLen: maxSeqLen,
                turboQuantRows: turboQuantRows,
                elementsPerToken: elementsPerToken,
                precision: precision,
                headDim: headDim
            )
            keyBuffers.append(keyStorage.denseBuffer)
            turboKeyBuffers.append(keyStorage.turboBuffer)

            let valueStorage = try Self.allocateLayerStorage(
                device: device,
                layerStorage: valueStorageKinds[layerIndex],
                maxSeqLen: maxSeqLen,
                turboQuantRows: turboQuantRows,
                elementsPerToken: elementsPerToken,
                precision: precision,
                headDim: headDim
            )
            valueBuffers.append(valueStorage.denseBuffer)
            turboValueBuffers.append(valueStorage.turboBuffer)
        }

        self.keyBuffers = keyBuffers
        self.valueBuffers = valueBuffers
        self.turboQuantKeyBuffers = turboKeyBuffers
        self.turboQuantValueBuffers = turboValueBuffers
    }

    public func append(layer: Int, keys: [Float], values: [Float]) throws {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)

        switch precision {
        case .float32:
            let writePos = nextWritePosition(for: layer)
            let byteOffset = writePos * denseBytesPerToken!
            let keyBuffer = try denseBuffer(layer: layer, kind: .key)
            let valueBuffer = try denseBuffer(layer: layer, kind: .value)

            keys.withUnsafeBytes { bytes in
                keyBuffer.contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            values.withUnsafeBytes { bytes in
                valueBuffer.contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }

        case .turboquantV2, .turboQuantBalanced, .turboQuantAggressive:
            let writePos = nextWritePosition(for: layer)
            try appendTurboQuant(
                layer: layer,
                writePos: writePos,
                keys: keys,
                values: values
            )
        case .q8_0:
            let writePos = nextWritePosition(for: layer)
            appendQ8_0(
                layer: layer,
                writePos: writePos,
                keys: keys,
                values: values
            )

        default:
            throw KVCacheError.precisionMismatch
        }
    }

    public func appendF16(layer: Int, keys: [Float16], values: [Float16]) throws {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision == .float16 else { throw KVCacheError.precisionMismatch }
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)

        let writePos = nextWritePosition(for: layer)
        let byteOffset = writePos * denseBytesPerToken!
        let keyBuffer = try denseBuffer(layer: layer, kind: .key)
        let valueBuffer = try denseBuffer(layer: layer, kind: .value)

        keys.withUnsafeBytes { bytes in
            keyBuffer.contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        values.withUnsafeBytes { bytes in
            valueBuffer.contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }

    public func retrieve<T>(layer: Int, asType: T.Type) throws -> ([T], [T]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        try validateRequestedType(asType)

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else {
            return ([], [])
        }

        let totalElements = currentLen * elementsPerToken
        let keyPtr = try denseBuffer(layer: layer, kind: .key).contents().bindMemory(to: T.self, capacity: maxSeqLen * elementsPerToken)
        let valuePtr = try denseBuffer(layer: layer, kind: .value).contents().bindMemory(to: T.self, capacity: maxSeqLen * elementsPerToken)

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

    public func retrieveDecodedTurboQuant(layer: Int) throws -> ([Float], [Float]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else {
            throw KVCacheError.precisionMismatch
        }

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else { return ([], []) }

        var keys: [Float] = []
        var values: [Float] = []
        keys.reserveCapacity(currentLen * elementsPerToken)
        values.reserveCapacity(currentLen * elementsPerToken)

        let rowIndices = orderedRowIndices(for: state, currentLen: currentLen)
        for tokenIndex in rowIndices {
            for kvHead in 0..<numKVHeads {
                let rowIndex = tokenIndex * numKVHeads + kvHead
                switch keyStorageKinds[layer] {
                case .dense:
                    keys.append(contentsOf: decodeDenseF16Row(from: try denseBuffer(layer: layer, kind: .key), rowIndex: rowIndex))
                case .q8_0:
                    keys.append(contentsOf: decodeQ8Row(from: try denseBuffer(layer: layer, kind: .key), rowIndex: rowIndex))
                case let .turboQuant(preset, layout):
                    let encodedK = try readTurboQuantRow(
                        storage: try turboBuffer(layer: layer, kind: .key),
                        layout: layout,
                        rowIndex: rowIndex,
                        preset: preset
                    )
                    keys.append(contentsOf: try TurboQuantReferenceEncoder.approximateDecode(
                        encodedK,
                        rotationSeed: TurboQuantSeeds.keyRotation,
                        residualSeed: TurboQuantSeeds.keyResidual
                    ))
                }

                switch valueStorageKinds[layer] {
                case .dense:
                    values.append(contentsOf: decodeDenseF16Row(from: try denseBuffer(layer: layer, kind: .value), rowIndex: rowIndex))
                case .q8_0:
                    values.append(contentsOf: decodeQ8Row(from: try denseBuffer(layer: layer, kind: .value), rowIndex: rowIndex))
                case let .turboQuant(preset, layout):
                    let encodedV = try readTurboQuantRow(
                        storage: try turboBuffer(layer: layer, kind: .value),
                        layout: layout,
                        rowIndex: rowIndex,
                        preset: preset
                    )
                    values.append(contentsOf: try TurboQuantReferenceEncoder.approximateDecode(
                        encodedV,
                        residualWeight: TurboQuantV2Contract.valueResidualScale(forLayer: layer, layerCount: numLayers),
                        rotationSeed: TurboQuantSeeds.valueRotation,
                        residualSeed: TurboQuantSeeds.valueResidual
                    ))
                }
            }
        }

        return (keys, values)
    }

    public func retrieveTurboQuantRuntimeRows(layer: Int) throws -> (keys: [TurboQuantRuntimeRow], values: [TurboQuantRuntimeRow]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else {
            throw KVCacheError.precisionMismatch
        }

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else { return ([], []) }

        guard case let .turboQuant(keyPreset, keyLayout) = keyStorageKinds[layer],
              case let .turboQuant(valuePreset, valueLayout) = valueStorageKinds[layer] else {
            throw KVCacheError.compressedBufferAccessRequiresTurboQuantAPI
        }
        var keys: [TurboQuantRuntimeRow] = []
        var values: [TurboQuantRuntimeRow] = []
        keys.reserveCapacity(currentLen * numKVHeads)
        values.reserveCapacity(currentLen * numKVHeads)

        let rowIndices = orderedRowIndices(for: state, currentLen: currentLen)
        for tokenIndex in rowIndices {
            for kvHead in 0..<numKVHeads {
                let rowIndex = tokenIndex * numKVHeads + kvHead
                let encodedKey = try readTurboQuantRow(
                    storage: try turboBuffer(layer: layer, kind: .key),
                    layout: keyLayout,
                    rowIndex: rowIndex,
                    preset: keyPreset
                )
                let encodedValue = try readTurboQuantRow(
                    storage: try turboBuffer(layer: layer, kind: .value),
                    layout: valueLayout,
                    rowIndex: rowIndex,
                    preset: valuePreset
                )
                keys.append(try TurboQuantReferenceEncoder.makeRuntimeRow(from: encodedKey))
                values.append(try TurboQuantReferenceEncoder.makeRuntimeRow(from: encodedValue))
            }
        }

        return (keys, values)
    }

    public func metalBuffers(layer: Int) throws -> (MTLBuffer, MTLBuffer) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        return (try denseBuffer(layer: layer, kind: .key).rawValue, try denseBuffer(layer: layer, kind: .value).rawValue)
    }

    public func keyMetalBuffer(layer: Int) throws -> MTLBuffer? {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        return keyBuffers[layer]?.rawValue
    }

    public func valueMetalBuffer(layer: Int) throws -> MTLBuffer? {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        return valueBuffers[layer]?.rawValue
    }

    public func turboQuantMetalBuffers(layer: Int) throws -> (key: TurboQuantMetalBuffers, value: TurboQuantMetalBuffers) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else { throw KVCacheError.precisionMismatch }

        let keyStorage = try turboBuffer(layer: layer, kind: .key)
        let valueStorage = try turboBuffer(layer: layer, kind: .value)
        return (
            key: TurboQuantMetalBuffers(
                codes: keyStorage.codes.rawValue,
                residualSigns: keyStorage.residualSigns.rawValue,
                outlierMask: keyStorage.outlierMask.rawValue,
                metadata: keyStorage.metadata.rawValue
            ),
            value: TurboQuantMetalBuffers(
                codes: valueStorage.codes.rawValue,
                residualSigns: valueStorage.residualSigns.rawValue,
                outlierMask: valueStorage.outlierMask.rawValue,
                metadata: valueStorage.metadata.rawValue
            )
        )
    }

    public func turboQuantKeyMetalBuffers(layer: Int) throws -> TurboQuantMetalBuffers? {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else { return nil }
        guard let keyStorage = turboQuantKeyBuffers[layer] else { return nil }
        return TurboQuantMetalBuffers(
            codes: keyStorage.codes.rawValue,
            residualSigns: keyStorage.residualSigns.rawValue,
            outlierMask: keyStorage.outlierMask.rawValue,
            metadata: keyStorage.metadata.rawValue
        )
    }

    public func turboQuantValueMetalBuffers(layer: Int) throws -> TurboQuantMetalBuffers? {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else { return nil }
        guard let valueStorage = turboQuantValueBuffers[layer] else { return nil }
        return TurboQuantMetalBuffers(
            codes: valueStorage.codes.rawValue,
            residualSigns: valueStorage.residualSigns.rawValue,
            outlierMask: valueStorage.outlierMask.rawValue,
            metadata: valueStorage.metadata.rawValue
        )
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

    public func advanceWritePosition(layer: Int, count: Int) {
        layerStates.withLock { states in
            for _ in 0..<count {
                states[layer].writePos = (states[layer].writePos + 1) % maxSeqLen
                states[layer].totalWritten += 1
            }
        }
    }

    public func setPosition(_ position: Int) {
        layerStates.withLock { states in
            for index in states.indices {
                states[index].writePos = position % maxSeqLen
                states[index].totalWritten = position
            }
        }
    }

    private static func allocateTurboQuantLayer(
        device: MTLDevice,
        rowCount: Int,
        layout: TurboQuantLayout
    ) throws -> TurboQuantLayerStorage {
        let codeLength = rowCount * layout.codeWordsPerRow * MemoryLayout<UInt32>.stride
        let residualLength = rowCount * TurboQuantLayout.residualWordsPerRow * MemoryLayout<UInt32>.stride
        let maskLength = rowCount * TurboQuantLayout.outlierMaskWordsPerRow * MemoryLayout<UInt32>.stride
        let metadataLength = rowCount * TurboQuantLayout.metadataScalarsPerRow * MemoryLayout<Float>.stride

        guard let codes = device.makeBuffer(length: codeLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let residualSigns = device.makeBuffer(length: residualLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let outlierMask = device.makeBuffer(length: maskLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let metadata = device.makeBuffer(length: metadataLength, options: [.storageModeShared, .hazardTrackingModeUntracked]) else {
            throw KVCacheError.allocationFailed
        }

        return TurboQuantLayerStorage(
            codes: MetalBufferHandle(codes),
            residualSigns: MetalBufferHandle(residualSigns),
            outlierMask: MetalBufferHandle(outlierMask),
            metadata: MetalBufferHandle(metadata)
        )
    }

    private static func layerStorageKind(
        cacheType: TurboQuantLayerCacheType,
        preset: TurboQuantPreset?,
        headDim: Int
    ) throws -> LayerStorageKind {
        switch cacheType {
        case .dense:
            return .dense
        case .q8_0:
            guard headDim % Self.q8BlockElementCount == 0 else {
                throw KVCacheError.unsupportedStorage
            }
            return .q8_0
        case .turbo2, .turbo3, .turbo4:
            guard let preset else {
                throw KVCacheError.unsupportedStorage
            }
            return .turboQuant(
                preset: preset,
                layout: try TurboQuantLayout(preset: preset, dimension: headDim)
            )
        }
    }

    private static func allocateLayerStorage(
        device: MTLDevice,
        layerStorage: LayerStorageKind,
        maxSeqLen: Int,
        turboQuantRows: Int,
        elementsPerToken: Int,
        precision: Precision,
        headDim: Int
    ) throws -> (denseBuffer: MetalBufferHandle?, turboBuffer: TurboQuantLayerStorage?) {
        switch layerStorage {
        case .dense:
            let bytesPerElement: Int
            switch precision {
            case .float32:
                bytesPerElement = MemoryLayout<Float>.stride
            case .float16, .turboquantV2, .turboQuantBalanced, .turboQuantAggressive:
                bytesPerElement = MemoryLayout<Float16>.stride
            case .float8:
                bytesPerElement = 1
            case .q8_0:
                throw KVCacheError.unsupportedStorage
            }
            let bytesPerToken = elementsPerToken * bytesPerElement
            let bufferLength = maxSeqLen * bytesPerToken
            guard let buffer = device.makeBuffer(length: bufferLength, options: [.storageModeShared]) else {
                throw KVCacheError.allocationFailed
            }
            return (MetalBufferHandle(buffer), nil)
        case .q8_0:
            let rowByteCount = (headDim / Self.q8BlockElementCount) * Self.q8BlockByteCount
            let bufferLength = turboQuantRows * rowByteCount
            guard let buffer = device.makeBuffer(
                length: bufferLength,
                options: [.storageModeShared, .hazardTrackingModeUntracked]
            ) else {
                throw KVCacheError.allocationFailed
            }
            return (MetalBufferHandle(buffer), nil)
        case let .turboQuant(_, layout):
            return (nil, try allocateTurboQuantLayer(device: device, rowCount: turboQuantRows, layout: layout))
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

    private enum BufferKind {
        case key
        case value
    }

    private func denseBuffer(layer: Int, kind: BufferKind) throws -> MetalBufferHandle {
        let buffer = switch kind {
        case .key:
            keyBuffers[layer]
        case .value:
            valueBuffers[layer]
        }
        guard let buffer else {
            throw KVCacheError.compressedBufferAccessRequiresTurboQuantAPI
        }
        return buffer
    }

    private func turboBuffer(layer: Int, kind: BufferKind) throws -> TurboQuantLayerStorage {
        let buffer = switch kind {
        case .key:
            turboQuantKeyBuffers[layer]
        case .value:
            turboQuantValueBuffers[layer]
        }
        guard let buffer else {
            throw KVCacheError.compressedBufferAccessRequiresTurboQuantAPI
        }
        return buffer
    }

    private func validateRequestedType<T>(_ requestedType: T.Type) throws {
        guard !precision.isTurboQuant else {
            throw KVCacheError.compressedRetrieveRequiresDecodedAPI
        }

        switch precision {
        case .float32:
            guard requestedType == Float.self else { throw KVCacheError.precisionMismatch }
        case .float16:
            guard requestedType == Float16.self else { throw KVCacheError.precisionMismatch }
        case .float8:
            guard requestedType == UInt8.self else { throw KVCacheError.precisionMismatch }
        case .q8_0:
            throw KVCacheError.compressedRetrieveRequiresDecodedAPI
        case .turboquantV2, .turboQuantBalanced, .turboQuantAggressive:
            throw KVCacheError.compressedRetrieveRequiresDecodedAPI
        }
    }

    public func retrieveDecodedQ8_0(layer: Int) throws -> ([Float], [Float]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision == .q8_0 else { throw KVCacheError.precisionMismatch }

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else { return ([], []) }

        let keyBuffer = try denseBuffer(layer: layer, kind: .key)
        let valueBuffer = try denseBuffer(layer: layer, kind: .value)
        var keys: [Float] = []
        var values: [Float] = []
        keys.reserveCapacity(currentLen * elementsPerToken)
        values.reserveCapacity(currentLen * elementsPerToken)

        let rowIndices = orderedRowIndices(for: state, currentLen: currentLen)
        for tokenIndex in rowIndices {
            for kvHead in 0..<numKVHeads {
                let rowIndex = tokenIndex * numKVHeads + kvHead
                keys.append(contentsOf: decodeQ8Row(from: keyBuffer, rowIndex: rowIndex))
                values.append(contentsOf: decodeQ8Row(from: valueBuffer, rowIndex: rowIndex))
            }
        }

        return (keys, values)
    }

    private func appendTurboQuant(
        layer: Int,
        writePos: Int,
        keys: [Float],
        values: [Float]
    ) throws {
        guard precision.isTurboQuant else {
            throw KVCacheError.precisionMismatch
        }

        for kvHead in 0..<numKVHeads {
            let start = kvHead * headDim
            let end = start + headDim
            let rowIndex = writePos * numKVHeads + kvHead
            switch keyStorageKinds[layer] {
            case .dense:
                let keyBase = try denseBuffer(layer: layer, kind: .key).contents().assumingMemoryBound(to: UInt8.self)
                encodeDenseF16Row(
                    source: keys,
                    sourceOffset: start,
                    destination: keyBase.advanced(by: rowIndex * headDim * MemoryLayout<Float16>.stride)
                )
            case .q8_0:
                let keyBase = try denseBuffer(layer: layer, kind: .key).contents().assumingMemoryBound(to: UInt8.self)
                encodeQ8Row(
                    source: keys,
                    sourceOffset: start,
                    destination: keyBase.advanced(by: rowIndex * q8RowBytes)
                )
            case let .turboQuant(preset, layout):
                let encodedKey = try TurboQuantReferenceEncoder.encode(
                    Array(keys[start..<end]),
                    preset: preset,
                    outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                )
                writeTurboQuantRow(encodedKey, to: try turboBuffer(layer: layer, kind: .key), layout: layout, rowIndex: rowIndex)
            }

            switch valueStorageKinds[layer] {
            case .dense:
                let valueBase = try denseBuffer(layer: layer, kind: .value).contents().assumingMemoryBound(to: UInt8.self)
                encodeDenseF16Row(
                    source: values,
                    sourceOffset: start,
                    destination: valueBase.advanced(by: rowIndex * headDim * MemoryLayout<Float16>.stride)
                )
            case .q8_0:
                let valueBase = try denseBuffer(layer: layer, kind: .value).contents().assumingMemoryBound(to: UInt8.self)
                encodeQ8Row(
                    source: values,
                    sourceOffset: start,
                    destination: valueBase.advanced(by: rowIndex * q8RowBytes)
                )
            case let .turboQuant(preset, layout):
                let encodedValue = try TurboQuantReferenceEncoder.encode(
                    Array(values[start..<end]),
                    preset: preset,
                    outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
                writeTurboQuantRow(encodedValue, to: try turboBuffer(layer: layer, kind: .value), layout: layout, rowIndex: rowIndex)
            }
        }
    }

    private func appendQ8_0(
        layer: Int,
        writePos: Int,
        keys: [Float],
        values: [Float]
    ) {
        guard let keyBuffer = try? denseBuffer(layer: layer, kind: .key),
              let valueBuffer = try? denseBuffer(layer: layer, kind: .value) else {
            return
        }
        let keyBase = keyBuffer.contents().assumingMemoryBound(to: UInt8.self)
        let valueBase = valueBuffer.contents().assumingMemoryBound(to: UInt8.self)

        for kvHead in 0..<numKVHeads {
            let start = kvHead * headDim
            let rowIndex = writePos * numKVHeads + kvHead
            let rowOffset = rowIndex * q8RowBytes
            encodeQ8Row(
                source: keys,
                sourceOffset: start,
                destination: keyBase.advanced(by: rowOffset)
            )
            encodeQ8Row(
                source: values,
                sourceOffset: start,
                destination: valueBase.advanced(by: rowOffset)
            )
        }
    }

    private func encodeQ8Row(
        source: [Float],
        sourceOffset: Int,
        destination: UnsafeMutablePointer<UInt8>
    ) {
        for blockIndex in 0..<q8BlocksPerRow {
            let blockOffset = blockIndex * Self.q8BlockByteCount
            let sourceBase = sourceOffset + blockIndex * Self.q8BlockElementCount

            var maxAbs: Float = 0
            for lane in 0..<Self.q8BlockElementCount {
                maxAbs = max(maxAbs, abs(source[sourceBase + lane]))
            }

            let scale: Float = maxAbs > 0 ? (maxAbs / 127.0) : 0
            let halfScale = Float16(scale)
            withUnsafeBytes(of: halfScale.bitPattern.littleEndian) { bytes in
                destination.advanced(by: blockOffset).update(from: bytes.bindMemory(to: UInt8.self).baseAddress!, count: 2)
            }

            for lane in 0..<Self.q8BlockElementCount {
                let value = source[sourceBase + lane]
                let quantized: Int8
                if scale == 0 {
                    quantized = 0
                } else {
                    let rounded = Int((value / scale).rounded())
                    quantized = Int8(clamping: rounded)
                }
                destination[blockOffset + 2 + lane] = UInt8(bitPattern: quantized)
            }
        }
    }

    private func decodeQ8Row(from buffer: MetalBufferHandle, rowIndex: Int) -> [Float] {
        let base = buffer.contents().assumingMemoryBound(to: UInt8.self)
        let rowStart = base.advanced(by: rowIndex * q8RowBytes)
        var values = [Float](repeating: 0, count: headDim)

        for blockIndex in 0..<q8BlocksPerRow {
            let blockOffset = blockIndex * Self.q8BlockByteCount
            let scaleBits = rowStart.advanced(by: blockOffset).withMemoryRebound(to: UInt16.self, capacity: 1) { $0.pointee }
            let scale = Float(Float16(bitPattern: scaleBits))
            for lane in 0..<Self.q8BlockElementCount {
                let q = Int8(bitPattern: rowStart[blockOffset + 2 + lane])
                values[blockIndex * Self.q8BlockElementCount + lane] = scale * Float(q)
            }
        }

        return values
    }

    private func encodeDenseF16Row(source: [Float], sourceOffset: Int, destination: UnsafeMutablePointer<UInt8>) {
        let destinationF16 = UnsafeMutableRawPointer(destination).assumingMemoryBound(to: Float16.self)
        for lane in 0..<headDim {
            destinationF16[lane] = Float16(source[sourceOffset + lane])
        }
    }

    private func decodeDenseF16Row(from buffer: MetalBufferHandle, rowIndex: Int) -> [Float] {
        let rowStride = headDim * MemoryLayout<Float16>.stride
        let rowStart = UnsafeRawPointer(buffer.contents().advanced(by: rowIndex * rowStride)).assumingMemoryBound(to: Float16.self)
        return (0..<headDim).map { Float(rowStart[$0]) }
    }

    private func writeTurboQuantRow(
        _ encoded: TurboQuantEncodedRow,
        to storage: TurboQuantLayerStorage,
        layout: TurboQuantLayout,
        rowIndex: Int
    ) {
        let codeOffset = rowIndex * layout.codeWordsPerRow
        let residualOffset = rowIndex * TurboQuantLayout.residualWordsPerRow
        let maskOffset = rowIndex * TurboQuantLayout.outlierMaskWordsPerRow
        let metadataOffset = rowIndex * TurboQuantLayout.metadataScalarsPerRow

        let codePtr = storage.codes.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * layout.codeWordsPerRow)
        let residualPtr = storage.residualSigns.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * TurboQuantLayout.residualWordsPerRow)
        let maskPtr = storage.outlierMask.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * TurboQuantLayout.outlierMaskWordsPerRow)
        let metadataPtr = storage.metadata.contents().bindMemory(to: Float.self, capacity: turboQuantRows * TurboQuantLayout.metadataScalarsPerRow)

        for index in 0..<layout.codeWordsPerRow {
            codePtr[codeOffset + index] = encoded.primaryCodes[index]
        }
        for index in 0..<TurboQuantLayout.residualWordsPerRow {
            residualPtr[residualOffset + index] = encoded.residualSigns[index]
            maskPtr[maskOffset + index] = encoded.outlierMask[index]
        }
        metadataPtr[metadataOffset] = encoded.rowNorm
        metadataPtr[metadataOffset + 1] = encoded.residualNorm
    }

    private func readTurboQuantRow(
        storage: TurboQuantLayerStorage,
        layout: TurboQuantLayout,
        rowIndex: Int,
        preset: TurboQuantPreset
    ) throws -> TurboQuantEncodedRow {
        let codeOffset = rowIndex * layout.codeWordsPerRow
        let residualOffset = rowIndex * TurboQuantLayout.residualWordsPerRow
        let maskOffset = rowIndex * TurboQuantLayout.outlierMaskWordsPerRow
        let metadataOffset = rowIndex * TurboQuantLayout.metadataScalarsPerRow

        let codePtr = storage.codes.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * layout.codeWordsPerRow)
        let residualPtr = storage.residualSigns.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * TurboQuantLayout.residualWordsPerRow)
        let maskPtr = storage.outlierMask.contents().bindMemory(to: UInt32.self, capacity: turboQuantRows * TurboQuantLayout.outlierMaskWordsPerRow)
        let metadataPtr = storage.metadata.contents().bindMemory(to: Float.self, capacity: turboQuantRows * TurboQuantLayout.metadataScalarsPerRow)

        let codes = Array(UnsafeBufferPointer(start: codePtr.advanced(by: codeOffset), count: layout.codeWordsPerRow))
        let residualSigns = Array(UnsafeBufferPointer(start: residualPtr.advanced(by: residualOffset), count: TurboQuantLayout.residualWordsPerRow))
        let outlierMask = Array(UnsafeBufferPointer(start: maskPtr.advanced(by: maskOffset), count: TurboQuantLayout.outlierMaskWordsPerRow))

        return TurboQuantEncodedRow(
            preset: preset,
            dimension: headDim,
            primaryCodes: codes,
            residualSigns: residualSigns,
            outlierMask: outlierMask,
            rowNorm: metadataPtr[metadataOffset],
            residualNorm: metadataPtr[metadataOffset + 1]
        )
    }

    private func orderedRowIndices(for state: LayerState, currentLen: Int) -> [Int] {
        if state.totalWritten <= maxSeqLen {
            return Array(0..<currentLen)
        }

        var indices = Array(state.writePos..<maxSeqLen)
        indices.append(contentsOf: 0..<state.writePos)
        return indices
    }
}

public enum KVCacheError: Error, Sendable {
    case allocationFailed
    case invalidLayer(Int)
    case precisionMismatch
    case unsupportedStorage
    case compressedRetrieveRequiresDecodedAPI
    case compressedBufferAccessRequiresTurboQuantAPI
}
