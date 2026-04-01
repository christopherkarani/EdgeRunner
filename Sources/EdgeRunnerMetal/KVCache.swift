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
            case .turboQuantBalanced, .turboQuantAggressive:
                return nil
            }
        }

        fileprivate var rawValue: UInt32 {
            switch self {
            case .float32: return 0
            case .float16: return 1
            case .float8: return 2
            case .turboQuantBalanced: return 3
            case .turboQuantAggressive: return 4
            }
        }

        fileprivate var turboQuantPreset: TurboQuantPreset? {
            switch self {
            case .turboQuantBalanced:
                return .balanced
            case .turboQuantAggressive:
                return .aggressive
            default:
                return nil
            }
        }

        fileprivate var isTurboQuant: Bool {
            turboQuantPreset != nil
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

    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision

    private let keyBuffers: [MetalBufferHandle]
    private let valueBuffers: [MetalBufferHandle]
    private let turboQuantKeyBuffers: [TurboQuantLayerStorage]
    private let turboQuantValueBuffers: [TurboQuantLayerStorage]
    private let layerStates: Mutex<[LayerState]>
    private let turboQuantLayout: TurboQuantLayout?

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
        self.turboQuantLayout = try precision.turboQuantPreset.map {
            try TurboQuantLayout(preset: $0, dimension: headDim)
        }
        let elementsPerToken = numKVHeads * headDim
        let turboQuantRows = maxSeqLen * numKVHeads

        if let bytesPerElement = precision.denseBytesPerElement {
            let bytesPerToken = elementsPerToken * bytesPerElement
            let bufferLength = maxSeqLen * bytesPerToken
            var keyBuffers: [MetalBufferHandle] = []
            var valueBuffers: [MetalBufferHandle] = []
            keyBuffers.reserveCapacity(numLayers)
            valueBuffers.reserveCapacity(numLayers)

            for _ in 0..<numLayers {
                guard let keyBuffer = device.makeBuffer(
                    length: bufferLength,
                    options: [.storageModeShared]
                ),
                let valueBuffer = device.makeBuffer(
                    length: bufferLength,
                    options: [.storageModeShared]
                ) else {
                    throw KVCacheError.allocationFailed
                }
                keyBuffers.append(MetalBufferHandle(keyBuffer))
                valueBuffers.append(MetalBufferHandle(valueBuffer))
            }

            self.keyBuffers = keyBuffers
            self.valueBuffers = valueBuffers
            self.turboQuantKeyBuffers = []
            self.turboQuantValueBuffers = []
        } else if let layout = turboQuantLayout {
            self.keyBuffers = []
            self.valueBuffers = []
            self.turboQuantKeyBuffers = try Self.allocateTurboQuantLayers(
                device: device,
                layerCount: numLayers,
                rowCount: turboQuantRows,
                layout: layout
            )
            self.turboQuantValueBuffers = try Self.allocateTurboQuantLayers(
                device: device,
                layerCount: numLayers,
                rowCount: turboQuantRows,
                layout: layout
            )
        } else {
            throw KVCacheError.unsupportedStorage
        }
    }

    public func append(layer: Int, keys: [Float], values: [Float]) throws {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)

        switch precision {
        case .float32:
            let writePos = nextWritePosition(for: layer)
            let byteOffset = writePos * denseBytesPerToken!

            keys.withUnsafeBytes { bytes in
                keyBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            values.withUnsafeBytes { bytes in
                valueBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }

        case .turboQuantBalanced, .turboQuantAggressive:
            let writePos = nextWritePosition(for: layer)
            try appendTurboQuant(
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

        keys.withUnsafeBytes { bytes in
            keyBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        values.withUnsafeBytes { bytes in
            valueBuffers[layer].contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
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

    public func retrieveDecodedTurboQuant(layer: Int) throws -> ([Float], [Float]) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard let layout = turboQuantLayout, let preset = precision.turboQuantPreset else {
            throw KVCacheError.precisionMismatch
        }

        let state = layerStates.withLock { $0[layer] }
        let currentLen = min(state.totalWritten, maxSeqLen)
        guard currentLen > 0 else { return ([], []) }

        let keyStorage = turboQuantKeyBuffers[layer]
        let valueStorage = turboQuantValueBuffers[layer]
        var keys: [Float] = []
        var values: [Float] = []
        keys.reserveCapacity(currentLen * elementsPerToken)
        values.reserveCapacity(currentLen * elementsPerToken)

        let rowIndices = orderedRowIndices(for: state, currentLen: currentLen)
        for tokenIndex in rowIndices {
            for kvHead in 0..<numKVHeads {
                let rowIndex = tokenIndex * numKVHeads + kvHead
                let encodedK = try readTurboQuantRow(
                    storage: keyStorage,
                    layout: layout,
                    rowIndex: rowIndex,
                    preset: preset
                )
                let encodedV = try readTurboQuantRow(
                    storage: valueStorage,
                    layout: layout,
                    rowIndex: rowIndex,
                    preset: preset
                )
                keys.append(contentsOf: try TurboQuantReferenceEncoder.approximateDecode(
                    encodedK,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                ))
                values.append(contentsOf: try TurboQuantReferenceEncoder.approximateDecode(
                    encodedV,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                ))
            }
        }

        return (keys, values)
    }

    public func metalBuffers(layer: Int) throws -> (MTLBuffer, MTLBuffer) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard !precision.isTurboQuant else { throw KVCacheError.compressedBufferAccessRequiresTurboQuantAPI }
        return (keyBuffers[layer].rawValue, valueBuffers[layer].rawValue)
    }

    public func turboQuantMetalBuffers(layer: Int) throws -> (key: TurboQuantMetalBuffers, value: TurboQuantMetalBuffers) {
        guard (0..<numLayers).contains(layer) else { throw KVCacheError.invalidLayer(layer) }
        guard precision.isTurboQuant else { throw KVCacheError.precisionMismatch }

        let keyStorage = turboQuantKeyBuffers[layer]
        let valueStorage = turboQuantValueBuffers[layer]
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

    private static func allocateTurboQuantLayers(
        device: MTLDevice,
        layerCount: Int,
        rowCount: Int,
        layout: TurboQuantLayout
    ) throws -> [TurboQuantLayerStorage] {
        var storages: [TurboQuantLayerStorage] = []
        storages.reserveCapacity(layerCount)

        let codeLength = rowCount * layout.codeWordsPerRow * MemoryLayout<UInt32>.stride
        let residualLength = rowCount * TurboQuantLayout.residualWordsPerRow * MemoryLayout<UInt32>.stride
        let maskLength = rowCount * TurboQuantLayout.outlierMaskWordsPerRow * MemoryLayout<UInt32>.stride
        let metadataLength = rowCount * TurboQuantLayout.metadataScalarsPerRow * MemoryLayout<Float>.stride

        for _ in 0..<layerCount {
            guard let codes = device.makeBuffer(length: codeLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
                  let residualSigns = device.makeBuffer(length: residualLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
                  let outlierMask = device.makeBuffer(length: maskLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
                  let metadata = device.makeBuffer(length: metadataLength, options: [.storageModeShared, .hazardTrackingModeUntracked]) else {
                throw KVCacheError.allocationFailed
            }

            storages.append(
                TurboQuantLayerStorage(
                    codes: MetalBufferHandle(codes),
                    residualSigns: MetalBufferHandle(residualSigns),
                    outlierMask: MetalBufferHandle(outlierMask),
                    metadata: MetalBufferHandle(metadata)
                )
            )
        }

        return storages
    }

    private func nextWritePosition(for layer: Int) -> Int {
        layerStates.withLock { states in
            let writePos = states[layer].writePos
            states[layer].writePos = (states[layer].writePos + 1) % maxSeqLen
            states[layer].totalWritten += 1
            return writePos
        }
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
        case .turboQuantBalanced, .turboQuantAggressive:
            throw KVCacheError.compressedRetrieveRequiresDecodedAPI
        }
    }

    private func appendTurboQuant(
        layer: Int,
        writePos: Int,
        keys: [Float],
        values: [Float]
    ) throws {
        guard let preset = precision.turboQuantPreset, let layout = turboQuantLayout else {
            throw KVCacheError.precisionMismatch
        }

        let keyStorage = turboQuantKeyBuffers[layer]
        let valueStorage = turboQuantValueBuffers[layer]

        for kvHead in 0..<numKVHeads {
            let start = kvHead * headDim
            let end = start + headDim
            let rowIndex = writePos * numKVHeads + kvHead

            let encodedKey = try TurboQuantReferenceEncoder.encode(
                Array(keys[start..<end]),
                preset: preset,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )
            let encodedValue = try TurboQuantReferenceEncoder.encode(
                Array(values[start..<end]),
                preset: preset,
                rotationSeed: TurboQuantSeeds.valueRotation,
                residualSeed: TurboQuantSeeds.valueResidual
            )

            writeTurboQuantRow(encodedKey, to: keyStorage, layout: layout, rowIndex: rowIndex)
            writeTurboQuantRow(encodedValue, to: valueStorage, layout: layout, rowIndex: rowIndex)
        }
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
