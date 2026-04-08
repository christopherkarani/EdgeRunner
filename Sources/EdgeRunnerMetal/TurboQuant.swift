import Foundation

private enum TurboQuantEnvOverrides {
    static let keyType = "EDGERUNNER_TURBOQUANT_KEY_TYPE"
    static let valueType = "EDGERUNNER_TURBOQUANT_VALUE_TYPE"
    static let adaptiveMode = "EDGERUNNER_TURBOQUANT_LAYER_ADAPTIVE"
    static let keyPolicy = "EDGERUNNER_TURBOQUANT_KEY_POLICY"
    static let valuePolicy = "EDGERUNNER_TURBOQUANT_VALUE_POLICY"
    static let keyPreset = "EDGERUNNER_TURBOQUANT_KEY_PRESET"
    static let valuePreset = "EDGERUNNER_TURBOQUANT_VALUE_PRESET"
    static let promotedKeyPreset = "EDGERUNNER_TURBOQUANT_PROMOTED_KEY_PRESET"
    static let keyOutlierSelection = "EDGERUNNER_TURBOQUANT_KEY_OUTLIER_SELECTION"
    static let valueOutlierSelection = "EDGERUNNER_TURBOQUANT_VALUE_OUTLIER_SELECTION"
    static let innerQEnabled = "EDGERUNNER_TURBOQUANT_INNERQ"
    static let innerQSamples = "EDGERUNNER_TURBOQUANT_INNERQ_SAMPLES"
    static let innerQStrength = "EDGERUNNER_TURBOQUANT_INNERQ_STRENGTH"
    static let forkAdaptiveMode = "TURBO_LAYER_ADAPTIVE"
    static let forkInnerQSamples = "TURBO_INNERQ"
    static let forkInnerQStrength = "TURBO_INNERQ_STRENGTH"
}

public enum TurboQuantError: Error, Sendable, Equatable {
    case unsupportedDimension(Int)
    case unsupportedBitWidth(Int)
    case invalidCodeCount(expected: Int, actual: Int)
}

public enum TurboQuantPreset: String, Sendable, Equatable, Codable {
    case turbo2
    case turbo3
    case turbo4
    case balanced
    case balanced64
    case balanced96
    case fiveBit
    case sixBit
    case sevenBit
    case aggressive
    case aggressive64

    public var descriptor: TurboQuantPresetDescriptor {
        switch self {
        case .turbo2:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 2,
                highPrecisionBits: 2,
                highPrecisionChannelCount: 0
            )
        case .turbo3:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 3,
                highPrecisionBits: 3,
                highPrecisionChannelCount: 0
            )
        case .turbo4:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 4,
                highPrecisionBits: 4,
                highPrecisionChannelCount: 0
            )
        case .balanced:
            // Engineering assumption until the authors publish the exact 3.5-bit split.
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 3,
                highPrecisionBits: 5,
                highPrecisionChannelCount: 32
            )
        case .balanced64:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 3,
                highPrecisionBits: 5,
                highPrecisionChannelCount: 64
            )
        case .balanced96:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 3,
                highPrecisionBits: 5,
                highPrecisionChannelCount: 96
            )
        case .fiveBit:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 5,
                highPrecisionBits: 5,
                highPrecisionChannelCount: 0
            )
        case .sixBit:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 6,
                highPrecisionBits: 6,
                highPrecisionChannelCount: 0
            )
        case .sevenBit:
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 7,
                highPrecisionBits: 7,
                highPrecisionChannelCount: 0
            )
        case .aggressive:
            // Historical EdgeRunner aggressive preset. This is 2.25 effective bits.
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 2,
                highPrecisionBits: 3,
                highPrecisionChannelCount: 32
            )
        case .aggressive64:
            // Matches a true 2.5-bit split for d = 128.
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 2,
                highPrecisionBits: 3,
                highPrecisionChannelCount: 64
            )
        }
    }
}

public enum TurboQuantOutlierSelection: String, Sendable, Equatable {
    case magnitude
    case quantizationBenefit
}

public struct TurboQuantPresetDescriptor: Sendable, Equatable {
    public let preset: TurboQuantPreset
    public let regularBits: Int
    public let highPrecisionBits: Int
    public let highPrecisionChannelCount: Int

    public var regularChannelCount: Int { 128 - highPrecisionChannelCount }
    public var effectiveBits: Double {
        let totalBits = (regularChannelCount * regularBits) + (highPrecisionChannelCount * highPrecisionBits)
        return Double(totalBits) / 128.0
    }
}

public enum TurboQuantFixedType: String, Sendable, Equatable, Codable {
    case turbo2
    case turbo3
    case turbo4

    public var preset: TurboQuantPreset {
        switch self {
        case .turbo2: return .turbo2
        case .turbo3: return .turbo3
        case .turbo4: return .turbo4
        }
    }

    public var residualScale: Float {
        switch self {
        case .turbo2, .turbo3, .turbo4:
            return 0
        }
    }
}

public enum TurboQuantLayerCacheType: String, Sendable, Equatable, Codable {
    case q8_0
    case turbo2
    case turbo3
    case turbo4

    public init(fixedType: TurboQuantFixedType) {
        switch fixedType {
        case .turbo2:
            self = .turbo2
        case .turbo3:
            self = .turbo3
        case .turbo4:
            self = .turbo4
        }
    }

    public var fixedType: TurboQuantFixedType? {
        switch self {
        case .q8_0:
            return nil
        case .turbo2:
            return .turbo2
        case .turbo3:
            return .turbo3
        case .turbo4:
            return .turbo4
        }
    }

    public var isTurbo: Bool {
        fixedType != nil
    }
}

public enum TurboQuantAdaptiveMode: Int, Sendable, Equatable, Codable {
    case uniform = 0
    case firstLast4Q8 = 1
    case last8Q8 = 2
    case boundaryTurbo4 = 5
    case last8Turbo4 = 6
    case boundaryQ8 = 7
}

public enum TurboQuantValuePolicy: String, Sendable, Equatable, Codable {
    case uniform
    case boundaryTurbo4
    case last8Turbo4
}

public enum TurboQuantKeyPolicy: String, Sendable, Equatable, Codable {
    case uniform
    case boundaryPromoted
    case last8Promoted
}

public struct TurboQuantInnerQConfiguration: Sendable, Equatable {
    public let enabled: Bool
    public let calibrationSampleCount: Int
    public let strength: Float

    public init(enabled: Bool, calibrationSampleCount: Int = 0, strength: Float) {
        self.enabled = enabled
        self.calibrationSampleCount = calibrationSampleCount
        self.strength = strength
    }

    public static let disabled = TurboQuantInnerQConfiguration(enabled: false, calibrationSampleCount: 0, strength: 0)
}

public struct TurboQuantLayout: Sendable, Equatable {
    public static let supportedDimension = 128
    public static let residualWordsPerRow = 4
    public static let outlierMaskWordsPerRow = 4
    public static let metadataScalarsPerRow = 2

    public let preset: TurboQuantPreset
    public let dimension: Int
    public let paddedDimension: Int
    public let codeBitsPerRow: Int
    public let codeWordsPerRow: Int
    public let runtimeCodeBitsPerRow: Int
    public let runtimeCodeWordsPerRow: Int
    public let baseBitsPerRow: Int
    public let baseWordsPerRow: Int
    public let sidebandBitsPerRow: Int
    public let sidebandWordsPerRow: Int
    public let runtimeBaseBitsPerRow: Int
    public let runtimeBaseWordsPerRow: Int
    public let runtimeSidebandBitsPerRow: Int
    public let runtimeSidebandWordsPerRow: Int

    public init(preset: TurboQuantPreset, dimension: Int = supportedDimension) throws {
        guard (1...Self.supportedDimension).contains(dimension) else {
            throw TurboQuantError.unsupportedDimension(dimension)
        }

        let descriptor = preset.descriptor
        let codeBitsPerRow =
            descriptor.regularChannelCount * descriptor.regularBits
            + descriptor.highPrecisionChannelCount * descriptor.highPrecisionBits
        let baseBitsPerRow = dimension * descriptor.regularBits
        let sidebandBitsPerRow = descriptor.highPrecisionChannelCount * (descriptor.highPrecisionBits - descriptor.regularBits)
        let runtimeBaseBitsPerRow = dimension * descriptor.regularBits
        let runtimeSidebandBitsPerRow: Int
        runtimeSidebandBitsPerRow = sidebandBitsPerRow
        let runtimeCodeBitsPerRow = runtimeBaseBitsPerRow + runtimeSidebandBitsPerRow

        self.preset = preset
        self.dimension = dimension
        self.paddedDimension = Self.supportedDimension
        self.codeBitsPerRow = codeBitsPerRow
        self.codeWordsPerRow = (codeBitsPerRow + 31) / 32
        self.runtimeCodeBitsPerRow = runtimeCodeBitsPerRow
        self.runtimeCodeWordsPerRow = (runtimeCodeBitsPerRow + 31) / 32
        self.baseBitsPerRow = baseBitsPerRow
        self.baseWordsPerRow = (baseBitsPerRow + 31) / 32
        self.sidebandBitsPerRow = sidebandBitsPerRow
        self.sidebandWordsPerRow = (sidebandBitsPerRow + 31) / 32
        self.runtimeBaseBitsPerRow = runtimeBaseBitsPerRow
        self.runtimeBaseWordsPerRow = (runtimeBaseBitsPerRow + 31) / 32
        self.runtimeSidebandBitsPerRow = runtimeSidebandBitsPerRow
        self.runtimeSidebandWordsPerRow = (runtimeSidebandBitsPerRow + 31) / 32
    }

    public var bytesPerRow: Int {
        (runtimeCodeWordsPerRow + Self.residualWordsPerRow + Self.outlierMaskWordsPerRow)
            * MemoryLayout<UInt32>.stride
            + Self.metadataScalarsPerRow * MemoryLayout<Float>.stride
    }
}

/// Production-track TurboQuant contract.
///
/// `turboquantV2` intentionally uses explicit per-component row formats instead
/// of the old runtime-balanced/aggressive split. The on-device row layout for
/// both keys and values is:
///
/// 1. `runtimeCodeWordsPerRow` packed 32-bit words for primary/base + sideband codes
/// 2. `residualWordsPerRow` packed residual sign words
/// 3. `outlierMaskWordsPerRow` packed outlier mask words
/// 4. `metadataScalarsPerRow` floats: `rowNorm`, `residualNorm`
///
/// This keeps the storage contract stable while letting keys and values choose
/// different presets while we validate the runtime path against `q8_0`.
public enum TurboQuantV2Contract {
    public static var keyType: TurboQuantFixedType {
        fixedTypeOverride(
            envKey: TurboQuantEnvOverrides.keyType,
            defaultType: .turbo3
        )
    }

    public static var valueType: TurboQuantFixedType {
        fixedTypeOverride(
            envKey: TurboQuantEnvOverrides.valueType,
            defaultType: .turbo3
        )
    }

    public static var adaptiveMode: TurboQuantAdaptiveMode {
        adaptiveModeOverride(
            defaultMode: inferredAdaptiveMode(baseValueType: valueType)
        )
    }

    public static var valuePolicy: TurboQuantValuePolicy {
        valuePolicyOverride(
            envKey: TurboQuantEnvOverrides.valuePolicy,
            defaultPolicy: .uniform
        )
    }

    public static var keyPolicy: TurboQuantKeyPolicy {
        keyPolicyOverride(
            envKey: TurboQuantEnvOverrides.keyPolicy,
            defaultPolicy: .uniform
        )
    }

    public static let supportedDimension = TurboQuantLayout.supportedDimension
    public static let residualWordsPerRow = TurboQuantLayout.residualWordsPerRow
    public static let outlierMaskWordsPerRow = TurboQuantLayout.outlierMaskWordsPerRow
    public static let metadataScalarsPerRow = TurboQuantLayout.metadataScalarsPerRow
    public static var keyOutlierSelection: TurboQuantOutlierSelection {
        outlierSelectionOverride(
            envKey: TurboQuantEnvOverrides.keyOutlierSelection,
            defaultSelection: .magnitude
        )
    }
    public static var valueOutlierSelection: TurboQuantOutlierSelection {
        outlierSelectionOverride(
            envKey: TurboQuantEnvOverrides.valueOutlierSelection,
            defaultSelection: .quantizationBenefit
        )
    }
    public static var innerQ: TurboQuantInnerQConfiguration {
        let environment = ProcessInfo.processInfo.environment
        let forkSamples = environment[TurboQuantEnvOverrides.forkInnerQSamples].flatMap(Int.init) ?? 0
        let localEnabled = environment[TurboQuantEnvOverrides.innerQEnabled] == "1"
        let localSamples = environment[TurboQuantEnvOverrides.innerQSamples].flatMap(Int.init) ?? (localEnabled ? 128 : 0)
        let sampleCount = max(forkSamples, localSamples)
        guard sampleCount > 0 else { return .disabled }
        let rawStrength = environment[TurboQuantEnvOverrides.forkInnerQStrength]
            ?? environment[TurboQuantEnvOverrides.innerQStrength]
        let parsedStrength = rawStrength.flatMap(Float.init) ?? 0.5
        let strength = min(max(parsedStrength, 0), 1)
        return TurboQuantInnerQConfiguration(
            enabled: true,
            calibrationSampleCount: sampleCount,
            strength: strength
        )
    }

    public static var keyPreset: TurboQuantPreset {
        presetOverride(
            envKey: TurboQuantEnvOverrides.keyPreset,
            defaultPreset: keyType.preset
        )
    }

    public static var valuePreset: TurboQuantPreset {
        presetOverride(
            envKey: TurboQuantEnvOverrides.valuePreset,
            defaultPreset: valueType.preset
        )
    }

    public static var promotedKeyPreset: TurboQuantPreset {
        presetOverride(
            envKey: TurboQuantEnvOverrides.promotedKeyPreset,
            defaultPreset: .turbo4
        )
    }

    public static func keyCacheType(forLayer layerIndex: Int, layerCount: Int) -> TurboQuantLayerCacheType {
        let baseType = TurboQuantLayerCacheType(fixedType: keyType)
        switch adaptiveMode {
        case .uniform, .boundaryTurbo4, .last8Turbo4, .boundaryQ8:
            return baseType
        case .firstLast4Q8:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return (layerIndex < 4 || layerIndex >= layerCount - 4) ? .q8_0 : baseType
        case .last8Q8:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return layerIndex >= layerCount - 8 ? .q8_0 : baseType
        }
    }

    public static func valueCacheType(forLayer layerIndex: Int, layerCount: Int) -> TurboQuantLayerCacheType {
        let baseType = TurboQuantLayerCacheType(fixedType: valueType)
        switch adaptiveMode {
        case .uniform:
            return baseType
        case .firstLast4Q8:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return (layerIndex < 4 || layerIndex >= layerCount - 4) ? .q8_0 : baseType
        case .last8Q8:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return layerIndex >= layerCount - 8 ? .q8_0 : baseType
        case .boundaryTurbo4:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return (layerIndex < 2 || layerIndex >= layerCount - 2) ? .turbo4 : .turbo2
        case .last8Turbo4:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return layerIndex >= layerCount - 8 ? .turbo4 : .turbo2
        case .boundaryQ8:
            guard baseType.isTurbo, layerCount >= 8 else { return baseType }
            return (layerIndex < 2 || layerIndex >= layerCount - 2) ? .q8_0 : .turbo2
        }
    }

    public static func keyPreset(forLayer layerIndex: Int, layerCount: Int) -> TurboQuantPreset? {
        let cacheType = keyCacheType(forLayer: layerIndex, layerCount: layerCount)
        guard let fixedType = cacheType.fixedType else { return nil }
        let basePreset = presetOverride(
            envKey: TurboQuantEnvOverrides.keyPreset,
            defaultPreset: fixedType.preset
        )
        switch keyPolicy {
        case .uniform:
            return basePreset
        case .boundaryPromoted:
            guard layerCount >= 8 else { return basePreset }
            return (layerIndex < 2 || layerIndex >= layerCount - 2) ? promotedKeyPreset : basePreset
        case .last8Promoted:
            guard layerCount >= 8 else { return basePreset }
            return layerIndex >= layerCount - 8 ? promotedKeyPreset : basePreset
        }
    }

    public static func valuePreset(forLayer layerIndex: Int, layerCount: Int) -> TurboQuantPreset? {
        let cacheType = valueCacheType(forLayer: layerIndex, layerCount: layerCount)
        guard let fixedType = cacheType.fixedType else { return nil }
        if adaptiveMode == .boundaryTurbo4 || adaptiveMode == .last8Turbo4 {
            return fixedType.preset
        }
        let basePreset = presetOverride(
            envKey: TurboQuantEnvOverrides.valuePreset,
            defaultPreset: fixedType.preset
        )
        switch valuePolicy {
        case .uniform:
            return basePreset
        case .boundaryTurbo4:
            guard layerCount >= 8 else { return valuePreset }
            return (layerIndex < 2 || layerIndex >= layerCount - 2) ? .turbo4 : basePreset
        case .last8Turbo4:
            guard layerCount >= 8 else { return valuePreset }
            return layerIndex >= layerCount - 8 ? .turbo4 : basePreset
        }
    }

    public static func keyResidualScale(forLayer layerIndex: Int, layerCount: Int) -> Float {
        guard let preset = keyPreset(forLayer: layerIndex, layerCount: layerCount) else { return 0 }
        return residualScale(for: preset, fallbackType: keyType)
    }

    public static func valueResidualScale(forLayer layerIndex: Int, layerCount: Int) -> Float {
        guard let preset = valuePreset(forLayer: layerIndex, layerCount: layerCount) else { return 0 }
        return residualScale(for: preset, fallbackType: valueType)
    }

    public static var keyResidualScale: Float {
        keyResidualScale(forLayer: 0, layerCount: 1)
    }

    public static func makeKeyLayout(
        dimension: Int = supportedDimension,
        layerIndex: Int = 0,
        layerCount: Int = 1
    ) throws -> TurboQuantLayout {
        guard let preset = keyPreset(forLayer: layerIndex, layerCount: layerCount) else {
            throw TurboQuantError.unsupportedBitWidth(0)
        }
        return try TurboQuantLayout(
            preset: preset,
            dimension: dimension
        )
    }

    public static func makeValueLayout(
        dimension: Int = supportedDimension,
        layerIndex: Int = 0,
        layerCount: Int = 1
    ) throws -> TurboQuantLayout {
        guard let preset = valuePreset(forLayer: layerIndex, layerCount: layerCount) else {
            throw TurboQuantError.unsupportedBitWidth(0)
        }
        return try TurboQuantLayout(
            preset: preset,
            dimension: dimension
        )
    }

    private static func presetOverride(
        envKey: String,
        defaultPreset: TurboQuantPreset
    ) -> TurboQuantPreset {
        guard let rawValue = ProcessInfo.processInfo.environment[envKey],
              let preset = TurboQuantPreset(rawValue: rawValue) else {
            return defaultPreset
        }
        return preset
    }

    private static func fixedTypeOverride(
        envKey: String,
        defaultType: TurboQuantFixedType
    ) -> TurboQuantFixedType {
        guard let rawValue = ProcessInfo.processInfo.environment[envKey],
              let type = TurboQuantFixedType(rawValue: rawValue) else {
            return defaultType
        }
        return type
    }

    private static func valuePolicyOverride(
        envKey: String,
        defaultPolicy: TurboQuantValuePolicy
    ) -> TurboQuantValuePolicy {
        guard let rawValue = ProcessInfo.processInfo.environment[envKey],
              let policy = TurboQuantValuePolicy(rawValue: rawValue) else {
            return defaultPolicy
        }
        return policy
    }

    private static func keyPolicyOverride(
        envKey: String,
        defaultPolicy: TurboQuantKeyPolicy
    ) -> TurboQuantKeyPolicy {
        guard let rawValue = ProcessInfo.processInfo.environment[envKey],
              let policy = TurboQuantKeyPolicy(rawValue: rawValue) else {
            return defaultPolicy
        }
        return policy
    }

    private static func outlierSelectionOverride(
        envKey: String,
        defaultSelection: TurboQuantOutlierSelection
    ) -> TurboQuantOutlierSelection {
        guard let rawValue = ProcessInfo.processInfo.environment[envKey],
              let selection = TurboQuantOutlierSelection(rawValue: rawValue) else {
            return defaultSelection
        }
        return selection
    }

    private static func adaptiveModeOverride(defaultMode: TurboQuantAdaptiveMode) -> TurboQuantAdaptiveMode {
        let environment = ProcessInfo.processInfo.environment
        if let rawValue = environment[TurboQuantEnvOverrides.forkAdaptiveMode] ?? environment[TurboQuantEnvOverrides.adaptiveMode],
           let parsed = Int(rawValue),
           let mode = TurboQuantAdaptiveMode(rawValue: parsed) {
            return mode
        }
        return defaultMode
    }

    private static func inferredAdaptiveMode(baseValueType: TurboQuantFixedType) -> TurboQuantAdaptiveMode {
        baseValueType == .turbo2 ? .boundaryQ8 : .uniform
    }

    private static func residualScale(
        for preset: TurboQuantPreset,
        fallbackType: TurboQuantFixedType
    ) -> Float {
        switch preset {
        case .turbo2:
            return TurboQuantFixedType.turbo2.residualScale
        case .turbo3:
            return TurboQuantFixedType.turbo3.residualScale
        case .turbo4:
            return TurboQuantFixedType.turbo4.residualScale
        case .balanced, .balanced64, .balanced96, .fiveBit, .sixBit, .sevenBit, .aggressive, .aggressive64:
            return 0
        }
    }
}

public struct TurboQuantCodebook: Sendable, Equatable {
    public let bits: Int
    public let centroids: [Float]
    public let thresholds: [Float]

    public func code(for value: Float) -> UInt8 {
        guard let index = thresholds.firstIndex(where: { value < $0 }) else {
            return UInt8(centroids.count - 1)
        }
        return UInt8(index)
    }

    public func centroid(for code: UInt8) -> Float {
        centroids[Int(code)]
    }
}

public enum TurboQuantCodebooks {
    public static let twoBit = TurboQuantCodebook(
        bits: 2,
        centroids: [-0.133462, -0.039994, 0.039994, 0.133462],
        thresholds: [-0.086728, 0, 0.086728]
    )

    public static let threeBit = TurboQuantCodebook(
        bits: 3,
        centroids: [-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685],
        thresholds: [-0.1542585, -0.0917745, -0.0435885, 0, 0.0435885, 0.0917745, 0.1542585]
    )

    public static let fourBit = TurboQuantCodebook(
        bits: 4,
        centroids: [
            -0.173926, -0.117195, -0.089527, -0.068756,
            -0.051262, -0.035597, -0.020989, -0.006938,
            0.006938, 0.020989, 0.035597, 0.051262,
            0.068756, 0.089527, 0.117195, 0.173926
        ],
        thresholds: [
            -0.145560, -0.103361, -0.079142, -0.060009,
            -0.043430, -0.028293, -0.013963, 0,
            0.013963, 0.028293, 0.043430, 0.060009,
            0.079142, 0.103361, 0.145560
        ]
    )

    public static let fiveBit = TurboQuantCodebook(
        bits: 5,
        centroids: [
            -3.183201, -2.6029081, -2.222862, -1.9298121, -1.6865128, -1.4757856, -1.2881749, -1.1178436,
            -0.9608707, -0.8144341, -0.6763788, -0.54496944, -0.41874045, -0.29640123, -0.17677347, -0.058747794,
            0.058747794, 0.17677347, 0.29640123, 0.41874045, 0.54496944, 0.6763788, 0.8144341, 0.9608707,
            1.1178436, 1.2881749, 1.4757856, 1.6865128, 1.9298121, 2.222862, 2.6029081, 3.183201
        ],
        thresholds: [
            -2.8930547, -2.4128852, -2.076337, -1.8081625, -1.5811492, -1.3819802, -1.2030092, -1.0393572,
            -0.8876524, -0.74540645, -0.6106741, -0.48185492, -0.35757083, -0.23658736, -0.117760636, 0,
            0.117760636, 0.23658736, 0.35757083, 0.48185492, 0.6106741, 0.74540645, 0.8876524, 1.0393572,
            1.2030092, 1.3819802, 1.5811492, 1.8081625, 2.076337, 2.4128852, 2.8930547
        ]
    )

    public static let sixBit = TurboQuantCodebook(
        bits: 6,
        centroids: [
            -3.017255738, -2.423820320, -2.058930059, -1.797707297, -1.601922457, -1.448291610, -1.326356628, -1.225995089,
            -1.139137223, -1.062727519, -0.993135300, -0.928426027, -0.868524053, -0.811485776, -0.757007509, -0.704944748,
            -0.654529973, -0.606343034, -0.559378028, -0.513393457, -0.468229381, -0.424872197, -0.382620330, -0.340845414,
            -0.299256835, -0.258542997, -0.217923724, -0.178098846, -0.138179662, -0.098636745, -0.058835570, -0.018710570,
            0.021130944, 0.060815313, 0.100191218, 0.139750780, 0.179826144, 0.219826894, 0.260545027, 0.301575157,
            0.342993060, 0.384991868, 0.427889936, 0.471579352, 0.516263475, 0.561314135, 0.607624272, 0.655973460,
            0.705897157, 0.758131069, 0.812601781, 0.869977353, 0.930940401, 0.995699969, 1.065519595, 1.141443664,
            1.227592854, 1.328595886, 1.450287956, 1.602228166, 1.797860727, 2.058171284, 2.423209814, 3.012639275
        ],
        thresholds: [
            -2.720538029, -2.241375189, -1.928318678, -1.699814877, -1.525107034, -1.387324119, -1.276175858, -1.182566156,
            -1.100932371, -1.027931409, -0.960780663, -0.898475040, -0.840004914, -0.784246643, -0.730976129, -0.679737361,
            -0.630436503, -0.582860531, -0.536385743, -0.490811419, -0.446550789, -0.403746264, -0.361732872, -0.320051124,
            -0.278899916, -0.238233360, -0.198011285, -0.158139254, -0.118408203, -0.078736157, -0.038773070, 0.001210187,
            0.040973129, 0.080503265, 0.119970999, 0.159788462, 0.199826519, 0.240185961, 0.281060092, 0.322284109,
            0.363992464, 0.406440902, 0.449734644, 0.493921414, 0.538788805, 0.584469203, 0.631798866, 0.680935309,
            0.732014113, 0.785366425, 0.841289567, 0.900458877, 0.963320185, 1.030609782, 1.103481629, 1.184518259,
            1.278094370, 1.389441921, 1.526258061, 1.700044446, 1.928016005, 2.240690549, 2.717924544
        ]
    )

    public static let sevenBit = TurboQuantCodebook(
        bits: 7,
        centroids: [
            -3.161821656, -2.593655236, -2.260366751, -2.036143901, -1.877254125, -1.756981236, -1.662470419, -1.584808772,
            -1.517156127, -1.456758653, -1.402001919, -1.350976044, -1.303012122, -1.257290771, -1.214683970, -1.174284348,
            -1.135820017, -1.099401736, -1.064814314, -1.031291885, -0.998516547, -0.966935452, -0.935911034, -0.905753606,
            -0.876445620, -0.847857376, -0.820004703, -0.792610167, -0.765753537, -0.739575621, -0.713990833, -0.688968586,
            -0.663956599, -0.639361813, -0.615405244, -0.591867693, -0.568442898, -0.545359109, -0.522723513, -0.500390633,
            -0.478618508, -0.457086748, -0.435321291, -0.413745130, -0.392190012, -0.371026980, -0.350001767, -0.328982207,
            -0.308341784, -0.287940557, -0.267649618, -0.247588727, -0.227361475, -0.207182363, -0.187185623, -0.167462390,
            -0.147365963, -0.127493547, -0.107718842, -0.088070616, -0.068404744, -0.048587333, -0.028918753, -0.009434275,
            0.010149665, 0.029802156, 0.049171144, 0.068828458, 0.088722840, 0.108229260, 0.128145472, 0.148108296,
            0.167952409, 0.187973966, 0.208235593, 0.228389448, 0.248464332, 0.268681237, 0.288765152, 0.309276338,
            0.329687599, 0.350446090, 0.371599662, 0.392955936, 0.414318992, 0.435806255, 0.457389054, 0.478953617,
            0.500847597, 0.523094315, 0.545609565, 0.568220557, 0.591509701, 0.615326460, 0.639434160, 0.663947113,
            0.688244905, 0.712990034, 0.738163980, 0.763943700, 0.790592008, 0.818028338, 0.846147718, 0.874858881,
            0.904436875, 0.934984934, 0.965949978, 0.997473628, 1.029677998, 1.063044768, 1.097693392, 1.134172843,
            1.172229797, 1.212077571, 1.254602382, 1.299288409, 1.346488593, 1.397246923, 1.452061588, 1.512432831,
            1.580646704, 1.659743219, 1.754504101, 1.873331421, 2.031510542, 2.255090788, 2.588195086, 3.143419937
        ],
        thresholds: [
            -2.877738446, -2.427010993, -2.148255326, -1.956699013, -1.817117680, -1.709725828, -1.623639596, -1.550982450,
            -1.486957390, -1.429380286, -1.376488981, -1.326994083, -1.280151447, -1.235987371, -1.194484159, -1.155052183,
            -1.117610877, -1.082108025, -1.048053099, -1.014904216, -0.982725999, -0.951423243, -0.920832320, -0.891099613,
            -0.862151498, -0.833931039, -0.806307435, -0.779181852, -0.752664579, -0.726783227, -0.701479710, -0.676462593,
            -0.651659206, -0.627383529, -0.603636469, -0.580155296, -0.556901003, -0.534041311, -0.511557073, -0.489504570,
            -0.467852628, -0.446204019, -0.424533210, -0.402967571, -0.381608496, -0.360514373, -0.339491987, -0.318661995,
            -0.298141171, -0.277795088, -0.257619173, -0.237475101, -0.217271919, -0.197183993, -0.177324006, -0.157414176,
            -0.137429755, -0.117606194, -0.097894729, -0.078237680, -0.058496038, -0.038753043, -0.019176514, 0.000357695,
            0.019975910, 0.039486650, 0.058999801, 0.078775649, 0.098476050, 0.118187366, 0.138126884, 0.158030352,
            0.177963187, 0.198104780, 0.218312521, 0.238426890, 0.258572784, 0.278723194, 0.299020745, 0.319481968,
            0.340066844, 0.361022876, 0.382277799, 0.403637464, 0.425062623, 0.446597654, 0.468171335, 0.489900607,
            0.511970956, 0.534351940, 0.556915061, 0.579865129, 0.603418081, 0.627380310, 0.651690636, 0.676096009,
            0.700617469, 0.725577007, 0.751053840, 0.777267854, 0.804310173, 0.832088028, 0.860503300, 0.889647878,
            0.919710905, 0.950467456, 0.981711803, 1.013575813, 1.046361383, 1.080369080, 1.115933117, 1.153201320,
            1.192153684, 1.233339976, 1.276945396, 1.322888501, 1.371867758, 1.424654256, 1.482247210, 1.546539768,
            1.620194962, 1.707123660, 1.813917761, 1.952420982, 2.143300665, 2.421642937, 2.865807512
        ]
    )

    public static func forBits(_ bits: Int) throws -> TurboQuantCodebook {
        switch bits {
        case 2:
            return twoBit
        case 3:
            return threeBit
        case 4:
            return fourBit
        case 5:
            return fiveBit
        case 6:
            return sixBit
        case 7:
            return sevenBit
        default:
            throw TurboQuantError.unsupportedBitWidth(bits)
        }
    }
}

public struct TurboQuantEncodedRow: Sendable, Equatable {
    public let preset: TurboQuantPreset
    public let dimension: Int
    public let primaryCodes: [UInt32]
    public let residualSigns: [UInt32]
    public let outlierMask: [UInt32]
    public let rowNorm: Float
    public let residualNorm: Float

    public func storageBits() -> Int {
        let words = primaryCodes.count + residualSigns.count + outlierMask.count
        return words * UInt32.bitWidth + (2 * MemoryLayout<Float>.size * 8)
    }
}

public struct TurboQuantRuntimeRow: Sendable, Equatable {
    public let preset: TurboQuantPreset
    public let dimension: Int
    public let primaryCodes: [UInt32]
    public let residualSigns: [UInt32]
    public let outlierMask: [UInt32]
    public let rowNorm: Float
    public let residualNorm: Float
}

public struct TurboQuantScoreTerms: Sendable, Equatable {
    public let mseDot: Float
    public let residualDot: Float
    public let rowNorm: Float
    public let residualNorm: Float
    public let score: Float
}

public enum TurboQuantSeeds {
    public static let keyRotation: UInt64 = 0xC0DEC0DEBADC0FFE
    public static let keyResidual: UInt64 = 0xA02BDBF7BB3C0A11
    public static let valueRotation: UInt64 = 0xD1CEFA57F00DBAAD
    public static let valueResidual: UInt64 = 0x9137C25A0B1ED123
    public static let testRotation: UInt64 = 0x123456789ABCDEF0
    public static let testResidual: UInt64 = 0x0FEDCBA987654321
}

public enum TurboQuantTransform {
    public static let qjlScale = sqrt(Float.pi / 2)
    private static let forwardScale = 1 / sqrt(Float(TurboQuantLayout.supportedDimension))
    private static let signs1: [Float] = [
        -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
        -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
        -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
        -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
    ]
    private static let signs2: [Float] = [
        1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
        1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
        1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
        1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
    ]

    public static func randomizedHadamard(_ vector: [Float], seed: UInt64) throws -> [Float] {
        guard (1...TurboQuantLayout.supportedDimension).contains(vector.count) else {
            throw TurboQuantError.unsupportedDimension(vector.count)
        }

        var padded = vector
        if vector.count < TurboQuantLayout.supportedDimension {
            padded.append(contentsOf: repeatElement(0, count: TurboQuantLayout.supportedDimension - vector.count))
        }
        var transformed = zip(padded, signs1).map(*)
        hadamardInPlace(&transformed)
        for index in transformed.indices {
            transformed[index] *= signs2[index] * forwardScale
        }
        return transformed
    }

    public static func inverseRandomizedHadamard(_ vector: [Float], seed: UInt64) throws -> [Float] {
        guard vector.count == TurboQuantLayout.supportedDimension else {
            throw TurboQuantError.unsupportedDimension(vector.count)
        }

        var transformed = zip(vector, signs2).map(*)
        hadamardInPlace(&transformed)
        return zip(transformed, signs1).map { ($0 * $1) * forwardScale }
    }

    public static func signPattern(count: Int, seed: UInt64) -> [Float] {
        let base = count <= 64 ? Array(signs1.prefix(count)) : Array(signs1.prefix(count))
        return base
    }

    static func hadamardInPlace(_ values: inout [Float]) {
        var butterflyWidth = 1
        while butterflyWidth < values.count {
            let step = butterflyWidth << 1
            for base in Swift.stride(from: 0, to: values.count, by: step) {
                for index in 0..<butterflyWidth {
                    let lhs = values[base + index]
                    let rhs = values[base + index + butterflyWidth]
                    values[base + index] = lhs + rhs
                    values[base + index + butterflyWidth] = lhs - rhs
                }
            }
            butterflyWidth = step
        }
    }
}

public enum TurboQuantReferenceEncoder {
    public static func encode(
        _ vector: [Float],
        preset: TurboQuantPreset,
        outlierSelection: TurboQuantOutlierSelection = .magnitude,
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> TurboQuantEncodedRow {
        guard (1...TurboQuantLayout.supportedDimension).contains(vector.count) else {
            throw TurboQuantError.unsupportedDimension(vector.count)
        }

        let descriptor = preset.descriptor
        let layout = try TurboQuantLayout(preset: preset)
        let rowNorm = sqrt(vector.reduce(Float.zero) { $0 + ($1 * $1) })

        guard rowNorm > 0 else {
            return TurboQuantEncodedRow(
                preset: preset,
                dimension: vector.count,
                primaryCodes: [UInt32](repeating: 0, count: layout.codeWordsPerRow),
                residualSigns: [UInt32](repeating: 0, count: TurboQuantLayout.residualWordsPerRow),
                outlierMask: [UInt32](repeating: 0, count: TurboQuantLayout.outlierMaskWordsPerRow),
                rowNorm: 0,
                residualNorm: 0
            )
        }

        let normalized = vector.map { $0 / rowNorm }
        let rotated = try TurboQuantTransform.randomizedHadamard(normalized, seed: rotationSeed)
        let regularBook = try TurboQuantCodebooks.forBits(descriptor.regularBits)
        let highBook = try TurboQuantCodebooks.forBits(descriptor.highPrecisionBits)
        let outlierMask = topKMask(
            values: rotated,
            count: descriptor.highPrecisionChannelCount,
            regularBook: regularBook,
            highBook: highBook,
            selection: outlierSelection
        )

        var codes = [UInt8](repeating: 0, count: rotated.count)
        var reconstructedRotated = [Float](repeating: 0, count: rotated.count)
        for index in rotated.indices {
            let useHighPrecision = outlierMask[index]
            let book = useHighPrecision ? highBook : regularBook
            let code = book.code(for: rotated[index])
            codes[index] = code
            reconstructedRotated[index] = book.centroid(for: code)
        }

        let usesResidualPath = descriptor.highPrecisionChannelCount > 0 || descriptor.highPrecisionBits != descriptor.regularBits
        if !usesResidualPath {
            let reconstructedNorm = sqrt(reconstructedRotated.reduce(Float.zero) { $0 + ($1 * $1) })
            let correctedNorm = reconstructedNorm > 0 ? (rowNorm / reconstructedNorm) : rowNorm
            return TurboQuantEncodedRow(
                preset: preset,
                dimension: vector.count,
                primaryCodes: try BitPacker.packCodes(
                    codes,
                    outlierMask: outlierMask,
                    regularBits: descriptor.regularBits,
                    highPrecisionBits: descriptor.highPrecisionBits
                ),
                residualSigns: [UInt32](repeating: 0, count: TurboQuantLayout.residualWordsPerRow),
                outlierMask: BitPacker.packBooleans(outlierMask),
                rowNorm: correctedNorm,
                residualNorm: 0
            )
        }

        let mseApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            reconstructedRotated,
            seed: rotationSeed
        )
        let residual = zip(normalized, mseApproximation).map(-)
        let residualNorm = sqrt(residual.reduce(Float.zero) { $0 + ($1 * $1) })
        let projectedResidual = try TurboQuantTransform.randomizedHadamard(residual, seed: residualSeed)
        let residualSigns = BitPacker.packBooleans(projectedResidual.map { $0 >= 0 })

        return TurboQuantEncodedRow(
            preset: preset,
            dimension: vector.count,
            primaryCodes: try BitPacker.packCodes(
                codes,
                outlierMask: outlierMask,
                regularBits: descriptor.regularBits,
                highPrecisionBits: descriptor.highPrecisionBits
            ),
            residualSigns: residualSigns,
            outlierMask: BitPacker.packBooleans(outlierMask),
            rowNorm: rowNorm,
            residualNorm: residualNorm
        )
    }

    public static func encode(
        _ vector: [Float],
        preset: TurboQuantPreset,
        outlierSelection: TurboQuantOutlierSelection = .magnitude,
        seed: UInt64
    ) throws -> TurboQuantEncodedRow {
        try encode(
            vector,
            preset: preset,
            outlierSelection: outlierSelection,
            rotationSeed: seed,
            residualSeed: seed ^ 0x9E3779B97F4A7C15
        )
    }

    public static func approximateDecode(
        _ encoded: TurboQuantEncodedRow,
        residualWeight: Float = 1.0,
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> [Float] {
        guard (1...TurboQuantLayout.supportedDimension).contains(encoded.dimension) else {
            throw TurboQuantError.unsupportedDimension(encoded.dimension)
        }

        let descriptor = encoded.preset.descriptor
        let layout = try TurboQuantLayout(preset: encoded.preset, dimension: encoded.dimension)
        let paddedCount = layout.paddedDimension
        let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: paddedCount)
        let codes = try BitPacker.unpackCodes(
            encoded.primaryCodes,
            count: paddedCount,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits
        )
        let regularBook = try TurboQuantCodebooks.forBits(descriptor.regularBits)
        let highBook = try TurboQuantCodebooks.forBits(descriptor.highPrecisionBits)

        let reconstructedRotated = codes.enumerated().map { index, code -> Float in
            let book = outlierMask[index] ? highBook : regularBook
            return book.centroid(for: code)
        }

        let mseApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            reconstructedRotated,
            seed: rotationSeed
        )

        let residualSigns = BitPacker.unpackBooleans(encoded.residualSigns, count: paddedCount)
        let residualDirection = residualSigns.map { $0 ? 1.0 as Float : -1.0 }
        let qjlApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            residualDirection,
            seed: residualSeed
        ).map {
            $0
                * TurboQuantTransform.qjlScale
                * (1 / Float(TurboQuantLayout.supportedDimension))
                * encoded.residualNorm
                * residualWeight
        }

        return Array(zip(mseApproximation, qjlApproximation).map { ($0 + $1) * encoded.rowNorm }.prefix(encoded.dimension))
    }

    public static func approximateDecode(
        _ encoded: TurboQuantEncodedRow,
        residualWeight: Float = 1.0,
        seed: UInt64
    ) throws -> [Float] {
        try approximateDecode(
            encoded,
            residualWeight: residualWeight,
            rotationSeed: seed,
            residualSeed: seed ^ 0x9E3779B97F4A7C15
        )
    }

    public static func topKMask(
        values: [Float],
        count: Int,
        regularBook: TurboQuantCodebook,
        highBook: TurboQuantCodebook,
        selection: TurboQuantOutlierSelection
    ) -> [Bool] {
        guard count > 0 else { return [Bool](repeating: false, count: values.count) }
        let sorted = values.enumerated()
            .sorted {
                selectionScore(
                    value: $0.element,
                    regularBook: regularBook,
                    highBook: highBook,
                    selection: selection
                ) > selectionScore(
                    value: $1.element,
                    regularBook: regularBook,
                    highBook: highBook,
                    selection: selection
                )
            }
        var mask = [Bool](repeating: false, count: values.count)
        for index in sorted.prefix(min(count, values.count)).map(\.offset) {
            mask[index] = true
        }
        return mask
    }

    private static func selectionScore(
        value: Float,
        regularBook: TurboQuantCodebook,
        highBook: TurboQuantCodebook,
        selection: TurboQuantOutlierSelection
    ) -> Float {
        switch selection {
        case .magnitude:
            return abs(value)
        case .quantizationBenefit:
            return quantizationBenefit(value: value, regularBook: regularBook, highBook: highBook)
        }
    }

    private static func quantizationBenefit(
        value: Float,
        regularBook: TurboQuantCodebook,
        highBook: TurboQuantCodebook
    ) -> Float {
        let regularCentroid = regularBook.centroid(for: regularBook.code(for: value))
        let highCentroid = highBook.centroid(for: highBook.code(for: value))
        let regularError = value - regularCentroid
        let highError = value - highCentroid
        return (regularError * regularError) - (highError * highError)
    }

    public static func makeRuntimeRow(
        from encoded: TurboQuantEncodedRow
    ) throws -> TurboQuantRuntimeRow {
        let layout = try TurboQuantLayout(preset: encoded.preset, dimension: encoded.dimension)
        let descriptor = encoded.preset.descriptor
        let paddedCount = layout.paddedDimension
        let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: paddedCount)
        let logicalCodes = try BitPacker.unpackCodes(
            encoded.primaryCodes,
            count: paddedCount,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits
        )
        let runtimeCodes = try BitPacker.packRuntimeCodes(
            logicalCodes,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits,
            preset: encoded.preset,
            layout: layout
        )
        return TurboQuantRuntimeRow(
            preset: encoded.preset,
            dimension: encoded.dimension,
            primaryCodes: runtimeCodes,
            residualSigns: encoded.residualSigns,
            outlierMask: encoded.outlierMask,
            rowNorm: encoded.rowNorm,
            residualNorm: encoded.residualNorm
        )
    }

    public static func approximateDecode(
        runtimeRow: TurboQuantRuntimeRow,
        residualWeight: Float = 1.0,
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> [Float] {
        let descriptor = runtimeRow.preset.descriptor
        let layout = try TurboQuantLayout(preset: runtimeRow.preset, dimension: runtimeRow.dimension)
        let paddedCount = layout.paddedDimension
        let outlierMask = BitPacker.unpackBooleans(runtimeRow.outlierMask, count: paddedCount)
        let codes = try BitPacker.unpackRuntimeCodes(
            runtimeRow.primaryCodes,
            count: paddedCount,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits,
            preset: runtimeRow.preset,
            layout: layout
        )
        let logicalRow = TurboQuantEncodedRow(
            preset: runtimeRow.preset,
            dimension: runtimeRow.dimension,
            primaryCodes: try BitPacker.packCodes(
                codes,
                outlierMask: outlierMask,
                regularBits: descriptor.regularBits,
                highPrecisionBits: descriptor.highPrecisionBits
            ),
            residualSigns: runtimeRow.residualSigns,
            outlierMask: runtimeRow.outlierMask,
            rowNorm: runtimeRow.rowNorm,
            residualNorm: runtimeRow.residualNorm
        )
        return try approximateDecode(
            logicalRow,
            residualWeight: residualWeight,
            rotationSeed: rotationSeed,
            residualSeed: residualSeed
        )
    }

    public static func approximateScoreTerms(
        query: [Float],
        runtimeRow: TurboQuantRuntimeRow,
        residualWeight: Float = 1.0,
        rotationSeed: UInt64,
        residualSeed: UInt64,
        scale: Float
    ) throws -> TurboQuantScoreTerms {
        guard (1...TurboQuantLayout.supportedDimension).contains(query.count) else {
            throw TurboQuantError.unsupportedDimension(query.count)
        }
        guard (1...TurboQuantLayout.supportedDimension).contains(runtimeRow.dimension) else {
            throw TurboQuantError.unsupportedDimension(runtimeRow.dimension)
        }

        let descriptor = runtimeRow.preset.descriptor
        let layout = try TurboQuantLayout(preset: runtimeRow.preset, dimension: runtimeRow.dimension)
        let paddedCount = layout.paddedDimension
        let outlierMask = BitPacker.unpackBooleans(runtimeRow.outlierMask, count: paddedCount)
        let residualSigns = BitPacker.unpackBooleans(runtimeRow.residualSigns, count: paddedCount)
        let codes = try BitPacker.unpackRuntimeCodes(
            runtimeRow.primaryCodes,
            count: paddedCount,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits,
            preset: runtimeRow.preset,
            layout: layout
        )
        let qRot = try TurboQuantTransform.randomizedHadamard(query, seed: rotationSeed)
        let qResidual = try TurboQuantTransform.randomizedHadamard(query, seed: residualSeed)

        var mseDot: Float = 0
        var residualDot: Float = 0
        for dim in 0..<paddedCount {
            let bits = outlierMask[dim] ? descriptor.highPrecisionBits : descriptor.regularBits
            let codebook = try TurboQuantCodebooks.forBits(bits)
            mseDot += qRot[dim] * codebook.centroid(for: codes[dim])
            residualDot += qResidual[dim] * (residualSigns[dim] ? 1 : -1)
        }

        let score = runtimeRow.rowNorm
            * (mseDot + (TurboQuantTransform.qjlScale * residualWeight * runtimeRow.residualNorm * residualDot))
            * scale
        return TurboQuantScoreTerms(
            mseDot: mseDot,
            residualDot: residualDot,
            rowNorm: runtimeRow.rowNorm,
            residualNorm: runtimeRow.residualNorm,
            score: score
        )
    }

    public static func approximateScore(
        query: [Float],
        runtimeRow: TurboQuantRuntimeRow,
        residualWeight: Float = 1.0,
        rotationSeed: UInt64,
        residualSeed: UInt64,
        scale: Float
    ) throws -> Float {
        try approximateScoreTerms(
            query: query,
            runtimeRow: runtimeRow,
            residualWeight: residualWeight,
            rotationSeed: rotationSeed,
            residualSeed: residualSeed,
            scale: scale
        ).score
    }
}

private struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

enum BitPacker {
    static func packBooleans(_ values: [Bool]) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: (values.count + 31) / 32)
        for (index, value) in values.enumerated() where value {
            let wordIndex = index / 32
            let bitIndex = index % 32
            words[wordIndex] |= (UInt32(1) << UInt32(bitIndex))
        }
        return words
    }

    static func unpackBooleans(_ words: [UInt32], count: Int) -> [Bool] {
        (0..<count).map { index in
            let wordIndex = index / 32
            let bitIndex = index % 32
            guard wordIndex < words.count else { return false }
            return ((words[wordIndex] >> UInt32(bitIndex)) & 1) == 1
        }
    }

    static func packCodes(
        _ codes: [UInt8],
        outlierMask: [Bool],
        regularBits: Int,
        highPrecisionBits: Int
    ) throws -> [UInt32] {
        let baseBits = codes.count * regularBits
        let sidebandBits = outlierMask.reduce(0) { $0 + ($1 ? (highPrecisionBits - regularBits) : 0) }
        var words = [UInt32](repeating: 0, count: ((baseBits + sidebandBits) + 31) / 32)

        let sidebandWidth = highPrecisionBits - regularBits
        var sidebandOffset = baseBits
        for index in codes.indices {
            let code = UInt32(codes[index])
            let baseMask = (UInt32(1) << UInt32(regularBits)) - 1
            try insert(code: code & baseMask, width: regularBits, into: &words, bitOffset: index * regularBits)
            if outlierMask[index], sidebandWidth > 0 {
                try insert(
                    code: code >> UInt32(regularBits),
                    width: sidebandWidth,
                    into: &words,
                    bitOffset: sidebandOffset
                )
                sidebandOffset += sidebandWidth
            }
        }
        return words
    }

    static func unpackCodes(
        _ words: [UInt32],
        count: Int,
        outlierMask: [Bool],
        regularBits: Int,
        highPrecisionBits: Int
    ) throws -> [UInt8] {
        var codes = [UInt8]()
        codes.reserveCapacity(count)
        let baseBits = count * regularBits
        let sidebandWidth = highPrecisionBits - regularBits
        var sidebandOffset = baseBits
        for index in 0..<count {
            var raw = try extract(from: words, width: regularBits, bitOffset: index * regularBits)
            if outlierMask[index], sidebandWidth > 0 {
                let extra = try extract(from: words, width: sidebandWidth, bitOffset: sidebandOffset)
                raw |= extra << UInt32(regularBits)
                sidebandOffset += sidebandWidth
            }
            codes.append(UInt8(raw))
        }
        return codes
    }

    static func packRuntimeCodes(
        _ codes: [UInt8],
        outlierMask: [Bool],
        regularBits: Int,
        highPrecisionBits: Int,
        preset: TurboQuantPreset,
        layout: TurboQuantLayout
    ) throws -> [UInt32] {
        switch preset {
        case .turbo2, .turbo3, .turbo4, .balanced, .balanced64, .balanced96, .fiveBit, .sixBit, .sevenBit, .aggressive, .aggressive64:
            return try packCodes(
                codes,
                outlierMask: outlierMask,
                regularBits: regularBits,
                highPrecisionBits: highPrecisionBits
            )
        }
    }

    static func unpackRuntimeCodes(
        _ words: [UInt32],
        count: Int,
        outlierMask: [Bool],
        regularBits: Int,
        highPrecisionBits: Int,
        preset: TurboQuantPreset,
        layout: TurboQuantLayout
    ) throws -> [UInt8] {
        switch preset {
        case .turbo2, .turbo3, .turbo4, .balanced, .balanced64, .balanced96, .fiveBit, .sixBit, .sevenBit, .aggressive, .aggressive64:
            return try unpackCodes(
                words,
                count: count,
                outlierMask: outlierMask,
                regularBits: regularBits,
                highPrecisionBits: highPrecisionBits
            )
        }
    }

    private static func insert(
        code: UInt32,
        width: Int,
        into words: inout [UInt32],
        bitOffset: Int
    ) throws {
        guard width > 0 && width <= 8 else {
            throw TurboQuantError.unsupportedBitWidth(width)
        }

        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        let mask = (UInt32(1) << UInt32(width)) - 1
        words[wordIndex] |= (code & mask) << UInt32(shift)

        let spill = shift + width - 32
        if spill > 0 {
            words[wordIndex + 1] |= (code & mask) >> UInt32(width - spill)
        }
    }

    private static func extract(
        from words: [UInt32],
        width: Int,
        bitOffset: Int
    ) throws -> UInt32 {
        guard width > 0 && width <= 8 else {
            throw TurboQuantError.unsupportedBitWidth(width)
        }

        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        let mask = (UInt32(1) << UInt32(width)) - 1
        var value = (words[wordIndex] >> UInt32(shift)) & mask
        let spill = shift + width - 32
        if spill > 0, wordIndex + 1 < words.count {
            let highBits = words[wordIndex + 1] & ((UInt32(1) << UInt32(spill)) - 1)
            value |= highBits << UInt32(width - spill)
        }
        return value
    }
}
