import Foundation

public enum TurboQuantError: Error, Sendable, Equatable {
    case unsupportedDimension(Int)
    case unsupportedBitWidth(Int)
    case invalidCodeCount(expected: Int, actual: Int)
}

public enum TurboQuantPreset: String, Sendable, Equatable, Codable {
    case balanced
    case aggressive

    public var descriptor: TurboQuantPresetDescriptor {
        switch self {
        case .balanced:
            // Engineering assumption until the authors publish the exact 3.5-bit split.
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 3,
                highPrecisionBits: 5,
                highPrecisionChannelCount: 32,
                effectiveBits: 3.5
            )
        case .aggressive:
            // Matches the paper's explicit 2.5-bit example for d = 128.
            return TurboQuantPresetDescriptor(
                preset: self,
                regularBits: 2,
                highPrecisionBits: 3,
                highPrecisionChannelCount: 32,
                effectiveBits: 2.5
            )
        }
    }
}

public struct TurboQuantPresetDescriptor: Sendable, Equatable {
    public let preset: TurboQuantPreset
    public let regularBits: Int
    public let highPrecisionBits: Int
    public let highPrecisionChannelCount: Int
    public let effectiveBits: Double

    public var regularChannelCount: Int { 128 - highPrecisionChannelCount }
}

public struct TurboQuantLayout: Sendable, Equatable {
    public static let supportedDimension = 128
    public static let residualWordsPerRow = 4
    public static let outlierMaskWordsPerRow = 4
    public static let metadataScalarsPerRow = 2

    public let preset: TurboQuantPreset
    public let dimension: Int
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
        guard dimension == Self.supportedDimension else {
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
        centroids: [-1.5104176, -0.45278004, 0.45278004, 1.5104176],
        thresholds: [-0.9815988, 0, 0.9815988]
    )

    public static let threeBit = TurboQuantCodebook(
        bits: 3,
        centroids: [-2.1519456, -1.3439093, -0.7560053, -0.24509418, 0.24509418, 0.7560053, 1.3439093, 2.1519456],
        thresholds: [-1.7479275, -1.0499573, -0.50054973, 0, 0.50054973, 1.0499573, 1.7479275]
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

    public static func forBits(_ bits: Int) throws -> TurboQuantCodebook {
        switch bits {
        case 2:
            return twoBit
        case 3:
            return threeBit
        case 5:
            return fiveBit
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

public enum TurboQuantSeeds {
    public static let keyRotation: UInt64 = 0xC0DEC0DEBADC0FFE
    public static let keyResidual: UInt64 = 0xA02BDBF7BB3C0A11
    public static let valueRotation: UInt64 = 0xD1CEFA57F00DBAAD
    public static let valueResidual: UInt64 = 0x9137C25A0B1ED123
    public static let testRotation: UInt64 = 0x123456789ABCDEF0
    public static let testResidual: UInt64 = 0x0FEDCBA987654321
}

public enum TurboQuantTransform {
    public static let inverseScale = 1 / Float(TurboQuantLayout.supportedDimension)
    public static let qjlScale = sqrt(Float.pi / 2)

    public static func randomizedHadamard(_ vector: [Float], seed: UInt64) throws -> [Float] {
        guard vector.count == TurboQuantLayout.supportedDimension else {
            throw TurboQuantError.unsupportedDimension(vector.count)
        }

        let signs = signPattern(count: vector.count, seed: seed)
        var transformed = zip(vector, signs).map(*)
        hadamardInPlace(&transformed)
        return transformed
    }

    public static func inverseRandomizedHadamard(_ vector: [Float], seed: UInt64) throws -> [Float] {
        guard vector.count == TurboQuantLayout.supportedDimension else {
            throw TurboQuantError.unsupportedDimension(vector.count)
        }

        var transformed = vector
        hadamardInPlace(&transformed)
        let signs = signPattern(count: vector.count, seed: seed)
        return zip(transformed, signs).map { ($0 * $1) * inverseScale }
    }

    public static func signPattern(count: Int, seed: UInt64) -> [Float] {
        var generator = SplitMix64(state: seed == 0 ? 0x9E3779B97F4A7C15 : seed)
        return (0..<count).map { _ in
            (generator.next() & 1) == 0 ? -1 : 1
        }
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
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> TurboQuantEncodedRow {
        guard vector.count == TurboQuantLayout.supportedDimension else {
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
        let outlierMask = topKMask(values: rotated, count: descriptor.highPrecisionChannelCount)
        let regularBook = try TurboQuantCodebooks.forBits(descriptor.regularBits)
        let highBook = try TurboQuantCodebooks.forBits(descriptor.highPrecisionBits)

        var codes = [UInt8](repeating: 0, count: rotated.count)
        var reconstructedRotated = [Float](repeating: 0, count: rotated.count)
        for index in rotated.indices {
            let useHighPrecision = outlierMask[index]
            let book = useHighPrecision ? highBook : regularBook
            let code = book.code(for: rotated[index])
            codes[index] = code
            reconstructedRotated[index] = book.centroid(for: code)
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
        seed: UInt64
    ) throws -> TurboQuantEncodedRow {
        try encode(
            vector,
            preset: preset,
            rotationSeed: seed,
            residualSeed: seed ^ 0x9E3779B97F4A7C15
        )
    }

    public static func approximateDecode(
        _ encoded: TurboQuantEncodedRow,
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> [Float] {
        guard encoded.dimension == TurboQuantLayout.supportedDimension else {
            throw TurboQuantError.unsupportedDimension(encoded.dimension)
        }

        let descriptor = encoded.preset.descriptor
        let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: encoded.dimension)
        let codes = try BitPacker.unpackCodes(
            encoded.primaryCodes,
            count: encoded.dimension,
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

        let residualSigns = BitPacker.unpackBooleans(encoded.residualSigns, count: encoded.dimension)
        let residualDirection = residualSigns.map { $0 ? 1.0 as Float : -1.0 }
        let qjlApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            residualDirection,
            seed: residualSeed
        ).map { $0 * TurboQuantTransform.qjlScale * encoded.residualNorm }

        return zip(mseApproximation, qjlApproximation).map { ($0 + $1) * encoded.rowNorm }
    }

    public static func approximateDecode(
        _ encoded: TurboQuantEncodedRow,
        seed: UInt64
    ) throws -> [Float] {
        try approximateDecode(
            encoded,
            rotationSeed: seed,
            residualSeed: seed ^ 0x9E3779B97F4A7C15
        )
    }

    public static func topKMask(values: [Float], count: Int) -> [Bool] {
        guard count > 0 else { return [Bool](repeating: false, count: values.count) }
        let sorted = values.enumerated()
            .sorted { abs($0.element) > abs($1.element) }
        var mask = [Bool](repeating: false, count: values.count)
        for index in sorted.prefix(min(count, values.count)).map(\.offset) {
            mask[index] = true
        }
        return mask
    }

    public static func makeRuntimeRow(
        from encoded: TurboQuantEncodedRow
    ) throws -> TurboQuantRuntimeRow {
        let layout = try TurboQuantLayout(preset: encoded.preset, dimension: encoded.dimension)
        let descriptor = encoded.preset.descriptor
        let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: encoded.dimension)
        let logicalCodes = try BitPacker.unpackCodes(
            encoded.primaryCodes,
            count: encoded.dimension,
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
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> [Float] {
        let descriptor = runtimeRow.preset.descriptor
        let layout = try TurboQuantLayout(preset: runtimeRow.preset, dimension: runtimeRow.dimension)
        let outlierMask = BitPacker.unpackBooleans(runtimeRow.outlierMask, count: runtimeRow.dimension)
        let codes = try BitPacker.unpackRuntimeCodes(
            runtimeRow.primaryCodes,
            count: runtimeRow.dimension,
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
            rotationSeed: rotationSeed,
            residualSeed: residualSeed
        )
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
        case .balanced:
            return try packCodes(
                codes,
                outlierMask: outlierMask,
                regularBits: regularBits,
                highPrecisionBits: highPrecisionBits
            )
        case .aggressive:
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
        case .balanced:
            return try unpackCodes(
                words,
                count: count,
                outlierMask: outlierMask,
                regularBits: regularBits,
                highPrecisionBits: highPrecisionBits
            )
        case .aggressive:
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
