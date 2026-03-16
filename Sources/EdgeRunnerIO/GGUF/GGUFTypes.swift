import Foundation

public let ggufMagic: UInt32 = 0x46554747
public let ggufSupportedVersions: ClosedRange<UInt32> = 2...3

public enum GGUFMetadataValueType: UInt32, Sendable {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12
}

public enum GGUFTensorType: UInt32, Sendable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case q8_K = 15
    case i8 = 16
    case i16 = 17
    case i32 = 18
    case i64 = 19
    case f64 = 20
    case iq2_xxs = 21
    case iq2_xs = 22
    case iq3_xxs = 23
    case iq1_s = 24
    case iq4_nl = 25
    case iq3_s = 26
    case iq2_s = 27
    case iq4_xs = 28

    public var tensorDataType: TensorDataType? {
        switch self {
        case .f32: .float32
        case .f16: .float16
        case .q4_0: .q4_0
        case .q4_1: .q4_1
        case .q5_0: .q5_0
        case .q5_1: .q5_1
        case .q8_0: .q8_0
        case .q8_1: .q8_1
        case .q2_K: .q2_K
        case .q3_K: .q3_K
        case .q4_K: .q4_K
        case .q5_K: .q5_K
        case .q6_K: .q6_K
        case .q8_K: .q8_K
        case .i8: .i8
        case .i16: .i16
        case .i32: .i32
        case .i64: .i64
        case .f64: .f64
        case .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs:
            nil
        }
    }
}

public struct GGUFHeader: Sendable, Equatable {
    public let version: UInt32
    public let tensorCount: UInt64
    public let metadataKVCount: UInt64

    public static func parse(from data: Data) throws -> GGUFHeader {
        let reader = GGUFReader(data: data)
        return try reader.readHeader()
    }
}

public struct GGUFTensorInfo: Sendable, Equatable {
    public let name: String
    public let dimensions: [UInt64]
    public let type: GGUFTensorType
    public let offset: UInt64

    public var elementCount: Int {
        dimensions.reduce(1) { partialResult, dimension in
            partialResult * Int(dimension)
        }
    }
}
