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
    case iq2_xxs = 16
    case iq2_xs = 17
    case iq3_xxs = 18
    case iq1_s = 19
    case iq4_nl = 20
    case iq3_s = 21
    case iq2_s = 22
    case iq4_xs = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1_m = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35
    case mxfp4 = 39
    case nvfp4 = 40
    case q1_0_g128 = 41

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
        case .bf16: .bfloat16
        case .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs,
             .iq1_m, .tq1_0, .tq2_0, .mxfp4, .nvfp4:
            nil
        case .q1_0_g128:
            .q1_0_g128
        }
    }

    public var diagnosticName: String {
        switch self {
        case .f32: "F32"
        case .f16: "F16"
        case .q4_0: "Q4_0"
        case .q4_1: "Q4_1"
        case .q5_0: "Q5_0"
        case .q5_1: "Q5_1"
        case .q8_0: "Q8_0"
        case .q8_1: "Q8_1"
        case .q2_K: "Q2_K"
        case .q3_K: "Q3_K"
        case .q4_K: "Q4_K"
        case .q5_K: "Q5_K"
        case .q6_K: "Q6_K"
        case .q8_K: "Q8_K"
        case .iq2_xxs: "IQ2_XXS"
        case .iq2_xs: "IQ2_XS"
        case .iq3_xxs: "IQ3_XXS"
        case .iq1_s: "IQ1_S"
        case .iq4_nl: "IQ4_NL"
        case .iq3_s: "IQ3_S"
        case .iq2_s: "IQ2_S"
        case .iq4_xs: "IQ4_XS"
        case .i8: "I8"
        case .i16: "I16"
        case .i32: "I32"
        case .i64: "I64"
        case .f64: "F64"
        case .iq1_m: "IQ1_M"
        case .bf16: "BF16"
        case .tq1_0: "TQ1_0"
        case .tq2_0: "TQ2_0"
        case .mxfp4: "MXFP4"
        case .nvfp4: "NVFP4"
        case .q1_0_g128: "Q1_0_G128"
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
