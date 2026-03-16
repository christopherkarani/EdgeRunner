import Foundation

public enum SafeTensorError: Error, Sendable, Equatable {
    case fileTooSmall
    case headerSizeExceedsFile
    case invalidJSON(description: String)
    case missingField(tensor: String, field: String)
    case unknownDtype(String)
    case invalidDataOffsets(tensor: String)
}

public enum SafeTensorDtype: String, Sendable, Equatable {
    case float32 = "F32"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case int8 = "I8"
    case int16 = "I16"
    case int32 = "I32"
    case int64 = "I64"
    case float64 = "F64"

    public var byteSize: Int {
        switch self {
        case .float32, .int32:
            return 4
        case .float16, .bfloat16, .int16:
            return 2
        case .int8:
            return 1
        case .int64, .float64:
            return 8
        }
    }

    var tensorDataType: TensorDataType {
        switch self {
        case .float32:
            return .float32
        case .float16:
            return .float16
        case .bfloat16:
            return .bfloat16
        case .int8:
            return .i8
        case .int16:
            return .i16
        case .int32:
            return .i32
        case .int64:
            return .i64
        case .float64:
            return .f64
        }
    }
}

public struct DataOffsets: Sendable, Equatable {
    public let begin: Int
    public let end: Int

    public init(begin: Int, end: Int) {
        self.begin = begin
        self.end = end
    }
}

public struct SafeTensorTensorMeta: Sendable, Equatable {
    public let dtype: SafeTensorDtype
    public let shape: [Int]
    public let dataOffsets: DataOffsets

    public init(dtype: SafeTensorDtype, shape: [Int], dataOffsets: DataOffsets) {
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}

public struct SafeTensorHeader: Sendable, Equatable {
    public let dataOffset: Int
    public let metadata: [String: MetadataValue]
    public let tensors: [String: SafeTensorTensorMeta]

    public init(
        dataOffset: Int,
        metadata: [String: MetadataValue],
        tensors: [String: SafeTensorTensorMeta]
    ) {
        self.dataOffset = dataOffset
        self.metadata = metadata
        self.tensors = tensors
    }

    public static func parse(from data: Data) throws -> SafeTensorHeader {
        guard data.count >= 8 else {
            throw SafeTensorError.fileTooSmall
        }

        let rawHeaderSize = data.withUnsafeBytes { buffer in
            buffer.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        }
        let headerSize = Int(UInt64(littleEndian: rawHeaderSize))
        let dataOffset = 8 + headerSize

        guard dataOffset <= data.count else {
            throw SafeTensorError.headerSizeExceedsFile
        }

        let headerData = data.subdata(in: 8..<dataOffset)
        let decoder = JSONDecoder()
        let rawHeader: RawHeader
        do {
            rawHeader = try decoder.decode(RawHeader.self, from: headerData)
        } catch {
            throw SafeTensorError.invalidJSON(description: error.localizedDescription)
        }

        var tensors: [String: SafeTensorTensorMeta] = [:]
        for (name, rawTensor) in rawHeader.tensors {
            guard let dtypeName = rawTensor.dtype else {
                throw SafeTensorError.missingField(tensor: name, field: "dtype")
            }
            guard let dtype = SafeTensorDtype(rawValue: dtypeName) else {
                throw SafeTensorError.unknownDtype(dtypeName)
            }
            guard let shape = rawTensor.shape else {
                throw SafeTensorError.missingField(tensor: name, field: "shape")
            }
            guard let offsets = rawTensor.dataOffsets else {
                throw SafeTensorError.missingField(tensor: name, field: "data_offsets")
            }
            guard offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0] else {
                throw SafeTensorError.invalidDataOffsets(tensor: name)
            }

            tensors[name] = SafeTensorTensorMeta(
                dtype: dtype,
                shape: shape,
                dataOffsets: DataOffsets(begin: offsets[0], end: offsets[1])
            )
        }

        return SafeTensorHeader(
            dataOffset: dataOffset,
            metadata: rawHeader.metadata.mapValues(\.metadataValue),
            tensors: tensors
        )
    }
}

private struct RawHeader: Decodable {
    var metadata: [String: RawMetadataValue] = [:]
    var tensors: [String: RawTensorEntry] = [:]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicKey.self)
        var metadata: [String: RawMetadataValue] = [:]
        var tensors: [String: RawTensorEntry] = [:]

        for key in container.allKeys {
            if key.stringValue == "__metadata__" {
                metadata = try container.decode([String: RawMetadataValue].self, forKey: key)
            } else {
                tensors[key.stringValue] = try container.decode(RawTensorEntry.self, forKey: key)
            }
        }

        self.metadata = metadata
        self.tensors = tensors
    }
}

private struct RawTensorEntry: Decodable {
    let dtype: String?
    let shape: [Int]?
    let dataOffsets: [Int]?

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

private enum RawMetadataValue: Decodable {
    case string(String)
    case int(Int)
    case float(Double)
    case bool(Bool)
    case array([RawMetadataValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .float(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([RawMetadataValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.typeMismatch(
                RawMetadataValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unsupported SafeTensor metadata value"
                )
            )
        }
    }

    var metadataValue: MetadataValue {
        switch self {
        case .string(let value):
            let lowercased = value.lowercased()
            if lowercased == "true" {
                return .bool(true)
            }
            if lowercased == "false" {
                return .bool(false)
            }
            if let integer = Int(value) {
                return .int(integer)
            }
            if let float = Float(value) {
                return .float(float)
            }
            return .string(value)
        case .int(let value):
            return .int(value)
        case .float(let value):
            return .float(Float(value))
        case .bool(let value):
            return .bool(value)
        case .array(let value):
            return .array(value.map(\.metadataValue))
        }
    }
}

private struct DynamicKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}
