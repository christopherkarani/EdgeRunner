import Foundation

public enum GGUFMetadataValue: Sendable, Equatable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case array([GGUFMetadataValue])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var uint32Value: UInt32? {
        guard case .uint32(let value) = self else { return nil }
        return value
    }

    public var int32Value: Int32? {
        guard case .int32(let value) = self else { return nil }
        return value
    }

    public var float32Value: Float? {
        guard case .float32(let value) = self else { return nil }
        return value
    }

    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    public var uint64Value: UInt64? {
        guard case .uint64(let value) = self else { return nil }
        return value
    }

    public var int64Value: Int64? {
        guard case .int64(let value) = self else { return nil }
        return value
    }

    public var float64Value: Double? {
        guard case .float64(let value) = self else { return nil }
        return value
    }

    public var arrayValue: [GGUFMetadataValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    public var intValue: Int? {
        switch self {
        case .uint8(let value): Int(value)
        case .int8(let value): Int(value)
        case .uint16(let value): Int(value)
        case .int16(let value): Int(value)
        case .uint32(let value): Int(value)
        case .int32(let value): Int(value)
        case .uint64(let value): Int(exactly: value)
        case .int64(let value): Int(exactly: value)
        default: nil
        }
    }

    public var floatValue: Float? {
        switch self {
        case .uint8(let value): Float(value)
        case .int8(let value): Float(value)
        case .uint16(let value): Float(value)
        case .int16(let value): Float(value)
        case .uint32(let value): Float(value)
        case .int32(let value): Float(value)
        case .uint64(let value): Float(value)
        case .int64(let value): Float(value)
        case .float32(let value): value
        case .float64(let value): Float(value)
        default: nil
        }
    }

    public var metadataValue: MetadataValue? {
        switch self {
        case .uint8(let value): return MetadataValue.int(Int(value))
        case .int8(let value): return MetadataValue.int(Int(value))
        case .uint16(let value): return MetadataValue.int(Int(value))
        case .int16(let value): return MetadataValue.int(Int(value))
        case .uint32(let value): return MetadataValue.int(Int(value))
        case .int32(let value): return MetadataValue.int(Int(value))
        case .float32(let value): return MetadataValue.float(value)
        case .bool(let value): return MetadataValue.bool(value)
        case .string(let value): return MetadataValue.string(value)
        case .array(let value):
            return MetadataValue.array(value.compactMap(\.metadataValue))
        case .uint64(let value):
            guard let intValue = Int(exactly: value) else { return nil }
            return MetadataValue.int(intValue)
        case .int64(let value):
            guard let intValue = Int(exactly: value) else { return nil }
            return MetadataValue.int(intValue)
        case .float64(let value):
            return MetadataValue.float(Float(value))
        }
    }
}

extension ModelConfig {
    public static func from(ggufMetadata metadata: [String: GGUFMetadataValue]) throws -> ModelConfig {
        guard let architectureName = metadata["general.architecture"]?.stringValue else {
            throw WeightLoaderError.missingMetadata("general.architecture")
        }

        var converted: [String: MetadataValue] = [:]
        converted.reserveCapacity(metadata.count)
        for (key, value) in metadata {
            guard let metadataValue = value.metadataValue else {
                throw WeightLoaderError.invalidFormat(
                    "GGUF metadata value for \(key) cannot be represented in ModelConfig"
                )
            }
            converted[key] = metadataValue
        }

        return ModelConfig(architectureName: architectureName, metadata: converted)
    }
}
