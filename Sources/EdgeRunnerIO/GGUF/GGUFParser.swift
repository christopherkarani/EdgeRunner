import Foundation

public final class GGUFReader {
    private let data: Data
    private(set) public var currentOffset: Int = 0

    public init(data: Data) {
        self.data = data
    }

    public func readHeader() throws -> GGUFHeader {
        let magic = try readUInt32()
        guard magic == ggufMagic else {
            throw WeightLoaderError.invalidFormat(
                "Invalid GGUF magic: 0x\(String(magic, radix: 16))"
            )
        }

        let version = try readUInt32()
        guard ggufSupportedVersions.contains(version) else {
            throw WeightLoaderError.unsupportedVersion(version)
        }

        return GGUFHeader(
            version: version,
            tensorCount: try readUInt64(),
            metadataKVCount: try readUInt64()
        )
    }

    public func readMetadata(count: Int) throws -> [String: GGUFMetadataValue] {
        var metadata: [String: GGUFMetadataValue] = [:]
        metadata.reserveCapacity(count)
        for _ in 0..<count {
            let key = try readString()
            let typeRaw = try readUInt32()
            guard let valueType = GGUFMetadataValueType(rawValue: typeRaw) else {
                throw WeightLoaderError.invalidFormat("Unknown GGUF metadata type: \(typeRaw)")
            }
            metadata[key] = try readValue(ofType: valueType)
        }
        return metadata
    }

    public func readTensorInfos(count: Int) throws -> [GGUFTensorInfo] {
        var tensorInfos: [GGUFTensorInfo] = []
        tensorInfos.reserveCapacity(count)

        for _ in 0..<count {
            let name = try readString()
            let dimensionCount = Int(try readUInt32())
            var dimensions: [UInt64] = []
            dimensions.reserveCapacity(dimensionCount)
            for _ in 0..<dimensionCount {
                dimensions.append(try readUInt64())
            }

            let typeRaw = try readUInt32()
            guard let tensorType = GGUFTensorType(rawValue: typeRaw) else {
                throw WeightLoaderError.unsupportedDataType(typeRaw)
            }

            let offset = try readUInt64()
            tensorInfos.append(
                GGUFTensorInfo(name: name, dimensions: dimensions, type: tensorType, offset: offset)
            )
        }

        return tensorInfos
    }

    public func advance(to alignment: Int) {
        guard alignment > 1 else { return }
        let remainder = currentOffset % alignment
        if remainder != 0 {
            currentOffset += alignment - remainder
        }
    }

    public func readUInt8() throws -> UInt8 {
        try readInteger(UInt8.self)
    }

    public func readInt8() throws -> Int8 {
        Int8(bitPattern: try readUInt8())
    }

    public func readUInt16() throws -> UInt16 {
        try readInteger(UInt16.self)
    }

    public func readInt16() throws -> Int16 {
        Int16(bitPattern: try readUInt16())
    }

    public func readUInt32() throws -> UInt32 {
        try readInteger(UInt32.self)
    }

    public func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    public func readUInt64() throws -> UInt64 {
        try readInteger(UInt64.self)
    }

    public func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    public func readFloat32() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    public func readFloat64() throws -> Double {
        Double(bitPattern: try readUInt64())
    }

    public func readBool() throws -> Bool {
        try readUInt8() != 0
    }

    public func readString() throws -> String {
        let length = Int(try readUInt64())
        let bytes = try readBytes(count: length)
        guard let value = String(data: bytes, encoding: .utf8) else {
            throw WeightLoaderError.invalidFormat("Invalid UTF-8 string at offset \(currentOffset - length)")
        }
        return value
    }

    private func readValue(ofType type: GGUFMetadataValueType) throws -> GGUFMetadataValue {
        switch type {
        case .uint8:
            return GGUFMetadataValue.uint8(try readUInt8())
        case .int8:
            return GGUFMetadataValue.int8(try readInt8())
        case .uint16:
            return GGUFMetadataValue.uint16(try readUInt16())
        case .int16:
            return GGUFMetadataValue.int16(try readInt16())
        case .uint32:
            return GGUFMetadataValue.uint32(try readUInt32())
        case .int32:
            return GGUFMetadataValue.int32(try readInt32())
        case .float32:
            return GGUFMetadataValue.float32(try readFloat32())
        case .bool:
            return GGUFMetadataValue.bool(try readBool())
        case .string:
            return GGUFMetadataValue.string(try readString())
        case .array:
            let elementTypeRaw = try readUInt32()
            guard let elementType = GGUFMetadataValueType(rawValue: elementTypeRaw) else {
                throw WeightLoaderError.invalidFormat(
                    "Unknown GGUF array element type: \(elementTypeRaw)"
                )
            }
            let count = Int(try readUInt64())
            var values: [GGUFMetadataValue] = []
            values.reserveCapacity(count)
            for _ in 0..<count {
                values.append(try readValue(ofType: elementType))
            }
            return GGUFMetadataValue.array(values)
        case .uint64:
            return GGUFMetadataValue.uint64(try readUInt64())
        case .int64:
            return GGUFMetadataValue.int64(try readInt64())
        case .float64:
            return GGUFMetadataValue.float64(try readFloat64())
        }
    }

    private func readBytes(count: Int) throws -> Data {
        guard count >= 0, currentOffset + count <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of GGUF data at offset \(currentOffset)")
        }
        let range = currentOffset..<(currentOffset + count)
        currentOffset += count
        return data.subdata(in: range)
    }

    private func readInteger<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard currentOffset + size <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of GGUF data at offset \(currentOffset)")
        }
        let value = data.withUnsafeBytes { rawBuffer in
            rawBuffer.loadUnaligned(fromByteOffset: currentOffset, as: T.self)
        }
        currentOffset += size
        return T(littleEndian: value)
    }
}
