import Foundation

public enum NPYError: Error, Sendable, Equatable {
    case fileTooSmall
    case invalidMagic
    case unsupportedVersion(major: UInt8, minor: UInt8)
    case invalidHeader(String)
    case unsupportedDtype(String)
    case entryNotFound(String)
    case unsupportedCompressionMethod(UInt16)
}

public enum NPYDtype: Sendable, Equatable {
    case float32
    case float16
    case int8

    public var byteSize: Int {
        switch self {
        case .float32:
            return 4
        case .float16:
            return 2
        case .int8:
            return 1
        }
    }

    var tensorDataType: TensorDataType {
        switch self {
        case .float32:
            return .float32
        case .float16:
            return .float16
        case .int8:
            return .i8
        }
    }

    init(descrString: String) throws {
        switch descrString {
        case "<f4", "=f4":
            self = .float32
        case "<f2", "=f2":
            self = .float16
        case "|i1", "=i1", "<i1":
            self = .int8
        default:
            throw NPYError.unsupportedDtype(descrString)
        }
    }
}

public struct NPYHeader: Sendable, Equatable {
    public let dtype: NPYDtype
    public let shape: [Int]
    public let isFortranOrder: Bool

    public init(dtype: NPYDtype, shape: [Int], isFortranOrder: Bool) {
        self.dtype = dtype
        self.shape = shape
        self.isFortranOrder = isFortranOrder
    }

    public static func parse(from data: Data) throws -> NPYHeader {
        try parseWithOffset(from: data).0
    }

    public static func parseWithOffset(from data: Data) throws -> (NPYHeader, Int) {
        guard data.count >= 10 else {
            throw NPYError.fileTooSmall
        }

        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
        for index in 0..<magic.count where data[index] != magic[index] {
            throw NPYError.invalidMagic
        }

        let major = data[6]
        let minor = data[7]

        let headerLength: Int
        let headerStart: Int
        switch major {
        case 1:
            headerLength = Int(readUInt16(from: data, at: 8))
            headerStart = 10
        case 2:
            guard data.count >= 12 else {
                throw NPYError.fileTooSmall
            }
            headerLength = Int(readUInt32(from: data, at: 8))
            headerStart = 12
        default:
            throw NPYError.unsupportedVersion(major: major, minor: minor)
        }

        let dataOffset = headerStart + headerLength
        guard dataOffset <= data.count else {
            throw NPYError.fileTooSmall
        }

        let headerData = data.subdata(in: headerStart..<dataOffset)
        guard let headerString = String(data: headerData, encoding: .ascii) else {
            throw NPYError.invalidHeader("Cannot decode NPY header as ASCII")
        }

        return (try parseHeaderString(headerString), dataOffset)
    }

    private static func parseHeaderString(_ header: String) throws -> NPYHeader {
        let trimmed = header.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let descr = firstCapture(
            in: trimmed,
            pattern: "'descr':\\s*'([^']+)'"
        ) else {
            throw NPYError.invalidHeader("Missing 'descr' field")
        }
        let dtype = try NPYDtype(descrString: descr)

        guard let fortran = firstCapture(
            in: trimmed,
            pattern: "'fortran_order':\\s*(True|False)"
        ) else {
            throw NPYError.invalidHeader("Missing 'fortran_order' field")
        }

        guard let shapeContents = firstCapture(
            in: trimmed,
            pattern: "'shape':\\s*\\(([^)]*)\\)"
        ) else {
            throw NPYError.invalidHeader("Missing 'shape' field")
        }

        let shape = shapeContents
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .compactMap(Int.init)

        let isFortranOrder = fortran == "True"
        guard !isFortranOrder else {
            throw NPYError.invalidHeader(
                "Fortran-order (column-major) arrays are not supported; data would be silently transposed"
            )
        }

        return NPYHeader(
            dtype: dtype,
            shape: shape,
            isFortranOrder: false
        )
    }

    private static func firstCapture(in string: String, pattern: String) -> String? {
        guard let expression = try? NSRegularExpression(pattern: pattern) else {
            return nil
        }

        let nsRange = NSRange(string.startIndex..<string.endIndex, in: string)
        guard let match = expression.firstMatch(in: string, options: [], range: nsRange),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: string)
        else {
            return nil
        }

        return String(string[range])
    }
}

func readUInt16(from data: Data, at offset: Int) -> UInt16 {
    let raw = data.withUnsafeBytes { buffer in
        buffer.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
    }
    return UInt16(littleEndian: raw)
}

func readUInt32(from data: Data, at offset: Int) -> UInt32 {
    let raw = data.withUnsafeBytes { buffer in
        buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
    }
    return UInt32(littleEndian: raw)
}
