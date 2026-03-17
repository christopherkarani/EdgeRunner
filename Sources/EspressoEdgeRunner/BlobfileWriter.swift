import Foundation

public enum BlobfileWriter: Sendable {

    /// Builds the 128-byte BLOBFILE header.
    public static func makeHeader(payloadByteCount: Int) throws -> Data {
        guard payloadByteCount >= 0, payloadByteCount <= Int(UInt32.max) else {
            throw EspressoError.payloadTooLarge(payloadByteCount)
        }

        var header = Data(count: 128)

        // Bytes 64-67: magic 0xDEADBEEF little-endian
        header[64] = 0xEF
        header[65] = 0xBE
        header[66] = 0xAD
        header[67] = 0xDE

        // Bytes 72-75: payload byte count as UInt32 LE
        let payloadSize = UInt32(payloadByteCount)
        header[72] = UInt8(payloadSize & 0xFF)
        header[73] = UInt8((payloadSize >> 8) & 0xFF)
        header[74] = UInt8((payloadSize >> 16) & 0xFF)
        header[75] = UInt8((payloadSize >> 24) & 0xFF)

        // Bytes 80-83: data offset = 128 as UInt32 LE
        let offset = UInt32(128)
        header[80] = UInt8(offset & 0xFF)
        header[81] = UInt8((offset >> 8) & 0xFF)
        header[82] = UInt8((offset >> 16) & 0xFF)
        header[83] = UInt8((offset >> 24) & 0xFF)

        return header
    }

    /// Converts `[Float]` to packed fp16 `Data` (each value as UInt16 LE).
    public static func floatsToFP16(_ floats: [Float]) -> Data {
        var data = Data(capacity: floats.count * 2)
        for value in floats {
            let fp16 = Float16(value)
            let bits = fp16.bitPattern
            data.append(UInt8(bits & 0xFF))
            data.append(UInt8(bits >> 8))
        }
        return data
    }

    /// Writes `[Float]` as a complete BLOBFILE (128-byte header + fp16 payload).
    public static func write(floats: [Float], to url: URL) throws {
        let payload = floatsToFP16(floats)
        let header = try makeHeader(payloadByteCount: payload.count)
        var fileData = header
        fileData.append(payload)
        try fileData.write(to: url, options: .atomic)
    }
}
