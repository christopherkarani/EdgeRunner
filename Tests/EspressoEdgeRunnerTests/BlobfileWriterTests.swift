import Testing
import Foundation
@testable import EspressoEdgeRunner

@Suite("BlobfileWriter")
struct BlobfileWriterTests {

    @Test("Header is exactly 128 bytes")
    func headerSize() throws {
        let header = try BlobfileWriter.makeHeader(payloadByteCount: 100)
        #expect(header.count == 128)
    }

    @Test("Magic at bytes 64-67 is 0xDEADBEEF LE")
    func headerMagic() throws {
        let header = try BlobfileWriter.makeHeader(payloadByteCount: 0)
        #expect(header[64] == 0xEF)
        #expect(header[65] == 0xBE)
        #expect(header[66] == 0xAD)
        #expect(header[67] == 0xDE)
    }

    @Test("Payload size at bytes 72-75 matches input count * 2")
    func headerPayloadSize() throws {
        let floatCount = 50
        let expectedPayloadBytes = UInt32(floatCount * 2)
        let header = try BlobfileWriter.makeHeader(payloadByteCount: floatCount * 2)
        let stored = UInt32(header[72])
            | (UInt32(header[73]) << 8)
            | (UInt32(header[74]) << 16)
            | (UInt32(header[75]) << 24)
        #expect(stored == expectedPayloadBytes)
    }

    @Test("Data offset at bytes 80-83 is 128")
    func headerDataOffset() throws {
        let header = try BlobfileWriter.makeHeader(payloadByteCount: 0)
        let offset = UInt32(header[80])
            | (UInt32(header[81]) << 8)
            | (UInt32(header[82]) << 16)
            | (UInt32(header[83]) << 24)
        #expect(offset == 128)
    }

    @Test("FP16 round-trip within tolerance 1e-3")
    func fp16RoundTrip() {
        let inputs: [Float] = [0.0, 1.0, -1.0, 0.5, 3.14, 65504.0]
        let data = BlobfileWriter.floatsToFP16(inputs)
        #expect(data.count == inputs.count * 2)

        for (i, original) in inputs.enumerated() {
            let lo = data[i * 2]
            let hi = data[i * 2 + 1]
            let bits = UInt16(lo) | (UInt16(hi) << 8)
            let recovered = Float(Float16(bitPattern: bits))
            #expect(abs(recovered - original) < 1e-3, "Mismatch at index \(i): \(recovered) vs \(original)")
        }
    }

    @Test("Write creates valid file with header + payload")
    func writeCreatesFile() throws {
        let floats: [Float] = [1.0, 2.0, 3.0]
        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_blobfile_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try BlobfileWriter.write(floats: floats, to: url)

        let data = try Data(contentsOf: url)
        #expect(data.count == 128 + floats.count * 2)
        // Verify magic
        #expect(data[64] == 0xEF)
        #expect(data[65] == 0xBE)
    }

    @Test("Empty array produces header-only 128-byte file")
    func writeEmptyArray() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_blobfile_empty_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try BlobfileWriter.write(floats: [], to: url)

        let data = try Data(contentsOf: url)
        #expect(data.count == 128)
    }

    @Test("Payload exceeding UInt32.max throws payloadTooLarge")
    func payloadTooLargeThrows() {
        let oversized = Int(UInt32.max) + 1
        #expect(throws: EspressoError.payloadTooLarge(oversized)) {
            try BlobfileWriter.makeHeader(payloadByteCount: oversized)
        }
    }
}
