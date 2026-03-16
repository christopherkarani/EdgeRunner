import Foundation
import Testing
@testable import EdgeRunnerIO

private struct GGUFBuilder {
    var data = Data()

    mutating func writeUInt8(_ value: UInt8) {
        data.append(value)
    }

    mutating func writeUInt16(_ value: UInt16) {
        append(value.littleEndian)
    }

    mutating func writeUInt32(_ value: UInt32) {
        append(value.littleEndian)
    }

    mutating func writeUInt64(_ value: UInt64) {
        append(value.littleEndian)
    }

    mutating func writeInt8(_ value: Int8) {
        data.append(UInt8(bitPattern: value))
    }

    mutating func writeInt16(_ value: Int16) {
        append(value.littleEndian)
    }

    mutating func writeInt32(_ value: Int32) {
        append(value.littleEndian)
    }

    mutating func writeInt64(_ value: Int64) {
        append(value.littleEndian)
    }

    mutating func writeFloat32(_ value: Float) {
        append(value.bitPattern.littleEndian)
    }

    mutating func writeFloat64(_ value: Double) {
        append(value.bitPattern.littleEndian)
    }

    mutating func writeBool(_ value: Bool) {
        data.append(value ? 1 : 0)
    }

    mutating func writeString(_ value: String) {
        let utf8 = Array(value.utf8)
        writeUInt64(UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }

    mutating func writeArray(
        key: String,
        elementType: GGUFMetadataValueType,
        values: [UInt32]
    ) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.array.rawValue)
        writeUInt32(elementType.rawValue)
        writeUInt64(UInt64(values.count))
        for value in values {
            writeUInt32(value)
        }
    }

    mutating func writeKV(key: String, stringValue: String) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.string.rawValue)
        writeString(stringValue)
    }

    mutating func writeKV(key: String, uint32Value: UInt32) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.uint32.rawValue)
        writeUInt32(uint32Value)
    }

    mutating func writeKV(key: String, int32Value: Int32) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.int32.rawValue)
        writeInt32(int32Value)
    }

    mutating func writeKV(key: String, float32Value: Float) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.float32.rawValue)
        writeFloat32(float32Value)
    }

    mutating func writeKV(key: String, boolValue: Bool) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.bool.rawValue)
        writeBool(boolValue)
    }

    mutating func writeKV(key: String, uint64Value: UInt64) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.uint64.rawValue)
        writeUInt64(uint64Value)
    }

    mutating func writeKV(key: String, int64Value: Int64) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.int64.rawValue)
        writeInt64(int64Value)
    }

    mutating func writeKV(key: String, float64Value: Double) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.float64.rawValue)
        writeFloat64(float64Value)
    }

    static func minimalHeader(
        version: UInt32 = 3,
        tensorCount: UInt64 = 0,
        metadataKVCount: UInt64 = 0,
        build: (inout GGUFBuilder) -> Void = { _ in }
    ) -> Data {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(version)
        builder.writeUInt64(tensorCount)
        builder.writeUInt64(metadataKVCount)
        build(&builder)
        return builder.data
    }

    private mutating func append<T>(_ value: T) {
        withUnsafeBytes(of: value) { bytes in
            data.append(contentsOf: bytes)
        }
    }
}

@Suite("GGUF Header Parsing")
struct GGUFHeaderTests {

    @Test func parseMagicAndVersion() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 0)
        let header = try GGUFHeader.parse(from: data)
        #expect(header.version == 3)
        #expect(header.tensorCount == 0)
        #expect(header.metadataKVCount == 0)
    }

    @Test func rejectInvalidMagic() {
        var data = Data()
        withUnsafeBytes(of: UInt32(0xDEADBEEF).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: [UInt8](repeating: 0, count: 20))
        #expect(throws: WeightLoaderError.self) {
            try GGUFHeader.parse(from: data)
        }
    }

    @Test func rejectUnsupportedVersion() {
        let data = GGUFBuilder.minimalHeader(version: 1)
        #expect(throws: WeightLoaderError.self) {
            try GGUFHeader.parse(from: data)
        }
    }

    @Test func parseVersion2() throws {
        let data = GGUFBuilder.minimalHeader(version: 2, tensorCount: 5, metadataKVCount: 2)
        let header = try GGUFHeader.parse(from: data)
        #expect(header.version == 2)
        #expect(header.tensorCount == 5)
        #expect(header.metadataKVCount == 2)
    }

    @Test func parseStringMetadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeKV(key: "general.architecture", stringValue: "llama")
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["general.architecture"]?.stringValue == "llama")
    }

    @Test func parseUInt32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeKV(key: "llama.embedding_length", uint32Value: 4096)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["llama.embedding_length"]?.uint32Value == 4096)
    }

    @Test func parseInt32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeKV(key: "test.signed", int32Value: -42)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["test.signed"]?.int32Value == -42)
    }

    @Test func parseFloat32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeKV(key: "llama.rope.freq_base", float32Value: 500000.0)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["llama.rope.freq_base"]?.float32Value == 500000.0)
    }

    @Test func parseBoolMetadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeKV(key: "general.use_parallel", boolValue: true)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["general.use_parallel"]?.boolValue == true)
    }

    @Test func parseUInt64Int64AndFloat64Metadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 3) { builder in
            builder.writeKV(key: "general.file_size", uint64Value: 1_234_567_890)
            builder.writeKV(key: "general.delta", int64Value: -99)
            builder.writeKV(key: "general.scale", float64Value: 3.5)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["general.file_size"]?.uint64Value == 1_234_567_890)
        #expect(metadata["general.delta"]?.int64Value == -99)
        #expect(metadata["general.scale"]?.float64Value == 3.5)
    }

    @Test func parseArrayMetadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 1) { builder in
            builder.writeArray(
                key: "tokenizer.tokens",
                elementType: .uint32,
                values: [1, 2, 3, 4]
            )
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let array = metadata["tokenizer.tokens"]?.arrayValue
        #expect(array?.count == 4)
        #expect(array?[0].uint32Value == 1)
        #expect(array?[3].uint32Value == 4)
    }

    @Test func parseMultipleMetadataEntries() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 3) { builder in
            builder.writeKV(key: "general.architecture", stringValue: "llama")
            builder.writeKV(key: "llama.block_count", uint32Value: 32)
            builder.writeKV(key: "llama.attention.head_count", uint32Value: 32)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata.count == 3)
        #expect(metadata["general.architecture"]?.stringValue == "llama")
        #expect(metadata["llama.block_count"]?.uint32Value == 32)
        #expect(metadata["llama.attention.head_count"]?.uint32Value == 32)
    }

    @Test func extractModelConfigFromMetadata() throws {
        let data = GGUFBuilder.minimalHeader(metadataKVCount: 9) { builder in
            builder.writeKV(key: "general.architecture", stringValue: "llama")
            builder.writeKV(key: "llama.vocab_size", uint32Value: 32000)
            builder.writeKV(key: "llama.embedding_length", uint32Value: 4096)
            builder.writeKV(key: "llama.block_count", uint32Value: 32)
            builder.writeKV(key: "llama.attention.head_count", uint32Value: 32)
            builder.writeKV(key: "llama.attention.head_count_kv", uint32Value: 8)
            builder.writeKV(key: "llama.feed_forward_length", uint32Value: 11008)
            builder.writeKV(key: "llama.context_length", uint32Value: 4096)
            builder.writeKV(key: "llama.rope.freq_base", float32Value: 500000.0)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let config = try ModelConfig.from(ggufMetadata: metadata)

        #expect(config.architectureName == "llama")
        #expect(config.int(forKey: "llama.vocab_size") == 32000)
        #expect(config.int(forKey: "llama.embedding_length") == 4096)
        #expect(config.int(forKey: "llama.block_count") == 32)
        #expect(config.int(forKey: "llama.attention.head_count") == 32)
        #expect(config.int(forKey: "llama.attention.head_count_kv") == 8)
        #expect(config.int(forKey: "llama.feed_forward_length") == 11008)
        #expect(config.int(forKey: "llama.context_length") == 4096)
        #expect(config.float(forKey: "llama.rope.freq_base") == 500000.0)
    }
}
