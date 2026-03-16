import Foundation
import Metal
import Testing
@testable import EdgeRunnerIO

@Suite("GGUF Tensor Table Parsing")
struct GGUFTensorTableTests {

    @Test func parseSingleTensorInfo() throws {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(1)
        builder.writeUInt64(0)
        builder.writeString("output.weight")
        builder.writeUInt32(2)
        builder.writeUInt64(4096)
        builder.writeUInt64(32000)
        builder.writeUInt32(GGUFTensorType.f16.rawValue)
        builder.writeUInt64(0)

        let reader = GGUFReader(data: builder.data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let infos = try reader.readTensorInfos(count: Int(header.tensorCount))

        #expect(metadata.isEmpty)
        #expect(infos.count == 1)
        #expect(infos[0].name == "output.weight")
        #expect(infos[0].dimensions == [4096, 32000])
        #expect(infos[0].type == .f16)
        #expect(infos[0].offset == 0)
    }

    @Test func parseMultipleTensorInfos() throws {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(3)
        builder.writeUInt64(0)

        let tensors: [(String, [UInt64], GGUFTensorType, UInt64)] = [
            ("token_embd.weight", [32000, 4096], .f16, 0),
            ("blk.0.attn_q.weight", [4096, 4096], .q8_0, 262_144_000),
            ("blk.0.ffn_gate.weight", [4096, 11008], .q4_0, 295_698_432),
        ]

        for (name, dimensions, type, offset) in tensors {
            builder.writeString(name)
            builder.writeUInt32(UInt32(dimensions.count))
            for dimension in dimensions {
                builder.writeUInt64(dimension)
            }
            builder.writeUInt32(type.rawValue)
            builder.writeUInt64(offset)
        }

        let reader = GGUFReader(data: builder.data)
        let header = try reader.readHeader()
        _ = try reader.readMetadata(count: Int(header.metadataKVCount))
        let infos = try reader.readTensorInfos(count: Int(header.tensorCount))

        #expect(infos.count == 3)
        #expect(infos[0].name == "token_embd.weight")
        #expect(infos[0].type == .f16)
        #expect(infos[1].name == "blk.0.attn_q.weight")
        #expect(infos[1].type == .q8_0)
        #expect(infos[2].name == "blk.0.ffn_gate.weight")
        #expect(infos[2].type == .q4_0)
    }

    @Test func rejectUnknownTensorType() {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(1)
        builder.writeUInt64(0)
        builder.writeString("bad.weight")
        builder.writeUInt32(1)
        builder.writeUInt64(100)
        builder.writeUInt32(0xFF)
        builder.writeUInt64(0)

        let reader = GGUFReader(data: builder.data)
        #expect(throws: WeightLoaderError.self) {
            let header = try reader.readHeader()
            _ = try reader.readMetadata(count: Int(header.metadataKVCount))
            _ = try reader.readTensorInfos(count: Int(header.tensorCount))
        }
    }
}

@Suite("Memory-Mapped File")
struct MemoryMappedFileTests {

    @Test func mmapReadOnlyFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_mmap_\(UUID().uuidString).bin")
        let testData = Data(repeating: 0xAB, count: 4096)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        #expect(mapped.size == 4096)
        #expect(mapped.data[0] == 0xAB)
        #expect(mapped.data[4095] == 0xAB)
    }

    @Test func mmapAlignment() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_align_\(UUID().uuidString).bin")
        let pageSize = Int(getpagesize())
        let testData = Data(repeating: 0xCD, count: pageSize * 2)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        #expect(Int(bitPattern: mapped.basePointer) % pageSize == 0)
    }

    @Test func mmapNonexistentFileThrows() {
        let url = URL(fileURLWithPath: "/tmp/nonexistent_\(UUID().uuidString).bin")
        #expect(throws: WeightLoaderError.self) {
            _ = try MemoryMappedFile(url: url)
        }
    }

    @Test func mmapSliceAtOffset() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_slice_\(UUID().uuidString).bin")
        var testData = Data(repeating: 0x00, count: 256)
        for index in 128..<192 {
            testData[index] = UInt8(index - 128)
        }
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        let slice = mapped.slice(offset: 128, length: 64)
        #expect(slice.count == 64)
        #expect(slice[0] == 0)
        #expect(slice[1] == 1)
        #expect(slice[63] == 63)
    }
}

@Suite("GGUF Loader Integration")
struct GGUFLoaderTests {

    @Test func loaderConformsToProtocol() {
        let loader: any EdgeRunnerWeightLoader = GGUFLoader()
        #expect(loader.canLoad(url: URL(fileURLWithPath: "/tmp/model.gguf")))
        #expect(!loader.canLoad(url: URL(fileURLWithPath: "/tmp/model.safetensors")))
    }

    @Test func createMTLBufferFromMmapNoCopy() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_nocopy_\(UUID().uuidString).bin")
        let pageSize = Int(getpagesize())
        let bufferSize = pageSize * 4
        let testData = Data(repeating: 0x42, count: bufferSize)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        let buffer = try mapped.makeMetalBuffer(device: device, offset: 0, length: bufferSize)
        #expect(buffer.length == bufferSize)

        let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: bufferSize)
        #expect(pointer[0] == 0x42)
        #expect(pointer[bufferSize - 1] == 0x42)
    }
}

private struct GGUFBuilder {
    var data = Data()

    mutating func writeUInt32(_ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    mutating func writeUInt64(_ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    mutating func writeString(_ value: String) {
        let utf8 = Array(value.utf8)
        writeUInt64(UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }
}
