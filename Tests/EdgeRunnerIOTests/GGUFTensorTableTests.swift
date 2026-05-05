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

    @Test func parseModernGGUFTensorTypeOrdering() throws {
        #expect(GGUFTensorType(rawValue: 16) == .iq2_xxs)
        #expect(GGUFTensorType(rawValue: 23) == .iq4_xs)
        #expect(GGUFTensorType(rawValue: 24) == .i8)
        #expect(GGUFTensorType(rawValue: 28) == .f64)
        #expect(GGUFTensorType(rawValue: 29) == .iq1_m)
        #expect(GGUFTensorType(rawValue: 30) == .bf16)
        #expect(GGUFTensorType(rawValue: 34) == .tq1_0)
        #expect(GGUFTensorType(rawValue: 35) == .tq2_0)
        #expect(GGUFTensorType(rawValue: 39) == .mxfp4)
        #expect(GGUFTensorType(rawValue: 40) == .nvfp4)
        #expect(GGUFTensorType.bf16.tensorDataType == .bfloat16)
    }

    @Test func deferredUpstreamTensorTypesAreExplicitlyUnsupported() throws {
        let unsupported: [(GGUFTensorType, String)] = [
            (.iq2_xxs, "IQ2_XXS"),
            (.iq2_xs, "IQ2_XS"),
            (.iq3_xxs, "IQ3_XXS"),
            (.iq1_s, "IQ1_S"),
            (.iq4_nl, "IQ4_NL"),
            (.iq3_s, "IQ3_S"),
            (.iq2_s, "IQ2_S"),
            (.iq4_xs, "IQ4_XS"),
            (.iq1_m, "IQ1_M"),
            (.tq1_0, "TQ1_0"),
            (.tq2_0, "TQ2_0"),
            (.mxfp4, "MXFP4"),
            (.nvfp4, "NVFP4"),
        ]

        for (type, name) in unsupported {
            #expect(type.tensorDataType == nil)
            #expect(type.diagnosticName == name)
        }
    }

    @Test func parseBF16TensorInfo() throws {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(1)
        builder.writeUInt64(0)
        builder.writeString("per_layer_model_proj.weight")
        builder.writeUInt32(2)
        builder.writeUInt64(2560)
        builder.writeUInt64(10752)
        builder.writeUInt32(30)
        builder.writeUInt64(0)

        let reader = GGUFReader(data: builder.data)
        let header = try reader.readHeader()
        _ = try reader.readMetadata(count: Int(header.metadataKVCount))
        let infos = try reader.readTensorInfos(count: Int(header.tensorCount))

        #expect(infos.count == 1)
        #expect(infos[0].name == "per_layer_model_proj.weight")
        #expect(infos[0].type == .bf16)
        #expect(infos[0].type.tensorDataType == .bfloat16)
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

    @Test func loaderAcceptsMobileQuantByteCounts() async throws {
        let cases: [(GGUFTensorType, [UInt64], Int)] = [
            (.f16, [256], 512),
            (.bf16, [256], 512),
            (.q2_K, [256], 84),
            (.q3_K, [256], 110),
            (.q4_K, [256], 144),
            (.q5_K, [256], 176),
            (.q6_K, [256], 210),
            (.q8_0, [32], 34),
        ]

        for (type, dimensions, byteCount) in cases {
            let url = try Self.writeSingleTensorGGUF(
                type: type,
                dimensions: dimensions,
                payloadByteCount: byteCount
            )
            defer { try? FileManager.default.removeItem(at: url) }

            let map = try await GGUFLoader().load(from: url)
            let tensor = try #require(map["test.weight"])
            #expect(tensor.dataType == type.tensorDataType)
            #expect(tensor.elementCount == Int(dimensions.reduce(1, *)))
        }
    }

    @Test func loaderRejectsTruncatedMobileQuantPayloads() async throws {
        let cases: [(GGUFTensorType, [UInt64], Int)] = [
            (.f16, [256], 512),
            (.bf16, [256], 512),
            (.q2_K, [256], 84),
            (.q3_K, [256], 110),
            (.q4_K, [256], 144),
            (.q5_K, [256], 176),
            (.q6_K, [256], 210),
            (.q8_0, [32], 34),
        ]

        for (type, dimensions, byteCount) in cases {
            let url = try Self.writeSingleTensorGGUF(
                type: type,
                dimensions: dimensions,
                payloadByteCount: byteCount - 1
            )
            defer { try? FileManager.default.removeItem(at: url) }

            await #expect(throws: WeightLoaderError.self, "Expected truncated \(type.diagnosticName) payload to fail") {
                _ = try await GGUFLoader().load(from: url)
            }
        }
    }

    @Test func loaderRejectsDeferredUpstreamTensorTypes() async throws {
        let unsupported: [GGUFTensorType] = [
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs,
            .iq1_m, .tq1_0, .tq2_0, .mxfp4, .nvfp4,
        ]

        for type in unsupported {
            let url = try Self.writeSingleTensorGGUF(
                type: type,
                dimensions: [256],
                payloadByteCount: 0
            )
            defer { try? FileManager.default.removeItem(at: url) }

            await #expect(throws: WeightLoaderError.unsupportedDataType(type.rawValue)) {
                _ = try await GGUFLoader().load(from: url)
            }
        }
    }

    private static func writeSingleTensorGGUF(
        type: GGUFTensorType,
        dimensions: [UInt64],
        payloadByteCount: Int
    ) throws -> URL {
        var builder = GGUFBuilder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(1)
        builder.writeUInt64(1)

        builder.writeString("general.architecture")
        builder.writeUInt32(GGUFMetadataValueType.string.rawValue)
        builder.writeString("llama")

        builder.writeString("test.weight")
        builder.writeUInt32(UInt32(dimensions.count))
        for dimension in dimensions {
            builder.writeUInt64(dimension)
        }
        builder.writeUInt32(type.rawValue)
        builder.writeUInt64(0)

        while builder.data.count % 32 != 0 {
            builder.data.append(0)
        }
        builder.data.append(contentsOf: [UInt8](repeating: 0, count: payloadByteCount))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_gguf_quant_\(UUID().uuidString).gguf")
        try builder.data.write(to: url)
        return url
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
