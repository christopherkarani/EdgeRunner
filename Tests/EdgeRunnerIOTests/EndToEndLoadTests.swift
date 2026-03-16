import Foundation
import Metal
import Testing
@testable import EdgeRunnerIO

@Suite("End-to-End Load Tests")
struct EndToEndLoadTests: Sendable {
    @Test("Load GGUF through EdgeRunnerModel and apply all Llama weights")
    func ggufToLoadedLlamaModel() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let url = temporaryFileURL(ext: "gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        try SyntheticEndToEndArtifacts.writeGGUF(to: url)

        let model = try await EdgeRunnerModel.load(from: url)
        guard let llama = model as? LlamaModel else {
            Issue.record("Expected EdgeRunnerModel.load(from:) to return LlamaModel for GGUF")
            return
        }

        #expect(llama.layers.count == 1)
        #expect(llama.config.vocabSize == 16)
        #expect(llama.loadedWeights.count == llama.parameterNames.count)
        #expect(llama.loadedWeights["embedding.weight"] != nil)
        #expect(llama.loadedWeights["layers.0.feedForward.down.weight"] != nil)
        #expect(llama.loadedWeights["finalNorm.weight"] != nil)
        #expect(llama.loadedWeights["lmHead.weight"] != nil)
    }

    @Test("Load SafeTensor through EdgeRunnerModel and apply all Llama weights")
    func safeTensorToLoadedLlamaModel() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let url = temporaryFileURL(ext: "safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try SyntheticEndToEndArtifacts.writeSafeTensor(to: url)

        let model = try await EdgeRunnerModel.load(from: url)
        guard let llama = model as? LlamaModel else {
            Issue.record("Expected EdgeRunnerModel.load(from:) to return LlamaModel for SafeTensor")
            return
        }

        #expect(llama.layers.count == 1)
        #expect(llama.config.embeddingDim == 8)
        #expect(llama.loadedWeights.count == llama.parameterNames.count)
        #expect(llama.loadedWeights["embedding.weight"] != nil)
        #expect(llama.loadedWeights["layers.0.attention.wq.weight"] != nil)
        #expect(llama.loadedWeights["lmHead.weight"] != nil)
    }

    @Test("Load NPZ through EdgeRunnerModel using single registered architecture")
    func npzToLoadedStubModel() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let url = temporaryFileURL(ext: "npz")
        defer { try? FileManager.default.removeItem(at: url) }
        try SyntheticEndToEndArtifacts.writeNPZ(to: url)

        let registry = ModelRegistry()
        registry.register(StubArchitectureFactory())

        let model = try await EdgeRunnerModel.load(from: url, registry: registry)
        #expect(model.parameterNames == ["stub.weight"])

        let stub = model as? StubModel
        #expect(stub != nil)
        #expect(stub?.weightsLoaded == true)
    }

    @Test("Format detection works for all supported formats")
    func formatDetection() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.gguf")) == .gguf)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.safetensors")) == .safetensors)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.npz")) == .npz)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.bin")) == nil)
    }

    @Test("Memory pressure integration: handler adjusts during model lifecycle")
    func memoryPressureIntegration() async {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0, .q4_k_m, .q4_0],
            evictBufferCacheOnPressure: true
        )
        let handler = MemoryPressureHandler(policy: policy)

        #expect(handler.currentQuantisation == .q8_0)

        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_k_m)

        handler.reset()
        #expect(handler.currentQuantisation == .q8_0)
    }

    @Test("Weight name mapping round-trip: GGUF names -> EdgeRunner names")
    func weightNameMappingRoundTrip() {
        let mappings: [(String, String)] = [
            ("token_embd.weight", "embedding.weight"),
            ("blk.0.attn_q.weight", "layers.0.attention.wq.weight"),
            ("blk.0.attn_k.weight", "layers.0.attention.wk.weight"),
            ("blk.0.attn_v.weight", "layers.0.attention.wv.weight"),
            ("blk.0.attn_output.weight", "layers.0.attention.wo.weight"),
            ("blk.0.ffn_gate.weight", "layers.0.feedForward.gate.weight"),
            ("blk.0.ffn_up.weight", "layers.0.feedForward.up.weight"),
            ("blk.0.ffn_down.weight", "layers.0.feedForward.down.weight"),
            ("blk.0.attn_norm.weight", "layers.0.attentionNorm.weight"),
            ("blk.0.ffn_norm.weight", "layers.0.ffnNorm.weight"),
            ("output_norm.weight", "finalNorm.weight"),
            ("output.weight", "lmHead.weight"),
        ]

        for (ggufName, expectedName) in mappings {
            #expect(
                LlamaWeightNameMapper.mapGGUFName(ggufName) == expectedName,
                "Expected \(ggufName) to map to \(expectedName)"
            )
        }
    }
}

private enum SyntheticEndToEndArtifacts {
    static let metadata: [String: MetadataValue] = [
        "general.architecture": "llama",
        "llama.embedding_length": 8,
        "llama.block_count": 1,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 1,
        "llama.vocab_size": 16,
        "llama.feed_forward_length": 12,
        "llama.rope.freq_base": 10_000.0,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
    ]

    static func writeGGUF(to url: URL) throws {
        try SyntheticGGUFFile.write(
            to: url,
            metadata: [
                ("general.architecture", .string("llama")),
                ("llama.attention.head_count", .uint32(2)),
                ("llama.attention.head_count_kv", .uint32(1)),
                ("llama.attention.layer_norm_rms_epsilon", .float32(1e-5)),
                ("llama.block_count", .uint32(1)),
                ("llama.embedding_length", .uint32(8)),
                ("llama.feed_forward_length", .uint32(12)),
                ("llama.rope.freq_base", .float32(10_000.0)),
                ("llama.vocab_size", .uint32(16)),
            ],
            tensors: ggufTensorSpecs()
        )
    }

    static func writeSafeTensor(to url: URL) throws {
        let blob = SyntheticSafeTensorFile.build(
            tensors: llamaParameterTensors(),
            metadata: metadata
        )
        try blob.write(to: url)
    }

    static func writeNPZ(to url: URL) throws {
        let npy = SyntheticNPZFile.buildNPY(
            dtype: .float32,
            shape: [1],
            data: floatData([1.0])
        )
        let archive = SyntheticNPZFile.buildNPZ(entries: [("stub.weight.npy", npy)])
        try archive.write(to: url)
    }

    private static func ggufTensorSpecs() -> [SyntheticGGUFFile.TensorSpec] {
        let names = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "output.weight",
        ]

        return names.enumerated().map { index, name in
            SyntheticGGUFFile.TensorSpec(
                name: name,
                dimensions: [1],
                type: .f32,
                values: [Float(index + 1)]
            )
        }
    }

    private static func llamaParameterTensors() -> [SyntheticSafeTensorFile.TensorSpec] {
        let config = LlamaConfig(
            embeddingDim: 8,
            layerCount: 1,
            headCount: 2,
            kvHeadCount: 1,
            vocabSize: 16,
            intermediateDim: 12,
            ropeFreqBase: 10_000.0,
            rmsNormEpsilon: 1e-5
        )
        let model = LlamaModel(config: config)

        return model.parameterNames.sorted().enumerated().map { index, name in
            SyntheticSafeTensorFile.TensorSpec(
                name: name,
                dtype: "F32",
                shape: [1],
                data: floatData([Float(index + 1)])
            )
        }
    }
}

private enum SyntheticGGUFFile {
    struct TensorSpec: Sendable {
        let name: String
        let dimensions: [UInt64]
        let type: GGUFTensorType
        let values: [Float]
    }

    enum MetadataSpec: Sendable {
        case string(String)
        case uint32(UInt32)
        case float32(Float)
    }

    static func write(
        to url: URL,
        metadata: [(String, MetadataSpec)],
        tensors: [TensorSpec]
    ) throws {
        var builder = Builder()
        builder.writeUInt32(ggufMagic)
        builder.writeUInt32(3)
        builder.writeUInt64(UInt64(tensors.count))
        builder.writeUInt64(UInt64(metadata.count))

        for (key, value) in metadata {
            builder.writeString(key)
            switch value {
            case .string(let string):
                builder.writeUInt32(GGUFMetadataValueType.string.rawValue)
                builder.writeString(string)
            case .uint32(let integer):
                builder.writeUInt32(GGUFMetadataValueType.uint32.rawValue)
                builder.writeUInt32(integer)
            case .float32(let float):
                builder.writeUInt32(GGUFMetadataValueType.float32.rawValue)
                builder.writeFloat32(float)
            }
        }

        var tensorData = Data()
        for tensor in tensors {
            builder.writeString(tensor.name)
            builder.writeUInt32(UInt32(tensor.dimensions.count))
            for dimension in tensor.dimensions {
                builder.writeUInt64(dimension)
            }
            builder.writeUInt32(tensor.type.rawValue)
            builder.writeUInt64(UInt64(tensorData.count))
            tensorData.append(floatData(tensor.values))
        }

        builder.pad(toMultipleOf: 32)
        builder.data.append(tensorData)
        try builder.data.write(to: url)
    }

    private struct Builder {
        var data = Data()

        mutating func writeUInt32(_ value: UInt32) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }

        mutating func writeUInt64(_ value: UInt64) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }

        mutating func writeFloat32(_ value: Float) {
            writeUInt32(value.bitPattern)
        }

        mutating func writeString(_ value: String) {
            let utf8 = Data(value.utf8)
            writeUInt64(UInt64(utf8.count))
            data.append(utf8)
        }

        mutating func pad(toMultipleOf alignment: Int) {
            let remainder = data.count % alignment
            if remainder != 0 {
                data.append(Data(repeating: 0, count: alignment - remainder))
            }
        }
    }
}

private enum SyntheticSafeTensorFile {
    struct TensorSpec: Sendable {
        let name: String
        let dtype: String
        let shape: [Int]
        let data: Data
    }

    static func build(
        tensors: [TensorSpec],
        metadata: [String: MetadataValue] = [:]
    ) -> Data {
        var entries: [String] = []
        var dataSection = Data()

        if !metadata.isEmpty {
            let metadataEntries = metadata.keys.sorted().map { key in
                let value = metadata[key] ?? .string("")
                return "\"\(escape(key))\":\(jsonFragment(for: value))"
            }
            entries.append("\"__metadata__\":{\(metadataEntries.joined(separator: ","))}")
        }

        for tensor in tensors {
            let begin = dataSection.count
            dataSection.append(tensor.data)
            let end = dataSection.count
            let shape = tensor.shape.map(String.init).joined(separator: ",")
            entries.append(
                """
                "\(escape(tensor.name))":{"dtype":"\(tensor.dtype)","shape":[\(shape)],"data_offsets":[\(begin),\(end)]}
                """
            )
        }

        let headerData = Data("{\(entries.sorted().joined(separator: ","))}".utf8)
        var headerSize = UInt64(headerData.count).littleEndian

        var blob = Data()
        withUnsafeBytes(of: &headerSize) { blob.append(contentsOf: $0) }
        blob.append(headerData)
        blob.append(dataSection)
        return blob
    }

    private static func jsonFragment(for value: MetadataValue) -> String {
        switch value {
        case .string(let string):
            "\"\(escape(string))\""
        case .int(let integer):
            String(integer)
        case .float(let float):
            String(float)
        case .bool(let bool):
            bool ? "true" : "false"
        case .array(let array):
            "[\(array.map(jsonFragment(for:)).joined(separator: ","))]"
        }
    }

    private static func escape(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}

private enum SyntheticNPZFile {
    static func buildNPY(dtype: NPYDtype, shape: [Int], data: Data) -> Data {
        let descr: String
        switch dtype {
        case .float32:
            descr = "<f4"
        case .float16:
            descr = "<f2"
        case .int8:
            descr = "|i1"
        }

        let shapeString = shape.map(String.init).joined(separator: ", ")
        let tupleSuffix = shape.count == 1 ? "," : ""
        let headerString =
            "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(shapeString)\(tupleSuffix)), }"

        let preambleLength = 10
        var paddedHeader = headerString
        let totalLength = preambleLength + paddedHeader.utf8.count + 1
        let padding = (64 - (totalLength % 64)) % 64
        paddedHeader += String(repeating: " ", count: padding) + "\n"

        var result = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        append(UInt16(paddedHeader.utf8.count), to: &result)
        result.append(Data(paddedHeader.utf8))
        result.append(data)
        return result
    }

    static func buildNPZ(entries: [(name: String, data: Data)]) -> Data {
        var archive = Data()
        var centralDirectory = Data()

        for entry in entries {
            let localHeaderOffset = UInt32(archive.count)
            let fileName = Data(entry.name.utf8)
            let uncompressedSize = UInt32(entry.data.count)

            append(UInt32(0x04034b50), to: &archive)
            append(UInt16(20), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt32(0), to: &archive)
            append(uncompressedSize, to: &archive)
            append(uncompressedSize, to: &archive)
            append(UInt16(fileName.count), to: &archive)
            append(UInt16(0), to: &archive)
            archive.append(fileName)
            archive.append(entry.data)

            append(UInt32(0x02014b50), to: &centralDirectory)
            append(UInt16(20), to: &centralDirectory)
            append(UInt16(20), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt32(0), to: &centralDirectory)
            append(uncompressedSize, to: &centralDirectory)
            append(uncompressedSize, to: &centralDirectory)
            append(UInt16(fileName.count), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt32(0), to: &centralDirectory)
            append(localHeaderOffset, to: &centralDirectory)
            centralDirectory.append(fileName)
        }

        let centralDirectoryOffset = UInt32(archive.count)
        archive.append(centralDirectory)

        append(UInt32(0x06054b50), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(entries.count), to: &archive)
        append(UInt16(entries.count), to: &archive)
        append(UInt32(centralDirectory.count), to: &archive)
        append(centralDirectoryOffset, to: &archive)
        append(UInt16(0), to: &archive)

        return archive
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}

private func temporaryFileURL(ext: String) -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension(ext)
}

private func floatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
}

private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}
