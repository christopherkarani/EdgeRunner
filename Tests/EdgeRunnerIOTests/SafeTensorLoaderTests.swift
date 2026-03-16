import Foundation
import Metal
import Testing
@testable import EdgeRunnerIO

private struct SyntheticSafeTensor: Sendable {
    struct TensorSpec: Sendable {
        let name: String
        let dtype: String
        let shape: [Int]
        let data: Data
    }

    static func build(
        tensors: [TensorSpec],
        metadata: [String: String] = [:]
    ) -> Data {
        var dataSection = Data()
        var entries: [String] = []

        if !metadata.isEmpty {
            let metadataEntries = metadata.keys.sorted().map { key in
                let value = metadata[key] ?? ""
                return "\"\(escape(key))\":\"\(escape(value))\""
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
        withUnsafeBytes(of: &headerSize) { pointer in
            blob.append(contentsOf: pointer)
        }
        blob.append(headerData)
        blob.append(dataSection)
        return blob
    }

    private static func escape(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}

@Suite("SafeTensor Loader Tests")
struct SafeTensorLoaderTests: Sendable {
    @Test("Parse JSON header from 8-byte size prefix")
    func parseHeader() throws {
        let floats: [Float] = [1, 2, 3, 4]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let blob = SyntheticSafeTensor.build(
            tensors: [
                .init(name: "weight", dtype: "F32", shape: [2, 2], data: floatData)
            ]
        )

        let header = try SafeTensorHeader.parse(from: blob)

        #expect(header.tensors.count == 1)
        #expect(header.tensors["weight"] != nil)
        #expect(header.dataOffset > 8)
    }

    @Test("Extract tensor metadata: dtype, shape, data_offsets")
    func extractMetadata() throws {
        let floats: [Float] = [1, 2, 3, 4, 5, 6]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let blob = SyntheticSafeTensor.build(
            tensors: [
                .init(name: "layer.0.weight", dtype: "F32", shape: [2, 3], data: floatData)
            ]
        )

        let header = try SafeTensorHeader.parse(from: blob)
        guard let meta = header.tensors["layer.0.weight"] else {
            Issue.record("Missing tensor metadata for layer.0.weight")
            return
        }
        #expect(meta.dtype == .float32)
        #expect(meta.shape == [2, 3])
        #expect(meta.dataOffsets.end - meta.dataOffsets.begin == 24)
    }

    @Test("Memory-map binary data section and read float32 values")
    func mmapDataSection() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let floats: [Float] = [1, 2, 3, 4]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let blob = SyntheticSafeTensor.build(
            tensors: [
                .init(name: "embed", dtype: "F32", shape: [4], data: floatData)
            ]
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try SafeTensorLoader(url: tmpURL)
        let storage = try loader.loadTensor(named: "embed")

        #expect(storage.shape == [4])
        #expect(storage.dataType == .float32)
        #expect(readFloat32(storage, count: 4) == floats)
    }

    @Test("Load multiple tensors from single file")
    func multipleTensors() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let w1Data = [Float](repeating: 1, count: 6).withUnsafeBufferPointer { Data(buffer: $0) }
        let w2Data = [Float](repeating: 2, count: 4).withUnsafeBufferPointer { Data(buffer: $0) }
        let blob = SyntheticSafeTensor.build(
            tensors: [
                .init(name: "attn.weight", dtype: "F32", shape: [2, 3], data: w1Data),
                .init(name: "ffn.weight", dtype: "F32", shape: [2, 2], data: w2Data),
            ],
            metadata: [
                "architecture": "llama",
                "hidden_size": "4096",
                "rope_theta": "500000.0",
                "use_bias": "false",
            ]
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try SafeTensorLoader(url: tmpURL)
        #expect(loader.tensorNames == ["attn.weight", "ffn.weight"])
        #expect(loader.modelConfig.architectureName == "llama")
        #expect(loader.modelConfig.int(forKey: "hidden_size") == 4096)
        #expect(loader.modelConfig.float(forKey: "rope_theta") == 500000)
        #expect(loader.modelConfig.bool(forKey: "use_bias") == false)

        let weightMap = try await loader.load(from: tmpURL)
        #expect(weightMap.count == 2)
        #expect(weightMap["attn.weight"]?.shape == [2, 3])
        #expect(weightMap["ffn.weight"]?.shape == [2, 2])
    }

    @Test("Float16 dtype parsing")
    func float16Dtype() throws {
        let blob = SyntheticSafeTensor.build(
            tensors: [
                .init(name: "half_tensor", dtype: "F16", shape: [4], data: Data(repeating: 0, count: 8))
            ]
        )

        let header = try SafeTensorHeader.parse(from: blob)
        guard let meta = header.tensors["half_tensor"] else {
            Issue.record("Missing tensor metadata for half_tensor")
            return
        }
        #expect(meta.dtype == .float16)
    }

    @Test("Invalid header throws descriptive error")
    func invalidHeader() {
        let invalid = Data([0, 0, 0, 0, 0, 0, 0, 0])
        #expect(throws: SafeTensorError.self) {
            _ = try SafeTensorHeader.parse(from: invalid)
        }
    }
}

private func readFloat32(_ storage: TensorStorage, count: Int) -> [Float] {
    let pointer = storage.buffer.contents()
        .advanced(by: storage.byteOffset)
        .bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}
