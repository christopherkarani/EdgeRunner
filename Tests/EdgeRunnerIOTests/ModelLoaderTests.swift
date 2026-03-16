import Foundation
import Testing
@testable import EdgeRunnerIO

@Suite("EdgeRunner Model Loader Tests")
struct ModelLoaderTests: Sendable {
    @Test("Detect GGUF format from extension")
    func detectGGUF() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.gguf")) == .gguf)
    }

    @Test("Detect SafeTensor format from extension")
    func detectSafeTensor() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.safetensors")) == .safetensors)
    }

    @Test("Detect NPZ format from extension")
    func detectNPZ() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.npz")) == .npz)
    }

    @Test("Unknown extension returns nil")
    func unknownExtension() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.bin")) == nil)
    }

    @Test("load(from:) loads weights into model and returns LoadableModel")
    func loadReturnsTypeErasedWithWeights() async throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_stub_\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try SyntheticGGUFBuilder.write(
            to: tmpURL,
            architecture: "stub",
            tensors: ["stub.weight": [1.0, 2.0, 3.0, 4.0]]
        )

        let registry = ModelRegistry()
        registry.register(StubArchitectureFactory())

        let model = try await EdgeRunnerModel.load(from: tmpURL, registry: registry)
        #expect(model.parameterNames == ["stub.weight"])

        let stub = model as? StubModel
        #expect(stub != nil)
        #expect(stub?.weightsLoaded == true)
    }
}

private struct SyntheticGGUFBuilder {
    private var data = Data()

    mutating func writeUInt32(_ value: UInt32) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    mutating func writeUInt64(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
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

    static func write(
        to url: URL,
        architecture: String,
        tensors: [String: [Float]]
    ) throws {
        var builder = SyntheticGGUFBuilder()
        builder.writeUInt32(0x46554747)
        builder.writeUInt32(3)
        builder.writeUInt64(UInt64(tensors.count))
        builder.writeUInt64(1)

        builder.writeString("general.architecture")
        builder.writeUInt32(GGUFMetadataValueType.string.rawValue)
        builder.writeString(architecture)

        var tensorData = Data()
        for (name, values) in tensors.sorted(by: { $0.key < $1.key }) {
            builder.writeString(name)
            builder.writeUInt32(1)
            builder.writeUInt64(UInt64(values.count))
            builder.writeUInt32(GGUFTensorType.f32.rawValue)
            builder.writeUInt64(UInt64(tensorData.count))

            values.forEach { value in
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { tensorData.append(contentsOf: $0) }
            }
        }

        builder.pad(toMultipleOf: 32)
        builder.data.append(tensorData)
        try builder.data.write(to: url)
    }
}
