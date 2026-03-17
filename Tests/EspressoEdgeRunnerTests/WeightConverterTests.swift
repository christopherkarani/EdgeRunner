import Testing
import Foundation
import Metal
@testable import EspressoEdgeRunner
import EdgeRunnerIO

@Suite("WeightConverter")
struct WeightConverterTests {

    private func makeDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EspressoError.metalDeviceUnavailable
        }
        return device
    }

    private func makeTempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_test_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("Single f32 tensor round-trip produces valid BLOBFILE")
    func singleF32RoundTrip() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [4], name: "token_embd.weight"
        )

        var weightMap = WeightMap()
        weightMap["token_embd.weight"] = tensor

        let count = try await converter.convert(
            weightMap: weightMap, architecture: "llama", outputDirectory: outputDir
        )
        #expect(count == 1)

        let blobURL = outputDir.appendingPathComponent("weights/token_embedding.bin")
        let data = try Data(contentsOf: blobURL)
        #expect(data.count == 128 + 4 * 2) // header + 4 fp16 values
        #expect(data[64] == 0xEF) // magic check
    }

    @Test("Transpose applied for gpt2 matrix weight")
    func transposeForGPT2() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let floats: [Float] = [1, 2, 3, 4, 5, 6]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 3], name: "blk.0.attn_q.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.0.attn_q.weight",
            architecture: "gpt2", outputDirectory: outputDir
        )

        // Read back and verify transposed: [[1,4],[2,5],[3,6]]
        let blobURL = outputDir.appendingPathComponent("weights/layers/0/wq.bin")
        let data = try Data(contentsOf: blobURL)
        let payload = data.subdata(in: 128..<data.count)
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        for (i, exp) in expected.enumerated() {
            let lo = payload[i * 2]
            let hi = payload[i * 2 + 1]
            let bits = UInt16(lo) | (UInt16(hi) << 8)
            let recovered = Float(Float16(bitPattern: bits))
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("No transpose for llama architecture")
    func noTransposeForLlama() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1, 2, 3, 4, 5, 6]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 3], name: "blk.0.attn_q.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.0.attn_q.weight",
            architecture: "llama", outputDirectory: outputDir
        )

        let blobURL = outputDir.appendingPathComponent("weights/layers/0/wq.bin")
        let data = try Data(contentsOf: blobURL)
        let payload = data.subdata(in: 128..<data.count)
        let expected: [Float] = [1, 2, 3, 4, 5, 6]
        for (i, exp) in expected.enumerated() {
            let lo = payload[i * 2]
            let hi = payload[i * 2 + 1]
            let bits = UInt16(lo) | (UInt16(hi) << 8)
            let recovered = Float(Float16(bitPattern: bits))
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Directory structure created for nested paths")
    func directoryStructure() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1.0]
        let buffer = device.makeBuffer(bytes: floats, length: 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [1], name: "blk.5.ffn_gate.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.5.ffn_gate.weight",
            architecture: "llama", outputDirectory: outputDir
        )

        let layerDir = outputDir.appendingPathComponent("weights/layers/5")
        var isDir: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: layerDir.path, isDirectory: &isDir))
        #expect(isDir.boolValue)
    }

    @Test("Unmapped tensors are skipped in batch convert")
    func unmappedSkipped() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1.0]
        let buffer = device.makeBuffer(bytes: floats, length: 4, options: .storageModeShared)!

        var weightMap = WeightMap()
        weightMap["unknown.tensor.name"] = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [1], name: "unknown.tensor.name"
        )
        weightMap["token_embd.weight"] = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [1], name: "token_embd.weight"
        )

        let count = try await converter.convert(
            weightMap: weightMap, architecture: "llama", outputDirectory: outputDir
        )
        #expect(count == 1) // Only the mapped tensor
    }

    @Test("Transpose validates floats count matches shape")
    func transposeShapeMismatchThrows() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        // Shape says [2, 3] (6 elements) but buffer only has 4 floats worth of data
        // The shape [2,3] declares 6 elements, buffer has 16 bytes = 4 floats
        // DequantDispatcher will read 6 floats (24 bytes) which exceeds 16-byte buffer
        let buffer = device.makeBuffer(length: 16, options: .storageModeShared)!
        memset(buffer.contents(), 0, 16)
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 3], name: "blk.0.attn_q.weight"
        )

        do {
            try await converter.convertTensor(
                tensor, ggufName: "blk.0.attn_q.weight",
                architecture: "gpt2", outputDirectory: outputDir
            )
            Issue.record("Expected error to be thrown")
        } catch let error as EspressoError {
            // Should throw bufferOutOfBounds since shape requires 24 bytes but only 16 available
            if case .bufferOutOfBounds = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
