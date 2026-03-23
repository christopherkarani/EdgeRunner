import Testing
import Foundation
import Metal
@testable import EspressoEdgeRunner
import EdgeRunnerIO

@Suite("WeightConverter", .serialized)
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

    private func payloadFloats(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let payload = data.subdata(in: 128..<data.count)
        return stride(from: 0, to: payload.count, by: 2).map { index in
            let lo = payload[index]
            let hi = payload[index + 1]
            let bits = UInt16(lo) | (UInt16(hi) << 8)
            return Float(Float16(bitPattern: bits))
        }
    }

    private func float32Payload(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let scalarSize = MemoryLayout<UInt32>.stride
        precondition(data.count.isMultiple(of: scalarSize))
        return data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: scalarSize).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }

    private func withEnvironment<T>(
        _ key: String,
        value: String?,
        operation: () async throws -> T
    ) async throws -> T {
        let previousValue = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previousValue {
                setenv(key, previousValue, 1)
            } else {
                unsetenv(key)
            }
        }
        return try await operation()
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
            dataType: .float32, shape: [4], name: "output_norm.weight"
        )

        var weightMap = WeightMap()
        weightMap["output_norm.weight"] = tensor

        let count = try await converter.convert(
            weightMap: weightMap, architecture: "llama", outputDirectory: outputDir
        )
        #expect(count == 1)

        let blobURL = outputDir.appendingPathComponent("rms_final.bin")
        let data = try Data(contentsOf: blobURL)
        #expect(data.count == 128 + 4 * 2) // header + 4 fp16 values
        #expect(data[64] == 0xEF) // magic check
    }

    @Test("Q/K norm tensors are emitted when present")
    func qkNormsExportedWhenPresent() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let qNorm: [Float] = [1, 2, 3, 4]
        let kNorm: [Float] = [5, 6, 7, 8]
        let qBuffer = device.makeBuffer(bytes: qNorm, length: qNorm.count * 4, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: kNorm, length: kNorm.count * 4, options: .storageModeShared)!

        var weightMap = WeightMap()
        weightMap["blk.0.attn_q_norm.weight"] = TensorStorage(
            buffer: qBuffer, byteOffset: 0,
            dataType: .float32, shape: [qNorm.count], name: "blk.0.attn_q_norm.weight"
        )
        weightMap["blk.0.attn_k_norm.weight"] = TensorStorage(
            buffer: kBuffer, byteOffset: 0,
            dataType: .float32, shape: [kNorm.count], name: "blk.0.attn_k_norm.weight"
        )

        let count = try await converter.convert(
            weightMap: weightMap, architecture: "qwen3", outputDirectory: outputDir
        )
        #expect(count == 2)

        let qPath = outputDir.appendingPathComponent("layers/0/q_norm.bin")
        let kPath = outputDir.appendingPathComponent("layers/0/k_norm.bin")
        #expect(FileManager.default.fileExists(atPath: qPath.path))
        #expect(FileManager.default.fileExists(atPath: kPath.path))
        #expect(try payloadFloats(at: qPath).elementsEqual(qNorm, by: { abs($0 - $1) < 1e-2 }))
        #expect(try payloadFloats(at: kPath).elementsEqual(kNorm, by: { abs($0 - $1) < 1e-2 }))
    }

    @Test("Tied embeddings do not require lm_head.weight")
    func tiedEmbeddingsRemainOptional() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let embedding: [Float] = [1, 2, 3, 4, 5, 6]
        let buffer = device.makeBuffer(bytes: embedding, length: embedding.count * 4, options: .storageModeShared)!

        var weightMap = WeightMap()
        weightMap["token_embd.weight"] = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 3], name: "token_embd.weight"
        )

        let count = try await converter.convert(
            weightMap: weightMap, architecture: "llama", outputDirectory: outputDir
        )
        #expect(count == 1)

        let tokenPath = outputDir.appendingPathComponent("embeddings/token.bin")
        let lmHeadPath = outputDir.appendingPathComponent("lm_head.bin")
        #expect(FileManager.default.fileExists(atPath: tokenPath.path))
        #expect(!FileManager.default.fileExists(atPath: lmHeadPath.path))
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
        let blobURL = outputDir.appendingPathComponent("layers/0/wq.bin")
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Llama matrix weights stay in loader order by default")
    func llamaKeepsLoaderOrderByDefault() async throws {
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

        let blobURL = outputDir.appendingPathComponent("layers/0/wq.bin")
        let recovered = try payloadFloats(at: blobURL)
        #expect(recovered.elementsEqual(floats, by: { abs($0 - $1) < 1e-2 }))
    }

    @Test("Llama matrix transpose can still be forced for bisects")
    func llamaTransposeCanBeForced() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1, 2, 3, 4, 5, 6]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 3], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.forceLlamaMatrixTransposeEnvKey, value: "1") {
            try await converter.convertTensor(
                tensor, ggufName: "blk.0.attn_v.weight",
                architecture: "llama", outputDirectory: outputDir
            )
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Llama lm_head export also writes an exact float32 sidecar")
    func llamaLMHeadWritesExactFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1.0003, -2.0007, 3.14159, -4.4444]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 2], name: "output.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "output.weight",
            architecture: "llama",
            outputDirectory: outputDir
        )

        let blobURL = outputDir.appendingPathComponent("lm_head.bin")
        let sidecarURL = outputDir.appendingPathComponent("lm_head.float32.bin")
        let blobValues = try payloadFloats(at: blobURL)
        let sidecarValues = try float32Payload(at: sidecarURL)
        #expect(sidecarValues == floats)
        #expect(blobValues != floats)
    }

    @Test("Llama final RMS export also writes an exact float32 sidecar")
    func llamaFinalRMSWritesExactFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [0.12345679, 0.33333334, -0.9876543]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [floats.count], name: "output_norm.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "output_norm.weight",
            architecture: "llama",
            outputDirectory: outputDir
        )

        let blobURL = outputDir.appendingPathComponent("rms_final.bin")
        let sidecarURL = outputDir.appendingPathComponent("rms_final.float32.bin")
        let blobValues = try payloadFloats(at: blobURL)
        let sidecarValues = try float32Payload(at: sidecarURL)
        #expect(sidecarValues == floats)
        #expect(blobValues != floats)
    }

    @Test("Llama token embedding export also writes an exact float32 sidecar")
    func llamaTokenEmbeddingWritesExactFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [0.1111, -0.2222, 0.3333, -0.4444]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 2], name: "token_embd.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "token_embd.weight",
            architecture: "llama",
            outputDirectory: outputDir
        )

        let sidecarURL = outputDir.appendingPathComponent("embeddings/token.float32.bin")
        #expect(try float32Payload(at: sidecarURL) == floats)
    }

    @Test("Full sidecar policy writes Qwen layer tensor float32 sidecars")
    func fullSidecarPolicyWritesQwenLayerTensorFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [0.1111, -0.2222, 0.3333, -0.4444]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 2], name: "blk.0.ffn_down.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "blk.0.ffn_down.weight",
            architecture: "qwen3",
            outputDirectory: outputDir,
            exactFloat32SidecarPolicy: .full
        )

        let sidecarURL = outputDir.appendingPathComponent("layers/0/w2.float32.bin")
        #expect(try float32Payload(at: sidecarURL) == floats)
    }

    @Test("Essential sidecar policy skips Qwen layer tensor sidecars")
    func essentialSidecarPolicySkipsQwenLayerTensorFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [0.1111, -0.2222, 0.3333, -0.4444]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 2], name: "blk.0.ffn_down.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "blk.0.ffn_down.weight",
            architecture: "qwen3",
            outputDirectory: outputDir,
            exactFloat32SidecarPolicy: .essential
        )

        let sidecarURL = outputDir.appendingPathComponent("layers/0/w2.float32.bin")
        #expect(!FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    @Test("Selected sidecar policy writes requested Qwen layer tensor sidecars")
    func selectedSidecarPolicyWritesRequestedQwenLayerTensorFloat32Sidecar() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [0.1111, -0.2222, 0.3333, -0.4444]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 2], name: "blk.0.ffn_down.weight"
        )

        try await converter.convertTensor(
            tensor,
            ggufName: "blk.0.ffn_down.weight",
            architecture: "qwen3",
            outputDirectory: outputDir,
            exactFloat32SidecarPolicy: .selected(["blk.0.ffn_down.weight"])
        )

        let sidecarURL = outputDir.appendingPathComponent("layers/0/w2.float32.bin")
        #expect(try float32Payload(at: sidecarURL) == floats)
    }

    @Test("Llama Q and K weights remain in loader order by default when layout is provided")
    func llamaQKWeightsKeepLoaderOrderByDefault() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 4)
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!

        let qTensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_q.weight"
        )
        try await converter.convertTensor(
            qTensor, ggufName: "blk.0.attn_q.weight",
            architecture: "qwen3", outputDirectory: outputDir,
            llamaProjectionLayout: layout
        )
        let qExpected: [Float] = floats
        let qBlobURL = outputDir.appendingPathComponent("layers/0/wq.bin")
        for (i, recovered) in try payloadFloats(at: qBlobURL).enumerated() {
            let exp = qExpected[i]
            #expect(abs(recovered - exp) < 1e-2, "Q index \(i): \(recovered) vs \(exp)")
        }

        let kTensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_k.weight"
        )
        try await converter.convertTensor(
            kTensor, ggufName: "blk.0.attn_k.weight",
            architecture: "qwen3", outputDirectory: outputDir,
            llamaProjectionLayout: layout
        )
        let kBlobURL = outputDir.appendingPathComponent("layers/0/wk.bin")
        for (i, recovered) in try payloadFloats(at: kBlobURL).enumerated() {
            let exp = floats[i]
            #expect(abs(recovered - exp) < 1e-2, "K index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Llama Q and K inverse interleave can still be forced for bisects")
    func llamaQKWeightsCanBeInverseInterleavedWhenForced() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 4)
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!

        let qTensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_q.weight"
        )
        try await withEnvironment(WeightConverter.forceQKInverseInterleaveEnvKey, value: "1") {
            try await converter.convertTensor(
                qTensor, ggufName: "blk.0.attn_q.weight",
                architecture: "qwen3", outputDirectory: outputDir,
                llamaProjectionLayout: layout
            )
        }
        let expected: [Float] = [1, 2, 5, 6, 3, 4, 7, 8]
        let qBlobURL = outputDir.appendingPathComponent("layers/0/wq.bin")
        for (i, recovered) in try payloadFloats(at: qBlobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Q index \(i): \(recovered) vs \(exp)")
        }

        let kTensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_k.weight"
        )
        try await withEnvironment(WeightConverter.forceQKInverseInterleaveEnvKey, value: "1") {
            try await converter.convertTensor(
                kTensor, ggufName: "blk.0.attn_k.weight",
                architecture: "qwen3", outputDirectory: outputDir,
                llamaProjectionLayout: layout
            )
        }
        let kBlobURL = outputDir.appendingPathComponent("layers/0/wk.bin")
        for (i, recovered) in try payloadFloats(at: kBlobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "K index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Non-QK llama tensors remain in loader order when projection layout is provided")
    func nonQKTensorNotInverseInterleaved() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 4)
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_v.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.0.attn_v.weight",
            architecture: "qwen3", outputDirectory: outputDir,
            llamaProjectionLayout: layout
        )

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("V-weight inverse interleave gate defaults to disabled")
    func vWeightInverseInterleaveGateDefaultsDisabled() {
        #expect(!WeightConverter.isEnabled(nil))
        #expect(!WeightConverter.isEnabled("0"))
        #expect(!WeightConverter.isEnabled("false"))
        #expect(!WeightConverter.isEnabled("off"))
        #expect(WeightConverter.isEnabled("1"))
        #expect(WeightConverter.isEnabled(" true "))
        #expect(WeightConverter.isEnabled("YES"))
    }

    @Test("V-weight inverse interleave experiment applies only when enabled")
    func vWeightInverseInterleaveExperiment() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 4)
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 4], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.inverseInterleaveVWeightEnvKey, value: "1") {
            try await converter.convertTensor(
                tensor, ggufName: "blk.0.attn_v.weight",
                architecture: "qwen3", outputDirectory: outputDir,
                llamaProjectionLayout: layout
            )
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [1, 2, 5, 6, 3, 4, 7, 8]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("V-weight forward interleave experiment applies only when enabled")
    func vWeightForwardInterleaveExperiment() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 8)
        let floats: [Float] = Array(1...16).map(Float.init)
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 8], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.forwardInterleaveVWeightEnvKey, value: "1") {
            try await converter.convertTensor(
                tensor, ggufName: "blk.0.attn_v.weight",
                architecture: "qwen3", outputDirectory: outputDir,
                llamaProjectionLayout: layout
            )
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("V-weight dim-major to head-major experiment applies only when enabled")
    func vWeightDimMajorToHeadMajorExperiment() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 2, kvHeadCount: 2, headDim: 4)
        let floats: [Float] = [
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 8], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.dimMajorToHeadMajorVWeightEnvKey, value: "1") {
            try await converter.convertTensor(
                tensor, ggufName: "blk.0.attn_v.weight",
                architecture: "qwen3", outputDirectory: outputDir,
                llamaProjectionLayout: layout
            )
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [
            1, 2, 5, 6, 9, 10, 13, 14,
            3, 4, 7, 8, 11, 12, 15, 16,
        ]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("V-weight forward gate takes precedence over inverse gate")
    func vWeightForwardGatePrecedence() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 1, kvHeadCount: 1, headDim: 8)
        let floats: [Float] = Array(1...16).map(Float.init)
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 8], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.inverseInterleaveVWeightEnvKey, value: "1") {
            try await withEnvironment(WeightConverter.forwardInterleaveVWeightEnvKey, value: "1") {
                try await converter.convertTensor(
                    tensor, ggufName: "blk.0.attn_v.weight",
                    architecture: "qwen3", outputDirectory: outputDir,
                    llamaProjectionLayout: layout
                )
            }
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("V-weight dim-major gate takes precedence over interleave gates")
    func vWeightDimMajorGatePrecedence() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let layout = WeightConverter.LlamaProjectionLayout(qHeadCount: 2, kvHeadCount: 2, headDim: 4)
        let floats: [Float] = [
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [2, 8], name: "blk.0.attn_v.weight"
        )

        try await withEnvironment(WeightConverter.forwardInterleaveVWeightEnvKey, value: "1") {
            try await withEnvironment(WeightConverter.inverseInterleaveVWeightEnvKey, value: "1") {
                try await withEnvironment(WeightConverter.dimMajorToHeadMajorVWeightEnvKey, value: "1") {
                    try await converter.convertTensor(
                        tensor, ggufName: "blk.0.attn_v.weight",
                        architecture: "qwen3", outputDirectory: outputDir,
                        llamaProjectionLayout: layout
                    )
                }
            }
        }

        let blobURL = outputDir.appendingPathComponent("layers/0/wv.bin")
        let expected: [Float] = [
            1, 2, 5, 6, 9, 10, 13, 14,
            3, 4, 7, 8, 11, 12, 15, 16,
        ]
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
            #expect(abs(recovered - exp) < 1e-2, "Index \(i): \(recovered) vs \(exp)")
        }
    }

    @Test("Llama 1D norms are NOT transposed")
    func llamaNormsNotTransposed() async throws {
        let device = try makeDevice()
        let converter = try WeightConverter(device: device)
        let outputDir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let floats: [Float] = [1, 2, 3, 4]
        let buffer = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [4], name: "blk.0.attn_norm.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.0.attn_norm.weight",
            architecture: "llama", outputDirectory: outputDir
        )

        let blobURL = outputDir.appendingPathComponent("layers/0/rms_att.bin")
        let expected: [Float] = [1, 2, 3, 4]  // NOT transposed
        for (i, recovered) in try payloadFloats(at: blobURL).enumerated() {
            let exp = expected[i]
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
            dataType: .float32, shape: [1], name: "blk.5.ffn_norm.weight"
        )

        try await converter.convertTensor(
            tensor, ggufName: "blk.5.ffn_norm.weight",
            architecture: "llama", outputDirectory: outputDir
        )

        let layerDir = outputDir.appendingPathComponent("layers/5")
        // ffn_norm is a 1D tensor → no transpose, just written as BLOBFILE
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
        weightMap["output_norm.weight"] = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float32, shape: [1], name: "output_norm.weight"
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
