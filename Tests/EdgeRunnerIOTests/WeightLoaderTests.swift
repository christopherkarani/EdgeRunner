import Metal
import Testing
@testable import EdgeRunnerIO

@Suite("WeightMap")
struct WeightMapTests {

    @Test func emptyWeightMapHasZeroCount() {
        let map = WeightMap()
        #expect(map.count == 0)
        #expect(map.tensorNames.isEmpty)
    }

    @Test func insertAndRetrieveTensorStorage() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }
        var map = WeightMap()
        let byteCount = 128 * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw WeightLoaderError.allocationFailed(byteCount: byteCount)
        }
        let storage = TensorStorage(
            buffer: buffer,
            dataType: .float32,
            shape: [4, 32],
            name: "model.layers.0.attention.wq.weight"
        )
        map["model.layers.0.attention.wq.weight"] = storage

        #expect(map.count == 1)
        #expect(map.tensorNames == ["model.layers.0.attention.wq.weight"])

        let retrieved = map["model.layers.0.attention.wq.weight"]
        #expect(retrieved != nil)
        #expect(retrieved?.shape == [4, 32])
        #expect(retrieved?.dataType == .float32)
    }

    @Test func subscriptReturnsNilForMissingKey() {
        let map = WeightMap()
        #expect(map["nonexistent"] == nil)
    }

    @Test func multipleInsertionsTrackAllNames() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }
        var map = WeightMap()
        for i in 0..<5 {
            let byteCount = 64 * MemoryLayout<Float>.stride
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                throw WeightLoaderError.allocationFailed(byteCount: byteCount)
            }
            let storage = TensorStorage(
                buffer: buffer,
                dataType: .float16,
                shape: [8, 8],
                name: "layer.\(i).weight"
            )
            map["layer.\(i).weight"] = storage
        }
        #expect(map.count == 5)
    }
}

@Suite("ModelConfig")
struct ModelConfigTests {

    @Test func initWithRequiredFields() {
        let config = ModelConfig(
            architectureName: "llama",
            metadata: [
                "llama.vocab_size": .int(32000),
                "llama.embedding_length": .int(4096),
                "llama.block_count": .int(32),
                "llama.attention.head_count": .int(32),
                "llama.attention.head_count_kv": .int(8),
                "llama.feed_forward_length": .int(11008),
                "llama.context_length": .int(4096),
                "llama.rope.freq_base": .float(500000.0),
                "llama.attention.layer_norm_rms_epsilon": .float(1e-5),
            ]
        )
        #expect(config.architectureName == "llama")
        #expect(config.int(forKey: "llama.vocab_size") == 32000)
        #expect(config.int(forKey: "llama.attention.head_count") == 32)
        #expect(config.int(forKey: "llama.attention.head_count_kv") == 8)
    }

    @Test func typedMetadataAccessors() {
        let config = ModelConfig(
            architectureName: "llama",
            metadata: [
                "general.architecture": .string("llama"),
                "llama.block_count": .int(16),
                "llama.rope.freq_base": .float(500000.0),
                "general.quantized": .bool(true),
            ]
        )
        #expect(config.string(forKey: "general.architecture") == "llama")
        #expect(config.int(forKey: "llama.block_count") == 16)
        #expect(config.float(forKey: "llama.rope.freq_base") == 500000.0)
        #expect(config.bool(forKey: "general.quantized") == true)
    }
}

@Suite("WeightLoaderProtocol")
struct WeightLoaderProtocolTests {

    @Test func protocolRequiresLoadMethod() {
        let _: any EdgeRunnerWeightLoader = MockWeightLoader()
    }

    @Test func protocolRequiresCanLoadMethod() {
        let loader: any EdgeRunnerWeightLoader = MockWeightLoader()
        #expect(loader.canLoad(url: URL(fileURLWithPath: "/tmp/test.mock")))
        #expect(!loader.canLoad(url: URL(fileURLWithPath: "/tmp/test.bin")))
    }
}

private struct MockWeightLoader: EdgeRunnerWeightLoader {
    let modelConfig = ModelConfig(
        architectureName: "llama",
        metadata: ["general.architecture": .string("llama")]
    )

    func canLoad(url: URL) -> Bool {
        url.pathExtension == "mock"
    }

    func load(from url: URL) async throws -> WeightMap {
        WeightMap()
    }
}
