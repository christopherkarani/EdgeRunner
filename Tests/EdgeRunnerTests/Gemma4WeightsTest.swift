import Metal
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerIO

@Suite("Gemma4Weights binding")
struct Gemma4WeightsTests {
    @Test("Binds all required tensor handles from weight map")
    func bindsAllTensors() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let weightMap = try Gemma4WeightsTests.makeStubWeightMap(
            config: config,
            device: device
        )

        let weights = try Gemma4Weights(
            weightMap: weightMap,
            config: config,
            device: device
        )

        #expect(weights.blocks.count == 42)
        let block0 = weights.blocks[0]
        #expect(block0.attnK != nil && block0.attnV != nil)

        let block24 = weights.blocks[24]
        #expect(block24.attnK == nil && block24.attnV == nil)

        let block41 = weights.blocks[41]
        #expect(block41.attnK == nil && block41.attnV == nil)
    }

    @Test("Rejects weight map missing model-level tensor")
    func rejectsMissingTokenEmbedding() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        var weightMap = try Gemma4WeightsTests.makeStubWeightMap(
            config: config,
            device: device
        )
        weightMap["token_embd.weight"] = nil

        #expect(throws: Gemma4LoadError.missingTensor("token_embd.weight")) {
            try Gemma4Weights(
                weightMap: weightMap,
                config: config,
                device: device
            )
        }
    }

    @Test("Rejects weight map missing PLE tensors")
    func rejectsMissingPLE() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        var weightMap = try Gemma4WeightsTests.makeStubWeightMap(
            config: config,
            device: device
        )
        weightMap["per_layer_token_embd.weight"] = nil

        #expect(throws: Gemma4LoadError.missingPLETensor("per_layer_token_embd.weight")) {
            try Gemma4Weights(
                weightMap: weightMap,
                config: config,
                device: device
            )
        }
    }

    @Test("Rejects PLE embedding with unsupported quantization")
    func rejectsUnsupportedPLEQuant() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        var weightMap = try Gemma4WeightsTests.makeStubWeightMap(
            config: config,
            device: device
        )
        weightMap["per_layer_token_embd.weight"] = try Gemma4WeightsTests.makeTensorStorage(
            device: device,
            dataType: .q4_K,
            shape: [config.perLayerVocabSize, config.perLayerDim],
            name: "per_layer_token_embd.weight"
        )

        #expect(throws: Gemma4LoadError.self) {
            try Gemma4Weights(
                weightMap: weightMap,
                config: config,
                device: device
            )
        }
    }

    @Test("Rejects weight map missing layer attention weights")
    func rejectsMissingLayerAttention() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        var weightMap = try Gemma4WeightsTests.makeStubWeightMap(
            config: config,
            device: device
        )
        weightMap["blk.0.attn_q.weight"] = nil

        #expect(throws: Gemma4LoadError.missingTensor("blk.0.attn_q.weight")) {
            try Gemma4Weights(
                weightMap: weightMap,
                config: config,
                device: device
            )
        }
    }

    // MARK: - Helpers

    /// Builds a minimal valid weight map with zero-filled TensorStorage values for every
    /// tensor Gemma4Weights requires. KV tensors are only emitted for layers that own
    /// their K/V (i.e. layers below `numHiddenLayers - numKVSharedLayers`).
    static func makeStubWeightMap(
        config: Gemma4ModelConfig,
        device: MTLDevice
    ) throws -> WeightMap {
        var map = WeightMap()

        map["token_embd.weight"] = try makeTensorStorage(
            device: device,
            dataType: .q8_0,
            shape: [config.vocabSize, config.hiddenSize],
            name: "token_embd.weight"
        )
        map["output_norm.weight"] = try makeTensorStorage(
            device: device,
            dataType: .float32,
            shape: [config.hiddenSize],
            name: "output_norm.weight"
        )
        // per_layer_token_embd must be Q8_0 (or another allowed quant) — use Q8_0 so
        // the default stub passes the PLE quant gate.
        map["per_layer_token_embd.weight"] = try makeTensorStorage(
            device: device,
            dataType: .q8_0,
            shape: [config.perLayerVocabSize, config.perLayerDim],
            name: "per_layer_token_embd.weight"
        )
        map["per_layer_model_proj.weight"] = try makeTensorStorage(
            device: device,
            dataType: .float32,
            shape: [config.hiddenSize, config.perLayerDim],
            name: "per_layer_model_proj.weight"
        )
        map["per_layer_proj_norm.weight"] = try makeTensorStorage(
            device: device,
            dataType: .float32,
            shape: [config.perLayerDim],
            name: "per_layer_proj_norm.weight"
        )

        for layer in 0..<config.numHiddenLayers {
            let prefix = "blk.\(layer)"
            map["\(prefix).attn_norm.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize],
                name: "\(prefix).attn_norm.weight"
            )
            map["\(prefix).attn_q.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.numAttentionHeads * config.headDim, config.hiddenSize],
                name: "\(prefix).attn_q.weight"
            )
            if config.kvSourceLayer(for: layer) == layer {
                map["\(prefix).attn_k.weight"] = try makeTensorStorage(
                    device: device,
                    dataType: .float32,
                    shape: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
                    name: "\(prefix).attn_k.weight"
                )
                map["\(prefix).attn_v.weight"] = try makeTensorStorage(
                    device: device,
                    dataType: .float32,
                    shape: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
                    name: "\(prefix).attn_v.weight"
                )
            }
            map["\(prefix).attn_output.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize, config.numAttentionHeads * config.headDim],
                name: "\(prefix).attn_output.weight"
            )
            map["\(prefix).post_attention_norm.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize],
                name: "\(prefix).post_attention_norm.weight"
            )
            map["\(prefix).ffn_gate.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.intermediateSize, config.hiddenSize],
                name: "\(prefix).ffn_gate.weight"
            )
            map["\(prefix).ffn_up.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.intermediateSize, config.hiddenSize],
                name: "\(prefix).ffn_up.weight"
            )
            map["\(prefix).ffn_down.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize, config.intermediateSize],
                name: "\(prefix).ffn_down.weight"
            )
            map["\(prefix).post_ffw_norm.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize],
                name: "\(prefix).post_ffw_norm.weight"
            )
            map["\(prefix).inp_gate.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize, config.perLayerDim],
                name: "\(prefix).inp_gate.weight"
            )
            map["\(prefix).proj.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize, config.perLayerDim],
                name: "\(prefix).proj.weight"
            )
            map["\(prefix).post_norm.weight"] = try makeTensorStorage(
                device: device,
                dataType: .float32,
                shape: [config.hiddenSize],
                name: "\(prefix).post_norm.weight"
            )
        }

        return map
    }

    static func makeTensorStorage(
        device: MTLDevice,
        dataType: TensorDataType,
        shape: [Int],
        name: String
    ) throws -> TensorStorage {
        // One byte per element is enough for stub zero-filled storage — the tests only
        // read dataType/shape/name metadata, never interpret the buffer contents.
        let byteCount = max(1, shape.reduce(1, *))
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw WeightLoaderError.allocationFailed(byteCount: byteCount)
        }
        return TensorStorage(
            buffer: buffer,
            dataType: dataType,
            shape: shape,
            name: name
        )
    }
}
