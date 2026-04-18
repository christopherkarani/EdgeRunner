import Testing
@testable import EdgeRunner
@testable import EdgeRunnerIO

@Suite("Gemma4ModelConfig parsing")
struct Gemma4ModelConfigTests {
    @Test("Parses E4B hparams from GGUF metadata")
    func parsesE4BHparams() throws {
        let metadata = Gemma4ModelConfigTests.makeReferenceMetadata()
        let config = try Gemma4ModelConfig(metadata: metadata)

        #expect(config.numHiddenLayers == 42)
        #expect(config.hiddenSize == 2560)
        #expect(config.intermediateSize == 10240)
        #expect(config.numAttentionHeads == 8)
        #expect(config.numKeyValueHeads == 2)
        #expect(config.headDim == 256)
        #expect(config.globalHeadDim == 512)
        #expect(config.vocabSize == 262144)
        #expect(config.maxPositionEmbeddings == 131072)
        #expect(config.rmsNormEps == 1e-6)
        #expect(config.finalLogitSoftcapping == 30.0)
        #expect(config.perLayerDim == 256)
        #expect(config.perLayerVocabSize == 262144)
        #expect(config.numKVSharedLayers == 18)
        #expect(config.slidingWindow == 512)
        #expect(config.layerTypes.count == 42)

        let globalLayers = config.layerTypes.enumerated()
            .compactMap { $0.element == .global ? $0.offset : nil }
        #expect(globalLayers == [5, 11, 17, 23, 29, 35, 41])
    }

    @Test("KV share map routes layers 24..41 to nearest same-type predecessor")
    func kvShareMapIsCorrect() throws {
        let metadata = Gemma4ModelConfigTests.makeReferenceMetadata()
        let config = try Gemma4ModelConfig(metadata: metadata)

        #expect(config.kvSourceLayer(for: 0) == 0)
        #expect(config.kvSourceLayer(for: 23) == 23)
        #expect(config.kvSourceLayer(for: 24) == 22)  // sliding; 23 is sliding
        #expect(config.kvSourceLayer(for: 29) == 23)  // global; last global before was 23
        #expect(config.kvSourceLayer(for: 35) == 23)
        #expect(config.kvSourceLayer(for: 41) == 23)
    }

    static func makeReferenceMetadata() -> [String: GGUFMetadataValue] {
        [
            "general.architecture": .string("gemma4"),
            "gemma4.block_count": .uint32(42),
            "gemma4.embedding_length": .uint32(2560),
            "gemma4.feed_forward_length": .uint32(10240),
            "gemma4.attention.head_count": .uint32(8),
            "gemma4.attention.head_count_kv": .uint32(2),
            "gemma4.attention.key_length": .uint32(256),
            "gemma4.attention.value_length": .uint32(256),
            "gemma4.attention.key_length_global": .uint32(512),
            "gemma4.attention.value_length_global": .uint32(512),
            "gemma4.vocab_size": .uint32(262144),
            "gemma4.context_length": .uint32(131072),
            "gemma4.attention.layer_norm_rms_epsilon": .float32(1e-6),
            "gemma4.final_logit_softcapping": .float32(30.0),
            "gemma4.embedding_length_per_layer": .uint32(256),
            "gemma4.per_layer_vocab_size": .uint32(262144),
            "gemma4.attention.shared_kv_layers": .uint32(18),
            "gemma4.attention.sliding_window": .uint32(512),
            "gemma4.layer_types": .string(
                "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global"
            ),
        ]
    }
}
