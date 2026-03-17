import Testing
@testable import EspressoEdgeRunner

@Suite("EspressoTensorNameMapper")
struct EspressoTensorNameMapperTests {

    @Test("Global tensor names map correctly")
    func globalMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "token_embd.weight") == "weights/token_embedding.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output_norm.weight") == "weights/output_norm.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output.weight") == "weights/output.bin")
    }

    @Test("Layer attention weights map correctly")
    func layerAttentionMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_q.weight") == "weights/layers/0/wq.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.5.attn_k.weight") == "weights/layers/5/wk.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.31.attn_v.weight") == "weights/layers/31/wv.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_output.weight") == "weights/layers/0/wo.bin")
    }

    @Test("Layer FFN weights map correctly")
    func layerFFNMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_gate.weight") == "weights/layers/2/w1.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_down.weight") == "weights/layers/2/w2.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_up.weight") == "weights/layers/2/w3.bin")
    }

    @Test("Layer norms map correctly")
    func layerNormMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_norm.weight") == "weights/layers/0/attn_norm.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_norm.weight") == "weights/layers/0/ffn_norm.bin")
    }

    @Test("Unknown tensor name returns nil")
    func unknownReturnsNil() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "some.random.tensor") == nil)
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.unknown_suffix.weight") == nil)
    }

    @Test("GPT-2 matrix weights require transpose")
    func gpt2Transpose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_q.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_gate.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_down.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_up.weight", architecture: "gpt2") == true)
    }

    @Test("GPT-2 norms do NOT require transpose")
    func gpt2NormNoTranspose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_norm.weight", architecture: "gpt2") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_norm.weight", architecture: "gpt2") == false)
    }

    @Test("Non-GPT-2 architectures never require transpose")
    func llamaNoTranspose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_q.weight", architecture: "llama") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_gate.weight", architecture: "llama") == false)
    }
}
