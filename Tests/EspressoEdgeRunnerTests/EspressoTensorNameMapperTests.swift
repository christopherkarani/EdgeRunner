import Testing
@testable import EspressoEdgeRunner

@Suite("EspressoTensorNameMapper")
struct EspressoTensorNameMapperTests {

    // MARK: - LLaMA mappings

    @Test("LLaMA global tensor names map correctly")
    func llamaGlobalMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "token_embd.weight", architecture: "llama") == "embeddings/token.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output_norm.weight", architecture: "llama") == "rms_final.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output.weight", architecture: "llama") == "lm_head.bin")
    }

    @Test("LLaMA layer attention weights map correctly")
    func llamaLayerAttentionMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_q.weight", architecture: "llama") == "layers/0/wq.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.5.attn_k.weight", architecture: "llama") == "layers/5/wk.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.31.attn_v.weight", architecture: "llama") == "layers/31/wv.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_output.weight", architecture: "llama") == "layers/0/wo.bin")
    }

    @Test("LLaMA layer FFN weights map correctly")
    func llamaLayerFFNMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_gate.weight", architecture: "llama") == "layers/2/w1.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_down.weight", architecture: "llama") == "layers/2/w2.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.2.ffn_up.weight", architecture: "llama") == "layers/2/w3.bin")
    }

    @Test("LLaMA layer norms map to rms_att/rms_ffn")
    func llamaLayerNormMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_norm.weight", architecture: "llama") == "layers/0/rms_att.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_norm.weight", architecture: "llama") == "layers/0/rms_ffn.bin")
    }

    @Test("LLaMA Q/K norm tensors map to stable artifact paths")
    func llamaLayerQKNormMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_q_norm.weight", architecture: "llama") == "layers/0/q_norm.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_k_norm.weight", architecture: "llama") == "layers/0/k_norm.bin")
    }

    @Test("Qwen-family models reuse llama-family Q/K norm artifact paths")
    func qwenLayerQKNormMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_q_norm.weight", architecture: "qwen3") == "layers/0/q_norm.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_k_norm.weight", architecture: "qwen3") == "layers/0/k_norm.bin")
    }

    // MARK: - GPT-2 mappings

    @Test("GPT-2 global tensor names map correctly")
    func gpt2GlobalMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "token_embd.weight", architecture: "gpt2") == "embeddings/token.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "position_embd.weight", architecture: "gpt2") == "embeddings/position.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output_norm.weight", architecture: "gpt2") == "final_norm_gamma.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output_norm.bias", architecture: "gpt2") == "final_norm_beta.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "output.weight", architecture: "gpt2") == "lm_head.bin")
    }

    @Test("GPT-2 layer norms map to ln_1_gamma/ln_2_gamma")
    func gpt2LayerNormMappings() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_norm.weight", architecture: "gpt2") == "layers/0/ln_1_gamma.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_norm.bias", architecture: "gpt2") == "layers/0/ln_1_beta.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_norm.weight", architecture: "gpt2") == "layers/0/ln_2_gamma.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_norm.bias", architecture: "gpt2") == "layers/0/ln_2_beta.bin")
    }

    @Test("GPT-2 layer bias tensors map correctly")
    func gpt2BiasMapping() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_q.bias", architecture: "gpt2") == "layers/0/bq.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_k.bias", architecture: "gpt2") == "layers/0/bk.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_v.bias", architecture: "gpt2") == "layers/0/bv.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_output.bias", architecture: "gpt2") == "layers/0/bo.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_up.bias", architecture: "gpt2") == "layers/0/b1.bin")
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.ffn_down.bias", architecture: "gpt2") == "layers/0/b2.bin")
    }

    // MARK: - Shared behavior

    @Test("Unknown tensor name returns nil")
    func unknownReturnsNil() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "some.random.tensor") == nil)
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.unknown_suffix.weight") == nil)
    }

    @Test("Default architecture is llama")
    func defaultArchitecture() {
        #expect(EspressoTensorNameMapper.espressoPath(for: "blk.0.attn_norm.weight") == "layers/0/rms_att.bin")
    }

    // MARK: - Transpose

    @Test("GPT-2 matrix weights require transpose")
    func gpt2Transpose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_q.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_gate.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_down.weight", architecture: "gpt2") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_up.weight", architecture: "gpt2") == true)
    }

    @Test("GPT-2 norms and biases do NOT require transpose")
    func gpt2NormNoTranspose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_norm.weight", architecture: "gpt2") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_norm.weight", architecture: "gpt2") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_q.bias", architecture: "gpt2") == false)
    }

    @Test("All architectures require transpose for matrix weights (GGML convention)")
    func llamaAlsoTransposes() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_q.weight", architecture: "llama") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_gate.weight", architecture: "llama") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_down.weight", architecture: "llama") == true)
    }

    @Test("Global matrix weights require transpose")
    func globalTranspose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "output.weight", architecture: "llama") == true)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "token_embd.weight", architecture: "llama") == true)
    }

    @Test("1D tensors do not require transpose")
    func normsNoTranspose() {
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.attn_norm.weight", architecture: "llama") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "blk.0.ffn_norm.weight", architecture: "llama") == false)
        #expect(EspressoTensorNameMapper.requiresTranspose(ggufName: "output_norm.weight", architecture: "llama") == false)
    }
}
