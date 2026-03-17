import Testing
@testable import EdgeRunnerIO

@Suite("Llama Block Tests")
struct LlamaBlockTests: Sendable {
    @Test("LlamaBlock has correct sub-module structure")
    func blockStructure() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 1,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let block = LlamaBlock(config: config, layerIndex: 0)
        #expect(block.attentionNorm.dim == 64)
        #expect(block.ffnNorm.dim == 64)
        #expect(block.attention.config == config)
        #expect(block.feedForward.hiddenDim == 128)
        #expect(block.parameterNames.count == 11)
    }

    @Test("Weight name mapping from GGUF tensor names")
    func weightNameMapping() {
        #expect(LlamaWeightNameMapper.mapGGUFName("blk.0.attn_q.weight") == "layers.0.attention.wq.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("blk.5.attn_norm.weight") == "layers.5.attentionNorm.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("blk.3.ffn_gate.weight") == "layers.3.feedForward.gate.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("blk.2.attn_q_norm.weight") == "layers.2.attention.qNorm.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("blk.2.attn_k_norm.weight") == "layers.2.attention.kNorm.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("output.weight") == "lmHead.weight")
        #expect(LlamaWeightNameMapper.mapGGUFName("token_embd.weight") == "embedding.weight")
    }
}
