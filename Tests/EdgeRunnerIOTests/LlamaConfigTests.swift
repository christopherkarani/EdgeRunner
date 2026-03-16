import Foundation
import Testing
@testable import EdgeRunnerIO

@Suite("Llama Config Tests")
struct LlamaConfigTests: Sendable {
    @Test("Parse LlamaConfig from GGUF metadata")
    func parseFromGGUF() throws {
        let metadata: [String: MetadataValue] = [
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.vocab_size": 128256,
            "llama.feed_forward_length": 14336,
            "llama.rope.freq_base": 500000.0,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
        ]

        let config = try LlamaConfig(fromGGUFMetadata: metadata)
        #expect(config.embeddingDim == 4096)
        #expect(config.layerCount == 32)
        #expect(config.headCount == 32)
        #expect(config.kvHeadCount == 8)
        #expect(config.vocabSize == 128256)
        #expect(config.intermediateDim == 14336)
        #expect(config.ropeFreqBase == 500000.0)
        #expect(abs(config.rmsNormEpsilon - 1e-5) < 1e-7)
    }

    @Test("Computed properties: headDim, GQA ratio")
    func computedProperties() {
        let config = LlamaConfig(
            embeddingDim: 4096,
            layerCount: 32,
            headCount: 32,
            kvHeadCount: 8,
            vocabSize: 128256,
            intermediateDim: 14336,
            ropeFreqBase: 500000.0,
            rmsNormEpsilon: 1e-5
        )

        #expect(config.headDim == 128)
        #expect(config.gqaRatio == 4)
    }

    @Test("Missing metadata key throws descriptive error")
    func missingKey() {
        let metadata: [String: MetadataValue] = [
            "llama.embedding_length": 4096,
        ]

        #expect(throws: LlamaConfigError.self) {
            _ = try LlamaConfig(fromGGUFMetadata: metadata)
        }
    }
}
