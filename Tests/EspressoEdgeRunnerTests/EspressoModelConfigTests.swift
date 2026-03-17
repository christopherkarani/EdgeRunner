import Testing
@testable import EspressoEdgeRunner
import EdgeRunnerIO

@Suite("EspressoModelConfig")
struct EspressoModelConfigTests {

    private func makeModelConfig(arch: String, prefixed: Bool) -> ModelConfig {
        let prefix = prefixed ? "\(arch)." : ""
        return ModelConfig(
            architectureName: arch,
            metadata: [
                "\(prefix)embedding_length": .int(4096),
                "\(prefix)attention.head_count": .int(32),
                "\(prefix)attention.head_count_kv": .int(8),
                "\(prefix)block_count": .int(32),
                "\(prefix)feed_forward_length": .int(11008),
                "\(prefix)context_length": .int(2048),
                "\(prefix)attention.layer_norm_rms_epsilon": .float(1e-5),
            ]
        )
    }

    @Test("Init from prefixed ModelConfig metadata")
    func prefixedInit() throws {
        let config = try EspressoModelConfig(from: makeModelConfig(arch: "llama", prefixed: true))
        #expect(config.embeddingDim == 4096)
        #expect(config.headCount == 32)
        #expect(config.kvHeadCount == 8)
        #expect(config.blockCount == 32)
        #expect(config.feedForwardLength == 11008)
        #expect(config.contextLength == 2048)
        #expect(abs(config.rmsNormEpsilon - 1e-5) < 1e-9)
        #expect(config.architectureName == "llama")
    }

    @Test("Init from unprefixed ModelConfig metadata")
    func unprefixedFallback() throws {
        let config = try EspressoModelConfig(from: makeModelConfig(arch: "llama", prefixed: false))
        #expect(config.embeddingDim == 4096)
        #expect(config.headCount == 32)
    }

    @Test("Missing key throws configMissingKey")
    func missingKeyThrows() {
        let modelConfig = ModelConfig(architectureName: "llama", metadata: [:])
        #expect(throws: EspressoError.configMissingKey("embedding_length")) {
            try EspressoModelConfig(from: modelConfig)
        }
    }

    @Test("headDim is computed correctly")
    func headDimComputed() throws {
        let config = try EspressoModelConfig(from: makeModelConfig(arch: "llama", prefixed: true))
        #expect(config.headDim == 128) // 4096 / 32
    }

    @Test("Equatable conformance")
    func equatable() {
        let a = EspressoModelConfig(
            embeddingDim: 4096, headCount: 32, kvHeadCount: 8,
            blockCount: 32, feedForwardLength: 11008, contextLength: 2048,
            rmsNormEpsilon: 1e-5, architectureName: "llama"
        )
        let b = EspressoModelConfig(
            embeddingDim: 4096, headCount: 32, kvHeadCount: 8,
            blockCount: 32, feedForwardLength: 11008, contextLength: 2048,
            rmsNormEpsilon: 1e-5, architectureName: "llama"
        )
        #expect(a == b)
    }
}
