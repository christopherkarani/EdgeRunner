import Testing
@testable import EdgeRunner

@Suite("Gemma4RuntimeOptions")
struct Gemma4RuntimeOptionsTests {
    @Test("Snapshots hot-path Gemma environment flags once")
    func snapshotsHotPathEnvironmentFlags() {
        let options = Gemma4RuntimeOptions(environment: [
            "EDGERUNNER_GEMMA4_BUFFER_NATIVE_PRELUDE": "1",
            "EDGERUNNER_GEMMA4_GPU_LAYER_RUNNER": "0",
            "EDGERUNNER_GEMMA4_Q4_PACKED": "1",
            "EDGERUNNER_GEMMA4_Q4_LLAMA_STYLE": "1",
            "EDGERUNNER_GEMMA4_Q4_LLAMA_STYLE_DUAL": "1",
            "EDGERUNNER_GEMMA4_Q4_TILED": "1",
            "EDGERUNNER_GEMMA4_Q4_FUSED_GEGLU": "1",
            "EDGERUNNER_GEMMA4_Q4_2ROW": "1",
            "EDGERUNNER_GEMMA4_Q6_TOP1": "1",
            "EDGERUNNER_GEMMA4_Q6_PACKED": "1"
        ])

        #expect(options.useBufferNativePrelude)
        #expect(!options.useGPULayerRunner)
        #expect(options.useQ4Packed)
        #expect(options.useQ4LlamaStyle)
        #expect(options.useQ4LlamaStyleDual)
        #expect(options.useQ4Tiled)
        #expect(options.useQ4FusedGeGLU)
        #expect(options.useQ4TwoRow)
        #expect(options.useQ6Top1)
        #expect(options.useQ6Packed)
    }

    @Test("Defaults match current Gemma runtime behavior")
    func defaultsMatchCurrentRuntimeBehavior() {
        let options = Gemma4RuntimeOptions(environment: [:])

        #expect(!options.useBufferNativePrelude)
        #expect(options.useGPULayerRunner)
        #expect(!options.useQ4Packed)
        #expect(!options.useQ4LlamaStyle)
        #expect(!options.useQ4LlamaStyleDual)
        #expect(!options.useQ4Tiled)
        #expect(!options.useQ4FusedGeGLU)
        #expect(!options.useQ4TwoRow)
        #expect(!options.useQ6Top1)
        #expect(!options.useQ6Packed)
    }
}
