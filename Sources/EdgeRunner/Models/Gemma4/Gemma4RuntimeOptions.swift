struct Gemma4RuntimeOptions: Sendable, Equatable {
    let useBufferNativePrelude: Bool
    let useGPULayerRunner: Bool
    let useQ4Packed: Bool
    let useQ4LlamaStyle: Bool
    let useQ4LlamaStyleDual: Bool
    let useQ4Tiled: Bool
    let useQ4FusedGeGLU: Bool
    let useQ4TwoRow: Bool
    let useQ6Top1: Bool
    let useQ6Packed: Bool

    init(environment: [String: String]) {
        self.useBufferNativePrelude = environment["EDGERUNNER_GEMMA4_BUFFER_NATIVE_PRELUDE"] == "1"
        self.useGPULayerRunner = environment["EDGERUNNER_GEMMA4_GPU_LAYER_RUNNER"] != "0"
        self.useQ4Packed = environment["EDGERUNNER_GEMMA4_Q4_PACKED"] == "1"
        self.useQ4LlamaStyle = environment["EDGERUNNER_GEMMA4_Q4_LLAMA_STYLE"] == "1"
        self.useQ4LlamaStyleDual = environment["EDGERUNNER_GEMMA4_Q4_LLAMA_STYLE_DUAL"] == "1"
        self.useQ4Tiled = environment["EDGERUNNER_GEMMA4_Q4_TILED"] == "1"
        self.useQ4FusedGeGLU = environment["EDGERUNNER_GEMMA4_Q4_FUSED_GEGLU"] == "1"
        self.useQ4TwoRow = environment["EDGERUNNER_GEMMA4_Q4_2ROW"] == "1"
        self.useQ6Top1 = environment["EDGERUNNER_GEMMA4_Q6_TOP1"] == "1"
        self.useQ6Packed = environment["EDGERUNNER_GEMMA4_Q6_PACKED"] == "1"
    }
}
