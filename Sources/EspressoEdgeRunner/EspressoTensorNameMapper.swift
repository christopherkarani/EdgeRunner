public enum EspressoTensorNameMapper: Sendable {

    // MARK: - Architecture-specific layer suffix mappings

    private static let llamaLayerSuffixMap: [String: String] = [
        "attn_q.weight": "wq.bin",
        "attn_k.weight": "wk.bin",
        "attn_v.weight": "wv.bin",
        "attn_output.weight": "wo.bin",
        "ffn_gate.weight": "w1.bin",
        "ffn_down.weight": "w2.bin",
        "ffn_up.weight": "w3.bin",
        "attn_norm.weight": "rms_att.bin",
        "ffn_norm.weight": "rms_ffn.bin",
    ]

    private static let gpt2LayerSuffixMap: [String: String] = [
        "attn_q.weight": "wq.bin",
        "attn_k.weight": "wk.bin",
        "attn_v.weight": "wv.bin",
        "attn_output.weight": "wo.bin",
        "attn_q.bias": "bq.bin",
        "attn_k.bias": "bk.bin",
        "attn_v.bias": "bv.bin",
        "attn_output.bias": "bo.bin",
        "ffn_up.weight": "w1.bin",
        "ffn_down.weight": "w2.bin",
        "ffn_up.bias": "b1.bin",
        "ffn_down.bias": "b2.bin",
        "attn_norm.weight": "ln_1_gamma.bin",
        "attn_norm.bias": "ln_1_beta.bin",
        "ffn_norm.weight": "ln_2_gamma.bin",
        "ffn_norm.bias": "ln_2_beta.bin",
    ]

    // MARK: - Architecture-specific global mappings

    private static let gpt2GlobalMap: [String: String] = [
        "token_embd.weight": "embeddings/token.bin",
        "position_embd.weight": "embeddings/position.bin",
        "output_norm.weight": "final_norm_gamma.bin",
        "output_norm.bias": "final_norm_beta.bin",
        "output.weight": "lm_head.bin",
    ]

    private static let llamaGlobalMap: [String: String] = [
        "token_embd.weight": "embeddings/token.bin",
        "output_norm.weight": "rms_final.bin",
        "output.weight": "lm_head.bin",
    ]

    /// Maps a GGUF tensor name to its Espresso filesystem path, or `nil` if unmapped.
    /// Architecture-aware: GPT-2 uses ln_1_gamma/ln_2_gamma + bias tensors,
    /// LLaMA uses rms_att/rms_ffn with no biases.
    public static func espressoPath(for ggufName: String, architecture: String = "llama") -> String? {
        let globalMap = architecture == "gpt2" ? gpt2GlobalMap : llamaGlobalMap
        if let global = globalMap[ggufName] {
            return global
        }

        let parts = ggufName.split(separator: ".")
        guard parts.count >= 3,
              parts[0] == "blk",
              let layerIndex = Int(parts[1]) else {
            return nil
        }

        let suffix = parts.dropFirst(2).joined(separator: ".")
        let layerMap = architecture == "gpt2" ? gpt2LayerSuffixMap : llamaLayerSuffixMap
        guard let mapped = layerMap[suffix] else {
            return nil
        }

        return "layers/\(layerIndex)/\(mapped)"
    }

    /// Matrix weight suffixes that require transpose for GPT-2.
    private static let transposeSuffixes: Set<String> = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_down.weight",
        "ffn_up.weight",
    ]

    /// Global tensors that require transposition.
    private static let globalTransposeNames: Set<String> = [
        "output.weight",        // lm_head: [inDim, vocab] → [vocab, inDim]
        "token_embd.weight",    // embedding: [dim, vocab] → [vocab, dim]
    ]

    /// Returns `true` if the tensor requires transposition.
    /// GGUF stores ALL matrix weights in GGML convention [inDim, outDim],
    /// but ANE conv1x1 expects [outChannels, inChannels, 1, 1] and
    /// Espresso's CPU embedding expects [vocab, dim] row-major.
    /// All 2D matrix weights from GGUF need transposition regardless of architecture.
    /// 1D tensors (norms, biases) do NOT need transposition.
    public static func requiresTranspose(ggufName: String, architecture: String) -> Bool {
        // Check global names first
        if globalTransposeNames.contains(ggufName) {
            return true
        }

        // Check layer suffixes
        let parts = ggufName.split(separator: ".")
        guard parts.count >= 3, parts[0] == "blk" else { return false }

        let suffix = parts.dropFirst(2).joined(separator: ".")
        return transposeSuffixes.contains(suffix)
    }
}
