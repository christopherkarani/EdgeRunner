public enum EspressoTensorNameMapper: Sendable {

    // MARK: - Layer suffix mappings

    private static let layerSuffixMap: [String: String] = [
        "attn_q.weight": "wq.bin",
        "attn_k.weight": "wk.bin",
        "attn_v.weight": "wv.bin",
        "attn_output.weight": "wo.bin",
        "ffn_gate.weight": "w1.bin",
        "ffn_down.weight": "w2.bin",
        "ffn_up.weight": "w3.bin",
        "attn_norm.weight": "attn_norm.bin",
        "ffn_norm.weight": "ffn_norm.bin",
    ]

    // MARK: - Global mappings

    private static let globalMap: [String: String] = [
        "token_embd.weight": "weights/token_embedding.bin",
        "output_norm.weight": "weights/output_norm.bin",
        "output.weight": "weights/output.bin",
    ]

    /// Maps a GGUF tensor name to its Espresso filesystem path, or `nil` if unmapped.
    public static func espressoPath(for ggufName: String) -> String? {
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
        guard let mapped = layerSuffixMap[suffix] else {
            return nil
        }

        return "weights/layers/\(layerIndex)/\(mapped)"
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

    /// Returns `true` if the tensor requires transposition (GPT-2 matrix weights only).
    public static func requiresTranspose(ggufName: String, architecture: String) -> Bool {
        guard architecture == "gpt2" else { return false }

        let parts = ggufName.split(separator: ".")
        guard parts.count >= 3, parts[0] == "blk" else { return false }

        let suffix = parts.dropFirst(2).joined(separator: ".")
        return transposeSuffixes.contains(suffix)
    }
}
