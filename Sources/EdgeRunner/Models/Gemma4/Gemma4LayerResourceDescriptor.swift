struct Gemma4LayerResourceDescriptor: Sendable, Equatable {
    let layer: Int
    let inputNorm: String
    let attnQ: String
    let attnK: String?
    let attnV: String?
    let attnO: String
    let attnQNorm: String
    let attnKNorm: String
    let postAttentionNorm: String
    let ffnNorm: String
    let ffnGate: String
    let ffnUp: String
    let ffnDown: String
    let postFFNNorm: String
    let perLayerInputGate: String
    let perLayerProjection: String
    let postPerLayerInputNorm: String
    let layerOutputScale: String

    static func makeDescriptors(config: Gemma4ModelConfig) -> [Gemma4LayerResourceDescriptor] {
        (0..<config.numHiddenLayers).map { layer in
            let prefix = "blk.\(layer)"
            let ownsKV = config.kvSourceLayer(for: layer) == layer
            return Gemma4LayerResourceDescriptor(
                layer: layer,
                inputNorm: "\(prefix).attn_norm.weight",
                attnQ: "\(prefix).attn_q.weight",
                attnK: ownsKV ? "\(prefix).attn_k.weight" : nil,
                attnV: ownsKV ? "\(prefix).attn_v.weight" : nil,
                attnO: "\(prefix).attn_output.weight",
                attnQNorm: "\(prefix).attn_q_norm.weight",
                attnKNorm: "\(prefix).attn_k_norm.weight",
                postAttentionNorm: "\(prefix).post_attention_norm.weight",
                ffnNorm: "\(prefix).ffn_norm.weight",
                ffnGate: "\(prefix).ffn_gate.weight",
                ffnUp: "\(prefix).ffn_up.weight",
                ffnDown: "\(prefix).ffn_down.weight",
                postFFNNorm: "\(prefix).post_ffw_norm.weight",
                perLayerInputGate: "\(prefix).inp_gate.weight",
                perLayerProjection: "\(prefix).proj.weight",
                postPerLayerInputNorm: "\(prefix).post_norm.weight",
                layerOutputScale: "\(prefix).layer_output_scale.weight"
            )
        }
    }
}
