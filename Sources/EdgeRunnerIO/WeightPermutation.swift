/// Utility for un-permuting Q/K weight matrices from GGUF interleaved (NeoX) layout
/// to natural contiguous layout suitable for NeoX RoPE kernels.
///
/// GGUF stores Llama/Qwen/Mistral Q and K weights with interleaved dimension ordering:
/// `[d0, d_half, d1, d_half+1, d2, d_half+2, ...]` (within each head).
///
/// The NeoX RoPE kernel expects the natural layout: `[d0, d1, d2, ..., d_half, d_half+1, ...]`.
/// This inverse interleaving converts from GGUF order to natural order.
public enum WeightPermutation {

    /// Inverse-interleave Q or K weight rows from GGUF interleaved order to natural order.
    ///
    /// - Parameters:
    ///   - weights: Flat row-major weight array of shape `[outDim, inDim]`
    ///              where `outDim = numHeads * headDim`.
    ///   - numHeads: Number of attention heads (Q heads or KV heads).
    ///   - headDim: Dimension per head.
    /// - Returns: Weight array with rows re-ordered so dimension pairs are contiguous.
    public static func inverseInterleaving(
        weights: [Float],
        numHeads: Int,
        headDim: Int
    ) -> [Float] {
        let outDim = numHeads * headDim
        let inDim = weights.count / outDim
        let halfDim = headDim / 2
        var result = [Float](repeating: 0, count: weights.count)

        for head in 0..<numHeads {
            for interleavedD in 0..<headDim {
                // GGUF interleaved: even indices map to first half, odd to second half
                let naturalD: Int
                if interleavedD % 2 == 0 {
                    naturalD = interleavedD / 2
                } else {
                    naturalD = halfDim + interleavedD / 2
                }

                let srcRow = head * headDim + interleavedD
                let dstRow = head * headDim + naturalD

                for col in 0..<inDim {
                    result[dstRow * inDim + col] = weights[srcRow * inDim + col]
                }
            }
        }

        return result
    }
}
