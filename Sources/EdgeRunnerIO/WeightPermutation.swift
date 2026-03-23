/// Utility for un-permuting Q/K weight matrices from GGUF interleaved (NeoX) layout
/// to natural contiguous layout suitable for NeoX RoPE kernels.
///
/// GGUF stores Llama/Qwen/Mistral Q and K weights with interleaved dimension ordering:
/// `[d0, d_half, d1, d_half+1, d2, d_half+2, ...]` (within each head).
///
/// The NeoX RoPE kernel expects the natural layout: `[d0, d1, d2, ..., d_half, d_half+1, ...]`.
/// This inverse interleaving converts from GGUF order to natural order.
public enum WeightPermutation {

    /// Re-order weight rows from dim-major-by-head layout to head-major-by-dim layout.
    ///
    /// Some converter/runtime boundaries can materialize rows as:
    /// `[dim0_head0, dim0_head1, ..., dim1_head0, dim1_head1, ...]`.
    /// The shared decode path expects:
    /// `[head0_dim0, head0_dim1, ..., head1_dim0, head1_dim1, ...]`.
    ///
    /// - Parameters:
    ///   - weights: Flat row-major weight array of shape `[outDim, inDim]`
    ///              where `outDim = numHeads * headDim`.
    ///   - numHeads: Number of heads represented in the output rows.
    ///   - headDim: Dimension per head.
    /// - Returns: Weight array with rows re-ordered into head-major layout.
    public static func dimMajorToHeadMajor(
        weights: [Float],
        numHeads: Int,
        headDim: Int
    ) -> [Float] {
        let outDim = numHeads * headDim
        let inDim = weights.count / outDim
        var result = [Float](repeating: 0, count: weights.count)

        for dimIndex in 0..<headDim {
            for head in 0..<numHeads {
                let srcRow = dimIndex * numHeads + head
                let dstRow = head * headDim + dimIndex
                for col in 0..<inDim {
                    result[dstRow * inDim + col] = weights[srcRow * inDim + col]
                }
            }
        }

        return result
    }

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

    /// Forward-interleave weight rows from natural order back to GGUF interleaved order.
    ///
    /// - Parameters:
    ///   - weights: Flat row-major weight array of shape `[outDim, inDim]`
    ///              where `outDim = numHeads * headDim`.
    ///   - numHeads: Number of attention heads.
    ///   - headDim: Dimension per head.
    /// - Returns: Weight array with rows re-ordered into interleaved GGUF order.
    public static func forwardInterleaving(
        weights: [Float],
        numHeads: Int,
        headDim: Int
    ) -> [Float] {
        let outDim = numHeads * headDim
        let inDim = weights.count / outDim
        let halfDim = headDim / 2
        var result = [Float](repeating: 0, count: weights.count)

        for head in 0..<numHeads {
            for naturalD in 0..<headDim {
                let interleavedD: Int
                if naturalD < halfDim {
                    interleavedD = naturalD * 2
                } else {
                    interleavedD = (naturalD - halfDim) * 2 + 1
                }

                let srcRow = head * headDim + naturalD
                let dstRow = head * headDim + interleavedD

                for col in 0..<inDim {
                    result[dstRow * inDim + col] = weights[srcRow * inDim + col]
                }
            }
        }

        return result
    }
}
