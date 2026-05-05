import Metal
import EdgeRunnerMetal

extension KVCache {
    /// Gemma 4 factory with dual head_dim and KV-share map.
    ///
    /// - Sliding-attention layers allocate with `config.headDim`
    ///   (Gemma 4 E4B: 256).
    /// - Global-attention layers allocate with `config.globalHeadDim`
    ///   (Gemma 4 E4B: 512).
    /// - Layers `i` with `config.kvSourceLayer(for: i) != i` alias the source
    ///   layer's `MTLBuffer` (`===` identity holds).
    ///
    /// - Parameters:
    ///   - device: Metal device used for buffer allocation.
    ///   - config: Gemma 4 hyperparameters (expects
    ///       `numHiddenLayers == layerTypes.count`).
    ///   - maxSeqLen: Ring-buffer length (tokens) per layer.
    ///   - compression: KV cache compression policy. Gemma 4's heterogeneous
    ///       KV layout does not yet support TurboQuant — requesting
    ///       `turboQuantBalanced` or `turboQuantAggressive` throws
    ///       ``KVCacheError/unsupportedStorage``. `disabled` and `automatic`
    ///       use dense float32 storage.
    /// - Returns: A `KVCache` configured for Gemma 4's dual head_dim + KV
    ///     sharing layout.
    /// - Throws: ``KVCacheError/unsupportedStorage`` when compression is
    ///     `turboQuantBalanced` or `turboQuantAggressive`.
    public static func gemma4(
        device: MTLDevice,
        config: Gemma4ModelConfig,
        maxSeqLen: Int,
        compression: KVCacheCompression = .disabled
    ) throws -> KVCache {
        let numLayers = config.numHiddenLayers
        var headDimByLayer: [Int] = []
        headDimByLayer.reserveCapacity(numLayers)
        var kvSourceLayers: [Int] = []
        kvSourceLayers.reserveCapacity(numLayers)

        for layer in 0..<numLayers {
            let dim: Int
            switch config.layerTypes[layer] {
            case .sliding:
                dim = config.headDim
            case .global:
                dim = config.globalHeadDim
            }
            headDimByLayer.append(dim)
            kvSourceLayers.append(config.kvSourceLayer(for: layer))
        }

        return try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numKVHeads: config.numKeyValueHeads,
            headDimByLayer: headDimByLayer,
            kvSourceLayers: kvSourceLayers,
            precision: try Self.resolveGemma4Precision(compression)
        )
    }

    private static func resolveGemma4Precision(
        _ compression: KVCacheCompression
    ) throws -> Precision {
        switch compression {
        case .disabled, .automatic:
            return .float32
        case .q8_0, .turboquantV2, .turboQuantBalanced, .turboQuantAggressive:
            // Gemma 4's heterogeneous KV layout (dual head_dim + shared
            // buffers) is incompatible with TurboQuant's homogeneous
            // packing. Fail loudly instead of silently falling back so
            // callers catch configuration mismatches at setup time.
            throw KVCacheError.unsupportedStorage
        }
    }
}
