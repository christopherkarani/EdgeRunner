struct Gemma4LayerRuntimePlan: Sendable, Equatable {
    let layer: Int
    let pleInputOffset: Int
    let headDim: Int
    let qRows: Int
    let kvRows: Int
    let kvSourceLayer: Int
    let ropeTheta: Float
    let rotaryFactor: Float

    static func makePlans(
        config: Gemma4ModelConfig,
        globalRotaryFactor: Float?
    ) -> [Gemma4LayerRuntimePlan] {
        (0..<config.numHiddenLayers).map { layer in
            let isGlobal = config.layerTypes[layer] == .global
            let headDim = isGlobal ? config.globalHeadDim : config.headDim
            let rotaryFactor: Float
            if isGlobal, let globalRotaryFactor {
                rotaryFactor = globalRotaryFactor
            } else {
                let rotaryDimension = isGlobal ? config.globalRotaryDimension : config.localRotaryDimension
                rotaryFactor = Float(rotaryDimension) / Float(headDim)
            }

            return Gemma4LayerRuntimePlan(
                layer: layer,
                pleInputOffset: layer * config.perLayerDim * MemoryLayout<Float>.stride,
                headDim: headDim,
                qRows: config.numAttentionHeads * headDim,
                kvRows: config.numKeyValueHeads * headDim,
                kvSourceLayer: config.kvSourceLayer(for: layer),
                ropeTheta: isGlobal ? config.ropeThetaGlobal : config.ropeThetaLocal,
                rotaryFactor: rotaryFactor
            )
        }
    }
}
