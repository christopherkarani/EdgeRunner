import Testing
@testable import EdgeRunner

@Suite("Gemma4LayerRuntimePlan")
struct Gemma4LayerRuntimePlanTests {
    @Test("Precomputes per-layer decode dimensions and offsets")
    func precomputesPerLayerDecodeDimensionsAndOffsets() throws {
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let plans = Gemma4LayerRuntimePlan.makePlans(config: config, globalRotaryFactor: nil)

        #expect(plans.count == config.numHiddenLayers)

        let sliding = plans[0]
        #expect(sliding.layer == 0)
        #expect(sliding.pleInputOffset == 0)
        #expect(sliding.headDim == config.headDim)
        #expect(sliding.qRows == config.numAttentionHeads * config.headDim)
        #expect(sliding.kvRows == config.numKeyValueHeads * config.headDim)
        #expect(sliding.kvSourceLayer == 0)
        #expect(sliding.ropeTheta == config.ropeThetaLocal)
        #expect(sliding.rotaryFactor == Float(config.localRotaryDimension) / Float(config.headDim))

        let global = plans[5]
        #expect(global.layer == 5)
        #expect(global.pleInputOffset == 5 * config.perLayerDim * MemoryLayout<Float>.stride)
        #expect(global.headDim == config.globalHeadDim)
        #expect(global.qRows == config.numAttentionHeads * config.globalHeadDim)
        #expect(global.kvRows == config.numKeyValueHeads * config.globalHeadDim)
        #expect(global.kvSourceLayer == 5)
        #expect(global.ropeTheta == config.ropeThetaGlobal)
        #expect(global.rotaryFactor == Float(config.globalRotaryDimension) / Float(config.globalHeadDim))

        let shared = plans[29]
        #expect(shared.kvSourceLayer == 23)
        #expect(shared.pleInputOffset == 29 * config.perLayerDim * MemoryLayout<Float>.stride)
    }

    @Test("Uses rope-frequency rotary factor for global layers when provided")
    func usesRopeFrequencyRotaryFactorForGlobalLayers() throws {
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let plans = Gemma4LayerRuntimePlan.makePlans(config: config, globalRotaryFactor: 0.5)

        #expect(plans[0].rotaryFactor == Float(config.localRotaryDimension) / Float(config.headDim))
        #expect(plans[5].rotaryFactor == 0.5)
        #expect(plans[11].rotaryFactor == 0.5)
    }

    @Test("Describes invariant per-layer resource names")
    func describesInvariantPerLayerResourceNames() throws {
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let descriptors = Gemma4LayerResourceDescriptor.makeDescriptors(config: config)

        #expect(descriptors.count == config.numHiddenLayers)
        #expect(descriptors[0].layer == 0)
        #expect(descriptors[0].attnQ == "blk.0.attn_q.weight")
        #expect(descriptors[0].attnK == "blk.0.attn_k.weight")
        #expect(descriptors[0].attnV == "blk.0.attn_v.weight")
        #expect(descriptors[0].ffnDown == "blk.0.ffn_down.weight")
        #expect(descriptors[0].postPerLayerInputNorm == "blk.0.post_norm.weight")

        #expect(descriptors[29].layer == 29)
        #expect(descriptors[29].attnQ == "blk.29.attn_q.weight")
        #expect(descriptors[29].attnK == nil)
        #expect(descriptors[29].attnV == nil)
        #expect(descriptors[29].perLayerInputGate == "blk.29.inp_gate.weight")
        #expect(descriptors[29].layerOutputScale == "blk.29.layer_output_scale.weight")
    }
}
