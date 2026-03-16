import Testing
@testable import EdgeRunnerIO

@Suite("Llama Model Tests")
struct LlamaModelTests: Sendable {
    @Test("LlamaModel initialises with correct layer count")
    func modelInit() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 4,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        #expect(model.layers.count == 4)
        #expect(model.config.vocabSize == 100)
    }

    @Test("LlamaModel conforms to LoadableModel")
    func conformance() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 1,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        let loadable: any LoadableModel = model
        #expect(loadable.parameterNames.contains("embedding.weight"))
        #expect(loadable.parameterNames.contains("lmHead.weight"))
    }

    @Test("Weight name list covers all parameters")
    func weightNames() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 2,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        let names = model.parameterNames
        #expect(names.contains("embedding.weight"))
        #expect(names.contains("lmHead.weight"))
        #expect(names.contains("finalNorm.weight"))
        #expect(names.contains("layers.0.attention.wq.weight"))
        #expect(names.contains("layers.0.attention.wk.weight"))
        #expect(names.contains("layers.0.attention.wv.weight"))
        #expect(names.contains("layers.0.attention.wo.weight"))
        #expect(names.contains("layers.0.feedForward.gate.weight"))
        #expect(names.contains("layers.0.feedForward.up.weight"))
        #expect(names.contains("layers.0.feedForward.down.weight"))
        #expect(names.contains("layers.1.attention.wq.weight"))
    }
}
