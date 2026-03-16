/// Configuration for GPT-2 model variants.
public struct GPT2Config: Sendable {
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numHeads: Int
    public let hiddenDim: Int
    public let layerNormEps: Float

    public var headDim: Int { hiddenDim / numHeads }
    public var intermediateSize: Int { hiddenDim * 4 }

    public init(
        vocabSize: Int = 50_257,
        maxSeqLen: Int = 1_024,
        numLayers: Int = 12,
        numHeads: Int = 12,
        hiddenDim: Int = 768,
        layerNormEps: Float = 1e-5
    ) {
        precondition(hiddenDim % numHeads == 0)
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.hiddenDim = hiddenDim
        self.layerNormEps = layerNormEps
    }
}
