/// Configuration for a decoder-only transformer model.
public struct TransformerConfig: Sendable {
    public let hiddenDim: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let intermediateSize: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float

    public var headDim: Int { hiddenDim / numHeads }
    public var kvGroupSize: Int { numHeads / numKVHeads }

    public init(
        hiddenDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        intermediateSize: Int,
        numLayers: Int,
        vocabSize: Int,
        maxSeqLen: Int,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000.0
    ) {
        precondition(hiddenDim % numHeads == 0, "hiddenDim must be divisible by numHeads")
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.intermediateSize = intermediateSize
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
    }
}
