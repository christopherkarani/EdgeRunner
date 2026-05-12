struct Gemma4PrefillChunkPlan: Sendable, Equatable {
    enum Error: Swift.Error, Equatable {
        case invalidDimensions
    }

    let tokenStartIndex: Int
    let tokenCount: Int
    let startPosition: Int
    let numLayers: Int
    let perLayerDim: Int

    var perLayerInputElementCount: Int {
        tokenCount * numLayers * perLayerDim
    }

    func position(forTokenOffset tokenOffset: Int) -> Int {
        startPosition + tokenOffset
    }

    func pleInputByteOffset(tokenOffset: Int, layer: Int) -> Int {
        Self.pleInputByteOffset(
            tokenOffset: tokenOffset,
            layer: layer,
            numLayers: numLayers,
            perLayerDim: perLayerDim
        )
    }

    static func pleInputElementOffset(
        tokenOffset: Int,
        layer: Int,
        numLayers: Int,
        perLayerDim: Int
    ) -> Int {
        (tokenOffset * numLayers + layer) * perLayerDim
    }

    static func pleInputByteOffset(
        tokenOffset: Int,
        layer: Int,
        numLayers: Int,
        perLayerDim: Int
    ) -> Int {
        pleInputElementOffset(
            tokenOffset: tokenOffset,
            layer: layer,
            numLayers: numLayers,
            perLayerDim: perLayerDim
        ) * MemoryLayout<Float>.stride
    }

    static func makeChunks(
        tokenCount: Int,
        startPosition: Int,
        chunkSize: Int,
        numLayers: Int,
        perLayerDim: Int
    ) throws -> [Gemma4PrefillChunkPlan] {
        guard tokenCount >= 0,
              startPosition >= 0,
              chunkSize > 0,
              numLayers > 0,
              perLayerDim > 0 else {
            throw Error.invalidDimensions
        }

        var chunks: [Gemma4PrefillChunkPlan] = []
        var tokenStartIndex = 0
        while tokenStartIndex < tokenCount {
            let count = min(chunkSize, tokenCount - tokenStartIndex)
            chunks.append(
                Gemma4PrefillChunkPlan(
                    tokenStartIndex: tokenStartIndex,
                    tokenCount: count,
                    startPosition: startPosition + tokenStartIndex,
                    numLayers: numLayers,
                    perLayerDim: perLayerDim
                )
            )
            tokenStartIndex += count
        }
        return chunks
    }
}
