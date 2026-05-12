import Testing
@testable import EdgeRunner

@Suite("Gemma4PrefillChunkPlan")
struct Gemma4PrefillChunkPlanTests {
    @Test("Splits prompt tokens into fixed-size chunks with absolute positions")
    func splitsPromptTokensIntoChunks() throws {
        let chunks = try Gemma4PrefillChunkPlan.makeChunks(
            tokenCount: 10,
            startPosition: 128,
            chunkSize: 4,
            numLayers: 42,
            perLayerDim: 256
        )

        #expect(chunks.map(\.tokenStartIndex) == [0, 4, 8])
        #expect(chunks.map(\.tokenCount) == [4, 4, 2])
        #expect(chunks.map(\.startPosition) == [128, 132, 136])
        #expect(chunks[0].position(forTokenOffset: 3) == 131)
        #expect(chunks[2].position(forTokenOffset: 1) == 137)
    }

    @Test("Computes PLE byte offsets in batch-layer-feature order")
    func computesPLEInputByteOffsets() throws {
        let chunks = try Gemma4PrefillChunkPlan.makeChunks(
            tokenCount: 6,
            startPosition: 0,
            chunkSize: 4,
            numLayers: 3,
            perLayerDim: 5
        )

        let chunk = try #require(chunks.first)
        #expect(chunk.perLayerInputElementCount == 4 * 3 * 5)
        #expect(chunk.pleInputByteOffset(tokenOffset: 0, layer: 0) == 0)
        #expect(chunk.pleInputByteOffset(tokenOffset: 0, layer: 2) == 10 * MemoryLayout<Float>.stride)
        #expect(chunk.pleInputByteOffset(tokenOffset: 1, layer: 0) == 15 * MemoryLayout<Float>.stride)
        #expect(chunk.pleInputByteOffset(tokenOffset: 3, layer: 2) == 55 * MemoryLayout<Float>.stride)
    }

    @Test("Rejects invalid chunk dimensions")
    func rejectsInvalidDimensions() throws {
        #expect(throws: Gemma4PrefillChunkPlan.Error.self) {
            _ = try Gemma4PrefillChunkPlan.makeChunks(
                tokenCount: 1,
                startPosition: 0,
                chunkSize: 0,
                numLayers: 42,
                perLayerDim: 256
            )
        }

        #expect(throws: Gemma4PrefillChunkPlan.Error.self) {
            _ = try Gemma4PrefillChunkPlan.makeChunks(
                tokenCount: 1,
                startPosition: -1,
                chunkSize: 4,
                numLayers: 42,
                perLayerDim: 256
            )
        }
    }
}
