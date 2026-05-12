import Metal
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerMetal

@Suite("Gemma4Scratch")
struct Gemma4ScratchTests {
    @Test("Allocates reusable GPU buffers for Gemma local and global layer shapes")
    func allocatesGemmaLayerScratchLayout() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let f32 = MemoryLayout<Float>.stride

        #expect(scratch.hiddenA.length == config.hiddenSize * f32)
        #expect(scratch.hiddenB.length == config.hiddenSize * f32)
        #expect(scratch.chunkHiddenA.length == Gemma4Scratch.prefillChunkCapacity * config.hiddenSize * f32)
        #expect(scratch.chunkHiddenB.length == Gemma4Scratch.prefillChunkCapacity * config.hiddenSize * f32)
        #expect(scratch.normed.length == config.hiddenSize * f32)
        #expect(scratch.attention.length == config.numAttentionHeads * config.globalHeadDim * f32)
        #expect(scratch.q.length == config.numAttentionHeads * config.globalHeadDim * f32)
        #expect(scratch.k.length == config.numKeyValueHeads * config.globalHeadDim * f32)
        #expect(scratch.v.length == config.numKeyValueHeads * config.globalHeadDim * f32)
        #expect(scratch.ffnGate.length == config.intermediateSize * f32)
        #expect(scratch.ffnUp.length == config.intermediateSize * f32)
        #expect(scratch.ffnActivated.length == config.intermediateSize * f32)
        #expect(scratch.ffnDown.length == config.hiddenSize * f32)
        #expect(scratch.pleInput.length == config.perLayerDim * f32)
        #expect(scratch.pleProjectionInput.length == config.numHiddenLayers * config.perLayerDim * f32)
        #expect(scratch.pleGate.length == config.perLayerDim * f32)
        #expect(scratch.pleActivated.length == config.perLayerDim * f32)
        #expect(scratch.pleProjection.length == config.hiddenSize * f32)
        #expect(scratch.logits.length == config.vocabSize * f32)
        #expect(scratch.top1PartialValues.length == GEMVKernel.q6KTop1PartialCount(rows: config.vocabSize) * f32)
        #expect(scratch.top1PartialIndices.length == GEMVKernel.q6KTop1PartialCount(rows: config.vocabSize) * MemoryLayout<UInt32>.stride)
        #expect(scratch.top1Token.length == MemoryLayout<UInt32>.stride)
    }

    @Test("Swaps hidden buffers without reallocating")
    func swapsHiddenBuffersWithoutAllocating() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let firstInput = scratch.currentHidden
        let firstOutput = scratch.nextHidden

        scratch.swapHiddenBuffers()

        #expect(scratch.currentHidden === firstOutput)
        #expect(scratch.nextHidden === firstInput)
    }

    @Test("Copies and reads hidden vectors through the active scratch buffer")
    func copiesAndReadsActiveHiddenBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let values = (0..<config.hiddenSize).map { Float(($0 % 23) - 11) / 17.0 }

        try scratch.copyHidden(values)
        let actual = try scratch.readHidden()

        #expect(actual.count == values.count)
        for index in actual.indices {
            #expect(actual[index] == values[index])
        }
    }

    @Test("Copies one hidden vector from a batched hidden array")
    func copiesHiddenSliceFromBatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let first = (0..<config.hiddenSize).map { Float($0 % 17) }
        let second = (0..<config.hiddenSize).map { Float(($0 % 23) + 100) }

        try scratch.copyHiddenBatch([first, second].flatMap { $0 }, tokenOffset: 1)
        let actual = try scratch.readHidden()

        #expect(actual.count == second.count)
        for index in second.indices {
            #expect(actual[index] == second[index])
        }
    }

    @Test("Copies PLE input vectors into reusable scratch storage")
    func copiesPLEInputBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let values = (0..<config.perLayerDim).map { Float(($0 % 13) - 6) / 9.0 }

        try scratch.copyPLEInput(values)

        let pointer = scratch.pleInput.contents().bindMemory(to: Float.self, capacity: config.perLayerDim)
        let actual = Array(UnsafeBufferPointer(start: pointer, count: config.perLayerDim))
        #expect(actual.count == values.count)
        for index in actual.indices {
            #expect(actual[index] == values[index])
        }
    }

    @Test("Copies FFN input vectors into reusable scratch storage")
    func copiesFFNInputBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)
        let values = (0..<config.hiddenSize).map { Float(($0 % 19) - 9) / 11.0 }

        try scratch.copyFFNInput(values)

        let pointer = scratch.ffnInput.contents().bindMemory(to: Float.self, capacity: config.hiddenSize)
        let actual = Array(UnsafeBufferPointer(start: pointer, count: config.hiddenSize))
        #expect(actual.count == values.count)
        for index in actual.indices {
            #expect(actual[index] == values[index])
        }
    }

    @Test("Rejects hidden vectors with wrong shape")
    func rejectsWrongHiddenShape() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)

        #expect(throws: Gemma4ScratchError.invalidHiddenShape(expected: config.hiddenSize, got: 3)) {
            try scratch.copyHidden([1, 2, 3])
        }
    }

    @Test("Rejects PLE input vectors with wrong shape")
    func rejectsWrongPLEInputShape() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)

        #expect(throws: Gemma4ScratchError.invalidPLEInputShape(expected: config.perLayerDim, got: 3)) {
            try scratch.copyPLEInput([1, 2, 3])
        }
    }

    @Test("Rejects FFN input vectors with wrong shape")
    func rejectsWrongFFNInputShape() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
        let scratch = try Gemma4Scratch(device: device, config: config)

        #expect(throws: Gemma4ScratchError.invalidFFNInputShape(expected: config.hiddenSize, got: 3)) {
            try scratch.copyFFNInput([1, 2, 3])
        }
    }
}
