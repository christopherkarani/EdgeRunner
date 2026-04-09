import Metal
import Testing
@testable import EdgeRunnerMetal

@Suite("TurboQuant Attention", .serialized)
struct TurboQuantAttentionTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    private var runtimeKeyPreset: TurboQuantPreset { TurboQuantV2Contract.keyPreset }
    private var runtimeValuePreset: TurboQuantPreset { TurboQuantV2Contract.valuePreset }

    @Test
    func contractPresetOverridesFollowEnvironment() throws {
        #expect(TurboQuantV2Contract.keyType == .turbo3)
        #expect(TurboQuantV2Contract.valueType == .turbo3)
        #expect(TurboQuantV2Contract.keyPreset == .turbo3)
        #expect(TurboQuantV2Contract.valuePreset == .turbo3)
        #expect(TurboQuantV2Contract.keyCacheType(forLayer: 0, layerCount: 28) == .turbo3)
        #expect(TurboQuantV2Contract.valueCacheType(forLayer: 0, layerCount: 28) == .turbo3)
        #expect(TurboQuantV2Contract.innerQ == .disabled)

        try withEnv("EDGERUNNER_TURBOQUANT_KEY_TYPE", value: "turbo4") {
            #expect(TurboQuantV2Contract.keyType == .turbo4)
            #expect(TurboQuantV2Contract.keyPreset == .turbo4)
            let keyLayout = try TurboQuantV2Contract.makeKeyLayout()
            #expect(keyLayout.preset == .turbo4)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_VALUE_POLICY", value: "boundaryTurbo4") {
            #expect(TurboQuantV2Contract.valuePolicy == .boundaryTurbo4)
            #expect(TurboQuantV2Contract.valuePreset(forLayer: 0, layerCount: 28) == .turbo4)
            #expect(TurboQuantV2Contract.valuePreset(forLayer: 13, layerCount: 28) == TurboQuantV2Contract.valuePreset)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_INNERQ", value: "1") {
            try withEnv("EDGERUNNER_TURBOQUANT_INNERQ_SAMPLES", value: "32") {
                try withEnv("EDGERUNNER_TURBOQUANT_INNERQ_STRENGTH", value: "0.75") {
                    #expect(TurboQuantV2Contract.innerQ.enabled)
                    #expect(TurboQuantV2Contract.innerQ.calibrationSampleCount == 32)
                    #expect(abs(TurboQuantV2Contract.innerQ.strength - 0.75) < 0.0001)
                }
            }
        }

        try withEnv("TURBO_INNERQ", value: "24") {
            try withEnv("TURBO_INNERQ_STRENGTH", value: "0.6") {
                #expect(TurboQuantV2Contract.innerQ.enabled)
                #expect(TurboQuantV2Contract.innerQ.calibrationSampleCount == 24)
                #expect(abs(TurboQuantV2Contract.innerQ.strength - 0.6) < 0.0001)
            }
        }

        try withEnv("EDGERUNNER_TURBOQUANT_KEY_PRESET", value: "balanced96") {
            #expect(TurboQuantV2Contract.keyPreset == .balanced96)
            let keyLayout = try TurboQuantV2Contract.makeKeyLayout()
            #expect(keyLayout.preset == .balanced96)
            #expect(keyLayout.runtimeCodeWordsPerRow == 18)
            #expect(TurboQuantV2Contract.keyResidualScale == 0)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_VALUE_PRESET", value: "balanced64") {
            #expect(TurboQuantV2Contract.valuePreset == .balanced64)
            let valueLayout = try TurboQuantV2Contract.makeValueLayout()
            #expect(valueLayout.preset == .balanced64)
            #expect(valueLayout.runtimeCodeWordsPerRow == 16)
            #expect(TurboQuantV2Contract.valueResidualScale(forLayer: 0, layerCount: 1) == 0)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_KEY_PRESET", value: "sixBit") {
            try withEnv("EDGERUNNER_TURBOQUANT_KEY_POLICY", value: "last8Promoted") {
                try withEnv("EDGERUNNER_TURBOQUANT_PROMOTED_KEY_PRESET", value: "sevenBit") {
                    #expect(TurboQuantV2Contract.keyPolicy == .last8Promoted)
                    #expect(TurboQuantV2Contract.keyPreset(forLayer: 0, layerCount: 28) == .sixBit)
                    #expect(TurboQuantV2Contract.keyPreset(forLayer: 27, layerCount: 28) == .sevenBit)
                    #expect(TurboQuantV2Contract.keyResidualScale(forLayer: 27, layerCount: 28) == 0)
                }
            }
        }

        try withEnv("EDGERUNNER_TURBOQUANT_KEY_OUTLIER_SELECTION", value: "quantizationBenefit") {
            #expect(TurboQuantV2Contract.keyOutlierSelection == .quantizationBenefit)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_VALUE_OUTLIER_SELECTION", value: "magnitude") {
            #expect(TurboQuantV2Contract.valueOutlierSelection == .magnitude)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS", value: "1") {
            #expect(TurboQuantV2Contract.keyCacheType(forLayer: 0, layerCount: 28) == .q8_0)
            #expect(TurboQuantV2Contract.valueCacheType(forLayer: 0, layerCount: 28) == .q8_0)
            #expect(TurboQuantV2Contract.keyCacheType(forLayer: 1, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.valueCacheType(forLayer: 1, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.keyPreset(forLayer: 0, layerCount: 28) == nil)
            #expect(TurboQuantV2Contract.valuePreset(forLayer: 0, layerCount: 28) == nil)
            #expect(TurboQuantV2Contract.keyPreset(forLayer: 1, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.valuePreset(forLayer: 1, layerCount: 28) == .turbo3)
        }

        try withEnv("EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS", value: "1") {
            #expect(TurboQuantV2Contract.keyCacheType(forLayer: 0, layerCount: 28) == .q8_0)
            #expect(TurboQuantV2Contract.valueCacheType(forLayer: 0, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.keyCacheType(forLayer: 1, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.valueCacheType(forLayer: 1, layerCount: 28) == .turbo3)
            #expect(TurboQuantV2Contract.keyPreset(forLayer: 0, layerCount: 28) == nil)
            #expect(TurboQuantV2Contract.valuePreset(forLayer: 0, layerCount: 28) == .turbo3)
        }
    }

    @Test
    func forkAdaptiveModeSevenMapsBoundaryVToQ8() throws {
        try withEnv("EDGERUNNER_TURBOQUANT_VALUE_TYPE", value: "turbo2") {
            try withEnv("TURBO_LAYER_ADAPTIVE", value: "7") {
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 0, layerCount: 28) == .q8_0)
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 1, layerCount: 28) == .q8_0)
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 2, layerCount: 28) == .turbo2)
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 25, layerCount: 28) == .turbo2)
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 26, layerCount: 28) == .q8_0)
                #expect(TurboQuantV2Contract.valueCacheType(forLayer: 27, layerCount: 28) == .q8_0)
            }
        }
    }

    @Test
    func keyOnlyQ8OverrideAllocatesHybridKVStorage() throws {
        try withEnv("EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS", value: "1") {
            let cache = try KVCache(
                device: device,
                maxSeqLen: 8,
                numLayers: 2,
                numKVHeads: 1,
                headDim: 128,
                precision: .turboquantV2
            )

            #expect(try cache.keyMetalBuffer(layer: 0) != nil)
            #expect(try cache.turboQuantKeyMetalBuffers(layer: 0) == nil)
            #expect(try cache.valueMetalBuffer(layer: 0) == nil)
            #expect(try cache.turboQuantValueMetalBuffers(layer: 0) != nil)
            #expect(try cache.keyMetalBuffer(layer: 1) == nil)
            #expect(try cache.turboQuantKeyMetalBuffers(layer: 1) != nil)
            #expect(try cache.valueMetalBuffer(layer: 1) == nil)
            #expect(try cache.turboQuantValueMetalBuffers(layer: 1) != nil)
        }
    }

    @Test
    func innerQLifecycleFinalizesAndUploadsScaleInv() throws {
        let state = try TurboQuantInnerQState(
            device: device,
            configuration: TurboQuantInnerQConfiguration(
                enabled: true,
                calibrationSampleCount: 4,
                strength: 0.5
            )
        )

        let skewedRowA: [Float] = (0..<128).map { $0 == 0 ? 8.0 : 1.0 }
        let skewedRowB: [Float] = (0..<128).map { $0 == 0 ? 7.0 : 1.2 }

        state.observe(rows: [skewedRowA, skewedRowB])
        #expect(!state.isActive)
        #expect(state.sampleCount == 2)
        #expect(state.currentScaleInv().allSatisfy { abs($0 - 1) < 0.0001 })

        state.observe(rows: [skewedRowA, skewedRowB])
        #expect(state.isActive)
        #expect(state.sampleCount == 4)

        let uploaded = state.currentScaleInv()
        #expect(uploaded.count == 128)
        #expect(uploaded[0] > 1.0)
        #expect(uploaded[1] < 1.0)
        #expect(uploaded.allSatisfy { $0.isFinite })
        #expect(uploaded.allSatisfy { $0 >= 0.5 && $0 <= 2.0 })
    }

    @Test
    func innerQAutoSkipsBalancedChannels() throws {
        let state = try TurboQuantInnerQState(
            device: device,
            configuration: TurboQuantInnerQConfiguration(
                enabled: true,
                calibrationSampleCount: 4,
                strength: 0.5
            )
        )

        let balancedRow = [Float](repeating: 1.0, count: 128)
        state.observe(rows: [balancedRow, balancedRow, balancedRow, balancedRow])

        #expect(!state.isActive)
        #expect(state.currentScaleInv().allSatisfy { abs($0 - 1) < 0.0001 })
    }

    @Test func attentionMatchesDecodedCPUReference() throws {
        try verifyAttentionKernel(useDecodePipeline: false)
    }

    @Test func decodeAttentionMatchesDecodedCPUReference() throws {
        try verifyAttentionKernel(useDecodePipeline: true)
    }

    @Test func turbo3AttentionMatchesDecodedCPUReference() throws {
        try verifyAttentionKernel(useDecodePipeline: false, keyPreset: .turbo3)
    }

    @Test func groupedDecodeAttentionMatchesDecodedCPUReference() throws {
        let numHeads = 4
        let numKVHeads = 2
        let groupSize = numHeads / numKVHeads
        let kvSeqLen = 3

        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: 128,
            precision: .turboquantV2
        )

        var keyRowsByToken: [[[Float]]] = []
        var decodedValuesByToken: [[[Float]]] = []
        for tokenIndex in 0..<kvSeqLen {
            let keyRows = (0..<numKVHeads).map { kvHead in
                makeSignal(phase: Float(tokenIndex) * 0.19 + Float(kvHead) * 0.31)
            }
            let valueRows = (0..<numKVHeads).map { kvHead in
                makeSignal(phase: 0.67 + Float(tokenIndex) * 0.13 + Float(kvHead) * 0.29)
            }
            try cache.append(
                layer: 0,
                keys: keyRows.flatMap { $0 },
                values: valueRows.flatMap { $0 }
            )
            keyRowsByToken.append(keyRows)
            decodedValuesByToken.append(valueRows)
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let (_, decodedValues) = try cache.retrieveDecodedTurboQuant(layer: 0)
        decodedValuesByToken = decodedValues.chunked(into: 128).chunked(into: numKVHeads)

        let qRows = (0..<numHeads).map { head in
            makeSignal(phase: 0.11 + Float(head) * 0.17)
        }
        let qBuffer = device.makeBuffer(
            bytes: qRows.flatMap { $0 },
            length: qRows.count * 128 * MemoryLayout<Float>.stride
        )!
        let outputBuffer = device.makeBuffer(length: qRows.count * 128 * MemoryLayout<Float>.stride)!
        let kernel = try TurboQuantKernel(device: device)
        var params = try makeAttentionParams(
            seqLen: 1,
            kvSeqLen: kvSeqLen,
            qOffset: kvSeqLen - 1,
            numHeads: numHeads,
            numKVHeads: numKVHeads
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(kernel.decodeAttentionPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(buffers.key.codes, offset: 0, index: 1)
        encoder.setBuffer(buffers.key.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(buffers.key.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(buffers.key.metadata, offset: 0, index: 4)
        encoder.setBuffer(buffers.value.codes, offset: 0, index: 5)
        encoder.setBuffer(buffers.value.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(buffers.value.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(buffers.value.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keyRotationBuffer(for: runtimeKeyPreset), offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueRotationBuffer(for: runtimeValuePreset), offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let gpu = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: qRows.count * 128),
                count: qRows.count * 128
            )
        )
        let expected = try cpuGroupedTurboQuantAttention(
            qRows: qRows,
            keyRowsByToken: keyRowsByToken,
            decodedValuesByToken: decodedValuesByToken,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            keyPreset: runtimeKeyPreset
        )

        for (lhs, rhs) in zip(gpu, expected) {
            #expect(abs(lhs - rhs) < 0.25)
        }
        #expect(groupSize == 2)
    }

    @Test func planar3GPUQuantizedRowsMatchReferenceRuntimeRows() throws {
        try verifyGPUQuantizedRowsMatchReferenceRuntimeRows(
            rowCount: 4,
            preset: .planar3,
            assertMetadata: true
        )
    }

    @Test func prefillAttentionMatchesDecodedCPUReference() throws {
        try verifyPrefillAttentionMatchesDecodedCPUReference(kvSeqLen: 4)
    }

    @Test func prefillAttentionMatchesDecodedCPUReferenceAtLongContext() throws {
        try verifyPrefillAttentionMatchesDecodedCPUReference(kvSeqLen: 130)
    }

    @Test func turbo3PrefillAttentionMatchesDecodedCPUReference() throws {
        try verifyPrefillAttentionMatchesDecodedCPUReference(kvSeqLen: 4, keyPreset: .turbo3, valuePreset: .turbo3)
    }

    @Test func turbo3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext() throws {
        try verifyPrefillAttentionMatchesDecodedCPUReference(kvSeqLen: 130, keyPreset: .turbo3, valuePreset: .turbo3)
    }

    @Test func groupedTurbo3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext() throws {
        try verifyGroupedPrefillAttentionMatchesDecodedCPUReference(
            kvSeqLen: 130,
            numHeads: 16,
            numKVHeads: 8,
            keyPreset: .turbo3,
            valuePreset: .turbo3
        )
    }

    private func verifyPrefillAttentionMatchesDecodedCPUReference(
        kvSeqLen: Int,
        keyPreset: TurboQuantPreset? = nil,
        valuePreset: TurboQuantPreset? = nil
    ) throws {
        let keyPreset = keyPreset ?? runtimeKeyPreset
        let _ = valuePreset ?? runtimeValuePreset
        let cache = try KVCache(
            device: device,
            maxSeqLen: max(kvSeqLen + 2, 8),
            numLayers: 1,
            numKVHeads: 1,
            headDim: 128,
            precision: .turboquantV2
        )

        let qRows = (0..<kvSeqLen).map { row in makeSignal(phase: 0.13 + Float(row) * 0.19) }
        let keyRows = (0..<kvSeqLen).map { row in makeSignal(phase: Float(row) * 0.23) }
        let valueRows = (0..<kvSeqLen).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }

        for index in keyRows.indices {
            try cache.append(layer: 0, keys: keyRows[index], values: valueRows[index])
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let (_, decodedValues) = try cache.retrieveDecodedTurboQuant(layer: 0)
        try verifyPrefillAttentionKernel(
            qRows: qRows,
            keyRows: keyRows,
            decodedValues: decodedValues,
            keyPreset: keyPreset,
            buffers: TurboQuantAttentionBuffers(key: buffers.key, value: buffers.value)
        )
    }

    private func verifyPrefillAttentionKernel(
        qRows: [[Float]],
        keyRows: [[Float]],
        decodedValues: [Float],
        keyPreset: TurboQuantPreset,
        buffers: TurboQuantAttentionBuffers
    ) throws {
        let kvSeqLen = keyRows.count
        let valuePreset: TurboQuantPreset = .turbo3
        #expect(decodedValues.count == kvSeqLen * 128)
        let flattenedQ = qRows.flatMap { $0 }
        let qBuffer = device.makeBuffer(bytes: flattenedQ, length: flattenedQ.count * MemoryLayout<Float>.stride)!
        let outputBuffer = device.makeBuffer(length: flattenedQ.count * MemoryLayout<Float>.stride)!
        let kernel = try TurboQuantKernel(device: device)
        var params = try makeAttentionParams(
            seqLen: qRows.count,
            kvSeqLen: keyRows.count,
            qOffset: 0,
            keyPreset: keyPreset,
            valuePreset: valuePreset
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(kernel.attentionPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(buffers.key.codes, offset: 0, index: 1)
        encoder.setBuffer(buffers.key.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(buffers.key.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(buffers.key.metadata, offset: 0, index: 4)
        encoder.setBuffer(buffers.value.codes, offset: 0, index: 5)
        encoder.setBuffer(buffers.value.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(buffers.value.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(buffers.value.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keyRotationBuffer(for: keyPreset), offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueRotationBuffer(for: valuePreset), offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let gpu = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: flattenedQ.count),
                count: flattenedQ.count
            )
        )
        #expect(gpu.count == flattenedQ.count)

        let qIndicesToCheck = kvSeqLen <= 8 ? Array(qRows.indices) : [0, 1, kvSeqLen - 1]
        for qIndex in qIndicesToCheck {
            let expected = try cpuTurboQuantAttention(
                q: qRows[qIndex],
                keyRows: Array(keyRows.prefix(qIndex + 1)),
                decodedValues: Array(decodedValues[0..<((qIndex + 1) * 128)]),
                keyPreset: keyPreset
            )
            let outputStart = qIndex * 128
            let actual = Array(gpu[outputStart..<(outputStart + 128)])
            for (lhs, rhs) in zip(actual, expected) {
                #expect(abs(lhs - rhs) < 0.30)
            }
        }
    }

    private func verifyGroupedPrefillAttentionMatchesDecodedCPUReference(
        kvSeqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        keyPreset: TurboQuantPreset,
        valuePreset: TurboQuantPreset
    ) throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: max(kvSeqLen + 2, 8),
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: 128,
            precision: .turboquantV2
        )

        var keyRowsByToken: [[[Float]]] = []
        var valueRowsByToken: [[[Float]]] = []
        for tokenIndex in 0..<kvSeqLen {
            let keyRows = (0..<numKVHeads).map { kvHead in
                makeSignal(phase: Float(tokenIndex) * 0.19 + Float(kvHead) * 0.31)
            }
            let valueRows = (0..<numKVHeads).map { kvHead in
                makeSignal(phase: 0.67 + Float(tokenIndex) * 0.13 + Float(kvHead) * 0.29)
            }
            try cache.append(
                layer: 0,
                keys: keyRows.flatMap { $0 },
                values: valueRows.flatMap { $0 }
            )
            keyRowsByToken.append(keyRows)
            valueRowsByToken.append(valueRows)
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let (_, decodedValues) = try cache.retrieveDecodedTurboQuant(layer: 0)
        let decodedValuesByToken = decodedValues.chunked(into: 128).chunked(into: numKVHeads)
        let qRowsByToken = (0..<kvSeqLen).map { tokenIndex in
            (0..<numHeads).map { headIndex in
                makeSignal(phase: 0.11 + Float(tokenIndex) * 0.07 + Float(headIndex) * 0.17)
            }
        }
        let flattenedQ = qRowsByToken.flatMap { $0.flatMap { $0 } }
        let qBuffer = device.makeBuffer(
            bytes: flattenedQ,
            length: flattenedQ.count * MemoryLayout<Float>.stride
        )!
        let outputBuffer = device.makeBuffer(length: flattenedQ.count * MemoryLayout<Float>.stride)!
        let kernel = try TurboQuantKernel(device: device)
        var params = try makeAttentionParams(
            seqLen: kvSeqLen,
            kvSeqLen: kvSeqLen,
            qOffset: 0,
            keyPreset: keyPreset,
            valuePreset: valuePreset,
            numHeads: numHeads,
            numKVHeads: numKVHeads
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(kernel.attentionPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(buffers.key.codes, offset: 0, index: 1)
        encoder.setBuffer(buffers.key.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(buffers.key.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(buffers.key.metadata, offset: 0, index: 4)
        encoder.setBuffer(buffers.value.codes, offset: 0, index: 5)
        encoder.setBuffer(buffers.value.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(buffers.value.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(buffers.value.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keyRotationBuffer(for: keyPreset), offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueRotationBuffer(for: valuePreset), offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        encoder.dispatchThreadgroups(
            MTLSize(width: (kvSeqLen + GQAKernel.blockSize - 1) / GQAKernel.blockSize, height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let gpu = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: flattenedQ.count),
                count: flattenedQ.count
            )
        )
        let tokenIndicesToCheck = kvSeqLen <= 8 ? Array(qRowsByToken.indices) : [0, 1, kvSeqLen - 1]
        for tokenIndex in tokenIndicesToCheck {
            let expected = try cpuGroupedTurboQuantAttention(
                qRows: qRowsByToken[tokenIndex],
                keyRowsByToken: Array(keyRowsByToken.prefix(tokenIndex + 1)),
                decodedValuesByToken: Array(decodedValuesByToken.prefix(tokenIndex + 1)),
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                keyPreset: keyPreset
            )
            let outputStart = tokenIndex * numHeads * 128
            let actual = Array(gpu[outputStart..<(outputStart + (numHeads * 128))])
            for (lhs, rhs) in zip(actual, expected) {
                #expect(abs(lhs - rhs) < 0.30)
            }
        }
    }

    @Test func gpuQuantizedRowsDecodeLikeReferenceRuntimeRows() throws {
        try verifyGPUQuantizedRowsDecodeLikeReferenceRuntimeRows(rowCount: 4)
    }

    @Test func gpuQuantizedRowsDecodeLikeReferenceRuntimeRowsAtLongContext() throws {
        try verifyGPUQuantizedRowsDecodeLikeReferenceRuntimeRows(rowCount: 130)
    }

    @Test func gpuFixedTurbo3RowsMatchReferenceRuntimeMetadata() throws {
        try verifyGPUQuantizedRowsMatchReferenceRuntimeRows(
            rowCount: 4,
            preset: .turbo3,
            assertMetadata: true
        )
    }

    private func verifyGPUQuantizedRowsDecodeLikeReferenceRuntimeRows(rowCount: Int) throws {
        try verifyGPUQuantizedRowsMatchReferenceRuntimeRows(
            rowCount: rowCount,
            preset: runtimeKeyPreset,
            assertMetadata: false
        )
    }

    private func verifyGPUQuantizedRowsMatchReferenceRuntimeRows(
        rowCount: Int,
        preset: TurboQuantPreset,
        assertMetadata: Bool
    ) throws {
        let kernel = try TurboQuantKernel(device: device)
        let layout = try TurboQuantLayout(preset: preset)
        let rows = (0..<rowCount).map { row in makeSignal(phase: Float(row) * 0.23) }
        let flattened = rows.flatMap { $0 }

        let sourceBuffer = device.makeBuffer(bytes: flattened, length: flattened.count * MemoryLayout<Float>.stride)!
        let codesBuffer = device.makeBuffer(length: rowCount * layout.runtimeCodeWordsPerRow * MemoryLayout<UInt32>.stride)!
        let residualSignsBuffer = device.makeBuffer(length: rowCount * TurboQuantLayout.residualWordsPerRow * MemoryLayout<UInt32>.stride)!
        let outlierMaskBuffer = device.makeBuffer(length: rowCount * TurboQuantLayout.outlierMaskWordsPerRow * MemoryLayout<UInt32>.stride)!
        let metadataBuffer = device.makeBuffer(length: rowCount * TurboQuantLayout.metadataScalarsPerRow * MemoryLayout<Float>.stride)!

        var params = try makeQuantizeParams(
            rowCount: rowCount,
            sourceRowStride: 128,
            destinationRowBase: 0,
            preset: preset
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(kernel.quantizePipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(codesBuffer, offset: 0, index: 1)
        encoder.setBuffer(residualSignsBuffer, offset: 0, index: 2)
        encoder.setBuffer(outlierMaskBuffer, offset: 0, index: 3)
        encoder.setBuffer(metadataBuffer, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantQuantizeParams>.stride, index: 5)
        encoder.setBuffer(kernel.keyRotationBuffer(for: preset), offset: 0, index: 6)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rowCount, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let codeWords = Array(
            UnsafeBufferPointer(
                start: codesBuffer.contents().bindMemory(to: UInt32.self, capacity: rowCount * layout.runtimeCodeWordsPerRow),
                count: rowCount * layout.runtimeCodeWordsPerRow
            )
        )
        let residualWords = Array(
            UnsafeBufferPointer(
                start: residualSignsBuffer.contents().bindMemory(to: UInt32.self, capacity: rowCount * TurboQuantLayout.residualWordsPerRow),
                count: rowCount * TurboQuantLayout.residualWordsPerRow
            )
        )
        let outlierWords = Array(
            UnsafeBufferPointer(
                start: outlierMaskBuffer.contents().bindMemory(to: UInt32.self, capacity: rowCount * TurboQuantLayout.outlierMaskWordsPerRow),
                count: rowCount * TurboQuantLayout.outlierMaskWordsPerRow
            )
        )
        let metadata = Array(
            UnsafeBufferPointer(
                start: metadataBuffer.contents().bindMemory(to: Float.self, capacity: rowCount * TurboQuantLayout.metadataScalarsPerRow),
                count: rowCount * TurboQuantLayout.metadataScalarsPerRow
            )
        )

        for rowIndex in 0..<rowCount {
            let runtimeRow = TurboQuantRuntimeRow(
                preset: preset,
                dimension: 128,
                primaryCodes: Array(codeWords[(rowIndex * layout.runtimeCodeWordsPerRow)..<((rowIndex + 1) * layout.runtimeCodeWordsPerRow)]),
                residualSigns: Array(residualWords[(rowIndex * TurboQuantLayout.residualWordsPerRow)..<((rowIndex + 1) * TurboQuantLayout.residualWordsPerRow)]),
                outlierMask: Array(outlierWords[(rowIndex * TurboQuantLayout.outlierMaskWordsPerRow)..<((rowIndex + 1) * TurboQuantLayout.outlierMaskWordsPerRow)]),
                rowNorm: metadata[rowIndex * TurboQuantLayout.metadataScalarsPerRow],
                residualNorm: metadata[rowIndex * TurboQuantLayout.metadataScalarsPerRow + 1]
            )
            let gpuDecoded = try TurboQuantReferenceEncoder.approximateDecode(
                runtimeRow: runtimeRow,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )
            let referenceEncoded = try TurboQuantReferenceEncoder.encode(
                rows[rowIndex],
                preset: preset,
                outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )
            let referenceRuntimeRow = try TurboQuantReferenceEncoder.makeRuntimeRow(from: referenceEncoded)
            let referenceDecoded = try TurboQuantReferenceEncoder.approximateDecode(
                runtimeRow: referenceRuntimeRow,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )

            #expect(runtimeRow.primaryCodes == referenceRuntimeRow.primaryCodes)
            #expect(runtimeRow.outlierMask == referenceRuntimeRow.outlierMask)
            #expect(runtimeRow.residualSigns == referenceRuntimeRow.residualSigns)
            let maxDelta = zip(gpuDecoded, referenceDecoded).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            #expect(maxDelta < 0.25)
            if assertMetadata {
                #expect(abs(runtimeRow.rowNorm - referenceRuntimeRow.rowNorm) < 1e-5)
                #expect(abs(runtimeRow.residualNorm - referenceRuntimeRow.residualNorm) < 1e-6)
            }
        }
    }

    @Test func scoreErrorVsValueErrorAttribution() throws {
        let q = makeSignal(phase: 0.13)
        let keyRows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let valueRows = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }

        let dense = denseAttention(q: q, keyRows: keyRows, valueRows: valueRows)
        let turboScoreExactV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: runtimeKeyPreset,
            valuePreset: runtimeValuePreset,
            useTurboScores: true,
            useDecodedK: false,
            useDecodedV: false
        )
        let exactKDecodedV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: runtimeKeyPreset,
            valuePreset: runtimeValuePreset,
            useTurboScores: false,
            useDecodedK: false,
            useDecodedV: true
        )
        let decodedKExactV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: runtimeKeyPreset,
            valuePreset: runtimeValuePreset,
            useTurboScores: false,
            useDecodedK: true,
            useDecodedV: false
        )
        let turboScoreDecodedV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: runtimeKeyPreset,
            valuePreset: runtimeValuePreset,
            useTurboScores: true,
            useDecodedK: false,
            useDecodedV: true
        )

        let turboScoreExactVError = meanSquaredError(lhs: dense, rhs: turboScoreExactV)
        let exactKDecodedVError = meanSquaredError(lhs: dense, rhs: exactKDecodedV)
        let decodedKExactVError = meanSquaredError(lhs: dense, rhs: decodedKExactV)
        let turboScoreDecodedVError = meanSquaredError(lhs: dense, rhs: turboScoreDecodedV)

        print("""
        [turboquant-attention-diagnostic]
          turbo_score_exact_v_mse=\(String(format: "%.6f", turboScoreExactVError))
          exact_k_decoded_v_mse=\(String(format: "%.6f", exactKDecodedVError))
          decoded_k_exact_v_mse=\(String(format: "%.6f", decodedKExactVError))
          turbo_score_decoded_v_mse=\(String(format: "%.6f", turboScoreDecodedVError))
        """)

        #expect(turboScoreExactVError.isFinite)
        #expect(exactKDecodedVError.isFinite)
        #expect(decodedKExactVError.isFinite)
        #expect(turboScoreDecodedVError.isFinite)
        #expect(abs(turboScoreExactVError - decodedKExactVError) < 1e-4)
        #expect(turboScoreDecodedVError > turboScoreExactVError)
        #expect(exactKDecodedVError < 0.010)
    }

    @Test func valueResidualScaleSweepDiagnostic() throws {
        let valueRows = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }
        let encodedRows = try valueRows.map {
            try TurboQuantReferenceEncoder.encode(
                $0,
                preset: runtimeValuePreset,
                outlierSelection: .quantizationBenefit,
                rotationSeed: TurboQuantSeeds.valueRotation,
                residualSeed: TurboQuantSeeds.valueResidual
            )
        }

        var bestScale: Float = 1.0
        var bestMSE: Float = .infinity
        for scale in stride(from: 0.0 as Float, through: 1.25 as Float, by: 0.05 as Float) {
            let mse = zip(valueRows, encodedRows).reduce(Float.zero) { partial, pair in
                let decoded = try! approximateDecodeValueRow(pair.1, residualScale: scale)
                return partial + meanSquaredError(lhs: pair.0, rhs: decoded)
            } / Float(valueRows.count)
            if mse < bestMSE {
                bestMSE = mse
                bestScale = scale
            }
        }

        print("""
        [turboquant-value-residual-sweep]
          best_scale=\(String(format: "%.2f", bestScale))
          best_mse=\(String(format: "%.6f", bestMSE))
        """)

        #expect(bestMSE.isFinite)
    }

    @Test func keyResidualScaleSweepDiagnostic() throws {
        let q = makeSignal(phase: 0.13)
        let keyRows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let valueRows = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }
        let dense = denseAttention(q: q, keyRows: keyRows, valueRows: valueRows)

        var bestScale: Float = 1.0
        var bestMSE: Float = .infinity
        for scale in stride(from: 0.0 as Float, through: 1.25 as Float, by: 0.05 as Float) {
            let turboScoreExactV = try cpuAttention(
                q: q,
                keyRows: keyRows,
                valueRows: valueRows,
                keyPreset: runtimeKeyPreset,
                valuePreset: runtimeValuePreset,
                useTurboScores: true,
                keyResidualScale: scale,
                useDecodedK: false,
                useDecodedV: false
            )
            let mse = meanSquaredError(lhs: dense, rhs: turboScoreExactV)
            if mse < bestMSE {
                bestMSE = mse
                bestScale = scale
            }
        }

        print("""
        [turboquant-key-residual-sweep]
          best_scale=\(String(format: "%.2f", bestScale))
          best_mse=\(String(format: "%.6f", bestMSE))
        """)

        #expect(bestMSE.isFinite)
    }

    @Test func decodeScoreTermsMatchCPUReference() throws {
        try verifyDecodeScoreTermsMatchCPUReference()
    }

    @Test func planar3DecodeScoreTermsMatchCPUReference() throws {
        try withEnv("EDGERUNNER_TURBOQUANT_KEY_PRESET", value: "planar3") {
            try verifyDecodeScoreTermsMatchCPUReference()
        }
    }

    private func verifyDecodeScoreTermsMatchCPUReference() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 128,
            precision: .turboquantV2
        )

        let q = makeSignal(phase: 0.13)
        let keyRows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let valueRows = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }
        for index in keyRows.indices {
            try cache.append(layer: 0, keys: keyRows[index], values: valueRows[index])
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let runtimeRows = try cache.retrieveTurboQuantRuntimeRows(layer: 0).keys
        let kernel = try TurboQuantKernel(device: device)
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride)!
        let outputBuffer = device.makeBuffer(
            length: runtimeRows.count * MemoryLayout<TurboQuantDebugScoreTerms>.stride
        )!

        var params = try makeAttentionParams(
            seqLen: 1,
            kvSeqLen: runtimeRows.count,
            qOffset: runtimeRows.count - 1
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(kernel.debugDecodeScoreTermsPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(buffers.key.codes, offset: 0, index: 1)
        encoder.setBuffer(buffers.key.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(buffers.key.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(buffers.key.metadata, offset: 0, index: 4)
        encoder.setBuffer(outputBuffer, offset: 0, index: 5)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 6)
        encoder.setBuffer(kernel.keyRotationBuffer(for: runtimeKeyPreset), offset: 0, index: 7)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: runtimeRows.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let gpuTerms = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(
                    to: TurboQuantDebugScoreTerms.self,
                    capacity: runtimeRows.count
                ),
                count: runtimeRows.count
            )
        )
        let cpuTerms = try runtimeRows.map {
            try TurboQuantReferenceEncoder.approximateScoreTerms(
                query: q,
                runtimeRow: $0,
                residualWeight: TurboQuantV2Contract.keyResidualScale,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual,
                scale: 1.0 / sqrt(128.0)
            )
        }

        var maxMSEDotDelta: Float = 0
        var maxResidualDotDelta: Float = 0
        var maxScoreDelta: Float = 0
        for index in runtimeRows.indices {
            let gpu = gpuTerms[index]
            let cpu = cpuTerms[index]
            maxMSEDotDelta = max(maxMSEDotDelta, abs(gpu.mseDot - cpu.mseDot))
            maxResidualDotDelta = max(maxResidualDotDelta, abs(gpu.residualDot - cpu.residualDot))
            maxScoreDelta = max(maxScoreDelta, abs(gpu.score - cpu.score))
            #expect(abs(gpu.mseDot - cpu.mseDot) < 1e-4)
            #expect(abs(gpu.residualDot - cpu.residualDot) < 1e-4)
            #expect(abs(gpu.rowNorm - cpu.rowNorm) < 1e-6)
            #expect(abs(gpu.residualNorm - cpu.residualNorm) < 1e-6)
            #expect(abs(gpu.score - cpu.score) < 1e-4)
        }
        print("""
        [turboquant-debug-score-terms]
          max_mse_dot_delta=\(String(format: "%.8f", maxMSEDotDelta))
          max_residual_dot_delta=\(String(format: "%.8f", maxResidualDotDelta))
          max_score_delta=\(String(format: "%.8f", maxScoreDelta))
        """)
    }

    private func verifyAttentionKernel(
        useDecodePipeline: Bool,
        keyPreset: TurboQuantPreset? = nil
    ) throws {
        let keyPreset = keyPreset ?? runtimeKeyPreset
        let valuePreset: TurboQuantPreset = .turbo3
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 128,
            precision: .turboquantV2
        )

        let rows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let values = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }
        for index in rows.indices {
            try cache.append(layer: 0, keys: rows[index], values: values[index])
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let (_, decodedValues) = try cache.retrieveDecodedTurboQuant(layer: 0)

        let q = makeSignal(phase: 0.13)
        let expected = try cpuTurboQuantAttention(
            q: q,
            keyRows: rows,
            decodedValues: decodedValues,
            keyPreset: keyPreset
        )

        let kernel = try TurboQuantKernel(device: device)
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride)!
        let outputBuffer = device.makeBuffer(length: q.count * MemoryLayout<Float>.stride)!
        var params = try makeAttentionParams(
            seqLen: 1,
            kvSeqLen: rows.count,
            qOffset: rows.count - 1,
            keyPreset: keyPreset,
            valuePreset: valuePreset
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(
            useDecodePipeline ? kernel.decodeAttentionPipeline : kernel.attentionPipeline
        )
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(buffers.key.codes, offset: 0, index: 1)
        encoder.setBuffer(buffers.key.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(buffers.key.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(buffers.key.metadata, offset: 0, index: 4)
        encoder.setBuffer(buffers.value.codes, offset: 0, index: 5)
        encoder.setBuffer(buffers.value.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(buffers.value.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(buffers.value.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keyRotationBuffer(for: keyPreset), offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueRotationBuffer(for: valuePreset), offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        if useDecodePipeline {
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let gpu = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: 128),
                count: 128
            )
        )

        for (lhs, rhs) in zip(gpu, expected) {
            #expect(abs(lhs - rhs) < 0.25)
        }
    }

    private func cpuTurboQuantAttention(
        q: [Float],
        keyRows: [[Float]],
        decodedValues: [Float],
        keyPreset: TurboQuantPreset
    ) throws -> [Float] {
        let qRot = try TurboQuantTransform.randomizedHadamard(q, seed: TurboQuantSeeds.keyRotation)
        let qResidual = try TurboQuantTransform.randomizedHadamard(q, seed: TurboQuantSeeds.keyResidual)
        let descriptor = keyPreset.descriptor
        var scores = [Float](repeating: 0, count: keyRows.count)
        for row in 0..<keyRows.count {
            let encoded = try TurboQuantReferenceEncoder.encode(
                keyRows[row],
                preset: keyPreset,
                outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )
            let outlierMask = unpackBits(encoded.outlierMask, count: 128)
            let residualSigns = unpackBits(encoded.residualSigns, count: 128)
            let codes = try BitPacker.unpackCodes(
                encoded.primaryCodes,
                count: 128,
                outlierMask: outlierMask,
                regularBits: descriptor.regularBits,
                highPrecisionBits: descriptor.highPrecisionBits
            )

            var mseDot: Float = 0
            var residualDot: Float = 0
            for dim in 0..<128 {
                let bits = outlierMask[dim] ? descriptor.highPrecisionBits : descriptor.regularBits
                let codebook = try TurboQuantCodebooks.forBits(bits)
                mseDot += qRot[dim] * codebook.centroid(for: codes[dim])
                residualDot += qResidual[dim] * (residualSigns[dim] ? 1 : -1)
            }
            scores[row] = encoded.rowNorm
                * (mseDot + (TurboQuantTransform.qjlScale * encoded.residualNorm * TurboQuantV2Contract.keyResidualScale * residualDot))
                / sqrt(128.0)
        }

        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp($0 - maxScore) }
        let sum = exps.reduce(Float.zero, +)
        var output = [Float](repeating: 0, count: 128)
        for row in 0..<keyRows.count {
            let weight = exps[row] / sum
            let base = row * 128
            for dim in 0..<128 {
                output[dim] += weight * decodedValues[base + dim]
            }
        }
        return output
    }

    private func cpuGroupedTurboQuantAttention(
        qRows: [[Float]],
        keyRowsByToken: [[[Float]]],
        decodedValuesByToken: [[[Float]]],
        numHeads: Int,
        numKVHeads: Int,
        keyPreset: TurboQuantPreset
    ) throws -> [Float] {
        let groupSize = numHeads / numKVHeads
        var output = [Float](repeating: 0, count: numHeads * 128)

        for headIndex in 0..<numHeads {
            let kvHead = headIndex / groupSize
            let keyRows = keyRowsByToken.map { $0[kvHead] }
            let decodedValues = decodedValuesByToken.flatMap { $0[kvHead] }
            let headOutput = try cpuTurboQuantAttention(
                q: qRows[headIndex],
                keyRows: keyRows,
                decodedValues: decodedValues,
                keyPreset: keyPreset
            )
            let base = headIndex * 128
            output.replaceSubrange(base..<(base + 128), with: headOutput)
        }

        return output
    }

    private func cpuAttention(
        q: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]],
        keyPreset: TurboQuantPreset,
        valuePreset: TurboQuantPreset,
        useTurboScores: Bool,
        keyResidualScale: Float = TurboQuantV2Contract.keyResidualScale,
        useDecodedK: Bool,
        useDecodedV: Bool
    ) throws -> [Float] {
        let effectiveKeys: [[Float]]
        if useDecodedK {
            effectiveKeys = try keyRows.map {
                let encoded = try TurboQuantReferenceEncoder.encode(
                    $0,
                    preset: keyPreset,
                    outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                )
                return try TurboQuantReferenceEncoder.approximateDecode(
                    encoded,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                )
            }
        } else {
            effectiveKeys = keyRows
        }

        let encodedValues: [TurboQuantEncodedRow]
        if useDecodedV {
            encodedValues = try valueRows.map {
                try TurboQuantReferenceEncoder.encode(
                    $0,
                    preset: valuePreset,
                    outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
            }
        } else {
            encodedValues = []
        }
        let effectiveValues: [[Float]] = useDecodedV ? [] : valueRows

        let scores: [Float]
        if useTurboScores {
            scores = try turboQuantScores(
                q: q,
                keyRows: keyRows,
                preset: keyPreset,
                keyResidualScale: keyResidualScale
            )
        } else {
            scores = effectiveKeys.map { row in
                zip(q, row).reduce(Float.zero) { partial, pair in partial + pair.0 * pair.1 } / sqrt(128.0)
            }
        }

        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp($0 - maxScore) }
        let sum = exps.reduce(Float.zero, +)
        if useDecodedV {
            return try decodeTurboValueOutput(encodedValues: encodedValues, weights: exps.map { $0 / sum })
        }

        var output = [Float](repeating: 0, count: 128)
        for row in effectiveValues.indices {
            let weight = exps[row] / sum
            for dim in 0..<128 {
                output[dim] += weight * effectiveValues[row][dim]
            }
        }
        return output
    }

    private func turboQuantScores(
        q: [Float],
        keyRows: [[Float]],
        preset: TurboQuantPreset,
        keyResidualScale: Float = TurboQuantV2Contract.keyResidualScale
    ) throws -> [Float] {
        return try keyRows.map { row in
            let encoded = try TurboQuantReferenceEncoder.encode(
                row,
                preset: preset,
                outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual
            )
            let runtimeRow = try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
            return try TurboQuantReferenceEncoder.approximateScore(
                query: q,
                runtimeRow: runtimeRow,
                residualWeight: keyResidualScale,
                rotationSeed: TurboQuantSeeds.keyRotation,
                residualSeed: TurboQuantSeeds.keyResidual,
                scale: 1.0 / sqrt(128.0)
            )
        }
    }

    private func decodeTurboValueOutput(
        encodedValues: [TurboQuantEncodedRow],
        weights: [Float]
    ) throws -> [Float] {
        let valueResidualScale = TurboQuantV2Contract.valueResidualScale(forLayer: 0, layerCount: 1)
        var rotatedMSE = [Float](repeating: 0, count: 128)
        var rotatedResidual = [Float](repeating: 0, count: 128)

        for (rowIndex, encoded) in encodedValues.enumerated() {
            let weight = weights[rowIndex]
            let descriptor = encoded.preset.descriptor
            let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: 128)
            let codes = try BitPacker.unpackCodes(
                encoded.primaryCodes,
                count: 128,
                outlierMask: outlierMask,
                regularBits: descriptor.regularBits,
                highPrecisionBits: descriptor.highPrecisionBits
            )
            for dim in 0..<128 {
                let bits = outlierMask[dim] ? descriptor.highPrecisionBits : descriptor.regularBits
                let codebook = try TurboQuantCodebooks.forBits(bits)
                rotatedMSE[dim] += weight * encoded.rowNorm * codebook.centroid(for: codes[dim])
            }

            if valueResidualScale != 0, encoded.residualNorm != 0 {
                let residualSigns = BitPacker.unpackBooleans(encoded.residualSigns, count: 128)
                let residualMagnitude = weight * encoded.rowNorm * encoded.residualNorm
                for dim in 0..<128 {
                    rotatedResidual[dim] += residualMagnitude * (residualSigns[dim] ? 1 : -1)
                }
            }
        }

        let mseOutput = try TurboQuantTransform.inverseRandomizedHadamard(
            rotatedMSE,
            seed: TurboQuantSeeds.valueRotation
        )
        let residualOutput = try TurboQuantTransform.inverseRandomizedHadamard(
            rotatedResidual,
            seed: TurboQuantSeeds.valueResidual
        )

        return zip(mseOutput, residualOutput).map {
            $0 + ($1 * TurboQuantTransform.qjlScale * valueResidualScale)
        }
    }

    private func denseAttention(q: [Float], keyRows: [[Float]], valueRows: [[Float]]) -> [Float] {
        var scores = [Float](repeating: 0, count: keyRows.count)
        for row in keyRows.indices {
            var dot: Float = 0
            for dim in 0..<128 {
                dot += q[dim] * keyRows[row][dim]
            }
            scores[row] = dot / sqrt(128.0)
        }

        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp($0 - maxScore) }
        let sum = exps.reduce(Float.zero, +)
        var output = [Float](repeating: 0, count: 128)
        for row in valueRows.indices {
            let weight = exps[row] / sum
            for dim in 0..<128 {
                output[dim] += weight * valueRows[row][dim]
            }
        }
        return output
    }

    private func meanSquaredError(lhs: [Float], rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta * delta
        } / Float(lhs.count)
    }

    private func makeSignal(phase: Float) -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.15 + phase) + 0.35 * cos(x * 0.05 - phase)
        }
    }

    private func unpackBits(_ words: [UInt32], count: Int) -> [Bool] {
        (0..<count).map { index in
            let wordIndex = index / 32
            let bitIndex = index % 32
            return ((words[wordIndex] >> UInt32(bitIndex)) & 1) == 1
        }
    }

    private func approximateDecodeValueRow(
        _ encoded: TurboQuantEncodedRow,
        residualScale: Float
    ) throws -> [Float] {
        let descriptor = encoded.preset.descriptor
        let outlierMask = BitPacker.unpackBooleans(encoded.outlierMask, count: encoded.dimension)
        let codes = try BitPacker.unpackCodes(
            encoded.primaryCodes,
            count: encoded.dimension,
            outlierMask: outlierMask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits
        )
        let regularBook = try TurboQuantCodebooks.forBits(descriptor.regularBits)
        let highBook = try TurboQuantCodebooks.forBits(descriptor.highPrecisionBits)
        let reconstructedRotated = codes.enumerated().map { index, code -> Float in
            let book = outlierMask[index] ? highBook : regularBook
            return book.centroid(for: code)
        }
        let mseApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            reconstructedRotated,
            seed: TurboQuantSeeds.valueRotation
        )
        let residualSigns = BitPacker.unpackBooleans(encoded.residualSigns, count: encoded.dimension)
        let residualDirection = residualSigns.map { $0 ? 1.0 as Float : -1.0 }
        let qjlApproximation = try TurboQuantTransform.inverseRandomizedHadamard(
            residualDirection,
            seed: TurboQuantSeeds.valueResidual
        ).map {
            $0
                * TurboQuantTransform.qjlScale
                * (1 / Float(TurboQuantLayout.supportedDimension))
                * encoded.residualNorm
                * residualScale
        }
        return zip(mseApproximation, qjlApproximation).map { ($0 + $1) * encoded.rowNorm }
    }

    private func runtimeKeyLayout() throws -> TurboQuantLayout {
        try TurboQuantV2Contract.makeKeyLayout()
    }

    private func runtimeValueLayout() throws -> TurboQuantLayout {
        try TurboQuantV2Contract.makeValueLayout()
    }

    private func makeAttentionParams(
        seqLen: Int,
        kvSeqLen: Int,
        qOffset: Int,
        keyPreset: TurboQuantPreset? = nil,
        valuePreset: TurboQuantPreset? = nil,
        numHeads: Int = 1,
        numKVHeads: Int = 1
    ) throws -> TurboQuantAttentionParams {
        let keyPreset = keyPreset ?? runtimeKeyPreset
        let valuePreset = valuePreset ?? runtimeValuePreset
        let keyLayout = try TurboQuantLayout(preset: keyPreset)
        let valueLayout = try TurboQuantLayout(preset: valuePreset)
        let keyDescriptor = keyPreset.descriptor
        let valueDescriptor = valuePreset.descriptor
        return TurboQuantAttentionParams(
            seqLen: UInt32(seqLen),
            headDim: 128,
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(numHeads / numKVHeads),
            scale: 1.0 / sqrt(128.0),
            keyResidualScale: keyPreset.descriptor.highPrecisionChannelCount > 0 ? TurboQuantV2Contract.keyResidualScale : 0,
            valueResidualScale: valuePreset.descriptor.highPrecisionChannelCount > 0 ? TurboQuantV2Contract.valueResidualScale(forLayer: 0, layerCount: 1) : 0,
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: UInt32(kvSeqLen),
            qOffset: UInt32(qOffset),
            codeWordsPerRow: UInt32(keyLayout.codeWordsPerRow),
            regularBits: UInt32(keyDescriptor.regularBits),
            highPrecisionBits: UInt32(keyDescriptor.highPrecisionBits),
            valueCodeWordsPerRow: UInt32(valueLayout.codeWordsPerRow),
            valueRegularBits: UInt32(valueDescriptor.regularBits),
            valueHighPrecisionBits: UInt32(valueDescriptor.highPrecisionBits),
            reserved: keyPreset == .planar3 ? 2 : 0
        )
    }

    private func makeQuantizeParams(
        rowCount: Int,
        sourceRowStride: Int,
        destinationRowBase: Int,
        preset: TurboQuantPreset
    ) throws -> TurboQuantQuantizeParams {
        let descriptor = preset.descriptor
        let layout = try TurboQuantLayout(preset: preset)
        return TurboQuantQuantizeParams(
            rowCount: UInt32(rowCount),
            sourceRowStride: UInt32(sourceRowStride),
            destinationRowBase: UInt32(destinationRowBase),
            codeWordsPerRow: UInt32(layout.codeWordsPerRow),
            regularBits: UInt32(descriptor.regularBits),
            highPrecisionBits: UInt32(descriptor.highPrecisionBits),
            highPrecisionChannelCount: UInt32(descriptor.highPrecisionChannelCount),
            reserved: preset == .planar3 ? 4 : 0
        )
    }

    private func withEnv(
        _ key: String,
        value: String,
        _ body: () throws -> Void
    ) throws {
        let previous = ProcessInfo.processInfo.environment[key]
        setenv(key, value, 1)
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }
}

private extension Array {
    func chunked(into chunkSize: Int) -> [[Element]] {
        stride(from: 0, to: count, by: chunkSize).map { start in
            Array(self[start..<Swift.min(start + chunkSize, count)])
        }
    }
}

private struct TurboQuantAttentionParams {
    var seqLen: UInt32
    var headDim: UInt32
    var numHeads: UInt32
    var numKVHeads: UInt32
    var groupSize: UInt32
    var scale: Float
    var keyResidualScale: Float
    var valueResidualScale: Float
    var causal: UInt32
    var kvBlockSize: UInt32
    var qBlockSize: UInt32
    var kvSeqLen: UInt32
    var qOffset: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var valueCodeWordsPerRow: UInt32
    var valueRegularBits: UInt32
    var valueHighPrecisionBits: UInt32
    var reserved: UInt32
}

private struct TurboQuantQuantizeParams {
    var rowCount: UInt32
    var sourceRowStride: UInt32
    var destinationRowBase: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var highPrecisionChannelCount: UInt32
    var reserved: UInt32
}

private struct TurboQuantDebugScoreTerms {
    var mseDot: Float
    var residualDot: Float
    var rowNorm: Float
    var residualNorm: Float
    var score: Float
}
