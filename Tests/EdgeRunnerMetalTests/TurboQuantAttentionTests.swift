import Metal
import Testing
@testable import EdgeRunnerMetal

@Suite("TurboQuant Attention")
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

    @Test func attentionMatchesDecodedCPUReference() throws {
        try verifyAttentionKernel(useDecodePipeline: false)
    }

    @Test func decodeAttentionMatchesDecodedCPUReference() throws {
        try verifyAttentionKernel(useDecodePipeline: true)
    }

    @Test func scoreErrorVsValueErrorAttribution() throws {
        let q = makeSignal(phase: 0.13)
        let keyRows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let valueRows = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }

        let dense = denseAttention(q: q, keyRows: keyRows, valueRows: valueRows)
        let exactKDecodedV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            preset: .aggressive,
            useDecodedK: false,
            useDecodedV: true
        )
        let decodedKExactV = try cpuAttention(
            q: q,
            keyRows: keyRows,
            valueRows: valueRows,
            preset: .aggressive,
            useDecodedK: true,
            useDecodedV: false
        )

        let exactKDecodedVError = meanSquaredError(lhs: dense, rhs: exactKDecodedV)
        let decodedKExactVError = meanSquaredError(lhs: dense, rhs: decodedKExactV)

        print("""
        [turboquant-attention-diagnostic]
          exact_k_decoded_v_mse=\(String(format: "%.6f", exactKDecodedVError))
          decoded_k_exact_v_mse=\(String(format: "%.6f", decodedKExactVError))
        """)

        #expect(exactKDecodedVError.isFinite)
        #expect(decodedKExactVError.isFinite)
    }

    private func verifyAttentionKernel(useDecodePipeline: Bool) throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 8,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 128,
            precision: .turboQuantAggressive
        )

        let rows = (0..<3).map { row in makeSignal(phase: Float(row) * 0.23) }
        let values = (0..<3).map { row in makeSignal(phase: 0.7 + Float(row) * 0.11) }
        for index in rows.indices {
            try cache.append(layer: 0, keys: rows[index], values: values[index])
        }

        let buffers = try cache.turboQuantMetalBuffers(layer: 0)
        let (decodedKeys, decodedValues) = try cache.retrieveDecodedTurboQuant(layer: 0)

        let q = makeSignal(phase: 0.13)
        let expected = try cpuTurboQuantAttention(
            q: q,
            keyRows: rows,
            decodedValues: decodedValues,
            preset: .aggressive
        )

        let kernel = try TurboQuantKernel(device: device)
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride)!
        let outputBuffer = device.makeBuffer(length: q.count * MemoryLayout<Float>.stride)!
        var params = TurboQuantAttentionParams(
            seqLen: 1,
            headDim: 128,
            numHeads: 1,
            numKVHeads: 1,
            groupSize: 1,
            scale: 1.0 / sqrt(128.0),
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: UInt32(rows.count),
            qOffset: UInt32(rows.count - 1),
            codeWordsPerRow: 9,
            regularBits: 2,
            highPrecisionBits: 3,
            reserved: 0
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer")
            return
        }

        encoder.setComputePipelineState(
            useDecodePipeline ? kernel.decodeAttentionAggressivePipeline : kernel.attentionPipeline
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
        encoder.setBuffer(kernel.keySigns.rotation, offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 13)
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
        preset: TurboQuantPreset
    ) throws -> [Float] {
        let qRot = try TurboQuantTransform.randomizedHadamard(q, seed: TurboQuantSeeds.keyRotation)
        let qResidual = try TurboQuantTransform.randomizedHadamard(q, seed: TurboQuantSeeds.keyResidual)
        let descriptor = preset.descriptor
        var scores = [Float](repeating: 0, count: keyRows.count)
        for row in 0..<keyRows.count {
            let encoded = try TurboQuantReferenceEncoder.encode(
                keyRows[row],
                preset: preset,
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
                * (mseDot + (TurboQuantTransform.qjlScale * encoded.residualNorm * residualDot))
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

    private func cpuAttention(
        q: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]],
        preset: TurboQuantPreset,
        useDecodedK: Bool,
        useDecodedV: Bool
    ) throws -> [Float] {
        let effectiveKeys: [[Float]]
        if useDecodedK {
            effectiveKeys = try keyRows.map {
                let encoded = try TurboQuantReferenceEncoder.encode(
                    $0,
                    preset: preset,
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

        let effectiveValues: [[Float]]
        if useDecodedV {
            effectiveValues = try valueRows.map {
                let encoded = try TurboQuantReferenceEncoder.encode(
                    $0,
                    preset: preset,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
                return try TurboQuantReferenceEncoder.approximateDecode(
                    encoded,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
            }
        } else {
            effectiveValues = valueRows
        }

        var scores = [Float](repeating: 0, count: effectiveKeys.count)
        for row in effectiveKeys.indices {
            var dot: Float = 0
            for dim in 0..<128 {
                dot += q[dim] * effectiveKeys[row][dim]
            }
            scores[row] = dot / sqrt(128.0)
        }

        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp($0 - maxScore) }
        let sum = exps.reduce(Float.zero, +)
        var output = [Float](repeating: 0, count: 128)
        for row in effectiveValues.indices {
            let weight = exps[row] / sum
            for dim in 0..<128 {
                output[dim] += weight * effectiveValues[row][dim]
            }
        }
        return output
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
}

private struct TurboQuantAttentionParams {
    var seqLen: UInt32
    var headDim: UInt32
    var numHeads: UInt32
    var numKVHeads: UInt32
    var groupSize: UInt32
    var scale: Float
    var causal: UInt32
    var kvBlockSize: UInt32
    var qBlockSize: UInt32
    var kvSeqLen: UInt32
    var qOffset: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var reserved: UInt32
}
