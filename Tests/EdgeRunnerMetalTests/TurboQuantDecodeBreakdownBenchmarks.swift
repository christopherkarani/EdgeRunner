import Foundation
import Metal
import Testing
@testable import EdgeRunnerMetal

@Suite("TurboQuant Decode Breakdown Benchmarks")
struct TurboQuantDecodeBreakdownBenchmarks {
    private static let runEnvKey = "EDGERUNNER_RUN_TURBOQUANT_DECODE_BREAKDOWN"

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

    @Test
    func aggressiveDecodeBreakdown() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }

        let kernel = try TurboQuantKernel(device: device)
        let registry = try KernelRegistry(device: device)
        let fusedQKVTurboPipeline = try registry.pipeline(for: "dequant_q8_0_fused_qkv_turbo")
        let phase1Pipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_phase1")
        let rotateOnlyPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_rotate_only")
        let selectOnlyPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_select_only")
        let selectOnlyBitonicPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_select_only_bitonic")
        let phase2Pipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_phase2")
        let layout = try TurboQuantLayout(preset: .aggressive)
        let rowCount = 8
        let sourceRowStride = 128
        let destinationRowBase = 0
        let dim = 1024
        let qRows = 2048
        let kvRows = 1024
        let blocksPerRow = dim / 32

        let kSource = makeRows(rowCount: rowCount, phase: 0.11)
        let vSource = makeRows(rowCount: rowCount, phase: 0.47)
        let q = makeQuery(headCount: 16)
        let hidden = makeHidden(dim: dim, phase: 0.19)
        let norm = makeNorm(dim: dim)

        let kSourceBuffer = device.makeBuffer(bytes: kSource, length: kSource.count * MemoryLayout<Float>.stride)!
        let vSourceBuffer = device.makeBuffer(bytes: vSource, length: vSource.count * MemoryLayout<Float>.stride)!
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride)!
        let hiddenBuffer = device.makeBuffer(bytes: hidden, length: hidden.count * MemoryLayout<Float>.stride)!
        let normBuffer = device.makeBuffer(bytes: norm, length: norm.count * MemoryLayout<Float>.stride)!
        let wqBuffer = device.makeBuffer(bytes: makeQ8Weights(rows: qRows, blocksPerRow: blocksPerRow, phase: 0x11), length: qRows * blocksPerRow * 34)!
        let wkBuffer = device.makeBuffer(bytes: makeQ8Weights(rows: kvRows, blocksPerRow: blocksPerRow, phase: 0x31), length: kvRows * blocksPerRow * 34)!
        let wvBuffer = device.makeBuffer(bytes: makeQ8Weights(rows: kvRows, blocksPerRow: blocksPerRow, phase: 0x51), length: kvRows * blocksPerRow * 34)!
        let fusedQOutput = device.makeBuffer(length: qRows * MemoryLayout<Float>.stride)!
        let fusedKOutput = device.makeBuffer(length: kvRows * MemoryLayout<Float>.stride)!
        let fusedVOutput = device.makeBuffer(length: kvRows * MemoryLayout<Float>.stride)!

        let singleK = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let fusedK = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let fusedV = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let attentionOutput = device.makeBuffer(length: 16 * 128 * MemoryLayout<Float>.stride)!
        let phaseNormalized = device.makeBuffer(length: rowCount * 128 * MemoryLayout<Float>.stride)!
        let phaseRotated = device.makeBuffer(length: rowCount * 128 * MemoryLayout<Float>.stride)!
        let phaseOutlierMask = device.makeBuffer(length: rowCount * 4 * MemoryLayout<UInt32>.stride)!
        let phaseRowNorm = device.makeBuffer(length: rowCount * MemoryLayout<Float>.stride)!

        var quantizeParams = TurboQuantQuantizeParamsBench(
            rowCount: UInt32(rowCount),
            sourceRowStride: UInt32(sourceRowStride),
            destinationRowBase: UInt32(destinationRowBase),
            codeWordsPerRow: UInt32(layout.codeWordsPerRow),
            regularBits: 2,
            highPrecisionBits: 3,
            highPrecisionChannelCount: 32,
            reserved: 0
        )
        var fusedQKVParams = FusedQKVParamsBench(
            qRows: UInt32(qRows),
            kvRows: UInt32(kvRows),
            cols: UInt32(dim),
            blocksPerRow: UInt32(blocksPerRow),
            tokenCount: 1,
            rmsEps: 1e-6
        )
        var attentionParams = TurboQuantAttentionParamsBench(
            seqLen: 1,
            headDim: 128,
            numHeads: 16,
            numKVHeads: 8,
            groupSize: 2,
            scale: 1.0 / sqrt(128.0),
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: 1024,
            qOffset: 1023,
            codeWordsPerRow: UInt32(layout.codeWordsPerRow),
            regularBits: 2,
            highPrecisionBits: 3,
            reserved: 0
        )

        let attentionCache = try makeAttentionCache(rowCount: 1024)

        let singleQuantize = try await benchmark(name: "turboquant_quantize_small_aggressive", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(kernel.quantizeAggressiveSmallPipeline)
                $0.setBuffer(kSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(singleK.codes, offset: 0, index: 1)
                $0.setBuffer(singleK.residualSigns, offset: 0, index: 2)
                $0.setBuffer(singleK.outlierMask, offset: 0, index: 3)
                $0.setBuffer(singleK.metadata, offset: 0, index: 4)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 5)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 6)
                $0.setBuffer(kernel.keySigns.residual, offset: 0, index: 7)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let fusedQuantize = try await benchmark(name: "turboquant_quantize_small_aggressive_kv", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(kernel.quantizeAggressiveSmallKVPipeline)
                $0.setBuffer(kSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(vSourceBuffer, offset: 0, index: 1)
                $0.setBuffer(fusedK.codes, offset: 0, index: 2)
                $0.setBuffer(fusedK.residualSigns, offset: 0, index: 3)
                $0.setBuffer(fusedK.outlierMask, offset: 0, index: 4)
                $0.setBuffer(fusedK.metadata, offset: 0, index: 5)
                $0.setBuffer(fusedV.codes, offset: 0, index: 6)
                $0.setBuffer(fusedV.residualSigns, offset: 0, index: 7)
                $0.setBuffer(fusedV.outlierMask, offset: 0, index: 8)
                $0.setBuffer(fusedV.metadata, offset: 0, index: 9)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 10)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 11)
                $0.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
                $0.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 13)
                $0.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let quantizePhase1 = try await benchmark(name: "turboquant_quantize_small_aggressive_phase1", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(phase1Pipeline)
                $0.setBuffer(kSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(phaseNormalized, offset: 0, index: 1)
                $0.setBuffer(phaseRotated, offset: 0, index: 2)
                $0.setBuffer(phaseOutlierMask, offset: 0, index: 3)
                $0.setBuffer(phaseRowNorm, offset: 0, index: 4)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 5)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 6)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let quantizeRotateOnly = try await benchmark(name: "turboquant_quantize_small_aggressive_rotate_only", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(rotateOnlyPipeline)
                $0.setBuffer(kSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(phaseRotated, offset: 0, index: 1)
                $0.setBuffer(phaseRowNorm, offset: 0, index: 2)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 3)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 4)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let quantizeSelectOnly = try await benchmark(name: "turboquant_quantize_small_aggressive_select_only", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(selectOnlyPipeline)
                $0.setBuffer(phaseRotated, offset: 0, index: 0)
                $0.setBuffer(phaseOutlierMask, offset: 0, index: 1)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 2)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let quantizeSelectOnlyBitonic = try await benchmark(name: "turboquant_quantize_small_aggressive_select_only_bitonic", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(selectOnlyBitonicPipeline)
                $0.setBuffer(phaseRotated, offset: 0, index: 0)
                $0.setBuffer(phaseOutlierMask, offset: 0, index: 1)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 2)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let quantizePhase2 = try await benchmark(name: "turboquant_quantize_small_aggressive_phase2", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(phase2Pipeline)
                $0.setBuffer(phaseNormalized, offset: 0, index: 0)
                $0.setBuffer(phaseRotated, offset: 0, index: 1)
                $0.setBuffer(phaseOutlierMask, offset: 0, index: 2)
                $0.setBuffer(phaseRowNorm, offset: 0, index: 3)
                $0.setBuffer(singleK.codes, offset: 0, index: 4)
                $0.setBuffer(singleK.residualSigns, offset: 0, index: 5)
                $0.setBuffer(singleK.outlierMask, offset: 0, index: 6)
                $0.setBuffer(singleK.metadata, offset: 0, index: 7)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 8)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 9)
                $0.setBuffer(kernel.keySigns.residual, offset: 0, index: 10)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let separateQuantizeKV = try await benchmark(name: "turboquant_quantize_small_aggressive_separate_kv", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(kernel.quantizeAggressiveSmallPipeline)
                $0.setBuffer(kSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(fusedK.codes, offset: 0, index: 1)
                $0.setBuffer(fusedK.residualSigns, offset: 0, index: 2)
                $0.setBuffer(fusedK.outlierMask, offset: 0, index: 3)
                $0.setBuffer(fusedK.metadata, offset: 0, index: 4)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 5)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 6)
                $0.setBuffer(kernel.keySigns.residual, offset: 0, index: 7)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )

                $0.setBuffer(vSourceBuffer, offset: 0, index: 0)
                $0.setBuffer(fusedV.codes, offset: 0, index: 1)
                $0.setBuffer(fusedV.residualSigns, offset: 0, index: 2)
                $0.setBuffer(fusedV.outlierMask, offset: 0, index: 3)
                $0.setBuffer(fusedV.metadata, offset: 0, index: 4)
                $0.setBytes(&quantizeParams, length: MemoryLayout<TurboQuantQuantizeParamsBench>.stride, index: 5)
                $0.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 6)
                $0.setBuffer(kernel.valueSigns.residual, offset: 0, index: 7)
                $0.dispatchThreadgroups(
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let fusedQKV = try await benchmark(name: "dequant_q8_0_fused_qkv_turbo", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(fusedQKVTurboPipeline)
                $0.setBuffer(wqBuffer, offset: 0, index: 0)
                $0.setBuffer(wkBuffer, offset: 0, index: 1)
                $0.setBuffer(wvBuffer, offset: 0, index: 2)
                $0.setBuffer(hiddenBuffer, offset: 0, index: 3)
                $0.setBuffer(fusedQOutput, offset: 0, index: 4)
                $0.setBuffer(fusedKOutput, offset: 0, index: 5)
                $0.setBuffer(fusedVOutput, offset: 0, index: 6)
                $0.setBytes(&fusedQKVParams, length: MemoryLayout<FusedQKVParamsBench>.stride, index: 7)
                $0.setBuffer(normBuffer, offset: 0, index: 8)
                $0.dispatchThreadgroups(
                    MTLSize(width: (qRows + kvRows + kvRows + 1) / 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
        }

        let decodeAttention = try await benchmark(name: "gqa_attention_turboquant_decode_aggressive", warmup: 3, iterations: 20) {
            try encodeAndWait {
                $0.setComputePipelineState(kernel.decodeAttentionAggressivePipeline)
                $0.setBuffer(qBuffer, offset: 0, index: 0)
                $0.setBuffer(attentionCache.key.codes, offset: 0, index: 1)
                $0.setBuffer(attentionCache.key.residualSigns, offset: 0, index: 2)
                $0.setBuffer(attentionCache.key.outlierMask, offset: 0, index: 3)
                $0.setBuffer(attentionCache.key.metadata, offset: 0, index: 4)
                $0.setBuffer(attentionCache.value.codes, offset: 0, index: 5)
                $0.setBuffer(attentionCache.value.residualSigns, offset: 0, index: 6)
                $0.setBuffer(attentionCache.value.outlierMask, offset: 0, index: 7)
                $0.setBuffer(attentionCache.value.metadata, offset: 0, index: 8)
                $0.setBuffer(attentionOutput, offset: 0, index: 9)
                $0.setBytes(&attentionParams, length: MemoryLayout<TurboQuantAttentionParamsBench>.stride, index: 10)
                $0.setBuffer(kernel.keySigns.rotation, offset: 0, index: 11)
                $0.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
                $0.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 13)
                $0.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
                $0.dispatchThreadgroups(
                    MTLSize(width: 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
                )
            }
        }

        let fusedQKVMs = String(format: "%.3f", fusedQKV.perIterationMs)
        let singleQuantizeMs = String(format: "%.3f", singleQuantize.perIterationMs)
        let fusedQuantizeMs = String(format: "%.3f", fusedQuantize.perIterationMs)
        let phase1Ms = String(format: "%.3f", quantizePhase1.perIterationMs)
        let rotateOnlyMs = String(format: "%.3f", quantizeRotateOnly.perIterationMs)
        let selectOnlyMs = String(format: "%.3f", quantizeSelectOnly.perIterationMs)
        let selectOnlyBitonicMs = String(format: "%.3f", quantizeSelectOnlyBitonic.perIterationMs)
        let phase2Ms = String(format: "%.3f", quantizePhase2.perIterationMs)
        let separateQuantizeKVms = String(format: "%.3f", separateQuantizeKV.perIterationMs)
        let decodeAttentionMs = String(format: "%.3f", decodeAttention.perIterationMs)
        let fusedPlusQuantize = fusedQKV.perIterationMs + fusedQuantize.perIterationMs
        let fusedPlusQuantizeMs = String(format: "%.3f", fusedPlusQuantize)
        let decodeVsSingle = String(
            format: "%.2f",
            decodeAttention.perIterationMs / max(singleQuantize.perIterationMs, 1e-9)
        )
        let decodeVsFused = String(
            format: "%.2f",
            decodeAttention.perIterationMs / max(fusedQuantize.perIterationMs, 1e-9)
        )
        let quantizeVsFusedQKV = String(
            format: "%.2f",
            fusedQuantize.perIterationMs / max(fusedQKV.perIterationMs, 1e-9)
        )
        let fusedVsSeparateKV = String(
            format: "%.2f",
            fusedQuantize.perIterationMs / max(separateQuantizeKV.perIterationMs, 1e-9)
        )

        print("BENCHMARK: turboquant_fused_qkv_ms \(fusedQKVMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_ms \(singleQuantizeMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_kv_ms \(fusedQuantizeMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_phase1_ms \(phase1Ms) ms/op")
        print("BENCHMARK: turboquant_small_quantize_rotate_only_ms \(rotateOnlyMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_select_only_ms \(selectOnlyMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_select_only_bitonic_ms \(selectOnlyBitonicMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_phase2_ms \(phase2Ms) ms/op")
        print("BENCHMARK: turboquant_small_quantize_separate_kv_ms \(separateQuantizeKVms) ms/op")
        print("BENCHMARK: turboquant_decode_attention_ms \(decodeAttentionMs) ms/op")
        print("BENCHMARK: turboquant_fused_qkv_plus_small_quantize_kv_ms \(fusedPlusQuantizeMs) ms/op")
        print("BENCHMARK: turboquant_decode_attention_vs_small_quantize \(decodeVsSingle)x")
        print("BENCHMARK: turboquant_decode_attention_vs_small_quantize_kv \(decodeVsFused)x")
        print("BENCHMARK: turboquant_small_quantize_kv_vs_fused_qkv \(quantizeVsFusedQKV)x")
        print("BENCHMARK: turboquant_small_quantize_kv_vs_separate_kv \(fusedVsSeparateKV)x")
    }

    private func encodeAndWait(
        _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestError.noMetal
        }
        try body(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
    }

    private func makeTurboBuffers(rowCount: Int, layout: TurboQuantLayout) throws -> TurboQuantMetalBuffers {
        let codeLength = rowCount * layout.codeWordsPerRow * MemoryLayout<UInt32>.stride
        let residualLength = rowCount * TurboQuantLayout.residualWordsPerRow * MemoryLayout<UInt32>.stride
        let maskLength = rowCount * TurboQuantLayout.outlierMaskWordsPerRow * MemoryLayout<UInt32>.stride
        let metadataLength = rowCount * TurboQuantLayout.metadataScalarsPerRow * MemoryLayout<Float>.stride

        guard let codes = device.makeBuffer(length: codeLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let residualSigns = device.makeBuffer(length: residualLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let outlierMask = device.makeBuffer(length: maskLength, options: [.storageModeShared, .hazardTrackingModeUntracked]),
              let metadata = device.makeBuffer(length: metadataLength, options: [.storageModeShared, .hazardTrackingModeUntracked]) else {
            throw TestError.noMetal
        }

        return TurboQuantMetalBuffers(
            codes: codes,
            residualSigns: residualSigns,
            outlierMask: outlierMask,
            metadata: metadata
        )
    }

    private func makeAttentionCache(rowCount: Int) throws -> (key: TurboQuantMetalBuffers, value: TurboQuantMetalBuffers) {
        let cache = try KVCache(
            device: device,
            maxSeqLen: rowCount,
            numLayers: 1,
            numKVHeads: 8,
            headDim: 128,
            precision: .turboQuantAggressive
        )

        for row in 0..<rowCount {
            let phase = Float(row) * 0.013
            let keys = (0..<8).flatMap { head in
                makeSignal(phase: phase + Float(head) * 0.07)
            }
            let values = (0..<8).flatMap { head in
                makeSignal(phase: 0.5 + phase + Float(head) * 0.03)
            }
            try cache.append(layer: 0, keys: keys, values: values)
        }
        return try cache.turboQuantMetalBuffers(layer: 0)
    }

    private func makeRows(rowCount: Int, phase: Float) -> [Float] {
        (0..<rowCount).flatMap { row in
            makeSignal(phase: phase + Float(row) * 0.09)
        }
    }

    private func makeQuery(headCount: Int) -> [Float] {
        (0..<headCount).flatMap { head in
            makeSignal(phase: 0.23 + Float(head) * 0.05)
        }
    }

    private func makeHidden(dim: Int, phase: Float) -> [Float] {
        (0..<dim).map { index in
            let x = Float(index)
            return sin(x * 0.021 + phase) + 0.4 * cos(x * 0.017 - phase)
        }
    }

    private func makeNorm(dim: Int) -> [Float] {
        (0..<dim).map { index in
            0.92 + Float(index % 11) * 0.007
        }
    }

    private func makeSignal(phase: Float) -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.15 + phase) + 0.35 * cos(x * 0.05 - phase)
        }
    }

    private func makeQ8Weights(rows: Int, blocksPerRow: Int, phase: UInt8) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: rows * blocksPerRow * 34)
        for row in 0..<rows {
            for block in 0..<blocksPerRow {
                let base = (row * blocksPerRow + block) * 34
                let scale = Float16(0.03125 + Double((row + block) % 13) * 0.002)
                let rawScale = scale.bitPattern
                bytes[base] = UInt8(rawScale & 0x00ff)
                bytes[base + 1] = UInt8((rawScale >> 8) & 0x00ff)
                for i in 0..<32 {
                    let v = Int8(bitPattern: UInt8(truncatingIfNeeded: row &+ block &+ i &+ Int(phase)))
                    bytes[base + 2 + i] = UInt8(bitPattern: v)
                }
            }
        }
        return bytes
    }
}

private struct TurboQuantQuantizeParamsBench {
    var rowCount: UInt32
    var sourceRowStride: UInt32
    var destinationRowBase: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var highPrecisionChannelCount: UInt32
    var reserved: UInt32
}

private struct TurboQuantAttentionParamsBench {
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

private struct FusedQKVParamsBench {
    var qRows: UInt32
    var kvRows: UInt32
    var cols: UInt32
    var blocksPerRow: UInt32
    var tokenCount: UInt32
    var rmsEps: Float
}
