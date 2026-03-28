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
        let layout = try TurboQuantLayout(preset: .aggressive)
        let rowCount = 8
        let sourceRowStride = 128
        let destinationRowBase = 0

        let kSource = makeRows(rowCount: rowCount, phase: 0.11)
        let vSource = makeRows(rowCount: rowCount, phase: 0.47)
        let q = makeQuery(headCount: 16)

        let kSourceBuffer = device.makeBuffer(bytes: kSource, length: kSource.count * MemoryLayout<Float>.stride)!
        let vSourceBuffer = device.makeBuffer(bytes: vSource, length: vSource.count * MemoryLayout<Float>.stride)!
        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride)!

        let singleK = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let fusedK = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let fusedV = try makeTurboBuffers(rowCount: rowCount, layout: layout)
        let attentionOutput = device.makeBuffer(length: 16 * 128 * MemoryLayout<Float>.stride)!

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

        let singleQuantizeMs = String(format: "%.3f", singleQuantize.perIterationMs)
        let fusedQuantizeMs = String(format: "%.3f", fusedQuantize.perIterationMs)
        let decodeAttentionMs = String(format: "%.3f", decodeAttention.perIterationMs)
        let decodeVsSingle = String(
            format: "%.2f",
            decodeAttention.perIterationMs / max(singleQuantize.perIterationMs, 1e-9)
        )
        let decodeVsFused = String(
            format: "%.2f",
            decodeAttention.perIterationMs / max(fusedQuantize.perIterationMs, 1e-9)
        )

        print("BENCHMARK: turboquant_small_quantize_ms \(singleQuantizeMs) ms/op")
        print("BENCHMARK: turboquant_small_quantize_kv_ms \(fusedQuantizeMs) ms/op")
        print("BENCHMARK: turboquant_decode_attention_ms \(decodeAttentionMs) ms/op")
        print("BENCHMARK: turboquant_decode_attention_vs_small_quantize \(decodeVsSingle)x")
        print("BENCHMARK: turboquant_decode_attention_vs_small_quantize_kv \(decodeVsFused)x")
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

    private func makeSignal(phase: Float) -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.15 + phase) + 0.35 * cos(x * 0.05 - phase)
        }
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
