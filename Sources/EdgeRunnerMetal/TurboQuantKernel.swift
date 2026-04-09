import Metal

public struct TurboQuantSignBuffers: @unchecked Sendable {
    public let rotation: MTLBuffer
    public let residual: MTLBuffer
}

public struct TurboQuantAttentionBuffers: @unchecked Sendable {
    public let key: TurboQuantMetalBuffers
    public let value: TurboQuantMetalBuffers
}

public final class TurboQuantKernel: @unchecked Sendable {
    public let quantizePipeline: MTLComputePipelineState
    public let quantizeAggressiveSmallPipeline: MTLComputePipelineState
    public let quantizeAggressiveSmallKPipeline: MTLComputePipelineState
    public let quantizeAggressiveSmallKVPipeline: MTLComputePipelineState
    public let attentionPipeline: MTLComputePipelineState
    public let q8KeyTurboValueAttentionPipeline: MTLComputePipelineState
    public let decodeAttentionPipeline: MTLComputePipelineState
    public let decodeAttentionQ8KeyTurboValuePipeline: MTLComputePipelineState
    public let debugDecodeScoreTermsPipeline: MTLComputePipelineState
    public let decodeAttentionDenseVPipeline: MTLComputePipelineState
    public let decodeAttentionAggressivePipeline: MTLComputePipelineState
    public let decodeAttentionAggressiveDenseVPipeline: MTLComputePipelineState
    public let decodeAttentionAggressiveHybridVPipeline: MTLComputePipelineState

    public let keySigns: TurboQuantSignBuffers
    public let valueSigns: TurboQuantSignBuffers
    public let keyPlanarRotation: MTLBuffer
    public let valuePlanarRotation: MTLBuffer

    public init(device: MTLDevice) throws {
        let registry = try KernelRegistry(device: device)
        self.quantizePipeline = try registry.pipeline(for: "turboquant_quantize_rows")
        self.quantizeAggressiveSmallPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive")
        self.quantizeAggressiveSmallKPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_k")
        self.quantizeAggressiveSmallKVPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_kv")
        self.attentionPipeline = try registry.pipeline(for: "gqa_attention_turboquant")
        self.q8KeyTurboValueAttentionPipeline = try registry.pipeline(for: "gqa_attention_q8k_turboquant")
        self.decodeAttentionPipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode")
        self.decodeAttentionQ8KeyTurboValuePipeline = try registry.pipeline(for: "gqa_attention_q8k_turboquant_decode")
        self.debugDecodeScoreTermsPipeline = try registry.pipeline(for: "turboquant_debug_decode_score_terms")
        self.decodeAttentionDenseVPipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode_f16v")
        self.decodeAttentionAggressivePipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode_aggressive")
        self.decodeAttentionAggressiveDenseVPipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode_aggressive_f16v")
        self.decodeAttentionAggressiveHybridVPipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode_aggressive_k_f16v")
        self.keySigns = try Self.makeSignBuffers(
            device: device,
            rotationSeed: TurboQuantSeeds.keyRotation,
            residualSeed: TurboQuantSeeds.keyResidual
        )
        self.keyPlanarRotation = try Self.makePlanarRotationBuffer(
            device: device,
            seed: TurboQuantSeeds.keyRotation
        )
        self.valueSigns = try Self.makeSignBuffers(
            device: device,
            rotationSeed: TurboQuantSeeds.valueRotation,
            residualSeed: TurboQuantSeeds.valueResidual
        )
        self.valuePlanarRotation = try Self.makePlanarRotationBuffer(
            device: device,
            seed: TurboQuantSeeds.valueRotation
        )
    }

    private static func makeSignBuffers(
        device: MTLDevice,
        rotationSeed: UInt64,
        residualSeed: UInt64
    ) throws -> TurboQuantSignBuffers {
        let rotation = try makeSignBuffer(device: device, seed: rotationSeed)
        let residual = try makeSignBuffer(device: device, seed: residualSeed)
        return TurboQuantSignBuffers(rotation: rotation, residual: residual)
    }

    private static func makeSignBuffer(device: MTLDevice, seed: UInt64) throws -> MTLBuffer {
        let signs = TurboQuantTransform.signPattern(
            count: TurboQuantLayout.supportedDimension,
            seed: seed
        )
        guard let buffer = device.makeBuffer(
            bytes: signs,
            length: signs.count * MemoryLayout<Float>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw GQAError.encodingFailed
        }
        return buffer
    }

    private static func makePlanarRotationBuffer(device: MTLDevice, seed: UInt64) throws -> MTLBuffer {
        let coefficients = TurboQuantTransform.planarRotationBuffer(seed: seed)
        guard let buffer = device.makeBuffer(
            bytes: coefficients,
            length: coefficients.count * MemoryLayout<Float>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw GQAError.encodingFailed
        }
        return buffer
    }

    public func keyRotationBuffer(for preset: TurboQuantPreset) -> MTLBuffer {
        preset == .planar3 ? keyPlanarRotation : keySigns.rotation
    }

    public func valueRotationBuffer(for preset: TurboQuantPreset) -> MTLBuffer {
        preset == .planar3 ? valuePlanarRotation : valueSigns.rotation
    }
}
