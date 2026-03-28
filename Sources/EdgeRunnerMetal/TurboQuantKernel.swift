import Metal

public struct TurboQuantSignBuffers: @unchecked Sendable {
    public let rotation: MTLBuffer
    public let residual: MTLBuffer
}

public struct TurboQuantAttentionBuffers: @unchecked Sendable {
    public let key: TurboQuantMetalBuffers
    public let value: TurboQuantMetalBuffers
}

public final class TurboQuantKernel: Sendable {
    public let quantizePipeline: MTLComputePipelineState
    public let quantizeAggressiveSmallPipeline: MTLComputePipelineState
    public let quantizeAggressiveSmallKVPipeline: MTLComputePipelineState
    public let attentionPipeline: MTLComputePipelineState
    public let decodeAttentionPipeline: MTLComputePipelineState
    public let decodeAttentionAggressivePipeline: MTLComputePipelineState

    public let keySigns: TurboQuantSignBuffers
    public let valueSigns: TurboQuantSignBuffers

    public init(device: MTLDevice) throws {
        let registry = try KernelRegistry(device: device)
        self.quantizePipeline = try registry.pipeline(for: "turboquant_quantize_rows")
        self.quantizeAggressiveSmallPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive")
        self.quantizeAggressiveSmallKVPipeline = try registry.pipeline(for: "turboquant_quantize_rows_small_aggressive_kv")
        self.attentionPipeline = try registry.pipeline(for: "gqa_attention_turboquant")
        self.decodeAttentionPipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode")
        self.decodeAttentionAggressivePipeline = try registry.pipeline(for: "gqa_attention_turboquant_decode_aggressive")
        self.keySigns = try Self.makeSignBuffers(
            device: device,
            rotationSeed: TurboQuantSeeds.keyRotation,
            residualSeed: TurboQuantSeeds.keyResidual
        )
        self.valueSigns = try Self.makeSignBuffers(
            device: device,
            rotationSeed: TurboQuantSeeds.valueRotation,
            residualSeed: TurboQuantSeeds.valueResidual
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
}
