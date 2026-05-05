import Metal
import EdgeRunnerSharedTypes

public final class GQAKernel: Sendable {
    public let pipelineF32: MTLComputePipelineState
    public let pipelineF32Wide: MTLComputePipelineState
    public let pipelineF32MaskedWide: MTLComputePipelineState
    public let pipelineF16KV: MTLComputePipelineState
    public let pipelineQ8KV: MTLComputePipelineState
    private let device: MTLDevice

    /// Must match the stack array sizes in GQA.metal (scores[16], probs[16]).
    public static let blockSize = 16
    public static let maxOptimizedHeadDim = 128
    public static let maxSupportedHeadDim = 512

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "gqa_attention_f32")
        self.pipelineF32Wide = try registry.pipeline(for: "gqa_attention_f32_wide")
        self.pipelineF32MaskedWide = try registry.pipeline(for: "gqa_attention_f32_masked_wide")
        self.pipelineF16KV = try registry.pipeline(for: "gqa_attention_f16kv")
        self.pipelineQ8KV = try registry.pipeline(for: "gqa_attention_q8kv")
    }

    public func execute(
        q: [Float],
        k: [Float],
        v: [Float],
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        causal: Bool,
        kvSeqLen: Int = 0,
        qOffset: Int = 0,
        additiveMask: [Float]? = nil,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let effectiveKVSeqLen = kvSeqLen > 0 ? kvSeqLen : seqLen
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        precondition(q.count == seqLen * numHeads * headDim)
        precondition(k.count == effectiveKVSeqLen * numKVHeads * headDim)
        precondition(v.count == effectiveKVSeqLen * numKVHeads * headDim)
        precondition(headDim > 0 && headDim <= Self.maxSupportedHeadDim, "headDim must be in 1...\(Self.maxSupportedHeadDim)")
        precondition(headDim % 4 == 0, "headDim must be divisible by 4")
        if let additiveMask {
            precondition(additiveMask.count == seqLen * effectiveKVSeqLen)
        }

        let outputCount = seqLen * numHeads * headDim

        let qBuffer = device.makeBuffer(
            bytes: q,
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let kBuffer = device.makeBuffer(
            bytes: k,
            length: k.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let vBuffer = device.makeBuffer(
            bytes: v,
            length: v.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let additiveMaskBuffer = additiveMask.map {
            device.makeBuffer(
                bytes: $0,
                length: $0.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )!
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GQAError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            qBuffer: qBuffer,
            kBuffer: kBuffer,
            vBuffer: vBuffer,
            outputBuffer: outputBuffer,
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            causal: causal,
            kvSeqLen: kvSeqLen,
            qOffset: qOffset,
            additiveMaskBuffer: additiveMaskBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    /// Encode a GQA attention dispatch into an existing command buffer without committing.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        qBuffer: MTLBuffer, kBuffer: MTLBuffer, vBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        seqLen: Int, headDim: Int,
        numHeads: Int, numKVHeads: Int,
        causal: Bool,
        kvSeqLen: Int = 0,
        qOffset: Int = 0,
        additiveMaskBuffer: MTLBuffer? = nil
    ) throws {
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        precondition(headDim > 0 && headDim <= Self.maxSupportedHeadDim, "headDim must be in 1...\(Self.maxSupportedHeadDim)")
        precondition(headDim % 4 == 0, "headDim must be divisible by 4")
        let groupSize = numHeads / numKVHeads

        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: 1.0 / sqrt(Float(headDim)),
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.blockSize),
            qBlockSize: UInt32(Self.blockSize),
            kvSeqLen: UInt32(kvSeqLen),
            qOffset: UInt32(qOffset)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GQAError.encodingFailed
        }

        let pipeline = selectPipeline(headDim: headDim, hasAdditiveMask: additiveMaskBuffer != nil)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)
        if let additiveMaskBuffer {
            encoder.setBuffer(additiveMaskBuffer, offset: 0, index: 5)
        }

        if headDim <= Self.maxOptimizedHeadDim && additiveMaskBuffer == nil {
            let qBlockCount = (seqLen + Self.blockSize - 1) / Self.blockSize
            let gridSize = MTLSize(width: qBlockCount, height: numHeads, depth: 1)
            let threadgroupSize = MTLSize(width: Self.blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else {
            let outputCount = seqLen * numHeads * headDim
            let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: outputCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()
    }

    private func selectPipeline(headDim: Int, hasAdditiveMask: Bool) -> MTLComputePipelineState {
        if hasAdditiveMask {
            return pipelineF32MaskedWide
        }
        if headDim <= Self.maxOptimizedHeadDim {
            return pipelineF32
        }
        return pipelineF32Wide
    }
}

public enum GQAError: Error, Sendable {
    case encodingFailed
}
