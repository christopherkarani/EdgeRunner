import Metal
import EdgeRunnerSharedTypes

public final class RoPEKernel: Sendable {
    public let pipelineF32: MTLComputePipelineState
    public let pipelineNeoX: MTLComputePipelineState
    public let pipelineNeoXF16Out: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipelineF32 = try registry.pipeline(for: "rope_f32")
        self.pipelineNeoX = try registry.pipeline(for: "rope_neox_f32")
        self.pipelineNeoXF16Out = try registry.pipeline(for: "rope_neox_f32_to_f16")
    }

    /// Apply RoPE rotation. `partialRotaryFactor` controls pRoPE: the fraction of
    /// `headDim` channels that get rotated. 1.0 = full rotation (standard RoPE, default).
    /// Values <1.0 enable partial rotary (e.g. Gemma 4 global layers use 0.25).
    public func execute(
        input: [Float],
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1,
        partialRotaryFactor: Float = 1.0,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(input.count == seqLen * numHeads * headDim)
        precondition(partialRotaryFactor >= 0 && partialRotaryFactor <= 1,
                     "partialRotaryFactor must be in [0, 1]")

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERRoPEParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            startPos: UInt32(startPos),
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RoPEError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)

        let halfDim = headDim / 2
        let gridSize = MTLSize(width: halfDim, height: numHeads, depth: seqLen)
        let threadgroupSize = MTLSize(
            width: min(halfDim, pipelineF32.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: pointer, count: input.count))
    }

    /// Apply RoPE to Q and K tensors in a single command buffer (1 sync point instead of 2).
    public func applyToQK(
        q: [Float],
        k: [Float],
        seqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1,
        partialRotaryFactor: Float = 1.0,
        commandQueue: MTLCommandQueue
    ) async throws -> ([Float], [Float]) {
        try await applyToQK(
            pipeline: pipelineF32,
            q: q,
            k: k,
            seqLen: seqLen,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor,
            commandQueue: commandQueue
        )
    }

    /// Apply NeoX-layout RoPE to Q and K tensors in a single command buffer.
    /// Gemma 4 uses NeoX-style split-half rotary layout with proportional pRoPE.
    public func applyToQKNeoX(
        q: [Float],
        k: [Float],
        seqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1,
        partialRotaryFactor: Float = 1.0,
        commandQueue: MTLCommandQueue
    ) async throws -> ([Float], [Float]) {
        try await applyToQK(
            pipeline: pipelineNeoX,
            q: q,
            k: k,
            seqLen: seqLen,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor,
            commandQueue: commandQueue
        )
    }

    private func applyToQK(
        pipeline: MTLComputePipelineState,
        q: [Float],
        k: [Float],
        seqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float,
        partialRotaryFactor: Float,
        commandQueue: MTLCommandQueue
    ) async throws -> ([Float], [Float]) {
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(q.count == seqLen * numHeads * headDim)
        precondition(k.count == seqLen * numKVHeads * headDim)
        precondition(partialRotaryFactor >= 0 && partialRotaryFactor <= 1,
                     "partialRotaryFactor must be in [0, 1]")

        let qInBuf = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let qOutBuf = device.makeBuffer(length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let kInBuf = device.makeBuffer(bytes: k, length: k.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let kOutBuf = device.makeBuffer(length: k.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw RoPEError.encodingFailed }
        let halfDim = headDim / 2

        // Encode Q RoPE
        do {
            var params = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numHeads), headDim: UInt32(headDim),
                                       startPos: UInt32(startPos), theta: theta, scalingFactor: scalingFactor,
                                       partialRotaryFactor: partialRotaryFactor)
            guard let enc = cmdBuf.makeComputeCommandEncoder() else { throw RoPEError.encodingFailed }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(qInBuf, offset: 0, index: 0)
            enc.setBuffer(qOutBuf, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
            let tgSize = MTLSize(width: min(halfDim, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            enc.dispatchThreads(MTLSize(width: halfDim, height: numHeads, depth: seqLen), threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        // Encode K RoPE (same command buffer)
        do {
            var params = ERRoPEParams(seqLen: UInt32(seqLen), numHeads: UInt32(numKVHeads), headDim: UInt32(headDim),
                                       startPos: UInt32(startPos), theta: theta, scalingFactor: scalingFactor,
                                       partialRotaryFactor: partialRotaryFactor)
            guard let enc = cmdBuf.makeComputeCommandEncoder() else { throw RoPEError.encodingFailed }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(kInBuf, offset: 0, index: 0)
            enc.setBuffer(kOutBuf, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
            let tgSize = MTLSize(width: min(halfDim, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            enc.dispatchThreads(MTLSize(width: halfDim, height: numKVHeads, depth: seqLen), threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        let qPtr = qOutBuf.contents().bindMemory(to: Float.self, capacity: q.count)
        let kPtr = kOutBuf.contents().bindMemory(to: Float.self, capacity: k.count)
        return (Array(UnsafeBufferPointer(start: qPtr, count: q.count)),
                Array(UnsafeBufferPointer(start: kPtr, count: k.count)))
    }

    /// Encode a RoPE dispatch into an existing command buffer without committing.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer, outputBuffer: MTLBuffer,
        seqLen: Int, numHeads: Int, headDim: Int,
        startPos: Int, theta: Float, scalingFactor: Float = 1,
        partialRotaryFactor: Float = 1.0
    ) throws {
        try encode(
            pipeline: pipelineF32,
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor
        )
    }

    public func encodeNeoX(
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer, outputBuffer: MTLBuffer,
        seqLen: Int, numHeads: Int, headDim: Int,
        startPos: Int, theta: Float, scalingFactor: Float = 1,
        partialRotaryFactor: Float = 1.0
    ) throws {
        try encode(
            pipeline: pipelineNeoX,
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor
        )
    }

    private func encode(
        pipeline: MTLComputePipelineState,
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float,
        partialRotaryFactor: Float
    ) throws {
        precondition(partialRotaryFactor >= 0 && partialRotaryFactor <= 1,
                     "partialRotaryFactor must be in [0, 1]")

        var params = ERRoPEParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            startPos: UInt32(startPos),
            theta: theta,
            scalingFactor: scalingFactor,
            partialRotaryFactor: partialRotaryFactor
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RoPEError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)

        let halfDim = headDim / 2
        let gridSize = MTLSize(width: halfDim, height: numHeads, depth: seqLen)
        let threadgroupSize = MTLSize(
            width: min(halfDim, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}

public enum RoPEError: Error, Sendable {
    case encodingFailed
}
