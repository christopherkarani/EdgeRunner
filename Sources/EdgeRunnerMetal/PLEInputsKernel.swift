import Foundation
import Metal

/// Parameters mirror the `PLEInputsParams` struct declared in `Shaders/PLE.metal`.
/// Defined in Swift (rather than EdgeRunnerSharedTypes) because this kernel has
/// no other Swift/C consumers — matches PLEGatherParams and similar small kernels.
private struct PLEInputsParams {
    var hidden: UInt32
    var perLayerDim: UInt32
    var numLayers: UInt32
    var batchSeq: UInt32
    var rmsEps: Float
    var scaleMix: Float
}

/// PLE inputs builder kernel for Gemma-style Per-Layer Embeddings.
///
/// For each `(batchSeq, layer)` slice computes
/// `per_layer_inputs[bs, ℓ, p] = (RMSNorm(proj[bs, ℓ, p]) + pleRows[bs, ℓ, p]) * scaleMix`
/// where `RMSNorm` is along the last dim (P) using Gemma's `(1 + w)` weight trick.
///
/// - `proj` is the already-1/√H-scaled output of the PLE projection GEMV with shape `[BS, L·P]`.
/// - `normWeight` is `per_layer_proj_norm.weight`, shape `[P]`.
/// - `pleRows` is the output of `PLEGatherKernel` (already scaled by √P), shape `[BS, L, P]`.
/// - Output shape: `[BS, L, P]`.
public struct PLEInputsKernel: Sendable {
    public let pipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        let registry = try KernelRegistry(device: device)
        self.pipeline = try registry.pipeline(for: "ple_inputs_build")
    }

    /// Test-only convenience — synchronous, creates its own command queue.
    ///
    /// The integration path should use an `encode(...)` variant that takes
    /// pre-allocated `MTLBuffer`s and a `MTLCommandBuffer` (follow-up).
    ///
    /// - Parameters:
    ///   - proj: Already-scaled projection buffer, shape `[BS, L·P]` laid out row-major.
    ///   - normWeight: Per-P RMSNorm weight (applied as `(1 + w)`), shape `[P]`.
    ///   - pleRows: Gathered PLE rows (already `√P`-scaled), shape `[BS, L, P]`.
    ///   - hiddenSize: Retained for API documentation / future validation; not consumed by the kernel.
    ///   - perLayerDim: P — per-layer embedding dim.
    ///   - numLayers: L — number of transformer layers.
    ///   - batchSeq: BS — batched sequence length.
    ///   - rmsEps: RMSNorm epsilon (Gemma default 1e-6).
    public func run(
        proj: [Float],
        normWeight: [Float],
        pleRows: [Float],
        hiddenSize: Int,
        perLayerDim: Int,
        numLayers: Int,
        batchSeq: Int,
        rmsEps: Float = 1e-6
    ) throws -> [Float] {
        precondition(
            proj.count == batchSeq * numLayers * perLayerDim,
            "proj shape mismatch: expected \(batchSeq * numLayers * perLayerDim), got \(proj.count)"
        )
        precondition(
            normWeight.count == perLayerDim,
            "normWeight shape mismatch: expected \(perLayerDim), got \(normWeight.count)"
        )
        precondition(
            pleRows.count == batchSeq * numLayers * perLayerDim,
            "pleRows shape mismatch: expected \(batchSeq * numLayers * perLayerDim), got \(pleRows.count)"
        )

        let outputCount = batchSeq * numLayers * perLayerDim
        guard outputCount > 0 else {
            return []
        }

        guard let commandQueue = device.makeCommandQueue() else {
            throw PLEInputsError.encodingFailed
        }

        guard let projBuffer = device.makeBuffer(
            bytes: proj,
            length: proj.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEInputsError.encodingFailed
        }
        guard let normBuffer = device.makeBuffer(
            bytes: normWeight,
            length: normWeight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEInputsError.encodingFailed
        }
        guard let pleBuffer = device.makeBuffer(
            bytes: pleRows,
            length: pleRows.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEInputsError.encodingFailed
        }
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw PLEInputsError.encodingFailed
        }

        let scaleMix = 1.0 / Float(2.0).squareRoot()
        var params = PLEInputsParams(
            hidden: UInt32(hiddenSize),
            perLayerDim: UInt32(perLayerDim),
            numLayers: UInt32(numLayers),
            batchSeq: UInt32(batchSeq),
            rmsEps: rmsEps,
            scaleMix: scaleMix
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLEInputsError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projBuffer, offset: 0, index: 0)
        encoder.setBuffer(normBuffer, offset: 0, index: 1)
        encoder.setBuffer(pleBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<PLEInputsParams>.stride, index: 4)

        let gridSize = MTLSize(
            width: perLayerDim,
            height: batchSeq * numLayers,
            depth: 1
        )
        let tgWidth = min(perLayerDim, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }
}

public enum PLEInputsError: Error, Sendable {
    case encodingFailed
}
