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
/// where `RMSNorm` is along the last dim (P) using Gemma 4's direct affine weight.
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

    /// Encode PLE input construction into an existing command buffer.
    ///
    /// The caller owns buffer allocation, command-buffer commit, and synchronization.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        projectionBuffer: MTLBuffer,
        normWeightBuffer: MTLBuffer,
        pleRowsBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        hiddenSize: Int,
        perLayerDim: Int,
        numLayers: Int,
        batchSeq: Int,
        rmsEps: Float = 1e-6
    ) throws {
        guard hiddenSize > 0, perLayerDim > 0, numLayers > 0, batchSeq >= 0 else {
            throw PLEInputsError.invalidShape(
                "hiddenSize, perLayerDim, and numLayers must be positive; batchSeq must be non-negative"
            )
        }

        let outputCount = batchSeq * numLayers * perLayerDim
        guard projectionBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLEInputsError.invalidShape("projectionBuffer is too small for \(outputCount) Float values")
        }
        guard normWeightBuffer.length >= perLayerDim * MemoryLayout<Float>.stride else {
            throw PLEInputsError.invalidShape("normWeightBuffer is too small for \(perLayerDim) Float values")
        }
        guard pleRowsBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLEInputsError.invalidShape("pleRowsBuffer is too small for \(outputCount) Float values")
        }
        guard outputBuffer.length >= outputCount * MemoryLayout<Float>.stride else {
            throw PLEInputsError.invalidShape("outputBuffer is too small for \(outputCount) Float values")
        }
        guard outputCount > 0 else {
            return
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

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PLEInputsError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projectionBuffer, offset: 0, index: 0)
        encoder.setBuffer(normWeightBuffer, offset: 0, index: 1)
        encoder.setBuffer(pleRowsBuffer, offset: 0, index: 2)
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
    }

    /// Test-only convenience — synchronous, creates its own command queue.
    ///
    /// The integration path should use an `encode(...)` variant that takes
    /// pre-allocated `MTLBuffer`s and a `MTLCommandBuffer` (follow-up).
    ///
    /// - Parameters:
    ///   - proj: Already-scaled projection buffer, shape `[BS, L·P]` laid out row-major.
    ///   - normWeight: Per-P RMSNorm weight, shape `[P]`.
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
        guard proj.count == batchSeq * numLayers * perLayerDim else {
            throw PLEInputsError.invalidShape(
                "proj must have batchSeq*numLayers*perLayerDim = \(batchSeq * numLayers * perLayerDim) elements, got \(proj.count)"
            )
        }
        guard normWeight.count == perLayerDim else {
            throw PLEInputsError.invalidShape(
                "normWeight must have perLayerDim = \(perLayerDim) elements, got \(normWeight.count)"
            )
        }
        guard pleRows.count == batchSeq * numLayers * perLayerDim else {
            throw PLEInputsError.invalidShape(
                "pleRows must have batchSeq*numLayers*perLayerDim = \(batchSeq * numLayers * perLayerDim) elements, got \(pleRows.count)"
            )
        }

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

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PLEInputsError.encodingFailed
        }

        try encode(
            commandBuffer: commandBuffer,
            projectionBuffer: projBuffer,
            normWeightBuffer: normBuffer,
            pleRowsBuffer: pleBuffer,
            outputBuffer: outputBuffer,
            hiddenSize: hiddenSize,
            perLayerDim: perLayerDim,
            numLayers: numLayers,
            batchSeq: batchSeq,
            rmsEps: rmsEps
        )

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
    case invalidShape(String)
}
