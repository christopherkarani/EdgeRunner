import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// Tests for dual-table RoPE supporting partial rotary factor (pRoPE),
/// required for Gemma 4 global-attention layers which rotate only the
/// first `partialRotaryFactor * headDim` channels and pass the remainder
/// through unchanged.
///
/// Note: the existing `RoPEKernel` API is array-in / array-out and async,
/// not in-place. Tests use `execute(...)` (interleaved pairs) which matches
/// the `rope_f32` shader variant used across the codebase when callers
/// bypass the fused kernels.
@Suite("RoPE dual tables (Gemma 4)")
struct RoPEDualTableTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = commandQueue
    }

    // MARK: - Global pRoPE (partial rotation)

    @Test("Global pRoPE rotates only first partial*head_dim channels (base=1e6, partial=0.25)")
    func pRoPERotatesOnlyPartial() async throws {
        let seqLen = 1
        let numHeads = 1
        let headDim = 512
        let partialRotaryFactor: Float = 0.25
        // All-ones input so any rotation is detectable as != 1.0.
        let input = [Float](repeating: 1.0, count: seqLen * numHeads * headDim)

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 10,
            theta: 1_000_000,
            scalingFactor: 1,
            partialRotaryFactor: partialRotaryFactor,
            commandQueue: commandQueue
        )

        // First `partialRotaryFactor * headDim` channels should have been rotated
        // (at least some will differ from 1.0 given nonzero startPos).
        let rotatedChannelCount = Int(Float(headDim) * partialRotaryFactor)
        #expect(rotatedChannelCount == 128,
                "Sanity: headDim=512 * partial=0.25 should yield 128 rotated channels")

        let rotatedChanged = (0..<rotatedChannelCount).contains { result[$0] != 1.0 }
        #expect(rotatedChanged, "Expected at least one of the first \(rotatedChannelCount) channels to change under pRoPE")

        // Remaining channels must pass through unchanged (exactly 1.0).
        for i in rotatedChannelCount..<headDim {
            #expect(result[i] == 1.0, "channel \(i) was rotated despite partialRotaryFactor=\(partialRotaryFactor)")
        }
    }

    // MARK: - Local RoPE (full rotation)

    @Test("Local RoPE rotates all channels when partial=1.0 (base=1e4)")
    func localRotatesAll() async throws {
        let seqLen = 1
        let numHeads = 1
        let headDim = 256
        let input = [Float](repeating: 1.0, count: seqLen * numHeads * headDim)

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 10,
            theta: 10_000,
            scalingFactor: 1,
            partialRotaryFactor: 1.0,
            commandQueue: commandQueue
        )

        // With interleaved pairs, each pair (2k, 2k+1) is a rotation of (1, 1).
        // At startPos=10 and theta=1e4, every pair has some nonzero rotation angle;
        // no pair can map (1, 1) back to (1, 1) except at angle=0 (position 0).
        // So every output channel should differ from 1.0 (modulo float rounding).
        // We check per-pair that at least ONE of the two channels changed, to avoid
        // the rare lucky case where a single channel happens to equal 1 after rotation.
        let halfDim = headDim / 2
        for pair in 0..<halfDim {
            let c0 = result[2 * pair]
            let c1 = result[2 * pair + 1]
            let pairChanged = (c0 != 1.0) || (c1 != 1.0)
            #expect(pairChanged, "pair \(pair) (channels \(2*pair),\(2*pair+1)) was NOT rotated despite partialRotaryFactor=1.0")
        }
    }

    // MARK: - Backward compatibility

    @Test("Backward compatibility: omitted partialRotaryFactor defaults to full rotation")
    func defaultsToFullRotation() async throws {
        let seqLen = 1
        let numHeads = 1
        let headDim = 128
        let input = [Float](repeating: 1.0, count: seqLen * numHeads * headDim)

        let kernel = try RoPEKernel(device: device)
        // Call without `partialRotaryFactor` — existing callers must keep working.
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 5,
            theta: 10_000,
            commandQueue: commandQueue
        )

        let halfDim = headDim / 2
        for pair in 0..<halfDim {
            let c0 = result[2 * pair]
            let c1 = result[2 * pair + 1]
            let pairChanged = (c0 != 1.0) || (c1 != 1.0)
            #expect(pairChanged, "pair \(pair) was NOT rotated despite default partialRotaryFactor (should be 1.0)")
        }
    }

    // MARK: - Cross-check: Gemma 4 global layer sizes

    @Test("Gemma 4 global layer shape: headDim=512, partial=0.25 rotates exactly 128 channels")
    func gemma4GlobalShape() async throws {
        let seqLen = 1
        let numHeads = 1
        let headDim = 512
        let partialRotaryFactor: Float = 0.25
        let input = (0..<headDim).map { i in Float(i) + 1.0 }  // distinct nonzero values

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            startPos: 7,
            theta: 1_000_000,
            scalingFactor: 1,
            partialRotaryFactor: partialRotaryFactor,
            commandQueue: commandQueue
        )

        // First 128 channels: at least one pair must have changed (rotated).
        var anyRotated = false
        for pair in 0..<64 {  // 128 / 2 = 64 pairs
            if result[2 * pair] != input[2 * pair] || result[2 * pair + 1] != input[2 * pair + 1] {
                anyRotated = true
                break
            }
        }
        #expect(anyRotated, "expected some rotation in first 128 channels")

        // Channels 128..<512 must be bit-exact pass-through.
        for i in 128..<headDim {
            #expect(result[i] == input[i], "channel \(i) changed despite being beyond partial-rotary boundary")
        }
    }

    // MARK: - NeoX layout pRoPE

    /// Dispatches the NeoX f32 RoPE kernel (`rope_neox_f32`) directly — the public
    /// `execute()` / `applyToQK()` APIs wrap `pipelineF32` (interleaved layout), so this
    /// test is the only coverage for the NeoX pass-through guard. The NeoX kernel is
    /// the real hot path used by Gemma 4 global-attention layers (partial=0.25).
    ///
    /// NeoX pair layout: (i, i + halfDim). With headDim=512, halfDim=256, partial=0.25
    /// → rotatedPairs = floor(256 * 0.25) = 64. Rotated channels are
    /// {0..63} ∪ {256..319}; pass-through channels are {64..255} ∪ {320..511}.
    @Test("NeoX pRoPE (partial=0.25): rotated pairs = (i, i+halfDim) for i<64; all other channels pass through")
    func neoxPRoPEPassThrough() async throws {
        let seqLen = 1
        let numHeads = 1
        let headDim = 512
        let halfDim = headDim / 2
        let partialRotaryFactor: Float = 0.25
        let rotatedPairs = Int(Float(halfDim) * partialRotaryFactor)
        #expect(rotatedPairs == 64, "Sanity: 256 * 0.25 should yield 64 NeoX rotated pairs")

        // Q and K: all ones, nonzero startPos so rotated channels differ from 1.0.
        let count = seqLen * numHeads * headDim
        let q = [Float](repeating: 1.0, count: count)

        let kernel = try RoPEKernel(device: device)
        var params = ERRoPEParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            startPos: 10,
            theta: 1_000_000,
            scalingFactor: 1,
            partialRotaryFactor: partialRotaryFactor
        )

        let inBuf = device.makeBuffer(bytes: q,
                                      length: count * MemoryLayout<Float>.stride,
                                      options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                       options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw TestError.noMetal
        }
        enc.setComputePipelineState(kernel.pipelineNeoX)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)
        let tgSize = MTLSize(
            width: min(halfDim, kernel.pipelineNeoX.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        enc.dispatchThreads(
            MTLSize(width: halfDim, height: numHeads, depth: seqLen),
            threadsPerThreadgroup: tgSize
        )
        enc.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()
        if let error = cmdBuf.error { throw error }

        let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let qOut = Array(UnsafeBufferPointer(start: ptr, count: count))

        // Pass-through regions: channels [rotatedPairs, halfDim) and [halfDim+rotatedPairs, headDim).
        for i in rotatedPairs..<halfDim {
            #expect(qOut[i] == 1.0,
                    "NeoX Q channel \(i) should be untouched (pass-through; partial=\(partialRotaryFactor))")
        }
        for i in (halfDim + rotatedPairs)..<headDim {
            #expect(qOut[i] == 1.0,
                    "NeoX Q channel \(i) should be untouched (pass-through upper half)")
        }

        // At least one rotated channel in the first `rotatedPairs` or its NeoX partner must differ
        // from 1.0. At startPos=10 with theta=1e6, at least some pairs will have nonzero rotation.
        let anyRotated = (0..<rotatedPairs).contains { i in
            qOut[i] != 1.0 || qOut[i + halfDim] != 1.0
        }
        #expect(anyRotated,
                "Expected rotation on at least one NeoX pair (i, i+halfDim) for i<\(rotatedPairs)")
    }
}
