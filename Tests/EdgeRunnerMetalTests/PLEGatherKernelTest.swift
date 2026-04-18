import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("PLE Q8_0 gather")
struct PLEGatherKernelTests {
    @Test("Gathers single PLE row and scales by sqrt(P)")
    func gathersAndScales() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let P = 256
        let L = 42
        let vocab = 8
        let totalElems = vocab * L * P
        let rows = (0..<totalElems).map { i in Float(i % 137) / 137.0 - 0.5 }

        let (q8Bytes, dequant) = Self.quantizeToQ8_0(floats: rows, blockSize: 32)
        // Sanity: dequantized floats should match rows within Q8_0 tolerance.

        let kernel = try PLEGatherKernel(device: device)
        let tokens: [Int32] = [3, 7, 1]
        let out = try kernel.run(q8Table: q8Bytes, tokens: tokens, perLayerDim: P, numLayers: L)

        #expect(out.count == tokens.count * L * P)

        let sqrtP = Float(16.0)  // sqrt(256)
        for (tIdx, tok) in tokens.enumerated() {
            for ell in 0..<L {
                for p in 0..<P {
                    let srcIdx = Int(tok) * L * P + ell * P + p
                    let outIdx = tIdx * L * P + ell * P + p
                    let expected = dequant[srcIdx] * sqrtP
                    #expect(abs(out[outIdx] - expected) < 1e-2, "mismatch at t=\(tIdx) ell=\(ell) p=\(p)")
                }
            }
        }
    }

    // Test helper: Q8_0 quantize per block of 32 elements.
    // Returns packed bytes + the reference-dequantized floats (for expected-value comparison,
    // so tolerance accounts for Q8 rounding).
    static func quantizeToQ8_0(floats: [Float], blockSize: Int) -> (bytes: [UInt8], dequant: [Float]) {
        precondition(floats.count % blockSize == 0)
        var bytes: [UInt8] = []
        bytes.reserveCapacity((floats.count / blockSize) * 34)
        var dequant = [Float](repeating: 0, count: floats.count)
        for blockStart in stride(from: 0, to: floats.count, by: blockSize) {
            let slice = floats[blockStart..<(blockStart + blockSize)]
            let amax = slice.map { abs($0) }.max() ?? 0
            let scale: Float = amax == 0 ? 0 : amax / 127
            let scaleHalf = Float16(scale)
            let scaleBits = scaleHalf.bitPattern
            bytes.append(UInt8(scaleBits & 0xFF))
            bytes.append(UInt8((scaleBits >> 8) & 0xFF))
            for (i, v) in slice.enumerated() {
                let q = scale == 0 ? Int8(0) : Int8(clamping: Int(round(v / scale)))
                bytes.append(UInt8(bitPattern: q))
                dequant[blockStart + i] = Float(scaleHalf) * Float(q)
            }
        }
        return (bytes, dequant)
    }
}
