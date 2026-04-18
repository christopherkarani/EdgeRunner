import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("PLE inputs builder")
struct PLEInputsKernelTests {
    @Test("Builds per_layer_inputs matching reference math (small shapes)")
    func buildsPerLayerInputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let H = 8, L = 3, P = 4, BS = 2
        let proj = (0..<(BS * L * P)).map { Float($0) * 0.01 - 0.05 }
        let normW = (0..<P).map { _ in Float(0.0) }
        let pleRows = (0..<(BS * L * P)).map { Float($0) * 0.02 - 0.1 }

        let kernel = try PLEInputsKernel(device: device)
        let out = try kernel.run(
            proj: proj,
            normWeight: normW,
            pleRows: pleRows,
            hiddenSize: H,
            perLayerDim: P,
            numLayers: L,
            batchSeq: BS,
            rmsEps: 1e-6
        )

        let expected = Self.referenceForward(
            proj: proj,
            norm: normW,
            pleRows: pleRows,
            L: L,
            P: P,
            BS: BS,
            rmsEps: 1e-6
        )
        #expect(out.count == expected.count)
        for i in 0..<out.count {
            #expect(abs(out[i] - expected[i]) < 1e-3, "mismatch at \(i): got \(out[i]) expected \(expected[i])")
        }
    }

    @Test("Non-zero normWeight: (1 + w) trick applied")
    func normWeightAppliedWithOnePlusTrick() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let H = 4, L = 1, P = 4, BS = 1
        let proj = [Float](repeating: 1.0, count: BS * L * P)
        let normW: [Float] = [0.5, -0.5, 1.0, 0.0]
        let pleRows = [Float](repeating: 0.0, count: BS * L * P)

        let kernel = try PLEInputsKernel(device: device)
        let out = try kernel.run(
            proj: proj,
            normWeight: normW,
            pleRows: pleRows,
            hiddenSize: H,
            perLayerDim: P,
            numLayers: L,
            batchSeq: BS,
            rmsEps: 1e-6
        )
        let scaleMix = Float(1.0) / Float(2.0).squareRoot()
        // Expected: (1/rms * (1+w) + 0) * scaleMix
        // rms for all-1s -> sqrt(1 + 1e-6) ~ 1
        let expectedMultipliers: [Float] = [1.5, 0.5, 2.0, 1.0]
        for p in 0..<P {
            let expected = expectedMultipliers[p] * scaleMix
            #expect(abs(out[p] - expected) < 1e-3)
        }
    }

    @Test("Rejects mismatched buffer shapes")
    func rejectsMismatchedShapes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try PLEInputsKernel(device: device)
        let H = 4, L = 1, P = 4, BS = 1
        let goodProj = [Float](repeating: 1.0, count: BS * L * P)
        let wrongProj = [Float](repeating: 1.0, count: BS * L * P + 1)
        let goodNormW = [Float](repeating: 0.0, count: P)
        let wrongNormW = [Float](repeating: 0.0, count: P + 1)
        let goodPleRows = [Float](repeating: 0.0, count: BS * L * P)

        #expect(throws: PLEInputsError.self) {
            _ = try kernel.run(
                proj: wrongProj,
                normWeight: goodNormW,
                pleRows: goodPleRows,
                hiddenSize: H,
                perLayerDim: P,
                numLayers: L,
                batchSeq: BS
            )
        }
        #expect(throws: PLEInputsError.self) {
            _ = try kernel.run(
                proj: goodProj,
                normWeight: wrongNormW,
                pleRows: goodPleRows,
                hiddenSize: H,
                perLayerDim: P,
                numLayers: L,
                batchSeq: BS
            )
        }
    }

    static func referenceForward(
        proj: [Float],
        norm: [Float],
        pleRows: [Float],
        L: Int,
        P: Int,
        BS: Int,
        rmsEps: Float = 1e-6
    ) -> [Float] {
        let scaleMix = 1.0 / Float(2.0).squareRoot()
        var out = [Float](repeating: 0, count: BS * L * P)
        for b in 0..<BS {
            for ell in 0..<L {
                var sumSq: Float = 0
                for p in 0..<P {
                    let v = proj[b * L * P + ell * P + p]
                    sumSq += v * v
                }
                let rms = sqrt(sumSq / Float(P) + rmsEps)
                for p in 0..<P {
                    let w = Float(1) + norm[p]
                    let idx = b * L * P + ell * P + p
                    let normed = proj[idx] / rms * w
                    out[idx] = (normed + pleRows[idx]) * scaleMix
                }
            }
        }
        return out
    }
}
