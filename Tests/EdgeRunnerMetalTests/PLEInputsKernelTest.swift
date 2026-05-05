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
            #expect(abs(out[i] - expected[i]) < 1e-3, "mismatch at \(i)")
        }
    }

    @Test("Non-zero normWeight: direct Gemma 4 weight applied")
    func normWeightAppliedDirectly() throws {
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
        // Expected: (1/rms * w + 0) * scaleMix
        // rms for all-1s -> sqrt(1 + 1e-6) ~ 1
        let expectedMultipliers: [Float] = [0.5, -0.5, 1.0, 0.0]
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

    @Test("Encodes per-layer input build into caller-owned command buffer")
    func encodesIntoCommandBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            Issue.record("Metal device unavailable")
            return
        }

        let H = 8
        let L = 2
        let P = 4
        let BS = 3
        let proj = (0..<(BS * L * P)).map { Float($0) * 0.03 - 0.25 }
        let normW: [Float] = [0.1, -0.2, 0.0, 0.3]
        let pleRows = (0..<(BS * L * P)).map { Float($0 % 7) * 0.05 - 0.15 }
        let outputCount = BS * L * P

        guard let projBuffer = device.makeBuffer(bytes: proj, length: proj.count * MemoryLayout<Float>.stride),
              let normBuffer = device.makeBuffer(bytes: normW, length: normW.count * MemoryLayout<Float>.stride),
              let pleBuffer = device.makeBuffer(bytes: pleRows, length: pleRows.count * MemoryLayout<Float>.stride),
              let outputBuffer = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride)
        else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        let kernel = try PLEInputsKernel(device: device)
        try kernel.encode(
            commandBuffer: commandBuffer,
            projectionBuffer: projBuffer,
            normWeightBuffer: normBuffer,
            pleRowsBuffer: pleBuffer,
            outputBuffer: outputBuffer,
            hiddenSize: H,
            perLayerDim: P,
            numLayers: L,
            batchSeq: BS,
            rmsEps: 1e-6
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let output = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount),
                count: outputCount
            )
        )
        let expected = Self.referenceForward(proj: proj, norm: normW, pleRows: pleRows, L: L, P: P, BS: BS)
        for i in 0..<outputCount {
            #expect(abs(output[i] - expected[i]) < 1e-3, "mismatch at \(i)")
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
                    let w = norm[p]
                    let idx = b * L * P + ell * P + p
                    let normed = proj[idx] / rms * w
                    out[idx] = (normed + pleRows[idx]) * scaleMix
                }
            }
        }
        return out
    }
}
