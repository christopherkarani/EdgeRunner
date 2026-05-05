import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("PLE side-channel finalize")
struct PLESideChannelKernelTests {
    @Test("Finalizes projected side-channel with RMSNorm and residual add")
    func finalizesProjectionIntoResidual() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }

        let H = 6
        let BS = 3
        let hidden = (0..<(BS * H)).map { Float($0) * 0.02 - 0.3 }
        let projection = (0..<(BS * H)).map { Float(($0 % 11) - 5) * 0.07 }
        let postNormWeight = (0..<H).map { Float($0) * 0.05 - 0.1 }

        let kernel = try PLESideChannelKernel(device: device)
        let output = try kernel.run(
            hidden: hidden,
            projection: projection,
            postNormWeight: postNormWeight,
            hiddenSize: H,
            batchSeq: BS,
            rmsEps: 1e-6
        )

        let expected = Self.referenceFinalize(
            hidden: hidden,
            projection: projection,
            postNormWeight: postNormWeight,
            hiddenSize: H,
            batchSeq: BS,
            rmsEps: 1e-6
        )

        #expect(output.count == expected.count)
        for i in 0..<expected.count {
            #expect(abs(output[i] - expected[i]) < 1e-4, "mismatch at \(i)")
        }
    }

    @Test("Encodes finalize into caller-owned command buffer")
    func encodesIntoCommandBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            Issue.record("Metal device unavailable")
            return
        }

        let H = 4
        let BS = 2
        let hidden: [Float] = [0.2, -0.1, 0.4, 0.5, -0.3, 0.8, -0.7, 0.1]
        let projection: [Float] = [0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1]
        let postNormWeight: [Float] = [0.0, 0.25, -0.5, 0.1]

        guard let hiddenBuffer = device.makeBuffer(bytes: hidden, length: hidden.count * MemoryLayout<Float>.stride),
              let projectionBuffer = device.makeBuffer(
                bytes: projection,
                length: projection.count * MemoryLayout<Float>.stride
              ),
              let postNormBuffer = device.makeBuffer(
                bytes: postNormWeight,
                length: postNormWeight.count * MemoryLayout<Float>.stride
              )
        else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        let kernel = try PLESideChannelKernel(device: device)
        try kernel.encode(
            commandBuffer: commandBuffer,
            hiddenBuffer: hiddenBuffer,
            projectionBuffer: projectionBuffer,
            postNormWeightBuffer: postNormBuffer,
            hiddenSize: H,
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
                start: hiddenBuffer.contents().bindMemory(to: Float.self, capacity: hidden.count),
                count: hidden.count
            )
        )
        let expected = Self.referenceFinalize(
            hidden: hidden,
            projection: projection,
            postNormWeight: postNormWeight,
            hiddenSize: H,
            batchSeq: BS,
            rmsEps: 1e-6
        )

        for i in 0..<expected.count {
            #expect(abs(output[i] - expected[i]) < 1e-4, "mismatch at \(i)")
        }
    }

    @Test("Rejects mismatched side-channel buffer shapes")
    func rejectsMismatchedShapes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }

        let kernel = try PLESideChannelKernel(device: device)
        #expect(throws: PLESideChannelError.self) {
            _ = try kernel.run(
                hidden: [Float](repeating: 0, count: 5),
                projection: [Float](repeating: 0, count: 4),
                postNormWeight: [Float](repeating: 0, count: 4),
                hiddenSize: 4,
                batchSeq: 1
            )
        }
        #expect(throws: PLESideChannelError.self) {
            _ = try kernel.run(
                hidden: [Float](repeating: 0, count: 4),
                projection: [Float](repeating: 0, count: 4),
                postNormWeight: [Float](repeating: 0, count: 5),
                hiddenSize: 4,
                batchSeq: 1
            )
        }
    }

    static func referenceFinalize(
        hidden: [Float],
        projection: [Float],
        postNormWeight: [Float],
        hiddenSize: Int,
        batchSeq: Int,
        rmsEps: Float
    ) -> [Float] {
        var output = hidden
        for bs in 0..<batchSeq {
            let base = bs * hiddenSize
            var sumSq: Float = 0
            for h in 0..<hiddenSize {
                let value = projection[base + h]
                sumSq += value * value
            }
            let rms = sqrt(sumSq / Float(hiddenSize) + rmsEps)
            for h in 0..<hiddenSize {
                let idx = base + h
                output[idx] += projection[idx] / rms * postNormWeight[h]
            }
        }
        return output
    }
}
