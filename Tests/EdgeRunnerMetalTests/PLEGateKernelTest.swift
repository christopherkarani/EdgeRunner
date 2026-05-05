import Foundation
import Metal
import Testing
@testable import EdgeRunnerMetal

@Suite("PLE gate kernel")
struct PLEGateKernelTests {
    @Test("Computes gelu_tanh(gate) multiplied by PLE input")
    func matchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try PLEGateKernel(device: device)
        let gate: [Float] = [-2, -0.5, 0, 0.75, 1.5]
        let ple: [Float] = [3, -4, 5, -6, 7]
        let result = try kernel.run(gate: gate, ple: ple)

        let expected = zip(gate, ple).map { gateValue, pleValue in
            let coefficient: Float = 0.7978845608028654
            let inner = coefficient * (gateValue + 0.044715 * gateValue * gateValue * gateValue)
            let gelu = gateValue * 0.5 * (1 + tanh(inner))
            return gelu * pleValue
        }

        for index in result.indices {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Large gates stay finite")
    func largeGatesStayFinite() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try PLEGateKernel(device: device)
        let result = try kernel.run(gate: [-100, 100], ple: [3, 4])

        #expect(result[0].isFinite)
        #expect(result[1].isFinite)
        #expect(abs(result[0]) < 1e-5)
        #expect(abs(result[1] - 400) < 1e-4)
    }

    @Test("Encodes with PLE input read from a buffer offset")
    func encodesWithPLEInputOffset() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer()
        else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try PLEGateKernel(device: device)
        let gate: [Float] = [-1, 0.25, 1.5, 3]
        let prefix: [Float] = [99, 98]
        let ple: [Float] = [2, -3, 4, -5]
        let combinedPLE = prefix + ple
        guard let gateBuffer = device.makeBuffer(
            bytes: gate,
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let pleBuffer = device.makeBuffer(
            bytes: combinedPLE,
            length: combinedPLE.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        try kernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            pleBuffer: pleBuffer,
            outputBuffer: outputBuffer,
            count: gate.count,
            pleBufferOffset: prefix.count * MemoryLayout<Float>.stride
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let actual = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: gate.count),
                count: gate.count
            )
        )
        let expected = zip(gate, ple).map { gateValue, pleValue in
            let coefficient: Float = 0.7978845608028654
            let inner = coefficient * (gateValue + 0.044715 * gateValue * gateValue * gateValue)
            let gelu = gateValue * 0.5 * (1 + tanh(inner))
            return gelu * pleValue
        }

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }
}
