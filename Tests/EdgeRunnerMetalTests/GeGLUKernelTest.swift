import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("GeGLU kernel")
struct GeGLUKernelTests {
    @Test("Matches reference gelu_tanh(gate)*up within 1e-5")
    func matchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try GeGLUKernel(device: device)
        let gate: [Float] = [-2, -1, 0, 0.5, 1, 2]
        let up: [Float]   = [ 1,  2, 3, 4.0, 5, 6]
        let out = try kernel.run(gate: gate, up: up)
        let expected = zip(gate, up).map { g, u in
            let c: Float = 0.7978845608028654
            let inner = c * (g + 0.044715 * g * g * g)
            return g * 0.5 * (1 + tanh(inner)) * u
        }
        for i in 0..<out.count {
            #expect(abs(out[i] - expected[i]) < 1e-5)
        }
    }

    @Test("Encodes into caller-owned command buffer")
    func encodesIntoExistingCommandBuffer() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try GeGLUKernel(device: device)
        let gate: [Float] = [-1.5, -0.25, 0.75, 1.25]
        let up: [Float] = [2.0, -3.0, 4.0, -5.0]
        guard let gateBuffer = device.makeBuffer(
            bytes: gate,
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let upBuffer = device.makeBuffer(
            bytes: up,
            length: up.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: gate.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = queue.makeCommandBuffer() else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        try kernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            outputBuffer: outputBuffer,
            count: gate.count
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let outPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: gate.count)
        let out = Array(UnsafeBufferPointer(start: outPointer, count: gate.count))
        let expected = zip(gate, up).map { g, u in
            let c: Float = 0.7978845608028654
            let inner = c * (g + 0.044715 * g * g * g)
            return g * 0.5 * (1 + tanh(inner)) * u
        }
        for i in 0..<out.count {
            #expect(abs(out[i] - expected[i]) < 1e-5)
        }
    }
}
