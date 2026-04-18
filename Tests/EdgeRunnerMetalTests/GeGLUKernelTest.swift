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
}
