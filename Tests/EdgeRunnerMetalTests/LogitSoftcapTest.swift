import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("Logit softcap kernel")
struct LogitSoftcapTests {
    @Test("Softcaps logits with cap=30 to match tanh(x/30)*30")
    func softcapsLogits() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try LogitSoftcapKernel(device: device)
        let inp: [Float] = [-100, -30, -1, 0, 1, 30, 100]
        let out = try kernel.run(logits: inp, cap: 30)
        for (i, x) in inp.enumerated() {
            let expected = tanh(x / 30) * 30
            #expect(abs(out[i] - expected) < 1e-5)
        }
    }
}
