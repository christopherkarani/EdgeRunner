import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("SWA mask")
struct SlidingWindowMaskTests {
    @Test("SWA mask masks positions outside [q-window+1, q]")
    func swaMaskIsCorrect() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let maker = try SlidingWindowMask(device: device)
        let mask = try maker.build(seqLen: 10, window: 3)

        // For q=5, allowed k = {3, 4, 5}
        #expect(mask[5 * 10 + 2] == -.infinity)
        #expect(mask[5 * 10 + 3] == 0)
        #expect(mask[5 * 10 + 4] == 0)
        #expect(mask[5 * 10 + 5] == 0)
        #expect(mask[5 * 10 + 6] == -.infinity)
    }

    @Test("Underflow guard: q=0 only attends to k=0 regardless of window")
    func underflowGuardAtStart() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let maker = try SlidingWindowMask(device: device)
        let mask = try maker.build(seqLen: 5, window: 3)
        #expect(mask[0] == 0)              // q=0, k=0
        #expect(mask[1] == -.infinity)     // q=0, k=1 (future)
    }

    @Test("Global mode (window >= seqLen) is pure causal")
    func globalModeIsCausal() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let maker = try SlidingWindowMask(device: device)
        let mask = try maker.build(seqLen: 4, window: 4)
        // q=3 attends to all k in [0, 3]
        for k in 0...3 {
            let index = 3 * 4 + k
            #expect(mask[index] == 0)
        }
        // q=1 attends only to k in [0, 1]
        let rowQ1 = 1 * 4
        #expect(mask[rowQ1 + 0] == 0)
        #expect(mask[rowQ1 + 1] == 0)
        let futureValue: Float = -.infinity
        #expect(mask[rowQ1 + 2] == futureValue)
    }
}
