import Foundation
import Synchronization
import Testing
@testable import EdgeRunnerIO

@Suite("Memory Pressure Handler Tests")
struct MemoryPressureHandlerTests: Sendable {
    @Test("Default fallback chain: Q8 -> Q4_K_M -> Q4_0")
    func defaultFallbackChain() {
        let policy = EdgeRunnerMemoryPolicy.default
        #expect(policy.fallbackChain == [.q8_0, .q4_k_m, .q4_0])
    }

    @Test("Handler triggers fallback on simulated pressure")
    func simulatedPressure() async {
        let handler = MemoryPressureHandler(policy: .default)
        #expect(handler.currentQuantisation == .q8_0)

        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_k_m)

        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Handler does not fall back past minimum")
    func minimumFallback() async {
        let handler = MemoryPressureHandler(policy: .default)

        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()

        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Custom policy with restricted fallback chain")
    func customPolicy() async {
        let handler = MemoryPressureHandler(
            policy: EdgeRunnerMemoryPolicy(
                fallbackChain: [.q8_0, .q4_0],
                evictBufferCacheOnPressure: false
            )
        )

        #expect(handler.currentQuantisation == .q8_0)
        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Buffer cache eviction flag is respected")
    func bufferCacheEviction() async {
        let handler = MemoryPressureHandler(
            policy: EdgeRunnerMemoryPolicy(
                fallbackChain: [.q8_0, .q4_0],
                evictBufferCacheOnPressure: true
            )
        )
        let evictionCount = Mutex(0)
        handler.onBufferCacheEviction = {
            evictionCount.withLock { count in
                count += 1
            }
        }

        await handler.simulateMemoryWarning()
        #expect(evictionCount.withLock { $0 } == 1)
    }

    @Test("Handler can reset to highest quality")
    func reset() async {
        let handler = MemoryPressureHandler(policy: .default)

        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_k_m)

        handler.reset()
        #expect(handler.currentQuantisation == .q8_0)
    }
}
