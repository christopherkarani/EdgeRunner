import Testing
@testable import EdgeRunnerIO

@Suite("Memory Policy Tests")
struct MemoryPolicyTests: Sendable {
    @Test("QuantisationLevel ordering")
    func quantisationOrdering() {
        #expect(QuantisationLevel.q8_0.bitsPerWeight > QuantisationLevel.q4_k_m.bitsPerWeight)
        #expect(QuantisationLevel.q4_k_m.bitsPerWeight > QuantisationLevel.q4_0.bitsPerWeight)
    }

    @Test("Policy with empty fallback chain uses Q4_0")
    func emptyFallbackChain() {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [],
            evictBufferCacheOnPressure: false
        )
        let handler = MemoryPressureHandler(policy: policy)
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Custom eviction threshold")
    func customEvictionThreshold() {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0],
            evictBufferCacheOnPressure: true,
            maxMemoryBytes: 2 * 1024 * 1024 * 1024
        )
        #expect(policy.maxMemoryBytes == 2 * 1024 * 1024 * 1024)
    }
}
