import Foundation
import Synchronization

public final class MemoryPressureHandler: Sendable {
    private struct State: Sendable {
        var currentIndex: Int
        var onBufferCacheEviction: (@Sendable () -> Void)?
    }

    private let policy: EdgeRunnerMemoryPolicy
    private let state: Mutex<State>

    public init(policy: EdgeRunnerMemoryPolicy) {
        self.policy = policy
        self.state = Mutex(State(currentIndex: 0, onBufferCacheEviction: nil))
    }

    public var onBufferCacheEviction: (@Sendable () -> Void)? {
        get {
            state.withLock { $0.onBufferCacheEviction }
        }
        set {
            state.withLock { state in
                state.onBufferCacheEviction = newValue
            }
        }
    }

    public var currentQuantisation: QuantisationLevel {
        state.withLock { state in
            quantisation(for: state.currentIndex)
        }
    }

    public func handleMemoryPressure() {
        let callback = state.withLock { state in
            let lastIndex = fallbackChain.count - 1
            if state.currentIndex < lastIndex {
                state.currentIndex += 1
            }
            return policy.evictBufferCacheOnPressure ? state.onBufferCacheEviction : nil
        }
        callback?()
    }

    public func simulateMemoryWarning() async {
        handleMemoryPressure()
    }

    public func reset() {
        state.withLock { state in
            state.currentIndex = 0
        }
    }

    private var fallbackChain: [QuantisationLevel] {
        policy.fallbackChain.isEmpty ? [.q4_0] : policy.fallbackChain
    }

    private func quantisation(for index: Int) -> QuantisationLevel {
        let chain = fallbackChain
        let boundedIndex = min(max(index, 0), chain.count - 1)
        return chain[boundedIndex]
    }
}
