import Foundation

public enum QuantisationLevel: String, Sendable, Equatable, CaseIterable {
    case q8_0
    case q4_k_m
    case q4_0

    public var bitsPerWeight: Double {
        switch self {
        case .q8_0:
            return 8.0
        case .q4_k_m:
            return 4.5
        case .q4_0:
            return 4.0
        }
    }
}

public struct EdgeRunnerMemoryPolicy: Sendable, Equatable {
    public let fallbackChain: [QuantisationLevel]
    public let evictBufferCacheOnPressure: Bool
    public let maxMemoryBytes: Int

    public init(
        fallbackChain: [QuantisationLevel],
        evictBufferCacheOnPressure: Bool,
        maxMemoryBytes: Int = 0
    ) {
        self.fallbackChain = fallbackChain
        self.evictBufferCacheOnPressure = evictBufferCacheOnPressure
        self.maxMemoryBytes = maxMemoryBytes
    }

    public static let `default` = EdgeRunnerMemoryPolicy(
        fallbackChain: [.q8_0, .q4_k_m, .q4_0],
        evictBufferCacheOnPressure: true
    )
}
