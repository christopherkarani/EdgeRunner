import Foundation

/// A token emitted during streaming generation.
public struct StreamToken: Sendable {
    public let id: Int
    public let text: String
    public let isEOS: Bool
    public init(id: Int, text: String, isEOS: Bool = false) {
        self.id = id; self.text = text; self.isEOS = isEOS
    }
}

/// Statistics collected during a generation session.
public struct GenerationStats: Sendable {
    public var tokenCount: Int = 0
    public var timeToFirstToken: Double = 0
    public var totalTime: Double = 0
    public var tokensPerSecond: Double {
        guard totalTime > 0 else { return 0 }
        return Double(tokenCount) / totalTime
    }
}
