import Foundation

/// Configuration for model loading and generation behavior.
public struct ModelConfiguration: Sendable {
    public var maxTokens: Int
    public var contextWindowSize: Int
    public var useMemoryMapping: Bool
    public var tokenizerURL: URL?

    public init(
        maxTokens: Int = 2048,
        contextWindowSize: Int = 4096,
        useMemoryMapping: Bool = true,
        tokenizerURL: URL? = nil
    ) {
        self.maxTokens = maxTokens
        self.contextWindowSize = contextWindowSize
        self.useMemoryMapping = useMemoryMapping
        self.tokenizerURL = tokenizerURL
    }
}
