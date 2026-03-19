import Foundation

struct LlamaDecodeOverrides: Sendable, Equatable {
    var forceBaseDecodePath: Bool?
    var disableMegaKernel: Bool?
    var disableFusedFinalNormLMHead: Bool?

    init(
        forceBaseDecodePath: Bool? = nil,
        disableMegaKernel: Bool? = nil,
        disableFusedFinalNormLMHead: Bool? = nil
    ) {
        self.forceBaseDecodePath = forceBaseDecodePath
        self.disableMegaKernel = disableMegaKernel
        self.disableFusedFinalNormLMHead = disableFusedFinalNormLMHead
    }
}

/// Configuration for model loading and generation behavior.
public struct ModelConfiguration: Sendable {
    public var maxTokens: Int
    public var contextWindowSize: Int
    public var useMemoryMapping: Bool
    public var tokenizerURL: URL?
    var llamaDecodeOverrides: LlamaDecodeOverrides?

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
        self.llamaDecodeOverrides = nil
    }
}
