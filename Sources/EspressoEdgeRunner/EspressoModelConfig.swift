import EdgeRunnerIO

public struct EspressoModelConfig: Sendable, Equatable {
    public let embeddingDim: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let blockCount: Int
    public let feedForwardLength: Int
    public let contextLength: Int
    public let rmsNormEpsilon: Float
    public let architectureName: String

    public var headDim: Int { embeddingDim / headCount }

    /// Creates config by reading GGUF metadata from a `ModelConfig`.
    /// Tries `"{arch}.{key}"` first, then bare `"{key}"`.
    public init(from config: ModelConfig) throws {
        let arch = config.architectureName
        self.architectureName = arch

        func intKey(_ key: String) throws -> Int {
            if let v = config.int(forKey: "\(arch).\(key)") { return v }
            if let v = config.int(forKey: key) { return v }
            throw EspressoError.configMissingKey(key)
        }

        func floatKey(_ key: String) throws -> Float {
            if let v = config.float(forKey: "\(arch).\(key)") { return v }
            if let v = config.float(forKey: key) { return v }
            throw EspressoError.configMissingKey(key)
        }

        self.embeddingDim = try intKey("embedding_length")
        self.headCount = try intKey("attention.head_count")
        self.kvHeadCount = try intKey("attention.head_count_kv")
        self.blockCount = try intKey("block_count")
        self.feedForwardLength = try intKey("feed_forward_length")
        self.contextLength = try intKey("context_length")
        self.rmsNormEpsilon = try floatKey("attention.layer_norm_rms_epsilon")
    }

    /// Memberwise initializer.
    public init(
        embeddingDim: Int,
        headCount: Int,
        kvHeadCount: Int,
        blockCount: Int,
        feedForwardLength: Int,
        contextLength: Int,
        rmsNormEpsilon: Float,
        architectureName: String
    ) {
        self.embeddingDim = embeddingDim
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.blockCount = blockCount
        self.feedForwardLength = feedForwardLength
        self.contextLength = contextLength
        self.rmsNormEpsilon = rmsNormEpsilon
        self.architectureName = architectureName
    }
}
