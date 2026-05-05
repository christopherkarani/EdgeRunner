import Foundation
import EdgeRunnerCore
import EdgeRunnerIO

/// Dedicated production entry point for Bonsai-family local models.
///
/// The current implementation wraps the existing llama-compatible runtime while
/// giving Bonsai its own backend identity and loader routing. This isolates
/// Bonsai-specific policy from the generic model path so the internals can be
/// replaced later without changing the app-facing interface.
public struct BonsaiLanguageModel: LogitsModel, @unchecked Sendable {
    public static let modelIdentifier = "bonsai"

    private let base: LlamaLanguageModel

    private init(base: LlamaLanguageModel) {
        self.base = base
    }

    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> BonsaiLanguageModel {
        let loader = try GGUFLoader(url: url)
        guard Self.supports(modelConfig: loader.modelConfig) else {
            throw GenerationError.modelLoadFailed(
                reason: "Requested Bonsai backend for a non-Bonsai model at \(url.path)"
            )
        }

        let base = try await LlamaLanguageModel.load(from: url, configuration: configuration)
        return BonsaiLanguageModel(base: base)
    }

    static func supports(modelConfig: ModelConfig) -> Bool {
        let architecture = modelConfig.architectureName.lowercased()
        let modelName = modelConfig.string(forKey: "general.name")?.lowercased() ?? ""
        let fileType = modelConfig.int(forKey: "general.file_type")

        if modelName.contains("bonsai") {
            return true
        }

        return architecture == "qwen3" && fileType == Int(TensorDataType.q1_0_g128.rawValue)
    }

    public func tokenize(_ text: String) -> [Int] {
        base.tokenize(text)
    }

    public func detokenize(_ ids: [Int]) -> String {
        base.detokenize(ids)
    }

    public var eosTokenID: Int { base.eosTokenID }
    public var bosTokenID: Int? { base.bosTokenID }
    public var vocabularySize: Int { base.vocabularySize }

    public func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool
    ) -> String? {
        base.applyChatTemplate(messages: messages, addGenerationPrompt: addGenerationPrompt)
    }

    public func nextToken(
        for tokenIDs: [Int],
        sampling: SamplingConfiguration
    ) async throws -> Int {
        try await base.nextToken(for: tokenIDs, sampling: sampling)
    }

    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        try await base.logits(for: tokenIDs)
    }

    func greedyToken(for tokenIDs: [Int]) async throws -> (token: Int, hasNonFinite: Bool) {
        try await base.greedyToken(for: tokenIDs)
    }

    func resetGenerationState(keepDecodeWarmup: Bool = true) {
        base.resetGenerationState(keepDecodeWarmup: keepDecodeWarmup)
    }

    public func greedyTokenSync(for tokenIDs: [Int]) throws -> (token: Int, hasNonFinite: Bool) {
        try base.greedyTokenSync(for: tokenIDs)
    }

    func measureLMHeadLatency(samples: Int = 5) async throws -> Double {
        try await base.measureLMHeadLatency(samples: samples)
    }

    func measureQ1ProjectionLatencies(samples: Int = 5) async throws -> [(name: String, milliseconds: Double)] {
        try await base.measureQ1ProjectionLatencies(samples: samples)
    }
}
