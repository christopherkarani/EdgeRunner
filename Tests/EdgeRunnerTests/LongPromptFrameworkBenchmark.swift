import Foundation
import Testing
@testable import EdgeRunner

@Suite("Long Prompt Framework Benchmark")
struct LongPromptFrameworkBenchmark {
    static let childRunEnvKey = "EDGERUNNER_LONG_PROMPT_CHILD_RUN"
    static let promptTokensPathEnvKey = "EDGERUNNER_LONG_PROMPT_TOKENS_PATH"
    static let outputPathEnvKey = "EDGERUNNER_LONG_PROMPT_OUTPUT_PATH"
    static let generateCountEnvKey = "EDGERUNNER_LONG_PROMPT_GENERATE_COUNT"
    static let contextWindowEnvKey = "EDGERUNNER_LONG_PROMPT_CONTEXT_WINDOW"
    static let modelPath = BenchmarkContract.pinned.model.localPath

    @Test
    func edgeRunnerChildRun() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env[Self.childRunEnvKey] == "1" else {
            Swift.print("SKIP: Set \(Self.childRunEnvKey)=1 to run the long-prompt EdgeRunner child benchmark.")
            return
        }

        guard let promptPath = env[Self.promptTokensPathEnvKey], !promptPath.isEmpty else {
            throw GenerationError.modelLoadFailed(reason: "Missing \(Self.promptTokensPathEnvKey)")
        }
        guard let outputPath = env[Self.outputPathEnvKey], !outputPath.isEmpty else {
            throw GenerationError.modelLoadFailed(reason: "Missing \(Self.outputPathEnvKey)")
        }

        let generateCount = try validatedPositiveInt(
            env[Self.generateCountEnvKey],
            defaultValue: 128,
            name: Self.generateCountEnvKey
        )
        let defaultContextWindow = BenchmarkContract.pinned.publishable.contextWindow
        let contextWindow = try validatedPositiveInt(
            env[Self.contextWindowEnvKey],
            defaultValue: defaultContextWindow,
            name: Self.contextWindowEnvKey
        )

        let promptTokens = try loadPromptTokens(from: promptPath)
        guard promptTokens.count + generateCount <= contextWindow else {
            throw GenerationError.decodingFailed(
                "Prompt tokens (\(promptTokens.count)) + generate count (\(generateCount)) exceed context window \(contextWindow)"
            )
        }

        let modelURL = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw GenerationError.modelLoadFailed(reason: "Pinned model not found at \(Self.modelPath)")
        }

        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: .pinnedBenchmarkConfiguration(contextWindow: contextWindow)
        )

        // Warm the exact prompt shape once so TTFT reflects steady-state prompt processing.
        _ = try await model.greedyToken(for: promptTokens)
        model.resetGenerationState(keepDecodeWarmup: true)

        let clock = ContinuousClock()

        let ttftStart = clock.now
        let first = try await model.greedyToken(for: promptTokens)
        let ttftEnd = clock.now

        let ttftSeconds = seconds(from: ttftStart.duration(to: ttftEnd))
        let promptTokensPerSecond = Double(promptTokens.count) / ttftSeconds

        var tokenIDs = promptTokens
        tokenIDs.append(first.token)

        var hasNonFinite = first.hasNonFinite
        var decodeSeconds = 0.0

        for _ in 1..<generateCount {
            let decodeStart = clock.now
            let result = try await model.greedyToken(for: tokenIDs)
            let decodeEnd = clock.now

            decodeSeconds += seconds(from: decodeStart.duration(to: decodeEnd))
            hasNonFinite = hasNonFinite || result.hasNonFinite
            tokenIDs.append(result.token)
        }

        let measuredDecodeCount = max(generateCount - 1, 0)
        let decodeTokensPerSecond = measuredDecodeCount > 0
            ? Double(measuredDecodeCount) / decodeSeconds
            : 0

        let result: [String: Any] = [
            "framework": "EdgeRunner",
            "model_path": Self.modelPath,
            "prompt_token_count": promptTokens.count,
            "generated_token_count": generateCount,
            "ttft_ms": ttftSeconds * 1000,
            "prompt_tok_s": promptTokensPerSecond,
            "decode_tok_s": decodeTokensPerSecond,
            "has_non_finite": hasNonFinite,
            "generated_prefix": Array(tokenIDs.suffix(min(tokenIDs.count, 8))),
        ]

        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outputURL)
    }

    private func loadPromptTokens(from path: String) throws -> [Int] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let object = try JSONSerialization.jsonObject(with: data)
        if let direct = object as? [Int] {
            return direct
        }
        if
            let dict = object as? [String: Any],
            let nested = dict["prompt_tokens"] as? [Int]
        {
            return nested
        }
        throw GenerationError.modelLoadFailed(reason: "Prompt token file at \(path) did not contain [Int] or {\"prompt_tokens\": [Int]}")
    }

    private func seconds(from duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }

    private func validatedPositiveInt(_ raw: String?, defaultValue: Int, name: String) throws -> Int {
        guard let raw, !raw.isEmpty else { return defaultValue }
        guard let value = Int(raw), value > 0 else {
            throw GenerationError.decodingFailed("\(name) must be a positive integer")
        }
        return value
    }
}
