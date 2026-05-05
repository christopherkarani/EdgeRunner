import Foundation
import EdgeRunner

public enum BenchmarkMode: String, CaseIterable, Codable, Equatable, Sendable {
    case rawDecode = "raw_decode"
    case streamedChat = "streamed_chat"

    public var displayName: String {
        switch self {
        case .rawDecode:
            return "Raw Decode"
        case .streamedChat:
            return "Chat Stream"
        }
    }
}

public struct BenchmarkAutomationConfiguration: Equatable, Sendable {
    public static let modelPathEnvironmentKey = "EDGERUNNER_AUTOBENCH_MODEL_PATH"
    public static let modelFilenameEnvironmentKey = "EDGERUNNER_AUTOBENCH_MODEL_FILENAME"
    public static let promptEnvironmentKey = "EDGERUNNER_AUTOBENCH_PROMPT"
    public static let resultFilenameEnvironmentKey = "EDGERUNNER_AUTOBENCH_RESULT_FILENAME"
    public static let maxTokensEnvironmentKey = "EDGERUNNER_AUTOBENCH_MAX_TOKENS"
    public static let contextWindowEnvironmentKey = "EDGERUNNER_AUTOBENCH_CONTEXT_WINDOW"
    public static let modeEnvironmentKey = "EDGERUNNER_AUTOBENCH_MODE"

    public let modelPath: String
    public let prompt: String
    public let resultURL: URL
    public let maxTokens: Int
    public let contextWindowSize: Int
    public let mode: BenchmarkMode

    public static func make(
        environment: [String: String],
        documentsDirectory: URL,
        resultDirectory: URL
    ) -> BenchmarkAutomationConfiguration? {
        let explicitModelPath = environment[modelPathEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let modelFilename = environment[modelFilenameEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let resolvedModelPath: String?
        if let explicitModelPath, !explicitModelPath.isEmpty {
            resolvedModelPath = explicitModelPath
        } else if let modelFilename, !modelFilename.isEmpty {
            resolvedModelPath = documentsDirectory
                .appendingPathComponent(modelFilename)
                .path
        } else {
            resolvedModelPath = nil
        }

        guard let modelPath = resolvedModelPath else {
            return nil
        }

        let prompt = environment[promptEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nonEmpty ?? "Explain quantum computing in simple terms."

        let resultFilename = environment[resultFilenameEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nonEmpty ?? "autobench-result.json"

        let maxTokens = parsedPositiveInt(
            from: environment[maxTokensEnvironmentKey],
            defaultValue: 128
        )
        let contextWindowSize = parsedPositiveInt(
            from: environment[contextWindowEnvironmentKey],
            defaultValue: 4096
        )
        let mode = BenchmarkMode(
            rawValue: environment[modeEnvironmentKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() ?? ""
        ) ?? .rawDecode

        return BenchmarkAutomationConfiguration(
            modelPath: modelPath,
            prompt: prompt,
            resultURL: resultDirectory.appendingPathComponent(resultFilename),
            maxTokens: maxTokens,
            contextWindowSize: contextWindowSize,
            mode: mode
        )
    }

    private static func parsedPositiveInt(from value: String?, defaultValue: Int) -> Int {
        guard let value else { return defaultValue }
        guard let parsed = Int(value.trimmingCharacters(in: .whitespacesAndNewlines)), parsed > 0 else {
            return defaultValue
        }
        return parsed
    }
}

public struct BenchmarkAutomationResult: Codable, Equatable, Sendable {
    public let modelPath: String
    public let prompt: String
    public let contextWindowSize: Int
    public let maxTokens: Int
    public let mode: BenchmarkMode
    public let generatedTokenCount: Int
    public let timeToFirstTokenSeconds: Double?
    public let decodeTokensPerSecond: Double
    public let endToEndTokensPerSecond: Double
    public let generationDurationSeconds: Double
    public let assistantResponse: String
    public let errorMessage: String?

    public init(
        modelPath: String,
        prompt: String,
        contextWindowSize: Int,
        maxTokens: Int,
        mode: BenchmarkMode,
        generatedTokenCount: Int,
        timeToFirstTokenSeconds: Double?,
        decodeTokensPerSecond: Double,
        endToEndTokensPerSecond: Double,
        generationDurationSeconds: Double,
        assistantResponse: String,
        errorMessage: String?
    ) {
        self.modelPath = modelPath
        self.prompt = prompt
        self.contextWindowSize = contextWindowSize
        self.maxTokens = maxTokens
        self.mode = mode
        self.generatedTokenCount = generatedTokenCount
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.endToEndTokensPerSecond = endToEndTokensPerSecond
        self.generationDurationSeconds = generationDurationSeconds
        self.assistantResponse = assistantResponse
        self.errorMessage = errorMessage
    }
}

public enum BenchmarkAutomationWriter {
    public static func write(_ result: BenchmarkAutomationResult, to url: URL) throws {
        let directoryURL = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directoryURL,
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        try data.write(to: url, options: .atomic)
    }
}

public struct EdgeRunnerDecodeBenchmarkResult: Equatable, Sendable {
    public let prompt: String
    public let maxTokens: Int
    public let generatedTokenCount: Int
    public let timeToFirstTokenSeconds: Double?
    public let decodeTokensPerSecond: Double
    public let endToEndTokensPerSecond: Double
    public let generationDurationSeconds: Double
    public let assistantResponse: String

    public init(
        prompt: String,
        maxTokens: Int,
        generatedTokenCount: Int,
        timeToFirstTokenSeconds: Double?,
        decodeTokensPerSecond: Double,
        endToEndTokensPerSecond: Double,
        generationDurationSeconds: Double,
        assistantResponse: String
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.generatedTokenCount = generatedTokenCount
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.endToEndTokensPerSecond = endToEndTokensPerSecond
        self.generationDurationSeconds = generationDurationSeconds
        self.assistantResponse = assistantResponse
    }
}

public protocol DecodeBenchmarkRunning: Sendable {
    func runDecodeBenchmark(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        contextWindowSize: Int
    ) async throws -> EdgeRunnerDecodeBenchmarkResult
}

public struct EdgeRunnerDecodeBenchmarker: DecodeBenchmarkRunning {
    public init() {}

    public func runDecodeBenchmark(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        contextWindowSize: Int
    ) async throws -> EdgeRunnerDecodeBenchmarkResult {
        let model = try await ModelLoader.load(
            from: URL(fileURLWithPath: modelPath),
            configuration: ModelConfiguration(contextWindowSize: contextWindowSize)
        )

        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        var tokenIDs = model.tokenize(prompt)
        var assistantResponse = ""
        var generatedTokenCount = 0
        let generationStart = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?

        for _ in 0..<maxTokens {
            let tokenID = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let now = ProcessInfo.processInfo.systemUptime
            if tokenID == model.eosTokenID {
                break
            }

            if firstTokenTime == nil {
                firstTokenTime = now
            }

            tokenIDs.append(tokenID)
            assistantResponse += model.detokenize([tokenID])
            generatedTokenCount += 1
        }

        let generationFinish = ProcessInfo.processInfo.systemUptime
        let generationDurationSeconds = max(0, generationFinish - generationStart)
        let timeToFirstTokenSeconds = firstTokenTime.map { max(0, $0 - generationStart) }

        let decodeTokensPerSecond: Double
        if let firstTokenTime, generatedTokenCount > 1 {
            let decodeDuration = max(0, generationFinish - firstTokenTime)
            decodeTokensPerSecond = decodeDuration > 0
                ? Double(generatedTokenCount - 1) / decodeDuration
                : 0
        } else {
            decodeTokensPerSecond = 0
        }

        let endToEndTokensPerSecond = generationDurationSeconds > 0
            ? Double(generatedTokenCount) / generationDurationSeconds
            : 0

        return EdgeRunnerDecodeBenchmarkResult(
            prompt: prompt,
            maxTokens: maxTokens,
            generatedTokenCount: generatedTokenCount,
            timeToFirstTokenSeconds: timeToFirstTokenSeconds,
            decodeTokensPerSecond: decodeTokensPerSecond,
            endToEndTokensPerSecond: endToEndTokensPerSecond,
            generationDurationSeconds: generationDurationSeconds,
            assistantResponse: assistantResponse
        )
    }
}

@MainActor
public extension ChatRuntime {
    func runAutomatedBenchmark(
        _ configuration: BenchmarkAutomationConfiguration
    ) async -> BenchmarkAutomationResult {
        resetConversation()
        modelPath = configuration.modelPath
        contextWindowSize = configuration.contextWindowSize
        maxResponseTokens = configuration.maxTokens
        benchmarkMode = configuration.mode

        switch configuration.mode {
        case .rawDecode:
            let benchmarkResult = await runRawBenchmark(prompt: configuration.prompt)
            return BenchmarkAutomationResult(
                modelPath: configuration.modelPath,
                prompt: configuration.prompt,
                contextWindowSize: configuration.contextWindowSize,
                maxTokens: configuration.maxTokens,
                mode: configuration.mode,
                generatedTokenCount: benchmarkResult.generatedTokenCount,
                timeToFirstTokenSeconds: benchmarkResult.timeToFirstTokenSeconds,
                decodeTokensPerSecond: benchmarkResult.decodeTokensPerSecond,
                endToEndTokensPerSecond: benchmarkResult.endToEndTokensPerSecond,
                generationDurationSeconds: benchmarkResult.generationDurationSeconds,
                assistantResponse: benchmarkResult.assistantResponse,
                errorMessage: errorMessage
            )
        case .streamedChat:
            await runStreamBenchmark(prompt: configuration.prompt)

            let assistantResponse = messages.last(where: { $0.role == .assistant })?.content ?? ""
            return BenchmarkAutomationResult(
                modelPath: configuration.modelPath,
                prompt: configuration.prompt,
                contextWindowSize: configuration.contextWindowSize,
                maxTokens: configuration.maxTokens,
                mode: configuration.mode,
                generatedTokenCount: metrics.generatedTokenCount,
                timeToFirstTokenSeconds: metrics.timeToFirstTokenSeconds,
                decodeTokensPerSecond: metrics.finalDecodeTokensPerSecond,
                endToEndTokensPerSecond: metrics.endToEndTokensPerSecond,
                generationDurationSeconds: metrics.generationDurationSeconds,
                assistantResponse: assistantResponse,
                errorMessage: errorMessage
            )
        }
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
