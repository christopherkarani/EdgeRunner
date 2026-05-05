import EdgeRunner
import Foundation
import Observation

public enum ChatRole: String, CaseIterable, Codable, Equatable, Sendable {
    case system
    case user
    case assistant
}

public struct ChatRequestMessage: Equatable, Sendable {
    public let role: ChatRole
    public let content: String

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatTranscriptEntry: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let role: ChatRole
    public var content: String

    public init(id: UUID = UUID(), role: ChatRole, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

public struct LocalModelOption: Identifiable, Equatable, Sendable {
    public let path: String

    public init(path: String) {
        self.path = path
    }

    public var id: String { path }

    public var displayName: String {
        URL(fileURLWithPath: path).lastPathComponent
    }
}

public struct GenerationMetrics: Equatable, Sendable {
    public var generatedTokenCount: Int
    public var timeToFirstTokenSeconds: Double?
    public var rollingDecodeTokensPerSecond: Double
    public var finalDecodeTokensPerSecond: Double
    public var endToEndTokensPerSecond: Double
    public var generationDurationSeconds: Double

    public init(
        generatedTokenCount: Int = 0,
        timeToFirstTokenSeconds: Double? = nil,
        rollingDecodeTokensPerSecond: Double = 0,
        finalDecodeTokensPerSecond: Double = 0,
        endToEndTokensPerSecond: Double = 0,
        generationDurationSeconds: Double = 0
    ) {
        self.generatedTokenCount = generatedTokenCount
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.rollingDecodeTokensPerSecond = rollingDecodeTokensPerSecond
        self.finalDecodeTokensPerSecond = finalDecodeTokensPerSecond
        self.endToEndTokensPerSecond = endToEndTokensPerSecond
        self.generationDurationSeconds = generationDurationSeconds
    }

    public static let zero = GenerationMetrics()
}

public struct ThroughputTracker: Sendable {
    private var generationStartTime: TimeInterval?
    private var firstTokenTime: TimeInterval?
    private var lastTokenTime: TimeInterval?
    private var generatedTokenCount = 0

    public init() {}

    public mutating func start(at time: TimeInterval) {
        generationStartTime = time
        firstTokenTime = nil
        lastTokenTime = nil
        generatedTokenCount = 0
    }

    @discardableResult
    public mutating func recordToken(at time: TimeInterval) -> GenerationMetrics {
        if generationStartTime == nil {
            start(at: time)
        }

        if firstTokenTime == nil {
            firstTokenTime = time
        }

        lastTokenTime = time
        generatedTokenCount += 1
        return snapshot(finishTime: time)
    }

    public mutating func finish(at time: TimeInterval) -> GenerationMetrics {
        snapshot(finishTime: time)
    }

    private func snapshot(finishTime: TimeInterval) -> GenerationMetrics {
        guard let generationStartTime else {
            return .zero
        }

        let totalDuration = max(0, finishTime - generationStartTime)
        let ttft = firstTokenTime.map { max(0, $0 - generationStartTime) }

        let rollingDecodeTokensPerSecond: Double
        if let firstTokenTime, let lastTokenTime, generatedTokenCount > 1 {
            let decodeDuration = max(0, lastTokenTime - firstTokenTime)
            if decodeDuration > 0 {
                rollingDecodeTokensPerSecond = Double(generatedTokenCount - 1) / decodeDuration
            } else {
                rollingDecodeTokensPerSecond = 0
            }
        } else {
            rollingDecodeTokensPerSecond = 0
        }

        let finalDecodeTokensPerSecond = rollingDecodeTokensPerSecond
        let endToEndTokensPerSecond: Double
        if totalDuration > 0 {
            endToEndTokensPerSecond = Double(generatedTokenCount) / totalDuration
        } else {
            endToEndTokensPerSecond = 0
        }

        return GenerationMetrics(
            generatedTokenCount: generatedTokenCount,
            timeToFirstTokenSeconds: ttft,
            rollingDecodeTokensPerSecond: rollingDecodeTokensPerSecond,
            finalDecodeTokensPerSecond: finalDecodeTokensPerSecond,
            endToEndTokensPerSecond: endToEndTokensPerSecond,
            generationDurationSeconds: totalDuration
        )
    }
}

struct StreamUpdateBuffer: Sendable {
    struct Update: Equatable, Sendable {
        let text: String
        let metrics: GenerationMetrics
    }

    private let maxBufferedTokens: Int
    private let maxLatencySeconds: TimeInterval
    private var pendingText = ""
    private var pendingTokenCount = 0
    private var lastFlushTime: TimeInterval?
    private var hasDeliveredFirstToken = false

    init(maxBufferedTokens: Int = 4, maxLatencySeconds: TimeInterval = 0.12) {
        self.maxBufferedTokens = max(maxBufferedTokens, 1)
        self.maxLatencySeconds = max(maxLatencySeconds, 0)
    }

    mutating func enqueue(
        _ token: String,
        at time: TimeInterval,
        metrics: GenerationMetrics
    ) -> Update? {
        pendingText += token
        pendingTokenCount += 1

        if !hasDeliveredFirstToken {
            hasDeliveredFirstToken = true
            return flush(at: time, metrics: metrics)
        }

        let referenceTime = lastFlushTime ?? time
        let shouldFlush =
            pendingTokenCount >= maxBufferedTokens
            || (time - referenceTime) >= maxLatencySeconds

        guard shouldFlush else {
            return nil
        }

        return flush(at: time, metrics: metrics)
    }

    mutating func finish(metrics: GenerationMetrics) -> Update? {
        guard pendingTokenCount > 0 else { return nil }
        return flush(at: lastFlushTime ?? 0, metrics: metrics)
    }

    private mutating func flush(at time: TimeInterval, metrics: GenerationMetrics) -> Update {
        let update = Update(text: pendingText, metrics: metrics)
        pendingText.removeAll(keepingCapacity: true)
        pendingTokenCount = 0
        lastFlushTime = time
        return update
    }
}

public protocol ChatGenerating: Sendable {
    func streamReply(
        for messages: [ChatRequestMessage],
        maxTokens: Int,
        sampling: SamplingConfiguration
    ) async -> AsyncThrowingStream<String, Error>
}

public protocol ChatGeneratorFactory: Sendable {
    func makeGenerator(
        modelPath: String,
        contextWindowSize: Int
    ) async throws -> any ChatGenerating
}

public struct EdgeRunnerChatGeneratorFactory: ChatGeneratorFactory {
    public init() {}

    public func makeGenerator(
        modelPath: String,
        contextWindowSize: Int
    ) async throws -> any ChatGenerating {
        let model = try await ModelLoader.load(
            from: URL(fileURLWithPath: modelPath),
            configuration: ModelConfiguration(contextWindowSize: contextWindowSize)
        )
        return EdgeRunnerChatGenerator(model: model)
    }
}

public struct EdgeRunnerChatGenerator: ChatGenerating {
    private let model: any EdgeRunnerLanguageModel

    public init(model: any EdgeRunnerLanguageModel) {
        self.model = model
    }

    public func streamReply(
        for messages: [ChatRequestMessage],
        maxTokens: Int,
        sampling: SamplingConfiguration
    ) async -> AsyncThrowingStream<String, Error> {
        let prompt = formattedPrompt(for: messages)
        let model = self.model

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    for _ in 0..<maxTokens {
                        try Task.checkCancellation()
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: sampling
                        )
                        if tokenID == model.eosTokenID { break }
                        tokenIDs.append(tokenID)
                        continuation.yield(model.detokenize([tokenID]))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: GenerationError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func formattedPrompt(for messages: [ChatRequestMessage]) -> String {
        let transcript: [EdgeRunnerCore.ChatMessage] = messages.map { message in
            EdgeRunnerCore.ChatMessage(
                role: edgeRunnerRole(for: message.role),
                content: message.content
            )
        }

        if let templated = model.applyChatTemplate(
            messages: transcript,
            addGenerationPrompt: true
        ) {
            return templated
        }

        var lines: [String] = []
        lines.reserveCapacity(messages.count + 1)
        for message in messages {
            switch message.role {
            case .system:
                lines.append("System: \(message.content)")
            case .user:
                lines.append("User: \(message.content)")
            case .assistant:
                lines.append("Assistant: \(message.content)")
            }
        }
        lines.append("Assistant:")
        return lines.joined(separator: "\n")
    }

    private func edgeRunnerRole(for role: ChatRole) -> String {
        switch role {
        case .system:
            return "system"
        case .user:
            return "user"
        case .assistant:
            return "assistant"
        }
    }
}

@MainActor
@Observable
public final class ChatRuntime {
    public var modelPath: String
    public var systemPrompt: String
    public var currentInput: String
    public var benchmarkPrompt: String
    public var benchmarkMode: BenchmarkMode
    public var availableLocalModels: [LocalModelOption]
    public var messages: [ChatTranscriptEntry]
    public var metrics: GenerationMetrics
    public var errorMessage: String?
    public var benchmarkStatusMessage: String?
    public var isGenerating: Bool
    public var contextWindowSize: Int
    public var maxResponseTokens: Int
    public var sampling: SamplingConfiguration

    @ObservationIgnored
    private let generatorFactory: any ChatGeneratorFactory

    @ObservationIgnored
    private let timeSource: @Sendable () -> TimeInterval

    @ObservationIgnored
    private var generationTask: Task<Void, Never>?

    @ObservationIgnored
    private let decodeBenchmarker: any DecodeBenchmarkRunning

    @ObservationIgnored
    private var cachedGenerator: CachedGenerator?

    public init(
        modelPath: String = ProcessInfo.processInfo.environment["EDGERUNNER_CHAT_MODEL_PATH"] ?? "",
        systemPrompt: String = "You are a concise, helpful assistant running entirely on-device.",
        currentInput: String = "",
        benchmarkPrompt: String = "Explain quantum computing in simple terms.",
        benchmarkMode: BenchmarkMode = .rawDecode,
        availableLocalModels: [LocalModelOption] = [],
        messages: [ChatTranscriptEntry] = [],
        metrics: GenerationMetrics = .zero,
        errorMessage: String? = nil,
        benchmarkStatusMessage: String? = nil,
        isGenerating: Bool = false,
        contextWindowSize: Int = 4096,
        maxResponseTokens: Int = 512,
        sampling: SamplingConfiguration = SamplingConfiguration(),
        generatorFactory: any ChatGeneratorFactory = EdgeRunnerChatGeneratorFactory(),
        decodeBenchmarker: any DecodeBenchmarkRunning = EdgeRunnerDecodeBenchmarker(),
        timeSource: @escaping @Sendable () -> TimeInterval = {
            ProcessInfo.processInfo.systemUptime
        }
    ) {
        self.modelPath = modelPath
        self.systemPrompt = systemPrompt
        self.currentInput = currentInput
        self.benchmarkPrompt = benchmarkPrompt
        self.benchmarkMode = benchmarkMode
        self.availableLocalModels = availableLocalModels
        self.messages = messages
        self.metrics = metrics
        self.errorMessage = errorMessage
        self.benchmarkStatusMessage = benchmarkStatusMessage
        self.isGenerating = isGenerating
        self.contextWindowSize = contextWindowSize
        self.maxResponseTokens = maxResponseTokens
        self.sampling = sampling
        self.generatorFactory = generatorFactory
        self.decodeBenchmarker = decodeBenchmarker
        self.timeSource = timeSource

        if availableLocalModels.isEmpty {
            refreshDiscoveredModels()
        }
    }

    public var modelDisplayName: String {
        let trimmedModelPath = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModelPath.isEmpty else { return "No model selected" }
        return URL(fileURLWithPath: trimmedModelPath).lastPathComponent
    }

    public func sendCurrentInput() {
        let trimmedInput = currentInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedInput.isEmpty else { return }
        guard generationTask == nil else { return }

        currentInput = ""
        generationTask = Task { [self] in
            await runGeneration(for: trimmedInput)
        }
    }

    public func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    public func resetConversation() {
        cancelGeneration()
        messages.removeAll()
        metrics = .zero
        errorMessage = nil
        benchmarkStatusMessage = nil
    }

    public func refreshDiscoveredModels(in directory: URL? = nil) {
        let documentsDirectory = directory ?? FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first

        guard let documentsDirectory else {
            availableLocalModels = []
            return
        }

        let discoveredModels = (try? FileManager.default.contentsOfDirectory(
            at: documentsDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ))?
        .filter { $0.pathExtension.caseInsensitiveCompare("gguf") == .orderedSame }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        .map { LocalModelOption(path: $0.path) } ?? []

        availableLocalModels = discoveredModels
    }

    public func selectModel(_ model: LocalModelOption) {
        modelPath = model.path
        errorMessage = nil
    }

    public func runBenchmark() async {
        guard generationTask == nil else { return }

        let trimmedBenchmarkPrompt = benchmarkPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedBenchmarkPrompt.isEmpty else {
            errorMessage = "Set a benchmark prompt before running."
            benchmarkStatusMessage = nil
            return
        }

        benchmarkStatusMessage = "Benchmark running..."
        errorMessage = nil

        let task = Task { [self] in
            switch benchmarkMode {
            case .rawDecode:
                _ = await runRawBenchmark(prompt: trimmedBenchmarkPrompt)
            case .streamedChat:
                await runStreamBenchmark(prompt: trimmedBenchmarkPrompt)
            }

            if Task.isCancelled {
                benchmarkStatusMessage = "Benchmark canceled."
            } else if errorMessage != nil {
                benchmarkStatusMessage = "Benchmark failed."
            } else {
                benchmarkStatusMessage = "Benchmark completed."
            }
        }

        generationTask = task
        await task.value
    }

    func runRawBenchmark(prompt: String) async -> EdgeRunnerDecodeBenchmarkResult {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedModelPath = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            errorMessage = "Set a benchmark prompt before running."
            return emptyRawBenchmarkResult(prompt: trimmedPrompt)
        }

        guard !trimmedModelPath.isEmpty else {
            errorMessage = "Set a local GGUF model path before running a benchmark."
            return emptyRawBenchmarkResult(prompt: trimmedPrompt)
        }

        messages = [
            ChatTranscriptEntry(role: .user, content: trimmedPrompt),
            ChatTranscriptEntry(role: .assistant, content: "")
        ]
        metrics = .zero
        isGenerating = true
        defer {
            isGenerating = false
            generationTask = nil
        }

        do {
            let result = try await decodeBenchmarker.runDecodeBenchmark(
                modelPath: trimmedModelPath,
                prompt: trimmedPrompt,
                maxTokens: maxResponseTokens,
                contextWindowSize: contextWindowSize
            )
            if Task.isCancelled {
                removeTrailingAssistantPlaceholderIfNeeded()
                return result
            }
            messages[messages.count - 1].content = result.assistantResponse
            metrics = GenerationMetrics(
                generatedTokenCount: result.generatedTokenCount,
                timeToFirstTokenSeconds: result.timeToFirstTokenSeconds,
                rollingDecodeTokensPerSecond: result.decodeTokensPerSecond,
                finalDecodeTokensPerSecond: result.decodeTokensPerSecond,
                endToEndTokensPerSecond: result.endToEndTokensPerSecond,
                generationDurationSeconds: result.generationDurationSeconds
            )
            return result
        } catch is CancellationError {
            removeTrailingAssistantPlaceholderIfNeeded()
            return emptyRawBenchmarkResult(prompt: trimmedPrompt)
        } catch {
            errorMessage = String(describing: error)
            removeTrailingAssistantPlaceholderIfNeeded()
            return emptyRawBenchmarkResult(prompt: trimmedPrompt)
        }
    }

    func runStreamBenchmark(prompt: String) async {
        await runGeneration(for: prompt)
    }

    func runGeneration(for input: String) async {
        let trimmedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedInput.isEmpty else { return }

        let trimmedModelPath = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModelPath.isEmpty else {
            errorMessage = "Set a local GGUF model path before sending a prompt."
            return
        }

        let userEntry = ChatTranscriptEntry(role: .user, content: trimmedInput)
        messages.append(userEntry)
        let requestMessages = buildRequestMessages()
        messages.append(ChatTranscriptEntry(role: .assistant, content: ""))
        errorMessage = nil
        metrics = .zero
        isGenerating = true

        var tracker = ThroughputTracker()
        tracker.start(at: timeSource())
        var streamUpdateBuffer = StreamUpdateBuffer()

        do {
            let generator = try await generator(for: trimmedModelPath)
            let stream = await generator.streamReply(
                for: requestMessages,
                maxTokens: maxResponseTokens,
                sampling: sampling
            )

            for try await token in stream {
                try Task.checkCancellation()
                let now = timeSource()
                let snapshot = tracker.recordToken(at: now)
                if let update = streamUpdateBuffer.enqueue(token, at: now, metrics: snapshot) {
                    applyStreamUpdate(update)
                }
            }

            let finalMetrics = tracker.finish(at: timeSource())
            if let update = streamUpdateBuffer.finish(metrics: finalMetrics) {
                applyStreamUpdate(update)
            } else {
                metrics = finalMetrics
            }
        } catch is CancellationError {
            let finalMetrics = tracker.finish(at: timeSource())
            if let update = streamUpdateBuffer.finish(metrics: finalMetrics) {
                applyStreamUpdate(update)
            } else {
                metrics = finalMetrics
            }
            removeTrailingAssistantPlaceholderIfNeeded()
        } catch {
            errorMessage = String(describing: error)
            let finalMetrics = tracker.finish(at: timeSource())
            if let update = streamUpdateBuffer.finish(metrics: finalMetrics) {
                applyStreamUpdate(update)
            } else {
                metrics = finalMetrics
            }
            removeTrailingAssistantPlaceholderIfNeeded()
        }

        isGenerating = false
        generationTask = nil
    }

    private func emptyRawBenchmarkResult(prompt: String) -> EdgeRunnerDecodeBenchmarkResult {
        EdgeRunnerDecodeBenchmarkResult(
            prompt: prompt,
            maxTokens: maxResponseTokens,
            generatedTokenCount: 0,
            timeToFirstTokenSeconds: nil,
            decodeTokensPerSecond: 0,
            endToEndTokensPerSecond: 0,
            generationDurationSeconds: 0,
            assistantResponse: ""
        )
    }

    private func buildRequestMessages() -> [ChatRequestMessage] {
        var requestMessages = [ChatRequestMessage]()

        let trimmedSystemPrompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystemPrompt.isEmpty {
            requestMessages.append(
                ChatRequestMessage(role: .system, content: trimmedSystemPrompt)
            )
        }

        requestMessages.append(
            contentsOf: messages.compactMap { entry in
                let trimmedContent = entry.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmedContent.isEmpty else { return nil }
                return ChatRequestMessage(role: entry.role, content: trimmedContent)
            }
        )

        return requestMessages
    }

    private func appendAssistantToken(_ token: String) {
        guard !messages.isEmpty, messages[messages.count - 1].role == .assistant else {
            messages.append(ChatTranscriptEntry(role: .assistant, content: token))
            return
        }

        messages[messages.count - 1].content += token
    }

    private func applyStreamUpdate(_ update: StreamUpdateBuffer.Update) {
        appendAssistantToken(update.text)
        metrics = update.metrics
    }

    private func removeTrailingAssistantPlaceholderIfNeeded() {
        guard let lastMessage = messages.last else { return }
        guard lastMessage.role == .assistant else { return }
        guard lastMessage.content.isEmpty else { return }
        messages.removeLast()
    }

    private func generator(for modelPath: String) async throws -> any ChatGenerating {
        if let cachedGenerator,
           cachedGenerator.modelPath == modelPath,
           cachedGenerator.contextWindowSize == contextWindowSize {
            return cachedGenerator.generator
        }

        let generator = try await generatorFactory.makeGenerator(
            modelPath: modelPath,
            contextWindowSize: contextWindowSize
        )
        cachedGenerator = CachedGenerator(
            modelPath: modelPath,
            contextWindowSize: contextWindowSize,
            generator: generator
        )
        return generator
    }

    private struct CachedGenerator {
        let modelPath: String
        let contextWindowSize: Int
        let generator: any ChatGenerating
    }
}
