import Foundation
import Testing
import EdgeRunner
@testable import EdgeRunnerChatAppCore

private actor RequestRecorder {
    private(set) var messages = [[ChatRequestMessage]]()

    func record(_ request: [ChatRequestMessage]) {
        messages.append(request)
    }

    func latest() -> [ChatRequestMessage]? {
        messages.last
    }
}

private struct MockGenerator: ChatGenerating {
    let streamBuilder: @Sendable ([ChatRequestMessage], Int) async -> AsyncThrowingStream<String, Error>

    func streamReply(
        for messages: [ChatRequestMessage],
        maxTokens: Int,
        sampling: SamplingConfiguration
    ) async -> AsyncThrowingStream<String, Error> {
        await streamBuilder(messages, maxTokens)
    }
}

private struct MockFactory: ChatGeneratorFactory {
    let makeStream: @Sendable ([ChatRequestMessage], Int) async -> AsyncThrowingStream<String, Error>

    func makeGenerator(
        modelPath: String,
        contextWindowSize: Int
    ) async throws -> any ChatGenerating {
        MockGenerator(streamBuilder: makeStream)
    }
}

private struct MockDecodeBenchmarker: DecodeBenchmarkRunning {
    let run: @Sendable (String, String, Int, Int) async throws -> EdgeRunnerDecodeBenchmarkResult

    func runDecodeBenchmark(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        contextWindowSize: Int
    ) async throws -> EdgeRunnerDecodeBenchmarkResult {
        try await run(modelPath, prompt, maxTokens, contextWindowSize)
    }
}

private final class DeterministicTimeSource: @unchecked Sendable {
    private let values: [TimeInterval]
    private var index = 0

    init(values: [TimeInterval]) {
        self.values = values
    }

    func next() -> TimeInterval {
        let value = values[min(index, values.count - 1)]
        index += 1
        return value
    }
}

@Suite("ThroughputTracker")
struct ThroughputTrackerTests {
    @Test func firstTokenCapturesTTFT() {
        var tracker = ThroughputTracker()

        tracker.start(at: 10.0)
        let snapshot = tracker.recordToken(at: 10.25)

        #expect(snapshot.generatedTokenCount == 1)
        #expect(abs((snapshot.timeToFirstTokenSeconds ?? 0) - 0.25) < 0.0001)
        #expect(snapshot.rollingDecodeTokensPerSecond == 0)
        #expect(snapshot.finalDecodeTokensPerSecond == 0)
    }

    @Test func decodeThroughputExcludesTTFT() {
        var tracker = ThroughputTracker()

        tracker.start(at: 5.0)
        _ = tracker.recordToken(at: 5.4)
        _ = tracker.recordToken(at: 5.9)
        let rolling = tracker.recordToken(at: 6.4)
        let final = tracker.finish(at: 6.5)

        #expect(rolling.generatedTokenCount == 3)
        #expect(abs((rolling.timeToFirstTokenSeconds ?? 0) - 0.4) < 0.0001)
        #expect(abs(rolling.rollingDecodeTokensPerSecond - 2.0) < 0.0001)
        #expect(abs(final.finalDecodeTokensPerSecond - 2.0) < 0.0001)
        #expect(abs(final.endToEndTokensPerSecond - 2.0) < 0.0001)
    }
}

@Suite("StreamUpdateBuffer")
struct StreamUpdateBufferTests {
    @Test func firstTokenFlushesImmediately() {
        var buffer = StreamUpdateBuffer(maxBufferedTokens: 4, maxLatencySeconds: 0.12)
        let firstMetrics = GenerationMetrics(
            generatedTokenCount: 1,
            timeToFirstTokenSeconds: 0.2,
            rollingDecodeTokensPerSecond: 0,
            finalDecodeTokensPerSecond: 0,
            endToEndTokensPerSecond: 5,
            generationDurationSeconds: 0.2
        )

        let firstUpdate = buffer.enqueue("Hello", at: 10.2, metrics: firstMetrics)

        #expect(firstUpdate?.text == "Hello")
        #expect(firstUpdate?.metrics == firstMetrics)
    }

    @Test func laterTokensBatchUntilThreshold() {
        var buffer = StreamUpdateBuffer(maxBufferedTokens: 3, maxLatencySeconds: 10)
        let firstMetrics = GenerationMetrics(generatedTokenCount: 1, timeToFirstTokenSeconds: 0.1)
        let secondMetrics = GenerationMetrics(generatedTokenCount: 2, timeToFirstTokenSeconds: 0.1)
        let thirdMetrics = GenerationMetrics(generatedTokenCount: 3, timeToFirstTokenSeconds: 0.1)
        let fourthMetrics = GenerationMetrics(generatedTokenCount: 4, timeToFirstTokenSeconds: 0.1)

        _ = buffer.enqueue("A", at: 0.1, metrics: firstMetrics)
        #expect(buffer.enqueue("B", at: 0.2, metrics: secondMetrics) == nil)
        #expect(buffer.enqueue("C", at: 0.3, metrics: thirdMetrics) == nil)

        let update = buffer.enqueue("D", at: 0.4, metrics: fourthMetrics)

        #expect(update?.text == "BCD")
        #expect(update?.metrics == fourthMetrics)
    }

    @Test func finishFlushesRemainingBufferedTokens() {
        var buffer = StreamUpdateBuffer(maxBufferedTokens: 8, maxLatencySeconds: 10)
        let firstMetrics = GenerationMetrics(generatedTokenCount: 1, timeToFirstTokenSeconds: 0.1)
        let secondMetrics = GenerationMetrics(generatedTokenCount: 2, timeToFirstTokenSeconds: 0.1)
        let finalMetrics = GenerationMetrics(
            generatedTokenCount: 3,
            timeToFirstTokenSeconds: 0.1,
            rollingDecodeTokensPerSecond: 12,
            finalDecodeTokensPerSecond: 12,
            endToEndTokensPerSecond: 8,
            generationDurationSeconds: 0.3
        )

        _ = buffer.enqueue("A", at: 0.1, metrics: firstMetrics)
        _ = buffer.enqueue("B", at: 0.2, metrics: secondMetrics)

        let update = buffer.finish(metrics: finalMetrics)

        #expect(update?.text == "B")
        #expect(update?.metrics == finalMetrics)
    }
}

@Suite("ChatRuntime")
@MainActor
struct ChatRuntimeTests {
    @Test func benchmarkConfigurationResolvesDocumentsModelPath() throws {
        let documentsDirectory = URL(fileURLWithPath: "/tmp/edgerunner-documents", isDirectory: true)
        let resultDirectory = URL(fileURLWithPath: "/tmp/edgerunner-results", isDirectory: true)
        let configuration = try #require(
            BenchmarkAutomationConfiguration.make(
                environment: [
                    BenchmarkAutomationConfiguration.modelFilenameEnvironmentKey: "Bonsai-8B-Q1_0.gguf",
                    BenchmarkAutomationConfiguration.promptEnvironmentKey: "Benchmark me",
                    BenchmarkAutomationConfiguration.maxTokensEnvironmentKey: "96",
                    BenchmarkAutomationConfiguration.contextWindowEnvironmentKey: "4096",
                    BenchmarkAutomationConfiguration.resultFilenameEnvironmentKey: "bonsai-result.json",
                ],
                documentsDirectory: documentsDirectory,
                resultDirectory: resultDirectory
            )
        )

        #expect(configuration.modelPath == "/tmp/edgerunner-documents/Bonsai-8B-Q1_0.gguf")
        #expect(configuration.prompt == "Benchmark me")
        #expect(configuration.maxTokens == 96)
        #expect(configuration.contextWindowSize == 4096)
        #expect(configuration.mode == .rawDecode)
        #expect(configuration.resultURL.path == "/tmp/edgerunner-results/bonsai-result.json")
    }

    @Test func benchmarkConfigurationDefaultsAreStable() throws {
        let documentsDirectory = URL(fileURLWithPath: "/tmp/edgerunner-documents", isDirectory: true)
        let resultDirectory = URL(fileURLWithPath: "/tmp/edgerunner-results", isDirectory: true)
        let configuration = try #require(
            BenchmarkAutomationConfiguration.make(
                environment: [
                    BenchmarkAutomationConfiguration.modelPathEnvironmentKey: "/private/tmp/Bonsai-8B-Q1_0.gguf",
                    BenchmarkAutomationConfiguration.maxTokensEnvironmentKey: "-1",
                    BenchmarkAutomationConfiguration.contextWindowEnvironmentKey: "not-a-number",
                ],
                documentsDirectory: documentsDirectory,
                resultDirectory: resultDirectory
            )
        )

        #expect(configuration.modelPath == "/private/tmp/Bonsai-8B-Q1_0.gguf")
        #expect(configuration.prompt == "Explain quantum computing in simple terms.")
        #expect(configuration.maxTokens == 128)
        #expect(configuration.contextWindowSize == 4096)
        #expect(configuration.mode == .rawDecode)
        #expect(configuration.resultURL.lastPathComponent == "autobench-result.json")
    }

    @Test func benchmarkConfigurationParsesStreamMode() throws {
        let documentsDirectory = URL(fileURLWithPath: "/tmp/edgerunner-documents", isDirectory: true)
        let resultDirectory = URL(fileURLWithPath: "/tmp/edgerunner-results", isDirectory: true)
        let configuration = try #require(
            BenchmarkAutomationConfiguration.make(
                environment: [
                    BenchmarkAutomationConfiguration.modelPathEnvironmentKey: "/tmp/Bonsai-1.7B.gguf",
                    BenchmarkAutomationConfiguration.modeEnvironmentKey: "streamed_chat",
                ],
                documentsDirectory: documentsDirectory,
                resultDirectory: resultDirectory
            )
        )

        #expect(configuration.mode == .streamedChat)
    }

    @Test func benchmarkWriterCreatesMissingDirectories() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let resultURL = root
            .appendingPathComponent("BenchmarkResults", isDirectory: true)
            .appendingPathComponent("autobench-result.json")
        let result = BenchmarkAutomationResult(
            modelPath: "/tmp/Bonsai-8B-Q1_0.gguf",
            prompt: "Test prompt",
            contextWindowSize: 4096,
            maxTokens: 16,
            mode: .rawDecode,
            generatedTokenCount: 0,
            timeToFirstTokenSeconds: nil,
            decodeTokensPerSecond: 0,
            endToEndTokensPerSecond: 0,
            generationDurationSeconds: 0,
            assistantResponse: "",
            errorMessage: "missing model"
        )

        try BenchmarkAutomationWriter.write(result, to: resultURL)

        #expect(FileManager.default.fileExists(atPath: resultURL.path))
    }

    @Test func refreshDiscoveredModelsListsLocalGGUFFiles() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("Bonsai-8B-Q1_0.gguf"))
        try Data().write(to: root.appendingPathComponent("Bonsai-1.7B.gguf"))
        try Data().write(to: root.appendingPathComponent("notes.txt"))

        let runtime = ChatRuntime()

        runtime.refreshDiscoveredModels(in: root)

        #expect(runtime.availableLocalModels.map(\.displayName) == [
            "Bonsai-1.7B.gguf",
            "Bonsai-8B-Q1_0.gguf",
        ])
    }

    @Test func selectModelUpdatesModelPath() {
        let runtime = ChatRuntime()
        let model = LocalModelOption(path: "/tmp/Bonsai-8B-Q1_0.gguf")

        runtime.selectModel(model)

        #expect(runtime.modelPath == "/tmp/Bonsai-8B-Q1_0.gguf")
    }

    @Test func automatedBenchmarkCollectsMetrics() async throws {
        let timeSource = DeterministicTimeSource(values: [10.0, 10.4, 10.9, 11.1])
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { _, _ in
                    AsyncThrowingStream { continuation in
                        continuation.yield("Hello")
                        continuation.yield(" world")
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                timeSource.next()
            }
        )

        let configuration = BenchmarkAutomationConfiguration(
            modelPath: "/tmp/Bonsai-8B-Q1_0.gguf",
            prompt: "Say hi",
            resultURL: URL(fileURLWithPath: "/tmp/autobench-result.json"),
            maxTokens: 128,
            contextWindowSize: 4096,
            mode: .streamedChat
        )

        let result = await runtime.runAutomatedBenchmark(configuration)

        #expect(result.modelPath == configuration.modelPath)
        #expect(result.prompt == "Say hi")
        #expect(result.mode == .streamedChat)
        #expect(result.generatedTokenCount == 2)
        #expect(abs((result.timeToFirstTokenSeconds ?? 0) - 0.4) < 0.0001)
        #expect(abs(result.decodeTokensPerSecond - 2.0) < 0.0001)
        #expect(abs(result.endToEndTokensPerSecond - (2.0 / 1.1)) < 0.0001)
        #expect(result.assistantResponse == "Hello world")
        #expect(result.errorMessage == nil)
    }

    @Test func automatedRawBenchmarkCollectsMetrics() async throws {
        let runtime = ChatRuntime(
            decodeBenchmarker: MockDecodeBenchmarker { modelPath, prompt, maxTokens, contextWindowSize in
                #expect(modelPath == "/tmp/Bonsai-8B-Q1_0.gguf")
                #expect(prompt == "Say hi")
                #expect(maxTokens == 128)
                #expect(contextWindowSize == 4096)
                return EdgeRunnerDecodeBenchmarkResult(
                    prompt: prompt,
                    maxTokens: maxTokens,
                    generatedTokenCount: 32,
                    timeToFirstTokenSeconds: 0.5,
                    decodeTokensPerSecond: 29.5,
                    endToEndTokensPerSecond: 18.0,
                    generationDurationSeconds: 1.8,
                    assistantResponse: "Hello world"
                )
            }
        )

        let configuration = BenchmarkAutomationConfiguration(
            modelPath: "/tmp/Bonsai-8B-Q1_0.gguf",
            prompt: "Say hi",
            resultURL: URL(fileURLWithPath: "/tmp/autobench-result.json"),
            maxTokens: 128,
            contextWindowSize: 4096,
            mode: .rawDecode
        )

        let result = await runtime.runAutomatedBenchmark(configuration)

        #expect(result.mode == .rawDecode)
        #expect(result.generatedTokenCount == 32)
        #expect(abs((result.timeToFirstTokenSeconds ?? 0) - 0.5) < 0.0001)
        #expect(abs(result.decodeTokensPerSecond - 29.5) < 0.0001)
        #expect(abs(result.endToEndTokensPerSecond - 18.0) < 0.0001)
        #expect(result.assistantResponse == "Hello world")
        #expect(result.errorMessage == nil)
    }

    @Test func runGenerationStreamsAssistantReply() async throws {
        let recorder = RequestRecorder()
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { messages, _ in
                    await recorder.record(messages)
                    return AsyncThrowingStream { continuation in
                        continuation.yield("Hello")
                        continuation.yield(" world")
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                ProcessInfo.processInfo.systemUptime
            }
        )

        runtime.modelPath = "/tmp/gemma-4-4b-it-q4.gguf"
        runtime.systemPrompt = "You are concise."

        await runtime.runGeneration(for: "Say hi")

        #expect(runtime.messages.count == 2)
        #expect(runtime.messages[0].role == .user)
        #expect(runtime.messages[0].content == "Say hi")
        #expect(runtime.messages[1].role == .assistant)
        #expect(runtime.messages[1].content == "Hello world")
        #expect(runtime.metrics.generatedTokenCount == 2)
        #expect(runtime.isGenerating == false)
        #expect(runtime.errorMessage == nil)

        let recorded = try #require(await recorder.latest())
        #expect(recorded.map(\.role) == [.system, .user])
        #expect(recorded.first?.content == "You are concise.")
        #expect(recorded.last?.content == "Say hi")
    }

    @Test func runBenchmarkUsesBenchmarkPromptAndUpdatesStatus() async throws {
        let recorder = RequestRecorder()
        let timeSource = DeterministicTimeSource(values: [20.0, 20.4, 20.9, 21.1])
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { messages, _ in
                    await recorder.record(messages)
                    return AsyncThrowingStream { continuation in
                        continuation.yield("Quantum")
                        continuation.yield(" chips")
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                timeSource.next()
            }
        )

        runtime.modelPath = "/tmp/Bonsai-8B-Q1_0.gguf"
        runtime.benchmarkPrompt = "Explain quantum computing in simple terms."
        runtime.benchmarkMode = .streamedChat

        await runtime.runBenchmark()

        #expect(runtime.messages.count == 2)
        #expect(runtime.messages[0].role == .user)
        #expect(runtime.messages[0].content == "Explain quantum computing in simple terms.")
        #expect(runtime.messages[1].role == .assistant)
        #expect(runtime.messages[1].content == "Quantum chips")
        #expect(runtime.metrics.generatedTokenCount == 2)
        #expect(runtime.benchmarkStatusMessage == "Benchmark completed.")
        #expect(runtime.errorMessage == nil)

        let recorded = try #require(await recorder.latest())
        #expect(recorded.map(\.role) == [.system, .user])
        #expect(recorded.last?.content == "Explain quantum computing in simple terms.")
    }

    @Test func rawBenchmarkUsesBenchmarkerAndUpdatesStatus() async throws {
        let runtime = ChatRuntime(
            decodeBenchmarker: MockDecodeBenchmarker { modelPath, prompt, maxTokens, contextWindowSize in
                #expect(modelPath == "/tmp/Bonsai-1.7B.gguf")
                #expect(prompt == "Explain quantum computing in simple terms.")
                #expect(maxTokens == 512)
                #expect(contextWindowSize == 4096)
                return EdgeRunnerDecodeBenchmarkResult(
                    prompt: prompt,
                    maxTokens: maxTokens,
                    generatedTokenCount: 64,
                    timeToFirstTokenSeconds: 0.42,
                    decodeTokensPerSecond: 31.0,
                    endToEndTokensPerSecond: 20.5,
                    generationDurationSeconds: 3.1,
                    assistantResponse: "Quantum chips"
                )
            }
        )

        runtime.modelPath = "/tmp/Bonsai-1.7B.gguf"
        runtime.benchmarkPrompt = "Explain quantum computing in simple terms."
        runtime.benchmarkMode = .rawDecode

        await runtime.runBenchmark()

        #expect(runtime.messages.count == 2)
        #expect(runtime.messages[0].role == .user)
        #expect(runtime.messages[0].content == "Explain quantum computing in simple terms.")
        #expect(runtime.messages[1].role == .assistant)
        #expect(runtime.messages[1].content == "Quantum chips")
        #expect(runtime.metrics.generatedTokenCount == 64)
        #expect(abs((runtime.metrics.timeToFirstTokenSeconds ?? 0) - 0.42) < 0.0001)
        #expect(abs(runtime.metrics.finalDecodeTokensPerSecond - 31.0) < 0.0001)
        #expect(runtime.benchmarkStatusMessage == "Benchmark completed.")
        #expect(runtime.errorMessage == nil)
    }

    @Test func runBenchmarkRejectsBlankPrompt() async {
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { _, _ in
                    AsyncThrowingStream { continuation in
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                ProcessInfo.processInfo.systemUptime
            }
        )

        runtime.modelPath = "/tmp/Bonsai-8B-Q1_0.gguf"
        runtime.benchmarkPrompt = "   "

        await runtime.runBenchmark()

        #expect(runtime.messages.isEmpty)
        #expect(runtime.errorMessage == "Set a benchmark prompt before running.")
        #expect(runtime.benchmarkStatusMessage == nil)
    }

    @Test func resetConversationClearsTranscriptAndMetrics() async {
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { _, _ in
                    AsyncThrowingStream { continuation in
                        continuation.yield("Done")
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                ProcessInfo.processInfo.systemUptime
            }
        )

        runtime.modelPath = "/tmp/gemma-4-4b-it-q4.gguf"
        await runtime.runGeneration(for: "Hello")
        runtime.benchmarkStatusMessage = "Benchmark completed."
        runtime.resetConversation()

        #expect(runtime.messages.isEmpty)
        #expect(runtime.metrics == .zero)
        #expect(runtime.errorMessage == nil)
        #expect(runtime.benchmarkStatusMessage == nil)
    }

    @Test func blankInputIsIgnored() {
        let runtime = ChatRuntime(
            generatorFactory: MockFactory(
                makeStream: { _, _ in
                    AsyncThrowingStream { continuation in
                        continuation.finish()
                    }
                }
            ),
            timeSource: {
                ProcessInfo.processInfo.systemUptime
            }
        )

        runtime.currentInput = "   "
        runtime.sendCurrentInput()

        #expect(runtime.messages.isEmpty)
        #expect(runtime.isGenerating == false)
    }
}
