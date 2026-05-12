import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

private struct GemmaBenchmarkCoherence: Codable, Equatable {
    let passed: Bool
    let reasons: [String]

    static func evaluate(generatedText: String, generatedTokenIDs: [Int]) -> GemmaBenchmarkCoherence {
        var reasons: [String] = []
        let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            reasons.append("empty_or_whitespace")
        }
        if generatedText.contains("\u{0000}") {
            reasons.append("contains_null")
        }
        if trimmed.localizedCaseInsensitiveContains("Thinking Process")
            || trimmed.localizedCaseInsensitiveContains("Analyze the Request") {
            reasons.append("reasoning_preamble_without_answer")
        }
        if generatedTokenIDs.isEmpty {
            reasons.append("no_generated_tokens")
        }
        if generatedTokenIDs.count >= 4 && Set(generatedTokenIDs).count <= max(1, generatedTokenIDs.count / 4) {
            reasons.append("token_repetition")
        }

        let words = trimmed
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
        if words.count >= 8 {
            var counts: [String: Int] = [:]
            for word in words {
                counts[word, default: 0] += 1
            }
            if let mostRepeated = counts.values.max(),
               Double(mostRepeated) / Double(words.count) >= 0.5 {
                reasons.append("word_repetition")
            }
            if let last = trimmed.last, !".!?".contains(last) {
                reasons.append("incomplete_sentence")
            }
        }

        return GemmaBenchmarkCoherence(passed: reasons.isEmpty, reasons: reasons)
    }
}

private struct GemmaBenchmarkPromptResult: Codable, Equatable {
    let label: String
    let prompt: String
    let promptTokenCount: Int
    let generatedTokenIDs: [Int]
    let generatedText: String
    let ttftSeconds: Double
    let promptProcessingSeconds: Double
    let promptTokS: Double
    let decodeSeconds: Double
    let decodeTokS: Double
    let generatedTokenCount: Int
    let coherence: GemmaBenchmarkCoherence
}

private struct GemmaBenchmarkArtifact: Codable, Equatable {
    let modelPath: String
    let modelFileSizeBytes: Int64
    let modelSha256: String
    let gitCommit: String
    let gitDirty: Bool
    let machine: String
    let osVersion: String
    let swiftVersion: String
    let command: String
    let env: [String: String]
    let limitations: [String]
    let shortPrompt: GemmaBenchmarkPromptResult
    let longPrompt: GemmaBenchmarkPromptResult
    let medianDecodeTokS: Double
    let medianTTFTSeconds: Double
    let runCount: Int
}

@Suite("Gemma 4 downloaded benchmark")
struct Gemma4DownloadedBenchmark {
    private static let modelPathEnvironmentKey = "EDGERUNNER_GEMMA4_BENCHMARK_MODEL"
    private static let artifactPathEnvironmentKey = "EDGERUNNER_GEMMA4_BENCHMARK_ARTIFACT"
    private static let maxTokens = 16
    private static let publishableMaxTokens = 64
    private static let contextWindowSize = 4096
    private static let warmupRuns = 1
    private static let medianRuns = 5

    @Test func runDownloadedGGUFThroughEdgeRunner() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )
        let prompt = shortPromptText(for: model)
        var tokenIDs = model.tokenize(prompt)
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        let start = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?
        var generatedTokens = 0
        var generatedTokenIDs: [Int] = []

        for _ in 0..<Self.maxTokens {
            let token: Int
            if generatedTokens == 0, let logitsModel = model as? any LogitsModel {
                let logits = try await logitsModel.logits(for: tokenIDs)
                let top = logits.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(8)
                    .map { (id: $0.offset, value: $0.element, text: model.detokenize([$0.offset])) }
                print("GEMMA4_BENCHMARK first_top_logits=\(top)")
                token = top.first?.id ?? 0
            } else {
                token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            }
            let now = ProcessInfo.processInfo.systemUptime
            if token == model.eosTokenID {
                break
            }
            firstTokenTime = firstTokenTime ?? now
            tokenIDs.append(token)
            generatedTokenIDs.append(token)
            generatedTokens += 1
        }

        let finish = ProcessInfo.processInfo.systemUptime
        let ttft = firstTokenTime.map { $0 - start } ?? 0
        let decodeDuration = firstTokenTime.map { finish - $0 } ?? 0
        let decodeTokens = max(0, generatedTokens - 1)
        let tokensPerSecond = decodeDuration > 0 ? Double(decodeTokens) / decodeDuration : 0
        let generatedText = model.detokenize(generatedTokenIDs)

        print("GEMMA4_BENCHMARK model=\(url.lastPathComponent)")
        print("GEMMA4_BENCHMARK generated_tokens=\(generatedTokens)")
        print("GEMMA4_BENCHMARK generated_token_ids=\(generatedTokenIDs)")
        print("GEMMA4_BENCHMARK generated_text=\(generatedText)")
        print("GEMMA4_BENCHMARK ttft_seconds=\(String(format: "%.6f", ttft))")
        print("GEMMA4_BENCHMARK decode_tok_s=\(String(format: "%.3f", tokensPerSecond))")

        #expect(generatedTokens > 0)
        #expect(!generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!generatedText.contains("\u{0000}"))
        #expect(Set(generatedTokenIDs).count > 1)
    }

    @Test func runDownloadedGGUFMedianBenchmark() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )

        var results: [(ttft: Double, decodeTokS: Double, generatedTokens: Int)] = []
        results.reserveCapacity(Self.medianRuns)
        for warmup in 0..<Self.warmupRuns {
            let result = try await runGeneration(model: model)
            print(
                "GEMMA4_MEDIAN_BENCHMARK warmup=\(warmup) generated_tokens=\(result.generatedTokens) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttft)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS))"
            )
        }
        for _ in 0..<Self.medianRuns {
            results.append(try await runGeneration(model: model))
        }

        for (index, result) in results.enumerated() {
            print(
                "GEMMA4_MEDIAN_BENCHMARK run=\(index) generated_tokens=\(result.generatedTokens) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttft)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS))"
            )
        }

        let decodeValues = results.map(\.decodeTokS).sorted()
        let ttftValues = results.map(\.ttft).sorted()
        let medianDecode = decodeValues[decodeValues.count / 2]
        let medianTTFT = ttftValues[ttftValues.count / 2]
        let bestDecode = decodeValues.last ?? 0
        let minDecode = decodeValues.first ?? 0

        print("GEMMA4_MEDIAN_BENCHMARK model=\(url.lastPathComponent)")
        print("GEMMA4_MEDIAN_BENCHMARK runs=\(Self.medianRuns)")
        print("GEMMA4_MEDIAN_BENCHMARK median_decode_tok_s=\(String(format: "%.3f", medianDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK best_decode_tok_s=\(String(format: "%.3f", bestDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK min_decode_tok_s=\(String(format: "%.3f", minDecode))")
        print("GEMMA4_MEDIAN_BENCHMARK median_ttft_seconds=\(String(format: "%.6f", medianTTFT))")
    }

    @Test func runDownloadedGGUFPublishableBenchmark() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )

        for warmup in 0..<Self.warmupRuns {
            let result = try await runGeneration(
                model: model,
                prompt: shortPromptText(for: model),
                label: "warmup_short",
                maxTokens: Self.publishableMaxTokens
            )
            print(
                "GEMMA4_PUBLISHABLE warmup=\(warmup) generated_tokens=\(result.generatedTokenCount) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttftSeconds)) "
                + "prompt_tok_s=\(String(format: "%.3f", result.promptTokS)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS)) "
                + "coherent=\(result.coherence.passed)"
            )
        }

        var shortResults: [GemmaBenchmarkPromptResult] = []
        shortResults.reserveCapacity(Self.medianRuns)
        for run in 0..<Self.medianRuns {
            let result = try await runGeneration(
                model: model,
                prompt: shortPromptText(for: model),
                label: "short",
                maxTokens: Self.publishableMaxTokens
            )
            shortResults.append(result)
            print(
                "GEMMA4_PUBLISHABLE prompt=short run=\(run) generated_tokens=\(result.generatedTokenCount) "
                + "ttft_seconds=\(String(format: "%.6f", result.ttftSeconds)) "
                + "prompt_tok_s=\(String(format: "%.3f", result.promptTokS)) "
                + "decode_tok_s=\(String(format: "%.3f", result.decodeTokS)) "
                + "coherent=\(result.coherence.passed)"
            )
        }

        let longResult = try await runGeneration(
            model: model,
            prompt: longPromptText(for: model),
            label: "long",
            maxTokens: Self.publishableMaxTokens
        )
        print(
            "GEMMA4_PUBLISHABLE prompt=long generated_tokens=\(longResult.generatedTokenCount) "
            + "prompt_tokens=\(longResult.promptTokenCount) "
            + "ttft_seconds=\(String(format: "%.6f", longResult.ttftSeconds)) "
            + "prompt_tok_s=\(String(format: "%.3f", longResult.promptTokS)) "
            + "decode_tok_s=\(String(format: "%.3f", longResult.decodeTokS)) "
            + "coherent=\(longResult.coherence.passed)"
        )

        let sortedDecode = shortResults.map(\.decodeTokS).sorted()
        let sortedTTFT = shortResults.map(\.ttftSeconds).sorted()
        let medianDecode = sortedDecode[sortedDecode.count / 2]
        let medianTTFT = sortedTTFT[sortedTTFT.count / 2]
        let medianShort = shortResults.sorted { $0.decodeTokS < $1.decodeTokS }[shortResults.count / 2]
        let artifact = GemmaBenchmarkArtifact(
            modelPath: url.path,
            modelFileSizeBytes: try Self.fileSize(url),
            modelSha256: try Self.sha256(url),
            gitCommit: Self.commandOutput(["git", "rev-parse", "--short", "HEAD"]) ?? "unknown",
            gitDirty: !(Self.commandOutput(["git", "status", "--porcelain"]) ?? "").isEmpty,
            machine: Self.commandOutput(["uname", "-m"]) ?? "unknown",
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            swiftVersion: Self.commandOutput(["swift", "--version"]) ?? "unknown",
            command: Self.benchmarkCommand(
                modelPath: url.path,
                artifactPath: Self.artifactURL().path
            ),
            env: Self.relevantEnvironment(),
            limitations: Self.limitations(
                medianDecodeTokS: medianDecode,
                shortResult: medianShort,
                longResult: longResult
            ),
            shortPrompt: medianShort,
            longPrompt: longResult,
            medianDecodeTokS: medianDecode,
            medianTTFTSeconds: medianTTFT,
            runCount: Self.medianRuns
        )

        try Self.writeArtifact(artifact)
        print("GEMMA4_PUBLISHABLE median_decode_tok_s=\(String(format: "%.3f", medianDecode))")
        print("GEMMA4_PUBLISHABLE median_ttft_seconds=\(String(format: "%.6f", medianTTFT))")
        print("GEMMA4_PUBLISHABLE artifact=\(Self.artifactURL().path)")
        print("GEMMA4_PUBLISHABLE short_generated_text=\(medianShort.generatedText)")
        print("GEMMA4_PUBLISHABLE long_generated_text=\(longResult.generatedText)")

        #expect(medianDecode >= 150.0, "Gemma publishable median decode is below target")
        #expect(medianShort.coherence.passed, "Short prompt coherence failed: \(medianShort.coherence.reasons)")
        #expect(longResult.coherence.passed, "Long prompt coherence failed: \(longResult.coherence.reasons)")
    }

    @Test func diagnosePublishableGenerationPhases() async throws {
        guard ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PUBLISHABLE_DIAGNOSTIC"] == "1",
              let modelPath = ProcessInfo.processInfo.environment[Self.modelPathEnvironmentKey],
              !modelPath.isEmpty
        else {
            return
        }

        let maxTokens = Int(ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_DIAGNOSTIC_MAX_TOKENS"] ?? "")
            ?? Self.publishableMaxTokens
        let url = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GenerationError.modelLoadFailed(reason: "Model not found at \(url.path)")
        }

        try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC phase=load_start model=\(url.path)")
        let model = try await ModelLoader.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: Self.contextWindowSize)
        )
        try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC phase=load_done")
        try await diagnoseGeneration(model: model, prompt: shortPromptText(for: model), label: "short", maxTokens: maxTokens)
        try await diagnoseGeneration(model: model, prompt: longPromptText(for: model), label: "long", maxTokens: maxTokens)
    }

    @Test func coherenceGateRejectsInvalidGeneratedText() {
        #expect(
            !GemmaBenchmarkCoherence.evaluate(
                generatedText: "      \n\t   ",
                generatedTokenIDs: [42, 43, 44]
            ).passed
        )
        #expect(
            !GemmaBenchmarkCoherence.evaluate(
                generatedText: "fast fast fast fast fast fast fast fast",
                generatedTokenIDs: [8, 8, 8, 8, 8, 8, 8, 8]
            ).passed
        )
        #expect(
            !GemmaBenchmarkCoherence.evaluate(
                generatedText: "local\u{0000} inference",
                generatedTokenIDs: [17, 18, 19]
            ).passed
        )
        #expect(
            !GemmaBenchmarkCoherence.evaluate(
                generatedText: "thought\nThinking Process:\n1. **Analyze the Request:** The",
                generatedTokenIDs: [100, 45518, 107, 120474, 12364, 236787]
            ).passed
        )
        #expect(
            !GemmaBenchmarkCoherence.evaluate(
                generatedText: "Reporting median decode speed matters because speed alone don'",
                generatedTokenIDs: [1, 2, 3, 4, 5, 6, 7, 8]
            ).passed
        )
    }

    @Test func publishableArtifactIncludesRequiredMetadata() throws {
        let artifact = GemmaBenchmarkArtifact(
            modelPath: "/tmp/gemma.gguf",
            modelFileSizeBytes: 123,
            modelSha256: "abc123",
            gitCommit: "deadbee",
            gitDirty: false,
            machine: "arm64",
            osVersion: "macOS 26.0",
            swiftVersion: "Swift 6.2",
            command: "swift test -c release --filter Gemma4DownloadedBenchmark",
            env: ["EDGERUNNER_GEMMA4_BENCHMARK_MODEL": "/tmp/gemma.gguf"],
            limitations: ["Gemma short median decode remains below the publishable target."],
            shortPrompt: GemmaBenchmarkPromptResult(
                label: "short",
                prompt: "Write one short sentence.",
                promptTokenCount: 12,
                generatedTokenIDs: [101, 102, 103],
                generatedText: "Local inference is fast.",
                ttftSeconds: 1.0,
                promptProcessingSeconds: 1.0,
                promptTokS: 12.0,
                decodeSeconds: 0.16,
                decodeTokS: 18.0,
                generatedTokenCount: 3,
                coherence: GemmaBenchmarkCoherence(passed: true, reasons: [])
            ),
            longPrompt: GemmaBenchmarkPromptResult(
                label: "long",
                prompt: String(repeating: "Context. ", count: 64),
                promptTokenCount: 128,
                generatedTokenIDs: [201, 202, 203],
                generatedText: "The summary keeps the same topic.",
                ttftSeconds: 2.0,
                promptProcessingSeconds: 2.0,
                promptTokS: 64.0,
                decodeSeconds: 0.18,
                decodeTokS: 17.0,
                generatedTokenCount: 3,
                coherence: GemmaBenchmarkCoherence(passed: true, reasons: [])
            ),
            medianDecodeTokS: 17.0,
            medianTTFTSeconds: 1.5,
            runCount: 5
        )

        let data = try JSONEncoder().encode(artifact)
        let decoded = try JSONDecoder().decode(GemmaBenchmarkArtifact.self, from: data)

        #expect(decoded.modelSha256 == "abc123")
        #expect(decoded.gitCommit == "deadbee")
        #expect(decoded.modelFileSizeBytes == 123)
        #expect(!decoded.swiftVersion.isEmpty)
        #expect(decoded.limitations.contains("Gemma short median decode remains below the publishable target."))
        #expect(decoded.shortPrompt.coherence.passed)
        #expect(decoded.longPrompt.promptTokenCount == 128)
        #expect(decoded.env["EDGERUNNER_GEMMA4_BENCHMARK_MODEL"] == "/tmp/gemma.gguf")
    }

    @Test func benchmarkCommandIsReproducibleSwiftTestCommand() {
        let command = Self.benchmarkCommand(
            modelPath: "/tmp/gemma.gguf",
            artifactPath: "benchmarks/gemma4_publishable_benchmark.json"
        )

        #expect(command.contains("EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/tmp/gemma.gguf"))
        #expect(command.contains("EDGERUNNER_GEMMA4_BENCHMARK_ARTIFACT=benchmarks/gemma4_publishable_benchmark.json"))
        #expect(command.contains("swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFPublishableBenchmark'"))
        #expect(!command.contains("swiftpm-testing-helper"))
    }

    @Test func generationSeparatesPromptProcessingFromDecodeForLogitsModels() async throws {
        let model = FakeBenchmarkLogitsModel()
        let result = try await runGeneration(
            model: model,
            prompt: "prompt",
            label: "fake",
            maxTokens: 3
        )

        #expect(result.promptTokenCount == 1)
        #expect(result.generatedTokenIDs == [1, 2, 3])
        #expect(result.promptProcessingSeconds > 0)
        #expect(result.ttftSeconds >= result.promptProcessingSeconds)
        #expect(result.decodeSeconds > 0)
        #expect(result.decodeTokS > 0)
    }

    private func runGeneration(
        model: any EdgeRunnerLanguageModel
    ) async throws -> (ttft: Double, decodeTokS: Double, generatedTokens: Int) {
        let prompt = shortPromptText(for: model)
        var tokenIDs = model.tokenize(prompt)
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        let start = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?
        var generatedTokens = 0

        for _ in 0..<Self.maxTokens {
            let token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let now = ProcessInfo.processInfo.systemUptime
            if token == model.eosTokenID {
                break
            }
            firstTokenTime = firstTokenTime ?? now
            tokenIDs.append(token)
            generatedTokens += 1
        }

        let finish = ProcessInfo.processInfo.systemUptime
        let ttft = firstTokenTime.map { $0 - start } ?? 0
        let decodeDuration = firstTokenTime.map { finish - $0 } ?? 0
        let decodeTokens = max(0, generatedTokens - 1)
        let tokensPerSecond = decodeDuration > 0 ? Double(decodeTokens) / decodeDuration : 0
        return (ttft, tokensPerSecond, generatedTokens)
    }

    private func runGeneration(
        model: any EdgeRunnerLanguageModel,
        prompt: String,
        label: String,
        maxTokens: Int
    ) async throws -> GemmaBenchmarkPromptResult {
        var tokenIDs = model.tokenize(prompt)
        let promptTokenCount = tokenIDs.count
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        let start = ProcessInfo.processInfo.systemUptime
        var firstTokenTime: TimeInterval?
        var promptProcessingSeconds: TimeInterval = 0
        var generatedTokenIDs: [Int] = []

        let firstToken: Int
        if let logitsModel = model as? any LogitsModel {
            let logits = try await logitsModel.logits(for: tokenIDs)
            let now = ProcessInfo.processInfo.systemUptime
            promptProcessingSeconds = now - start
            firstToken = sampling.toPipeline().sample(logits: logits, previousTokens: tokenIDs)
        } else {
            firstToken = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let now = ProcessInfo.processInfo.systemUptime
            promptProcessingSeconds = now - start
        }

        if firstToken != model.eosTokenID {
            let now = ProcessInfo.processInfo.systemUptime
            firstTokenTime = now
            tokenIDs.append(firstToken)
            generatedTokenIDs.append(firstToken)

            while generatedTokenIDs.count < maxTokens {
                let token = try await model.nextToken(for: tokenIDs, sampling: sampling)
                if token == model.eosTokenID {
                    break
                }
                tokenIDs.append(token)
                generatedTokenIDs.append(token)
            }
        }

        let finish = ProcessInfo.processInfo.systemUptime
        let ttft = firstTokenTime.map { $0 - start } ?? promptProcessingSeconds
        let decodeDuration = firstTokenTime.map { finish - $0 } ?? 0
        let decodeTokens = max(0, generatedTokenIDs.count - 1)
        let tokensPerSecond = decodeDuration > 0 ? Double(decodeTokens) / decodeDuration : 0
        let promptTokS = promptProcessingSeconds > 0 ? Double(promptTokenCount) / promptProcessingSeconds : 0
        let generatedText = model.detokenize(generatedTokenIDs)
        let coherence = GemmaBenchmarkCoherence.evaluate(
            generatedText: generatedText,
            generatedTokenIDs: generatedTokenIDs
        )

        return GemmaBenchmarkPromptResult(
            label: label,
            prompt: prompt,
            promptTokenCount: promptTokenCount,
            generatedTokenIDs: generatedTokenIDs,
            generatedText: generatedText,
            ttftSeconds: ttft,
            promptProcessingSeconds: promptProcessingSeconds,
            promptTokS: promptTokS,
            decodeSeconds: decodeDuration,
            decodeTokS: tokensPerSecond,
            generatedTokenCount: generatedTokenIDs.count,
            coherence: coherence
        )
    }

    private func shortPromptText(for model: any EdgeRunnerLanguageModel) -> String {
        let userPrompt = "Write one short sentence about fast local inference."
        return model.applyChatTemplate(
            messages: [EdgeRunnerCore.ChatMessage(role: "user", content: userPrompt)],
            addGenerationPrompt: true
        ) ?? userPrompt
    }

    private func longPromptText(for model: any EdgeRunnerLanguageModel) -> String {
        let context = String(
            repeating: "Local inference benchmarks must report TTFT, prompt processing, decode throughput, generated tokens, and output quality. ",
            count: 48
        )
        let userPrompt = "\(context)\n\nIn two concise sentences, summarize why median decode speed and coherent output must be reported together."
        return model.applyChatTemplate(
            messages: [EdgeRunnerCore.ChatMessage(role: "user", content: userPrompt)],
            addGenerationPrompt: true
        ) ?? userPrompt
    }

    private func diagnoseGeneration(
        model: any EdgeRunnerLanguageModel,
        prompt: String,
        label: String,
        maxTokens: Int
    ) async throws {
        var tokenIDs = model.tokenize(prompt)
        try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC label=\(label) prompt_tokens=\(tokenIDs.count) max_tokens=\(maxTokens)")
        print("GEMMA4_DIAGNOSTIC label=\(label) prompt_tokens=\(tokenIDs.count) max_tokens=\(maxTokens)")
        let sampling = SamplingConfiguration(
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1
        )

        var generatedTokenIDs: [Int] = []
        let start = ProcessInfo.processInfo.systemUptime
        let firstToken: Int
        if let logitsModel = model as? any LogitsModel {
            try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC label=\(label) phase=prompt_processing_start")
            let logits = try await logitsModel.logits(for: tokenIDs)
            let now = ProcessInfo.processInfo.systemUptime
            firstToken = sampling.toPipeline().sample(logits: logits, previousTokens: tokenIDs)
            try Self.appendDiagnosticLine(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=prompt_processing "
                + "seconds=\(String(format: "%.6f", now - start)) first_token=\(firstToken)"
            )
            print(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=prompt_processing "
                + "seconds=\(String(format: "%.6f", now - start)) first_token=\(firstToken)"
            )
        } else {
            try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC label=\(label) phase=first_next_token_start")
            firstToken = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let now = ProcessInfo.processInfo.systemUptime
            try Self.appendDiagnosticLine(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=first_next_token "
                + "seconds=\(String(format: "%.6f", now - start)) first_token=\(firstToken)"
            )
            print(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=first_next_token "
                + "seconds=\(String(format: "%.6f", now - start)) first_token=\(firstToken)"
            )
        }

        guard firstToken != model.eosTokenID else {
            try Self.appendDiagnosticLine("GEMMA4_DIAGNOSTIC label=\(label) eos_at_first_token=true")
            print("GEMMA4_DIAGNOSTIC label=\(label) eos_at_first_token=true")
            return
        }
        tokenIDs.append(firstToken)
        generatedTokenIDs.append(firstToken)

        while generatedTokenIDs.count < maxTokens {
            let before = ProcessInfo.processInfo.systemUptime
            let token = try await model.nextToken(for: tokenIDs, sampling: sampling)
            let after = ProcessInfo.processInfo.systemUptime
            if token == model.eosTokenID {
                try Self.appendDiagnosticLine(
                    "GEMMA4_DIAGNOSTIC label=\(label) phase=decode_step "
                    + "index=\(generatedTokenIDs.count) seconds=\(String(format: "%.6f", after - before)) eos=true"
                )
                print(
                    "GEMMA4_DIAGNOSTIC label=\(label) phase=decode_step "
                    + "index=\(generatedTokenIDs.count) seconds=\(String(format: "%.6f", after - before)) eos=true"
                )
                break
            }
            tokenIDs.append(token)
            generatedTokenIDs.append(token)
            try Self.appendDiagnosticLine(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=decode_step "
                + "index=\(generatedTokenIDs.count) seconds=\(String(format: "%.6f", after - before)) token=\(token)"
            )
            print(
                "GEMMA4_DIAGNOSTIC label=\(label) phase=decode_step "
                + "index=\(generatedTokenIDs.count) seconds=\(String(format: "%.6f", after - before)) token=\(token)"
            )
        }

        let generatedText = model.detokenize(generatedTokenIDs)
        let coherence = GemmaBenchmarkCoherence.evaluate(
            generatedText: generatedText,
            generatedTokenIDs: generatedTokenIDs
        )
        try Self.appendDiagnosticLine(
            "GEMMA4_DIAGNOSTIC label=\(label) generated_tokens=\(generatedTokenIDs.count) "
            + "coherent=\(coherence.passed) reasons=\(coherence.reasons) text=\(generatedText)"
        )
        print(
            "GEMMA4_DIAGNOSTIC label=\(label) generated_tokens=\(generatedTokenIDs.count) "
            + "coherent=\(coherence.passed) reasons=\(coherence.reasons) text=\(generatedText)"
        )
    }

    private static func appendDiagnosticLine(_ line: String) throws {
        let path = ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_DIAGNOSTIC_LOG"]
            ?? "tasks/gemma4_publishable_diagnostic.log"
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = (line + "\n").data(using: .utf8) ?? Data()
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } else {
            try data.write(to: url)
        }
    }

    private static func artifactURL() -> URL {
        if let path = ProcessInfo.processInfo.environment[artifactPathEnvironmentKey],
           !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("benchmarks")
            .appendingPathComponent("gemma4_publishable_benchmark.json")
    }

    private static func benchmarkCommand(modelPath: String, artifactPath: String) -> String {
        "EDGERUNNER_GEMMA4_BENCHMARK_MODEL=\(modelPath) "
            + "EDGERUNNER_GEMMA4_BENCHMARK_ARTIFACT=\(artifactPath) "
            + "swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFPublishableBenchmark'"
    }

    private static func writeArtifact(_ artifact: GemmaBenchmarkArtifact) throws {
        let url = artifactURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
    }

    private static func sha256(_ url: URL) throws -> String {
        guard let output = commandOutput(["shasum", "-a", "256", url.path]) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to compute sha256 for \(url.path)")
        }
        return output.split(separator: " ").first.map(String.init) ?? output
    }

    private static func fileSize(_ url: URL) throws -> Int64 {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values.fileSize ?? 0)
    }

    private static func relevantEnvironment() -> [String: String] {
        ProcessInfo.processInfo.environment.filter { key, _ in
            key.hasPrefix("EDGERUNNER_GEMMA4") || key == modelPathEnvironmentKey || key == artifactPathEnvironmentKey
        }
    }

    private static func limitations(
        medianDecodeTokS: Double,
        shortResult: GemmaBenchmarkPromptResult,
        longResult: GemmaBenchmarkPromptResult
    ) -> [String] {
        var values: [String] = []
        if medianDecodeTokS < 150 {
            values.append("Gemma short median decode remains below the publishable target.")
        }
        if !shortResult.coherence.passed {
            values.append("Short prompt coherence gate failed: \(shortResult.coherence.reasons.joined(separator: ","))")
        }
        if !longResult.coherence.passed {
            values.append("Long prompt coherence gate failed: \(longResult.coherence.reasons.joined(separator: ","))")
        }
        if longResult.promptProcessingSeconds > 60 {
            values.append("Long-prompt prompt processing is too slow for a publishable local benchmark claim.")
        }
        return values
    }

    private static func commandOutput(_ arguments: [String]) -> String? {
        guard let executable = arguments.first else { return nil }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(decoding: data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return output.isEmpty ? nil : output
        } catch {
            _ = executable
            return nil
        }
    }
}

private final class FakeBenchmarkLogitsModel: LogitsModel, @unchecked Sendable {
    static let modelIdentifier = "fake-benchmark"

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> FakeBenchmarkLogitsModel {
        FakeBenchmarkLogitsModel()
    }

    func tokenize(_ text: String) -> [Int] {
        [0]
    }

    func detokenize(_ ids: [Int]) -> String {
        ids.map(String.init).joined(separator: " ")
    }

    var eosTokenID: Int { 99 }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { 100 }

    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        if tokenIDs.count == 1 {
            return 42
        }
        return min(tokenIDs.count, 3)
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        try await Task.sleep(for: .milliseconds(5))
        let next = min(tokenIDs.count, 3)
        var logits = [Float](repeating: -100, count: vocabularySize)
        logits[next] = 100
        return logits
    }
}
