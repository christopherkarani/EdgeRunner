import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Bonsai 1.7B Benchmark")
struct BonsaiBenchmark {
    static let modelPath = (NSHomeDirectory() as NSString).appendingPathComponent("edgerunner-models/Bonsai-1.7B.gguf")

    @Test("Bonsai 1.7B Q1_0_g128 coherence check")
    func bonsaiCoherenceCheck() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        // Load weight map to check tensor details
        let loader = try GGUFLoader(url: URL(fileURLWithPath: Self.modelPath))
        let weightMap = try await loader.load(from: URL(fileURLWithPath: Self.modelPath))

        // Check embedding table
        let embdName = weightMap.tensorNames.first { $0.contains("token_embd") } ?? ""
        if let embdStorage = weightMap[embdName] {
            print("=== Embedding table ===")
            print("  name=\(embdName), shape=\(embdStorage.shape), dtype=\(embdStorage.dataType)")
            print("  elementCount=\(embdStorage.elementCount)")
            print("  byteCount=\(embdStorage.byteCount)")

            // Check BOS token (151643) embedding
            let dim = 2048
            let blocksPerRow = dim / 128
            let bytesPerRow = blocksPerRow * 18
            let bosOffset = 151643 * bytesPerRow
            let basePtr = embdStorage.buffer.contents() + embdStorage.byteOffset
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: embdStorage.byteCount)

            if bosOffset + 18 <= embdStorage.byteCount {
                let scaleBits = UInt16(rawPtr[bosOffset]) | (UInt16(rawPtr[bosOffset + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                let bits = rawPtr[bosOffset + 2]
                print("  BOS token embedding: scale=\(String(format: "%.4f", scale)), bits=0x\(String(bits, radix: 16))")
            }

            // Also check token 1 embedding
            let token1Offset = 1 * bytesPerRow
            if token1Offset + 18 <= embdStorage.byteCount {
                let scaleBits = UInt16(rawPtr[token1Offset]) | (UInt16(rawPtr[token1Offset + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                let bits = rawPtr[token1Offset + 2]
                print("  Token 1 embedding: scale=\(String(format: "%.4f", scale)), bits=0x\(String(bits, radix: 16))")
            }
        }

        // Load model and test
        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        print("\n=== Test: Seed [1] ===")
        var seedIDs = [1]
        for i in 0..<8 {
            let result = try await model.greedyToken(for: seedIDs)
            seedIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))', hasNaN=\(result.hasNonFinite)")
            if result.token == 151645 || result.token == 2 || result.token == 0 { break }
        }

        print("\n=== Test: 'Hello' ===")
        var helloIDs = model.tokenize("Hello")
        print("  Prompt IDs: \(helloIDs)")
        for i in 0..<4 {
            let result = try await model.greedyToken(for: helloIDs)
            helloIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }

        print("\n=== Test: realistic prompt ===")
        var promptIDs = model.tokenize("The capital of France is")
        print("  Prompt IDs: \(promptIDs)")
        for i in 0..<6 {
            let result = try await model.greedyToken(for: promptIDs)
            promptIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
            if result.token == model.eosTokenID { break }
        }
    }

    @Test("Bonsai 1.7B Q1_0_g128 end-to-end benchmark")
    func bonsaiEndToEndBenchmark() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let prompt = "Explain quantum computing in simple terms:"
        let tokenIDs = model.tokenize(prompt)
        let promptTokenIDs = tokenIDs

        // Warmup
        var warmupTokens = Array(tokenIDs)
        for _ in 0..<4 {
            let result = try await model.greedyToken(for: warmupTokens)
            warmupTokens.append(result.token)
        }
        model.resetGenerationState(keepDecodeWarmup: true)

        var timings: [Double] = []
        for run in 0..<5 {
            var tokenIDs = Array(promptTokenIDs)
            let clock = ContinuousClock()
            let startTime = clock.now
            // Prefill via async path (initializes KV cache + decode warmup)
            let firstResult = try await model.greedyToken(for: tokenIDs)
            let prefillDuration = startTime.duration(to: clock.now)
            let ttftMs = (Double(prefillDuration.components.seconds) + Double(prefillDuration.components.attoseconds) * 1e-18) * 1000.0
            tokenIDs.append(firstResult.token)
            var generatedCount = 1

            // Decode via sync path (eliminates async scheduling overhead)
            let decodeStart = clock.now
            for _ in 1..<128 {
                let result = try model.greedyTokenSync(for: tokenIDs)
                tokenIDs.append(result.token)
                generatedCount += 1
                if result.token == 151645 || result.token == 2 || result.token == 0 { break }
            }

            let decodeDuration = decodeStart.duration(to: clock.now)
            let decodeSeconds = Double(decodeDuration.components.seconds) + Double(decodeDuration.components.attoseconds) * 1e-18
            let decodeTokens = generatedCount - 1
            let decodeTokPerSec = decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0
            timings.append(decodeTokPerSec)
            print("Run \(run): decode=\(String(format: "%.1f", decodeTokPerSec)) tok/s  tokens=\(generatedCount)  ttft=\(String(format: "%.1f", ttftMs))ms")
            model.resetGenerationState(keepDecodeWarmup: true)
        }

        let sortedTimings = timings.sorted()
        let median = sortedTimings[sortedTimings.count / 2]
        print("\nMedian decode: \(String(format: "%.1f", median)) tok/s")
        #expect(Bool(true), "Bonsai benchmark completed: \(String(format: "%.1f", median)) tok/s")
    }

    @Test("Bonsai 1.7B LM head profile")
    func bonsaiLMHeadProfile() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        var warmupTokens = model.tokenize("Explain quantum computing in simple terms:")
        let warmup = try await model.greedyToken(for: warmupTokens)
        warmupTokens.append(warmup.token)

        let lmHeadMs = try await model.measureLMHeadLatency(samples: 5)
        print("BONSAI_PROFILE: lm_head_ms \(String(format: "%.3f", lmHeadMs))")
        #expect(Bool(true), "Bonsai LM head profile completed")
    }

    @Test("Bonsai 1.7B Q1 projection profile")
    func bonsaiQ1ProjectionProfile() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let warmupPrompt = model.tokenize("Explain quantum computing in simple terms:")
        _ = try await model.greedyToken(for: warmupPrompt)

        let results = try await model.measureQ1ProjectionLatencies(samples: 5)
        for (name, milliseconds) in results {
            print("BONSAI_PROFILE: q1_\(name)_ms \(String(format: "%.3f", milliseconds))")
        }
        #expect(!results.isEmpty, "Bonsai Q1 projection profile completed")
    }
}
