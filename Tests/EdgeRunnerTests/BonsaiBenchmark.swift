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
        let model = try await LlamaLanguageModel.load(
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
        if let bos = model.bosTokenID, helloIDs.first != bos {
            helloIDs.insert(bos, at: 0)
        }
        print("  Prompt IDs: \(helloIDs)")
        for i in 0..<4 {
            let result = try await model.greedyToken(for: helloIDs)
            helloIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }

        print("\n=== Test: BOS only ===")
        guard let bos = model.bosTokenID else { return }
        var bosIDs = [bos]
        for i in 0..<4 {
            let result = try await model.greedyToken(for: bosIDs)
            bosIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }
    }

    @Test("Bonsai 1.7B Q1_0_g128 end-to-end benchmark")
    func bonsaiEndToEndBenchmark() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let prompt = "Explain quantum computing in simple terms:"
        var tokenIDs = model.tokenize(prompt)
        if let bos = model.bosTokenID, tokenIDs.first != bos {
            tokenIDs.insert(bos, at: 0)
        }
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
            let firstResult = try await model.greedyToken(for: tokenIDs)
            let prefillDuration = startTime.duration(to: clock.now)
            let ttftMs = (Double(prefillDuration.components.seconds) + Double(prefillDuration.components.attoseconds) * 1e-18) * 1000.0
            tokenIDs.append(firstResult.token)
            var generatedCount = 1

            for _ in 1..<128 {
                let result = try await model.greedyToken(for: tokenIDs)
                tokenIDs.append(result.token)
                generatedCount += 1
                if result.token == 151645 || result.token == 2 || result.token == 0 { break }
            }

            let totalDuration = startTime.duration(to: clock.now)
            let totalSeconds = Double(totalDuration.components.seconds) + Double(totalDuration.components.attoseconds) * 1e-18
            let decodeSeconds = totalSeconds - (ttftMs / 1000.0)
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
}
