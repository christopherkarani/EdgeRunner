import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

@Suite("Qwen 3 0.6B Real Model Benchmark")
struct QwenBenchmark {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    @Test func loadModel() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("BENCHMARK: qwen_skip Model not found at \(Self.modelPath)")
            return
        }

        let clock = ContinuousClock()
        let start = clock.now

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let loadMs = seconds * 1000

        print("BENCHMARK: qwen_load_time \(String(format: "%.0f", loadMs)) ms")
        print("BENCHMARK: qwen_vocab_size \(model.vocabularySize)")

        #expect(model.vocabularySize > 100_000)
    }

    @Test func debugWeightShapes() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let loader = try GGUFLoader(url: url)
        let weightMap = try await loader.load(from: url)
        let config = try LlamaConfig(fromGGUFMetadata: loader.modelConfig.metadata)

        print("Config: embed=\(config.embeddingDim) layers=\(config.layerCount) heads=\(config.headCount) kv=\(config.kvHeadCount) headDim=\(config.headDim) ffn=\(config.intermediateDim) vocab=\(config.vocabSize)")

        // Check key weight shapes
        for name in ["token_embd.weight", "output_norm.weight",
                      "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                      "blk.0.attn_output.weight", "blk.0.attn_norm.weight",
                      "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                      "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"]
        {
            if let tensor = weightMap[name] {
                print("  \(name): shape=\(tensor.shape) type=\(tensor.dataType) elements=\(tensor.elementCount) bytes=\(tensor.byteCount)")
            } else {
                print("  \(name): NOT FOUND")
            }
        }

        #expect(config.embeddingDim > 0)
    }

    @Test func singleForwardPass() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Single token forward pass
        let tokenIDs = [1] // BOS token

        let clock = ContinuousClock()
        let start = clock.now
        let logits = try await model.logits(for: tokenIDs)
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18

        print("BENCHMARK: qwen_single_forward \(String(format: "%.0f", seconds * 1000)) ms")
        print("BENCHMARK: qwen_logits_size \(logits.count)")

        #expect(logits.count == model.vocabularySize)

        // Verify logits are finite
        let allFinite = logits.allSatisfy { $0.isFinite }
        #expect(allFinite)
    }

    @Test func prefillBenchmark() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Prefill with 8 tokens (keep short for a 0.6B model benchmark)
        let promptTokens = Array(0..<8)

        // Warmup
        _ = try await model.logits(for: [1])

        let clock = ContinuousClock()
        let start = clock.now
        let logits = try await model.logits(for: promptTokens)
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(promptTokens.count) / seconds

        print("BENCHMARK: qwen_prefill_8tok \(String(format: "%.1f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: qwen_prefill_latency \(String(format: "%.0f", seconds * 1000)) ms")

        #expect(logits.count == model.vocabularySize)
    }

    @Test func decodeBenchmark() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else { return }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Autoregressive decode: generate 4 tokens one at a time
        let generateCount = 4
        var tokenIDs = [1] // Start with BOS

        // Warmup
        _ = try await model.logits(for: tokenIDs)

        let clock = ContinuousClock()
        let start = clock.now

        for _ in 0..<generateCount {
            let logits = try await model.logits(for: tokenIDs)
            // Greedy decode
            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (i, v) in logits.enumerated() {
                if v > maxVal { maxVal = v; maxIdx = i }
            }
            tokenIDs.append(maxIdx)
        }

        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let tokensPerSec = Double(generateCount) / seconds
        let msPerToken = seconds / Double(generateCount) * 1000

        print("BENCHMARK: qwen_decode_throughput \(String(format: "%.2f", tokensPerSec)) tokens/sec")
        print("BENCHMARK: qwen_decode_latency \(String(format: "%.0f", msPerToken)) ms/token")
        print("BENCHMARK: qwen_generated_tokens \(tokenIDs)")

        #expect(tokenIDs.count == generateCount + 1)
    }
}
