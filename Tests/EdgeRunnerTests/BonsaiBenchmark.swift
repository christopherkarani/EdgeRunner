import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Bonsai 1.7B Benchmark")
struct BonsaiBenchmark {
    static let modelPath = "/tmp/edgerunner-models/Bonsai-1.7B.gguf"

    @Test("Bonsai 1.7B Q1_0_g128 coherence check")
    func bonsaiCoherenceCheck() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        // Load weight map to inspect Q1 tensor details
        let loader = try GGUFLoader(url: URL(fileURLWithPath: Self.modelPath))
        let weightMap = try await loader.load(from: URL(fileURLWithPath: Self.modelPath))

        // Check a specific Q1 tensor
        let qName = "blk.0.attn_q.weight"
        if let qStorage = weightMap[qName] {
            let ec = qStorage.elementCount
            let bc = ec / 128
            let expectedBytes = bc * 18
            print("=== \(qName) ===")
            print("  shape=\(qStorage.shape), elementCount=\(ec)")
            print("  blocks=\(bc), expectedBytes=\(expectedBytes)")
            print("  actualBufferLength=\(qStorage.byteCount)")

            // Read first block directly from GGUF
            let basePtr = qStorage.buffer.contents() + qStorage.byteOffset
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: min(36, qStorage.byteCount))
            let scale0 = Float(Float16(bitPattern: UInt16(rawPtr[0]) | (UInt16(rawPtr[1]) << 8)))
            let scale1 = Float(Float16(bitPattern: UInt16(rawPtr[18]) | (UInt16(rawPtr[19]) << 8)))
            print("  Block 0 scale=\(String(format: "%.4f", scale0)), bits=0x\(String(rawPtr[2], radix: 16))")
            print("  Block 1 scale=\(String(format: "%.4f", scale1)), bits=0x\(String(rawPtr[20], radix: 16))")

            // Dequantize first block manually
            let bits = rawPtr[2]
            var dequantBlock0 = [String]()
            for k in 0..<8 {
                let bit = (bits >> k) & 1
                let val = bit == 1 ? scale0 : -scale0
                dequantBlock0.append(String(format: "%+.4f", val))
            }
            print("  Block 0 first 8 weights: \(dequantBlock0.joined(separator: " "))")
        }

        // Check mapped name
        let mappedName = LlamaWeightNameMapper.mapGGUFName(qName)
        print("\nMapped name: \(qName) → \(mappedName)")
        if let mappedStorage = weightMap[mappedName] {
            print("  Found as \(mappedName): shape=\(mappedStorage.shape), dtype=\(mappedStorage.dataType)")
        } else {
            print("  NOT FOUND as \(mappedName)")
        }

        // Load model and verify weights
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        print("\n=== Test 1: Hello prompt ===")
        var tokenIDs = model.tokenize("Hello")
        if let bos = model.bosTokenID, tokenIDs.first != bos {
            tokenIDs.insert(bos, at: 0)
        }
        print("Prompt tokens: \(tokenIDs)")
        for i in 0..<4 {
            let result = try await model.greedyToken(for: tokenIDs)
            tokenIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }

        // Test 2: Use benchmark-style seed [1]
        print("\n=== Test 2: Seed [1] (like benchmark) ===")
        var seedIDs = [1]
        for i in 0..<8 {
            let result = try await model.greedyToken(for: seedIDs)
            seedIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))', hasNaN=\(result.hasNonFinite)")
            if result.token == 151645 || result.token == 2 || result.token == 0 { break }
        }

        // Test 3: BOS token only
        print("\n=== Test 3: BOS token [151643] ===")
        guard let bos = model.bosTokenID else { return }
        var bosIDs = [bos]
        for i in 0..<4 {
            let result = try await model.greedyToken(for: bosIDs)
            bosIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }

        // Test 4: Compare with Qwen3 on same seed [1]
        print("\n=== Test 4: Qwen3 seed [1] ===")
        let qwen3Path = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
        guard FileManager.default.fileExists(atPath: qwen3Path) else {
            print("Qwen3 not found")
            return
        }
        let qwen3 = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: qwen3Path),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )
        var qwen3IDs = [1]
        for i in 0..<8 {
            let result = try await qwen3.greedyToken(for: qwen3IDs)
            qwen3IDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(qwen3.detokenize([result.token]))'")
            if result.token == 151645 || result.token == 2 || result.token == 0 { break }
        }
    }
}
