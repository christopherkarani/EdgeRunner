import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// End-to-end coherence test: feeds a real BPE-tokenized prompt to the model
/// and checks if the output is coherent text (not garbage).
///
/// Uses pre-tokenized prompts (from Qwen3's BPE tokenizer) since EdgeRunner
/// doesn't yet have a built-in BPE tokenizer. The token-to-text mapping is
/// loaded from the GGUF file's embedded vocabulary.
@Suite("End-to-End Coherence")
struct CoherenceTest {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    // Pre-tokenized prompts (from `transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')`)
    // "The capital of France is" → [785, 6722, 315, 9625, 374]
    static let completionPrompt = [785, 6722, 315, 9625, 374]
    static let completionPromptText = "The capital of France is"

    // Chat format: <|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    static let chatPrompt = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
    static let chatPromptText = "<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n"

    // MARK: - Load Vocabulary from GGUF

    /// Load the BPE vocabulary from the GGUF file's metadata.
    /// Returns an array where index = token ID, value = token string.
    static func loadVocabulary(from path: String) throws -> [String] {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url, options: .mappedIfSafe)

        // Parse GGUF header
        var offset = 0
        func readU32() -> UInt32 {
            let val = data[offset..<offset+4].withUnsafeBytes { $0.load(as: UInt32.self) }
            offset += 4
            return val
        }
        func readU64() -> UInt64 {
            let val = data[offset..<offset+8].withUnsafeBytes { $0.load(as: UInt64.self) }
            offset += 8
            return val
        }
        func readString() -> String {
            let len = Int(readU64())
            let str = String(data: data[offset..<offset+len], encoding: .utf8) ?? ""
            offset += len
            return str
        }

        // Magic + version
        let _ = readU32() // magic "GGUF"
        let _ = readU32() // version
        let _ = readU64() // n_tensors
        let nMetadata = Int(readU64())

        // Scan metadata for tokenizer.ggml.tokens
        var vocabulary: [String] = []

        for _ in 0..<nMetadata {
            let key = readString()
            let vtype = readU32()

            if key == "tokenizer.ggml.tokens" {
                // Array of strings
                let arrType = readU32() // should be 8 (string)
                let arrLen = Int(readU64())
                guard arrType == 8 else {
                    // Skip non-string array
                    break
                }
                vocabulary.reserveCapacity(arrLen)
                for _ in 0..<arrLen {
                    vocabulary.append(readString())
                }
                break // Found what we need
            } else {
                // Skip this metadata value
                switch vtype {
                case 0: offset += 1  // uint8
                case 1: offset += 1  // int8
                case 2: offset += 2  // uint16
                case 3: offset += 2  // int16
                case 4: offset += 4  // uint32
                case 5: offset += 4  // int32
                case 6: offset += 4  // float32
                case 7: offset += 1  // bool
                case 8: let _ = readString()  // string
                case 9:  // array
                    let aType = readU32()
                    let aLen = Int(readU64())
                    switch aType {
                    case 4, 5, 6: offset += aLen * 4
                    case 8: for _ in 0..<aLen { let _ = readString() }
                    case 10: offset += aLen * 8
                    default: break
                    }
                case 10: offset += 8  // uint64
                default: break
                }
            }
        }

        return vocabulary
    }

    /// Decode token IDs to text using the GGUF vocabulary.
    /// Handles the GPT-2 byte encoding where `Ġ` = space, etc.
    static func detokenize(_ tokenIDs: [Int], vocabulary: [String]) -> String {
        var result = ""
        for id in tokenIDs {
            guard id >= 0 && id < vocabulary.count else { continue }
            var piece = vocabulary[id]
            // GPT-2/Qwen BPE uses byte-level encoding:
            // Ġ (U+0120) = space, Ċ (U+010A) = newline, etc.
            piece = piece.replacingOccurrences(of: "\u{0120}", with: " ")
            piece = piece.replacingOccurrences(of: "\u{010A}", with: "\n")
            result += piece
        }
        return result
    }

    // MARK: - Tests

    @Test func completionCoherence() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return
        }

        // Load model
        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        // Load vocabulary from GGUF
        let vocab = try Self.loadVocabulary(from: Self.modelPath)
        print("Loaded vocabulary: \(vocab.count) tokens")
        #expect(vocab.count == 151936)

        // Feed pre-tokenized prompt and generate 32 tokens
        var tokenIDs = Self.completionPrompt
        let generateCount = 32

        for _ in 0..<generateCount {
            let logits = try await model.logits(for: tokenIDs)

            // Logits must be finite
            #expect(!logits.contains(where: { !$0.isFinite }), "NaN/Inf in logits")

            // Greedy argmax
            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (i, v) in logits.enumerated() {
                if v > maxVal { maxVal = v; maxIdx = i }
            }
            tokenIDs.append(maxIdx)

            // Stop on EOS
            if maxIdx == 151645 { break }
        }

        // Decode to text
        let promptText = Self.detokenize(Self.completionPrompt, vocabulary: vocab)
        let generatedIDs = Array(tokenIDs.dropFirst(Self.completionPrompt.count))
        let generatedText = Self.detokenize(generatedIDs, vocabulary: vocab)
        let fullText = Self.detokenize(tokenIDs, vocabulary: vocab)

        print("")
        print("=" * 60)
        print("  COHERENCE TEST: Text Completion")
        print("=" * 60)
        print("  Prompt:    \"\(promptText)\"")
        print("  Generated: \"\(generatedText)\"")
        print("  Full:      \"\(fullText)\"")
        print("  Tokens:    \(tokenIDs.count) (\(Self.completionPrompt.count) prompt + \(generatedIDs.count) generated)")
        print("  Token IDs: \(generatedIDs.prefix(20))...")
        print("=" * 60)
        print("")

        // Basic coherence checks
        #expect(generatedIDs.count > 0, "Model should generate at least 1 token")
        #expect(!generatedText.isEmpty, "Generated text should not be empty")

        // The completion of "The capital of France is" should contain "Paris"
        let containsParis = fullText.lowercased().contains("paris")
        print("Contains 'Paris': \(containsParis)")
        #expect(containsParis, "Expected the model to mention Paris when asked about the capital of France")
    }

    @Test func chatCoherence() async throws {
        let url = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return
        }

        let model = try await LlamaLanguageModel.load(
            from: url,
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let vocab = try Self.loadVocabulary(from: Self.modelPath)

        // Feed chat-formatted prompt and generate 64 tokens
        var tokenIDs = Self.chatPrompt
        let generateCount = 64

        for _ in 0..<generateCount {
            let logits = try await model.logits(for: tokenIDs)
            #expect(!logits.contains(where: { !$0.isFinite }))

            var maxVal: Float = -.infinity
            var maxIdx = 0
            for (i, v) in logits.enumerated() {
                if v > maxVal { maxVal = v; maxIdx = i }
            }
            tokenIDs.append(maxIdx)

            if maxIdx == 151645 { break }  // <|im_end|>
        }

        let generatedIDs = Array(tokenIDs.dropFirst(Self.chatPrompt.count))
        let generatedText = Self.detokenize(generatedIDs, vocabulary: vocab)

        print("")
        print("=" * 60)
        print("  COHERENCE TEST: Chat (What is 2+2?)")
        print("=" * 60)
        print("  Generated: \"\(generatedText)\"")
        print("  Tokens:    \(generatedIDs.count) generated")
        print("=" * 60)
        print("")

        #expect(generatedIDs.count > 0, "Model should generate at least 1 token")

        // The answer to "What is 2+2?" should contain "4"
        let containsFour = generatedText.contains("4")
        print("Contains '4': \(containsFour)")
        #expect(containsFour, "Expected the model to answer '4' when asked 'What is 2+2?'")
    }
}

private func *(lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
}
