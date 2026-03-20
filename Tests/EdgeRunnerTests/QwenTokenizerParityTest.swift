import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

/// Validates BPE tokenizer output against HuggingFace reference token IDs.
/// Requires Qwen3-0.6B-Q8_0.gguf at /tmp/edgerunner-models/.
/// Enable with: EDGERUNNER_RUN_TOKENIZER_PARITY=1
@Suite("Qwen Tokenizer Parity")
struct QwenTokenizerParityTest {

    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    /// Reference token IDs from:
    /// transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B').encode(text, add_special_tokens=False)
    static let parityTestCases: [(text: String, expectedIDs: [Int])] = [
        ("Hello, world!", [9707, 11, 1879, 0]),
        ("The capital of France is", [785, 6722, 315, 9625, 374]),
        ("1+1=2", [16, 10, 16, 28, 17]),
        ("def foo():\n    return 42", [750, 15229, 3932, 262, 470, 220, 19, 17]),
        ("  spaces  and\ttabs", [220, 12621, 220, 323, 3244, 3435]),
        ("I'm don't can't", [40, 2776, 1513, 944, 646, 944]),
        ("Hello 你好 مرحبا", [9707, 220, 108386, 23364, 126860, 124671]),
        ("Price: $123,456.78", [6972, 25, 400, 16, 17, 18, 11, 19, 20, 21, 13, 22, 23]),
        ("emoji: 🎉🚀", [37523, 25, 11162, 236, 231, 145836]),
    ]

    private func loadTokenizer() throws -> BPETokenizer {
        let url = URL(fileURLWithPath: Self.modelPath)
        let loader = try GGUFLoader(url: url)
        let metadata = try loader.modelConfig.tokenizerMetadata()
        return try TokenizerFactory.create(from: metadata)
    }

    private func shouldRun() -> Bool {
        guard ProcessInfo.processInfo.environment["EDGERUNNER_RUN_TOKENIZER_PARITY"] == "1" else {
            print("SKIP: Set EDGERUNNER_RUN_TOKENIZER_PARITY=1 to run tokenizer parity tests")
            return false
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return false
        }
        return true
    }

    @Test func tokenizerLoadsFromGGUF() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()
        #expect(tokenizer.vocabularySize == 151936)
        #expect(tokenizer.eosTokenID == 151645)
    }

    @Test func encodeMatchesHuggingFaceReference() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()

        var passed = 0
        var failed = 0

        for (text, expectedIDs) in Self.parityTestCases {
            let actualIDs = tokenizer.encode(text)
            if actualIDs == expectedIDs {
                passed += 1
            } else {
                failed += 1
                print("MISMATCH: \(repr(text))")
                print("  Expected: \(expectedIDs)")
                print("  Actual:   \(actualIDs)")
                // Find first divergence
                for i in 0..<max(actualIDs.count, expectedIDs.count) {
                    let a = i < actualIDs.count ? actualIDs[i] : nil
                    let e = i < expectedIDs.count ? expectedIDs[i] : nil
                    if a != e {
                        print("  First divergence at index \(i): got \(a.map(String.init) ?? "EOF") expected \(e.map(String.init) ?? "EOF")")
                        break
                    }
                }
            }
        }

        print("\nTokenizer Parity: \(passed)/\(passed + failed) test cases match HuggingFace reference")
        #expect(failed == 0, "Tokenizer parity failed for \(failed) test cases")
    }

    @Test func roundTripEncodeDecodePreservesText() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()

        let testStrings = [
            "Hello, world!",
            "The capital of France is",
            "def foo():\n    return 42",
            "I'm don't can't",
            "Hello 你好 مرحبا",
        ]

        for text in testStrings {
            let ids = tokenizer.encode(text)
            let decoded = tokenizer.decode(ids)
            #expect(decoded == text, "Round-trip failed for: \(repr(text)) → decoded as: \(repr(decoded))")
        }
    }

    @Test func specialTokensEncodeCorrectly() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()

        // ChatML special tokens should encode as single token IDs
        let ids = tokenizer.encode("<|im_start|>user\nHello<|im_end|>")
        #expect(ids == [151644, 872, 198, 9707, 151645],
               "ChatML special tokens should match HuggingFace reference")
    }

    @Test func chatTemplateProducesCorrectFormat() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()

        let result = try tokenizer.applyChatTemplate(
            messages: [EdgeRunnerCore.ChatMessage(role: "user", content: "Hello")],
            addGenerationPrompt: true
        )

        guard let formatted = result else {
            #expect(Bool(false), "Chat template should be available for Qwen3")
            return
        }

        #expect(formatted.contains("<|im_start|>user"))
        #expect(formatted.contains("Hello"))
        #expect(formatted.contains("<|im_end|>"))
        #expect(formatted.contains("<|im_start|>assistant"))
    }

    private func repr(_ s: String) -> String {
        s.debugDescription
    }
}
