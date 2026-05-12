import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

@Suite("Gemma Tokenizer Parity")
struct GemmaTokenizerParityTest {

    static let modelPath = "/tmp/edgerunner-models/gemma-3-1b-it-Q4_K_M.gguf"
    static let gemma4ModelPath = "/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf"

    /// Reference IDs from:
    /// AutoTokenizer.from_pretrained('google/gemma-3-1b-it', use_fast=False).encode(text, add_special_tokens=False)
    static let parityTestCases: [(text: String, expectedIDs: [Int])] = [
        ("Hello, world!", [9259, 236764, 1902, 236888]),
        ("The capital of France is", [818, 5279, 529, 7001, 563]),
        ("1+1=2", [236770, 236862, 236770, 236784, 236778]),
        ("def foo():\n    return 42", [2063, 46293, 6141, 107, 140, 2060, 236743, 236812, 236778]),
        ("I'm don't can't", [236777, 236789, 236757, 1537, 236789, 236745, 740, 236789, 236745]),
        ("Hello 你好", [9259, 43758, 237389]),
        ("  spaces  and\ttabs", [138, 35220, 138, 624, 255968, 39218]),
        ("emoji: 🎉🚀", [67906, 236787, 204906, 242015]),
    ]

    private func shouldRun() -> Bool {
        guard ProcessInfo.processInfo.environment["EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY"] == "1" else {
            print("SKIP: Set EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1 to run")
            return false
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return false
        }
        return true
    }

    private func loadTokenizer() throws -> any Tokenizer {
        let url = URL(fileURLWithPath: Self.modelPath)
        let loader = try GGUFLoader(url: url)
        let metadata = try loader.modelConfig.tokenizerMetadata()
        return try TokenizerFactory.create(from: metadata)
    }

    private func loadGemma4Tokenizer() throws -> any Tokenizer {
        let url = URL(fileURLWithPath: Self.gemma4ModelPath)
        let loader = try GGUFLoader(url: url)
        let metadata = try loader.modelConfig.tokenizerMetadata()
        return try TokenizerFactory.create(from: metadata)
    }

    @Test func tokenizerLoadsAsSentencePiece() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()
        #expect(tokenizer is SentencePieceTokenizer)
        #expect(tokenizer.vocabularySize == 262144)
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
                print("MISMATCH: \(text.debugDescription)")
                print("  Expected: \(expectedIDs)")
                print("  Actual:   \(actualIDs)")
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

        print("\nGemma Parity: \(passed)/\(passed + failed) match HuggingFace reference")
        #expect(failed == 0, "Parity failed for \(failed) test cases")
    }

    @Test func roundTripEncodeDecodePreservesText() throws {
        guard shouldRun() else { return }
        let tokenizer = try loadTokenizer()

        let testStrings = [
            "Hello, world!",
            "The capital of France is",
            "I'm don't can't",
            "Hello 你好",
        ]

        for text in testStrings {
            let ids = tokenizer.encode(text)
            let decoded = tokenizer.decode(ids)
            #expect(decoded == text, "Round-trip failed for: \(text.debugDescription)")
        }
    }

    @Test func gemma4NormalTokensDecodeToVisibleText() throws {
        guard FileManager.default.fileExists(atPath: Self.gemma4ModelPath) else {
            print("SKIP: Model not found at \(Self.gemma4ModelPath)")
            return
        }

        let tokenizer = try loadGemma4Tokenizer()
        #expect(tokenizer is Gemma4BPETokenizer)
        let cases: [(Int, String)] = [
            (38_786, "Would"),
            (128_654, " dishonest"),
            (230_178, "Mayo")
        ]

        for (tokenID, expectedText) in cases {
            #expect(tokenizer.decode([tokenID], skipSpecialTokens: true) == expectedText)
        }
    }

    @Test func gemma4PromptTokensMatchLlamaCppReference() throws {
        guard FileManager.default.fileExists(atPath: Self.gemma4ModelPath) else {
            print("SKIP: Model not found at \(Self.gemma4ModelPath)")
            return
        }

        let tokenizer = try loadGemma4Tokenizer()
        let prompt = try Gemma4ChatTemplate.renderThrowing(
            messages: [
                Gemma4ChatMessage(
                    role: .user,
                    content: "Write one short sentence about fast local inference."
                )
            ],
            addGenerationPrompt: true
        )
        let expected = [
            2, 105, 2364, 107, 6974,
            886, 2822, 13315, 1003, 4592, 2263, 34711, 236761, 106, 107,
            105, 4368, 107,
        ]

        #expect(tokenizer.encode(prompt, addBOS: tokenizer.shouldAddBOS) == expected)
        #expect(!prompt.contains("<|think|>"))
    }
}
