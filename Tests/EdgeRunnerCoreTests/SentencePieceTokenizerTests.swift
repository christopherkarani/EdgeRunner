import Testing
@testable import EdgeRunnerCore

@Suite("SentencePieceTokenizer")
struct SentencePieceTokenizerTests {

    // MARK: - Test Vocabulary

    /// Synthetic vocabulary mimicking a tiny SentencePiece model.
    /// Tokens are indexed 0..22; scores control merge priority (higher = merge first).
    private func makeTestTokenizer(addSpacePrefix: Bool = true) -> SentencePieceTokenizer {
        let tokens = [
            "<unk>", "<s>", "</s>",           // 0-2: special
            "\u{2581}", "h", "e", "l", "o",   // 3-7: single chars
            "\u{2581}h", "he", "ll", "lo",     // 8-11: first merges
            "hel", "llo",                       // 12-13
            "hello",                            // 14
            "\u{2581}hello",                    // 15
            "\u{2581}w", "or", "ld",           // 16-18
            "orl", "orld", "world",            // 19-21
            "\u{2581}world",                    // 22
        ]
        let scores: [Float] = [
            0, 0, 0,                           // special
            -1, -2, -2, -2, -2,               // single chars (low priority)
            -0.5, -0.4, -0.4, -0.45,          // first merges
            -0.3, -0.35,                       // hel, llo
            -0.1,                              // hello
            -0.05,                             // ▁hello (highest priority)
            -0.5, -0.4, -0.4,                 // ▁w, or, ld
            -0.3, -0.2, -0.1,                 // orl, orld, world
            -0.05,                             // ▁world (highest priority)
        ]
        let vocabulary = TokenizerVocabulary(tokens: tokens)
        let specialTokens = SpecialTokens(
            bosToken: ("<s>", 1),
            eosToken: ("</s>", 2),
            padToken: nil,
            additionalSpecialTokens: ["<unk>": 0]
        )
        return SentencePieceTokenizer(
            vocabulary: vocabulary,
            specialTokens: specialTokens,
            tokenScores: Dictionary(uniqueKeysWithValues: zip(tokens, scores)),
            unknownTokenID: 0,
            addSpacePrefix: addSpacePrefix
        )
    }

    // MARK: - Encode Tests

    @Test func encodeSimpleWord() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello")
        // "hello" → prepend ▁ → "▁hello" → single token 15
        #expect(ids == [15])
    }

    @Test func encodeTwoWords() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello world")
        // "hello world" → prepend ▁ → "▁hello▁world"
        // Characters: ▁ h e l l o ▁ w o r l d
        // Greedy merges produce ▁hello (15) and ▁world (22)
        #expect(ids == [15, 22])
    }

    @Test func roundTripEncodeDecode() {
        let tokenizer = makeTestTokenizer()
        let decoded = tokenizer.decode(tokenizer.encode("hello world"))
        #expect(decoded == "hello world")
    }

    // MARK: - BOS / Special Token Tests

    @Test func encodeWithBOS() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello", addBOS: true)
        #expect(ids.first == 1)   // BOS token ID
        #expect(ids.last == 15)   // ▁hello
        #expect(ids.count == 2)
    }

    @Test func decodeSkipsSpecialTokens() {
        let tokenizer = makeTestTokenizer()
        // Decode [BOS, ▁hello, EOS] with skipSpecialTokens
        let decoded = tokenizer.decode([1, 15, 2], skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    // MARK: - Edge Cases

    @Test func unknownTokenIDProducesReplacementChar() {
        let tokenizer = makeTestTokenizer()
        let decoded = tokenizer.decode([99999])
        #expect(decoded == "\u{FFFD}")
    }

    @Test func emptyStringReturnsEmpty() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("")
        #expect(ids.isEmpty)
    }

    @Test func vocabularySizeIsTokenCount() {
        let tokenizer = makeTestTokenizer()
        #expect(tokenizer.vocabularySize == 23)
    }

    // MARK: - Merge Order

    @Test func mergeOrderFollowsScore() {
        let tokenizer = makeTestTokenizer()
        // "hello" with ▁ prefix → chars: ▁ h e l l o
        // Highest-score merges happen first:
        //   ▁hello (-0.05) is highest, but requires full sequence.
        //   Merges proceed: he(-0.4), ll(-0.4), then hel(-0.3), llo(-0.35),
        //   then hello(-0.1), then ▁hello(-0.05).
        // Final result should be a single token: ▁hello = 15
        let ids = tokenizer.encode("hello")
        #expect(ids == [15])
    }

    // MARK: - Space Prefix Control

    @Test func noSpacePrefixWhenDisabled() {
        let tokenizer = makeTestTokenizer(addSpacePrefix: false)
        let ids = tokenizer.encode("hello")
        // Without ▁ prefix: "hello" → chars: h e l l o
        // Merges: he(-0.4), ll(-0.4), hel(-0.3), llo(-0.35),
        //   hello(-0.1). Final: hello = 14
        #expect(ids == [14])
    }
}
