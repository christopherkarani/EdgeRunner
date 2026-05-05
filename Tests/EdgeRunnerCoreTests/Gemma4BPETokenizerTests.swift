import Testing
@testable import EdgeRunnerCore

@Suite("Gemma4BPETokenizer")
struct Gemma4BPETokenizerTests {
    private func makeTokenizer() -> Gemma4BPETokenizer {
        let tokens = [
            "<unk>", "<eos>", "<bos>",
            "\u{2581}", "H", "e", "l", "o", "w", "r", "d",
            "\u{2581}Hello", "\u{2581}world", "\n\n", "<0xF0>", "<0x9F>", "<0x8E>", "<0x89>",
            "<turn|>", "\u{2581}H", "\u{2581}He", "\u{2581}Hel", "\u{2581}Hell",
            "wo", "wor", "worl", "\u{2581}w", "\u{2581}wo", "\u{2581}wor", "\u{2581}worl",
        ]
        let merges: [(String, String)] = [
            ("\u{2581}", "H"),
            ("\u{2581}H", "e"),
            ("\u{2581}He", "l"),
            ("\u{2581}Hel", "l"),
            ("\u{2581}Hell", "o"),
            ("\u{2581}", "w"),
            ("\u{2581}w", "o"),
            ("\u{2581}wo", "r"),
            ("\u{2581}wor", "l"),
            ("\u{2581}worl", "d"),
            ("w", "o"),
            ("wo", "r"),
            ("wor", "l"),
            ("worl", "d"),
        ]
        let special = SpecialTokens(
            bosToken: ("<bos>", 2),
            eosToken: ("<eos>", 1),
            padToken: nil,
            additionalSpecialTokens: ["<turn|>": 18]
        )
        return Gemma4BPETokenizer(
            vocabulary: TokenizerVocabulary(tokens: tokens),
            specialTokens: special,
            merges: merges,
            shouldAddBOS: true
        )
    }

    @Test func encodesSpaceEscapedBPEWithoutGPT2ByteEncoding() {
        let tokenizer = makeTokenizer()
        #expect(tokenizer.encode(" Hello world", addBOS: false) == [11, 12])
    }

    @Test func encodesBOSWhenRequested() {
        let tokenizer = makeTokenizer()
        #expect(tokenizer.encode(" Hello", addBOS: tokenizer.shouldAddBOS) == [2, 11])
    }

    @Test func newlineRunUsesDirectVocabularyLookup() {
        let tokenizer = makeTokenizer()
        #expect(tokenizer.encode("\n\n", addBOS: false) == [13])
    }

    @Test func byteFallbackUsesHexByteTokens() {
        let tokenizer = makeTokenizer()
        #expect(tokenizer.encode("🎉", addBOS: false) == [14, 15, 16, 17])
        #expect(tokenizer.decode([14, 15, 16, 17], skipSpecialTokens: true) == "🎉")
    }

    @Test func specialTokensArePreserved() {
        let tokenizer = makeTokenizer()
        #expect(tokenizer.encode("<turn|>", addBOS: false) == [18])
        #expect(tokenizer.decode([18], skipSpecialTokens: false) == "<turn|>")
        #expect(tokenizer.decode([18], skipSpecialTokens: true) == "")
    }
}
