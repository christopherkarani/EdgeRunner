import Testing
@testable import EdgeRunnerCore

struct PassthroughPreTokenizer: PreTokenizer {
    func split(_ text: String) -> [String] { [text] }
}

@Suite("Tokenizer Protocol")
struct TokenizerProtocolTests {
    @Test func protocolConformance() {
        let vocab = TokenizerVocabulary(tokens: ["h", "e", "l", "lo", "hello", "\u{0120}", "world"])
        let special = SpecialTokens(bosToken: ("<s>", 7), eosToken: ("</s>", 8), padToken: ("<pad>", 9))
        let tokenizer = BPETokenizer(
            vocabulary: vocab,
            specialTokens: special,
            merges: [],
            preTokenizer: PassthroughPreTokenizer()
        )
        #expect(tokenizer.vocabularySize == 7)
        #expect(tokenizer.eosTokenID == 8)
        #expect(tokenizer.bosTokenID == 7)
        #expect(tokenizer.padTokenID == 9)
    }
}

@Suite("SpecialTokens")
struct SpecialTokensTests {
    @Test func specialTokenIDs() {
        let special = SpecialTokens(bosToken: ("<s>", 0), eosToken: ("</s>", 1), padToken: ("<pad>", 2))
        #expect(special.bosTokenID == 0)
        #expect(special.eosTokenID == 1)
        #expect(special.padTokenID == 2)
        #expect(special.bosTokenString == "<s>")
    }

    @Test func optionalTokens() {
        let special = SpecialTokens(bosToken: nil, eosToken: ("</s>", 1), padToken: nil)
        #expect(special.bosTokenID == nil)
        #expect(special.padTokenID == nil)
        #expect(special.eosTokenID == 1)
    }

    @Test func additionalSpecialTokensAreMerged() {
        let additional: [String: Int] = [
            "<|im_start|>": 100,
            "<|im_end|>": 101,
            "<|endoftext|>": 102,
        ]
        let special = SpecialTokens(
            bosToken: ("<s>", 0),
            eosToken: ("</s>", 1),
            padToken: nil,
            additionalSpecialTokens: additional
        )
        #expect(special.specialTokenIDs.count == 5)
        #expect(special.specialTokenMap["<|im_start|>"] == 100)
        #expect(special.specialTokenMap["<|im_end|>"] == 101)
        #expect(special.specialTokenMap["<s>"] == 0)
        #expect(special.specialTokenIDs.contains(100))
        #expect(special.specialTokenIDs.contains(101))
    }

    @Test func emptyAdditionalTokensWorks() {
        let special = SpecialTokens(
            bosToken: ("<s>", 0),
            eosToken: ("</s>", 1),
            padToken: nil,
            additionalSpecialTokens: [:]
        )
        #expect(special.specialTokenIDs.count == 2)
    }
}

@Suite("TokenizerVocabulary")
struct TokenizerVocabularyTests {
    @Test func lookupByToken() {
        let vocab = TokenizerVocabulary(tokens: ["cat", "dog", "fish"])
        #expect(vocab.tokenToID("cat") == 0)
        #expect(vocab.tokenToID("dog") == 1)
        #expect(vocab.tokenToID("fish") == 2)
        #expect(vocab.tokenToID("bird") == nil)
    }

    @Test func lookupByID() {
        let vocab = TokenizerVocabulary(tokens: ["cat", "dog", "fish"])
        #expect(vocab.idToToken(0) == "cat")
        #expect(vocab.idToToken(1) == "dog")
        #expect(vocab.idToToken(2) == "fish")
        #expect(vocab.idToToken(99) == nil)
    }

    @Test func count() {
        let vocab = TokenizerVocabulary(tokens: ["a", "b", "c"])
        #expect(vocab.count == 3)
    }

    @Test func withOffset() {
        let vocab = TokenizerVocabulary(tokens: ["cat", "dog"], offset: 10)
        #expect(vocab.tokenToID("cat") == 10)
        #expect(vocab.tokenToID("dog") == 11)
        #expect(vocab.idToToken(10) == "cat")
        #expect(vocab.idToToken(0) == nil)
    }
}

@Suite("BPETokenizer")
struct BPETokenizerTests {
    private func makeTestTokenizer() -> BPETokenizer {
        let tokens = [
            "h", "e", "l", "o", "\u{0120}", "w", "r", "d",
            "he", "ll", "lo",
            "hel", "llo",
            "hello",
            "wo", "wor", "worl", "world",
        ]
        let vocab = TokenizerVocabulary(tokens: tokens)
        let merges: [(String, String)] = [
            ("h", "e"), ("l", "l"), ("ll", "o"),
            ("he", "llo"), ("w", "o"),
            ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ]
        let special = SpecialTokens(bosToken: ("<s>", 18), eosToken: ("</s>", 19), padToken: nil)
        return BPETokenizer(
            vocabulary: vocab,
            specialTokens: special,
            merges: merges,
            preTokenizer: PassthroughPreTokenizer()
        )
    }

    @Test func encodeSimpleWord() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello")
        #expect(ids == [13])
    }

    @Test func encodeTwoWords() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello world")
        #expect(ids == [13, 4, 17])
    }

    @Test func decodeRoundTrip() {
        let tokenizer = makeTestTokenizer()
        let text = "hello world"
        let ids = tokenizer.encode(text)
        let decoded = tokenizer.decode(ids)
        #expect(decoded == text)
    }

    @Test func encodeWithBOS() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello", addBOS: true)
        #expect(ids.first == 18)
        #expect(ids.last == 13)
    }

    @Test func decodeSkipsSpecialTokens() {
        let tokenizer = makeTestTokenizer()
        let ids = [18, 13, 19]
        let decoded = tokenizer.decode(ids, skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    @Test func encodeUnknownFallsBackToBytes() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("her")
        #expect(ids == [8, 6])
    }

    @Test func emptyStringReturnsEmpty() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("")
        #expect(ids.isEmpty)
    }

    @Test func decodeSingleToken() {
        let tokenizer = makeTestTokenizer()
        let text = tokenizer.decode([13])
        #expect(text == "hello")
    }

    @Test func vocabularySizeIsVocabCount() {
        let tokenizer = makeTestTokenizer()
        #expect(tokenizer.vocabularySize == 18)
    }
}

@Suite("BPETokenizer Pipeline")
struct BPETokenizerPipelineTests {
    private func makeByteEncodedTokenizer() -> BPETokenizer {
        let tokens = [
            "H", "e", "l", "o", "\u{0120}", "w", "r", "d",
            "he", "ll", "lo",
            "hel", "llo",
            "hello",
            "\u{0120}w", "\u{0120}wo", "\u{0120}wor",
            "\u{0120}worl", "\u{0120}world",
            "<|im_start|>", "<|im_end|>",
        ]
        let vocab = TokenizerVocabulary(tokens: tokens)
        let merges: [(String, String)] = [
            ("h", "e"), ("l", "l"), ("ll", "o"),
            ("he", "llo"),
            ("\u{0120}", "w"), ("\u{0120}w", "o"), ("\u{0120}wo", "r"),
            ("\u{0120}wor", "l"), ("\u{0120}worl", "d"),
        ]
        let special = SpecialTokens(
            bosToken: nil,
            eosToken: ("</s>", 21),
            padToken: nil,
            additionalSpecialTokens: ["<|im_start|>": 19, "<|im_end|>": 20]
        )
        let preTokenizer = PreTokenizerPattern.resolve("gpt-2")
        return BPETokenizer(
            vocabulary: vocab,
            specialTokens: special,
            merges: merges,
            preTokenizer: preTokenizer
        )
    }

    @Test func encodeSimpleWordThroughPipeline() {
        let tokenizer = makeByteEncodedTokenizer()
        let ids = tokenizer.encode("hello")
        #expect(ids == [13])
    }

    @Test func encodeWithLeadingSpace() {
        let tokenizer = makeByteEncodedTokenizer()
        let ids = tokenizer.encode(" world")
        #expect(ids == [18])
    }

    @Test func encodeTwoWords() {
        let tokenizer = makeByteEncodedTokenizer()
        let ids = tokenizer.encode("hello world")
        #expect(ids == [13, 18])
    }

    @Test func roundTripEncodeDecode() {
        let tokenizer = makeByteEncodedTokenizer()
        let decoded = tokenizer.decode(tokenizer.encode("hello world"))
        #expect(decoded == "hello world")
    }

    @Test func specialTokensPreservedDuringEncode() {
        let tokenizer = makeByteEncodedTokenizer()
        let ids = tokenizer.encode("<|im_start|>hello world<|im_end|>")
        #expect(ids.first == 19)
        #expect(ids.last == 20)
        #expect(ids.contains(13))
    }

    @Test func decodeSkipsSpecialTokens() {
        let tokenizer = makeByteEncodedTokenizer()
        let decoded = tokenizer.decode([19, 13, 20], skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    @Test func unknownTokenIDProducesReplacementChar() {
        let tokenizer = makeByteEncodedTokenizer()
        let decoded = tokenizer.decode([99999])
        #expect(decoded == "\u{FFFD}")
    }

    @Test func vocabularySizeIsTokenCount() {
        let tokenizer = makeByteEncodedTokenizer()
        #expect(tokenizer.vocabularySize == 21)
    }
}
