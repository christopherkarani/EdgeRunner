import Testing
@testable import EdgeRunnerCore

@Suite("Tokenizer Protocol")
struct TokenizerProtocolTests {
    @Test func protocolConformance() {
        let vocab = TokenizerVocabulary(tokens: ["h", "e", "l", "lo", "hello", " ", "world"])
        let special = SpecialTokens(bosToken: ("<s>", 7), eosToken: ("</s>", 8), padToken: ("<pad>", 9))
        let tokenizer = BPETokenizer(vocabulary: vocab, specialTokens: special, merges: [])
        #expect(tokenizer.vocabularySize == 10)
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
            "h", "e", "l", "o", " ", "w", "r", "d",
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
        return BPETokenizer(vocabulary: vocab, specialTokens: special, merges: merges)
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

    @Test func vocabularySizeIncludesSpecial() {
        let tokenizer = makeTestTokenizer()
        #expect(tokenizer.vocabularySize == 20)
    }
}
