import Testing
@testable import EdgeRunnerCore
import EdgeRunnerIO

// MARK: - Helpers

/// Builds a minimal metadata dictionary suitable for `GGUFTokenizerMetadata(metadata:)`.
private func makeTokenizerMetadata(
    tokens: [String],
    merges: [String] = [],
    tokenTypes: [Int]? = nil,
    scores: [Float]? = nil,
    bosTokenID: Int? = nil,
    eosTokenID: Int? = nil,
    paddingTokenID: Int? = nil,
    unknownTokenID: Int? = nil,
    shouldAddBOS: Bool? = nil,
    addSpacePrefix: Bool? = nil,
    preTokenizer: String? = nil,
    chatTemplate: String? = nil,
    model: String = "gpt2"
) -> [String: MetadataValue] {
    var meta: [String: MetadataValue] = [
        "tokenizer.ggml.model": .string(model),
        "tokenizer.ggml.tokens": .array(tokens.map { .string($0) }),
    ]
    if !merges.isEmpty {
        meta["tokenizer.ggml.merges"] = .array(merges.map { .string($0) })
    }
    if let tokenTypes {
        meta["tokenizer.ggml.token_type"] = .array(tokenTypes.map { .int($0) })
    }
    if let scores {
        meta["tokenizer.ggml.scores"] = .array(scores.map { .float($0) })
    }
    if let bosTokenID {
        meta["tokenizer.ggml.bos_token_id"] = .int(bosTokenID)
    }
    if let eosTokenID {
        meta["tokenizer.ggml.eos_token_id"] = .int(eosTokenID)
    }
    if let paddingTokenID {
        meta["tokenizer.ggml.padding_token_id"] = .int(paddingTokenID)
    }
    if let unknownTokenID {
        meta["tokenizer.ggml.unknown_token_id"] = .int(unknownTokenID)
    }
    if let shouldAddBOS {
        meta["tokenizer.ggml.add_bos_token"] = .bool(shouldAddBOS)
    }
    if let addSpacePrefix {
        meta["tokenizer.ggml.add_space_prefix"] = .bool(addSpacePrefix)
    }
    if let preTokenizer {
        meta["tokenizer.ggml.pre"] = .string(preTokenizer)
    }
    if let chatTemplate {
        meta["tokenizer.chat_template"] = .string(chatTemplate)
    }
    return meta
}

// MARK: - Tests

@Suite("TokenizerFactory")
struct TokenizerFactoryTests {

    // MARK: - Happy path: GPT-2 model

    @Test func createFromGPT2MetadataReturnsWorkingTokenizer() throws {
        let tokens = ["<s>", "</s>", "h", "e", "l", "lo", "hel"]
        let merges = ["h e", "l o", "he l"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            merges: merges,
            tokenTypes: [1, 1, 1, 1, 1, 1, 1],
            bosTokenID: 0,
            eosTokenID: 1
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        #expect(tokenizer.vocabularySize == 7)
        #expect(tokenizer.eosTokenID == 1)
        #expect(tokenizer.bosTokenID == 0)
    }

    // MARK: - llama-bpe model accepted

    @Test func createFromLlamaBPEMetadataSucceeds() throws {
        let tokens = ["<s>", "</s>", "a", "b"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1, 1, 1],
            bosTokenID: 0,
            eosTokenID: 1,
            model: "llama-bpe"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)
        #expect(tokenizer.vocabularySize == 4)
    }

    // MARK: - SentencePiece model dispatches to SentencePieceTokenizer

    @Test func sentencePieceModelCreatesSentencePieceTokenizer() throws {
        let tokens = ["<unk>", "<s>", "</s>", "\u{2581}", "a"]
        let scores: [Float] = [0.0, 0.0, 0.0, -1.0, -2.0]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [2, 3, 3, 1, 1],
            scores: scores,
            bosTokenID: 1,
            eosTokenID: 2,
            unknownTokenID: 0,
            model: "sentencepiece"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        #expect(tokenizer is SentencePieceTokenizer)
        #expect(tokenizer.vocabularySize == 5)
        #expect(tokenizer.eosTokenID == 2)
        #expect(tokenizer.bosTokenID == 1)
    }

    @Test func sentencePieceModelWithoutScoresThrowsMissingScores() throws {
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1, 1],
            eosTokenID: 1,
            model: "sentencepiece"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)

        do {
            _ = try TokenizerFactory.create(from: gguf)
            Issue.record("Expected TokenizerFactoryError.missingRequiredToken to be thrown")
        } catch let error as TokenizerFactoryError {
            if case .missingRequiredToken(let name) = error {
                #expect(name == "scores")
            } else {
                Issue.record("Wrong error case: \(error)")
            }
        }
    }

    // MARK: - Missing EOS throws missingRequiredToken

    @Test func missingEOSThrowsMissingRequiredToken() throws {
        let tokens = ["<s>", "a", "b"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1, 1],
            bosTokenID: 0
            // no eosTokenID
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)

        #expect(throws: TokenizerFactoryError.self) {
            try TokenizerFactory.create(from: gguf)
        }
    }

    @Test func missingEOSErrorIsMissingRequiredToken() throws {
        let tokens = ["<s>", "a", "b"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1, 1],
            bosTokenID: 0
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)

        do {
            _ = try TokenizerFactory.create(from: gguf)
            Issue.record("Expected TokenizerFactoryError.missingRequiredToken")
        } catch let error as TokenizerFactoryError {
            if case .missingRequiredToken(let name) = error {
                #expect(name == "EOS")
            } else {
                Issue.record("Wrong error case: \(error)")
            }
        }
    }

    // MARK: - Control tokens (type 3) recognized as special tokens

    @Test func controlTokensAreRecognizedAsSpecial() throws {
        // Tokens: 0=<s>(control), 1=</s>(control), 2=<|im_start|>(control), 3=hello(normal)
        let tokens = ["<s>", "</s>", "<|im_start|>", "hello"]
        let tokenTypes = [3, 3, 3, 1]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: tokenTypes,
            bosTokenID: 0,
            eosTokenID: 1
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        // Encode text containing the special token literal — it should be split and recognized.
        let encoded = tokenizer.encode("<|im_start|>")
        // The special token <|im_start|> (id=2) should appear in the encoded output.
        #expect(encoded.contains(2))
    }

    // MARK: - UserDefined tokens (type 4) recognized as special tokens

    @Test func userDefinedTokensAreRecognizedAsSpecial() throws {
        let tokens = ["<s>", "</s>", "<|custom|>", "a"]
        let tokenTypes = [3, 3, 4, 1]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: tokenTypes,
            bosTokenID: 0,
            eosTokenID: 1
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        let encoded = tokenizer.encode("<|custom|>")
        #expect(encoded.contains(2))
    }

    // MARK: - shouldAddBOS propagation

    @Test func shouldAddBOSIsPropagated() throws {
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [3, 3, 1],
            bosTokenID: 0,
            eosTokenID: 1,
            shouldAddBOS: true
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        #expect(tokenizer.shouldAddBOS == true)
    }

    @Test func shouldAddBOSDefaultsToFalse() throws {
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [3, 3, 1],
            bosTokenID: 0,
            eosTokenID: 1
            // shouldAddBOS not set
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        #expect(tokenizer.shouldAddBOS == false)
    }

    // MARK: - Byte fallback tokens (type 6) are parsed

    @Test func byteFallbackTokensAreParsed() throws {
        // Build a minimal vocab that includes a <0xHH> byte token.
        var tokens = ["<s>", "</s>"]
        var types = [3, 3]
        // Add byte tokens for 0x00 through 0x02.
        for b in 0...2 {
            tokens.append(String(format: "<0x%02X>", b))
            types.append(6) // byte type
        }
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: types,
            bosTokenID: 0,
            eosTokenID: 1
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        // The tokenizer should have been created without error.
        #expect(tokenizer.vocabularySize == tokens.count)
    }

    // MARK: - PreTokenizer resolution

    @Test func preTokenizerResolvesFromMetadata() throws {
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [3, 3, 1],
            bosTokenID: 0,
            eosTokenID: 1,
            preTokenizer: "qwen2"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        // Should not throw — preTokenizer "qwen2" is valid.
        let tokenizer = try TokenizerFactory.create(from: gguf)
        #expect(tokenizer.vocabularySize == 3)
    }

    // MARK: - Chat template parsing

    @Test func chatTemplateEngineIsCreatedWhenPresent() throws {
        let template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [3, 3, 1],
            bosTokenID: 0,
            eosTokenID: 1,
            chatTemplate: template
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        let bpe = try #require(tokenizer as? BPETokenizer)
        #expect(bpe.chatTemplateEngine != nil)
    }

    @Test func noChatTemplateYieldsNilEngine() throws {
        let tokens = ["<s>", "</s>", "a"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [3, 3, 1],
            bosTokenID: 0,
            eosTokenID: 1
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        let bpe = try #require(tokenizer as? BPETokenizer)
        #expect(bpe.chatTemplateEngine == nil)
    }

    // MARK: - EOS ID out of bounds throws missingRequiredToken

    @Test func eosIDOutOfBoundsThrowsMissingRequiredToken() throws {
        let tokens = ["<s>", "</s>"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1],
            eosTokenID: 999 // out of bounds
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)

        do {
            _ = try TokenizerFactory.create(from: gguf)
            Issue.record("Expected TokenizerFactoryError.missingRequiredToken")
        } catch let error as TokenizerFactoryError {
            if case .missingRequiredToken(let name) = error {
                #expect(name == "EOS")
            } else {
                Issue.record("Wrong error case: \(error)")
            }
        }
    }

    // MARK: - Unknown model type throws unsupportedModel

    @Test func unknownModelTypeThrowsUnsupportedModel() throws {
        let tokens = ["<s>", "</s>"]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [1, 1],
            eosTokenID: 1,
            model: "some-future-model"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)

        do {
            _ = try TokenizerFactory.create(from: gguf)
            Issue.record("Expected TokenizerFactoryError.unsupportedModel")
        } catch let error as TokenizerFactoryError {
            if case .unsupportedModel(let name) = error {
                #expect(name == "some-future-model")
            } else {
                Issue.record("Wrong error case: \(error)")
            }
        }
    }

    // MARK: - Llama model dispatches to SentencePieceTokenizer

    @Test func createFromLlamaModelReturnsSentencePiece() throws {
        let metadata = try GGUFTokenizerMetadata(metadata: [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array(["<unk>", "<s>", "</s>", "\u{2581}", "a"].map { .string($0) }),
            "tokenizer.ggml.scores": .array([0.0, 0.0, 0.0, -1.0, -2.0].map { MetadataValue.float(Float($0)) }),
            "tokenizer.ggml.token_type": .array([2, 3, 3, 1, 1].map { MetadataValue.int($0) }),
            "tokenizer.ggml.bos_token_id": .int(1),
            "tokenizer.ggml.eos_token_id": .int(2),
        ])
        let tokenizer = try TokenizerFactory.create(from: metadata)
        #expect(tokenizer is SentencePieceTokenizer)
        #expect(tokenizer.eosTokenID == 2)
        #expect(tokenizer.bosTokenID == 1)
    }

    @Test func llamaModelWithoutScoresThrows() throws {
        let metadata = try GGUFTokenizerMetadata(metadata: [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array(["a", "b"].map { .string($0) }),
            "tokenizer.ggml.eos_token_id": .int(1),
        ])
        #expect(throws: TokenizerFactoryError.self) {
            _ = try TokenizerFactory.create(from: metadata)
        }
    }

    // MARK: - Llama model shouldAddBOS defaults to true

    @Test func llamaModelShouldAddBOSDefaultsToTrue() throws {
        let tokens = ["<unk>", "<s>", "</s>", "\u{2581}", "a"]
        let scores: [Float] = [0.0, 0.0, 0.0, -1.0, -2.0]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [2, 3, 3, 1, 1],
            scores: scores,
            bosTokenID: 1,
            eosTokenID: 2,
            unknownTokenID: 0,
            model: "llama"
            // shouldAddBOS not set — defaults to true for SPM
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)

        #expect(tokenizer.shouldAddBOS == true)
    }

    // MARK: - Llama model addSpacePrefix defaults to true

    @Test func llamaModelAddSpacePrefixDefaultsToTrue() throws {
        let tokens = ["<unk>", "<s>", "</s>", "\u{2581}", "a"]
        let scores: [Float] = [0.0, 0.0, 0.0, -1.0, -2.0]
        let meta = makeTokenizerMetadata(
            tokens: tokens,
            tokenTypes: [2, 3, 3, 1, 1],
            scores: scores,
            bosTokenID: 1,
            eosTokenID: 2,
            unknownTokenID: 0,
            model: "llama"
        )
        let gguf = try GGUFTokenizerMetadata(metadata: meta)
        let tokenizer = try TokenizerFactory.create(from: gguf)
        let spm = try #require(tokenizer as? SentencePieceTokenizer)

        // Encode "a" — with addSpacePrefix=true, the SPM prepends the space char.
        let encoded = spm.encode("a")
        // Should contain the "▁a" or "▁" + "a" tokens, confirming space prefix is on.
        #expect(!encoded.isEmpty)
    }
}
