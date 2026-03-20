import Testing
@testable import EdgeRunnerIO

@Suite("GGUF Tokenizer Metadata")
struct GGUFTokenizerMetadataTests {

    @Test func parseQwenStyleTokenizerMetadataFromRawGGUFMetadata() throws {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("gpt2"),
            "tokenizer.ggml.pre": .string("qwen2"),
            "tokenizer.ggml.tokens": .array([
                .string("h"),
                .string("ello"),
                .string(" world"),
            ]),
            "tokenizer.ggml.merges": .array([
                .string("h ello"),
                .string("ello world"),
            ]),
            "tokenizer.ggml.token_type": .array([
                .int32(1),
                .int32(1),
                .int32(6),
            ]),
            "tokenizer.ggml.bos_token_id": .uint32(151643),
            "tokenizer.ggml.eos_token_id": .uint32(151645),
            "tokenizer.ggml.padding_token_id": .uint32(151665),
            "tokenizer.ggml.add_bos_token": .bool(false),
            "tokenizer.chat_template": .string("<|im_start|>user\n{{ prompt }}<|im_end|>"),
        ]

        let tokenizer = try GGUFTokenizerMetadata(ggufMetadata: metadata)

        #expect(tokenizer.model == .gpt2)
        #expect(tokenizer.preTokenizer == "qwen2")
        #expect(tokenizer.tokens == ["h", "ello", " world"])
        #expect(tokenizer.vocabularySize == 3)
        #expect(tokenizer.merges == [
            GGUFTokenizerMerge(left: "h", right: "ello", rawValue: "h ello"),
            GGUFTokenizerMerge(left: "ello", right: "world", rawValue: "ello world"),
        ])
        #expect(tokenizer.tokenTypes == [.normal, .normal, .byte])
        #expect(tokenizer.bosTokenID == 151643)
        #expect(tokenizer.eosTokenID == 151645)
        #expect(tokenizer.paddingTokenID == 151665)
        #expect(tokenizer.shouldAddBOS == false)
        #expect(tokenizer.chatTemplate == "<|im_start|>user\n{{ prompt }}<|im_end|>")
    }

    @Test func parseTokenizerMetadataFromModelConfigBridge() throws {
        let config = ModelConfig(
            architectureName: "qwen3",
            metadata: [
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "qwen2",
                "tokenizer.ggml.tokens": ["A", "B", "C"],
                "tokenizer.ggml.merges": ["A B", "AB C"],
                "tokenizer.ggml.token_type": [1, 1, 3],
                "tokenizer.ggml.bos_token_id": 10,
                "tokenizer.ggml.eos_token_id": 11,
                "tokenizer.ggml.add_bos_token": true,
            ]
        )

        let tokenizer = try config.tokenizerMetadata()

        #expect(tokenizer.model == .gpt2)
        #expect(tokenizer.preTokenizer == "qwen2")
        #expect(tokenizer.tokens == ["A", "B", "C"])
        #expect(tokenizer.merges == [
            GGUFTokenizerMerge(left: "A", right: "B", rawValue: "A B"),
            GGUFTokenizerMerge(left: "AB", right: "C", rawValue: "AB C"),
        ])
        #expect(tokenizer.tokenTypes == [.normal, .normal, .control])
        #expect(tokenizer.bosTokenID == 10)
        #expect(tokenizer.eosTokenID == 11)
        #expect(tokenizer.shouldAddBOS == true)
    }

    @Test func missingTokenizerModelThrows() {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.tokens": .array([.string("a")]),
        ]

        #expect(throws: GGUFTokenizerMetadataError.missingKey("tokenizer.ggml.model")) {
            _ = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        }
    }

    @Test func missingTokensThrows() {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("gpt2"),
        ]

        #expect(throws: GGUFTokenizerMetadataError.missingKey("tokenizer.ggml.tokens")) {
            _ = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        }
    }

    @Test func tokenTypeCountMismatchThrows() {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("gpt2"),
            "tokenizer.ggml.tokens": .array([.string("a"), .string("b")]),
            "tokenizer.ggml.token_type": .array([.int32(1)]),
        ]

        #expect(throws: GGUFTokenizerMetadataError.invalidValue(
            key: "tokenizer.ggml.token_type",
            description: "Expected 2 token types, found 1"
        )) {
            _ = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        }
    }

    @Test func invalidMergeEntryThrows() {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("gpt2"),
            "tokenizer.ggml.tokens": .array([.string("a")]),
            "tokenizer.ggml.merges": .array([.string("ab")]),
        ]

        #expect(throws: GGUFTokenizerMetadataError.invalidValue(
            key: "tokenizer.ggml.merges",
            description: "Invalid merge entry 'ab'"
        )) {
            _ = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        }
    }

    @Test func unknownTokenizerModelIsPreserved() throws {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("custom-model"),
            "tokenizer.ggml.tokens": .array([.string("a")]),
        ]

        let tokenizer = try GGUFTokenizerMetadata(ggufMetadata: metadata)

        #expect(tokenizer.model == .unknown("custom-model"))
    }

    @Test func llamaModelStringMapsToDotLlama() throws {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array([.string("a"), .string("b")]),
            "tokenizer.ggml.scores": .array([.float32(-1.0), .float32(-2.0)]),
        ]
        let tokenizer = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        #expect(tokenizer.model == .llama)
        #expect(tokenizer.scores == [-1.0, -2.0])
    }

    @Test func scoresCountMismatchThrows() {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array([.string("a"), .string("b")]),
            "tokenizer.ggml.scores": .array([.float32(-1.0)]),
        ]
        #expect(throws: GGUFTokenizerMetadataError.self) {
            _ = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        }
    }

    @Test func addSpacePrefixIsParsed() throws {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array([.string("a")]),
            "tokenizer.ggml.add_space_prefix": .bool(false),
        ]
        let tokenizer = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        #expect(tokenizer.addSpacePrefix == false)
    }

    @Test func unknownTokenIDIsParsed() throws {
        let metadata: [String: GGUFMetadataValue] = [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array([.string("<unk>"), .string("a")]),
            "tokenizer.ggml.unknown_token_id": .uint32(0),
        ]
        let tokenizer = try GGUFTokenizerMetadata(ggufMetadata: metadata)
        #expect(tokenizer.unknownTokenID == 0)
    }
}
