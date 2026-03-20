# SentencePiece Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SentencePiece tokenizer support so EdgeRunner can tokenize for Gemma, Phi-3, and other models that use GGUF `tokenizer.ggml.model = "llama"`.

**Architecture:** New `SentencePieceTokenizer` conforming to existing `Tokenizer` protocol, using greedy bigram merging by float score (matching llama.cpp). Factory dispatches `.llama`/`.sentencePiece` models to this new tokenizer. Space handling uses `▁` (U+2581) instead of GPT-2 byte encoding.

**Tech Stack:** Swift 6.2, Swift Testing framework

**Spec:** `docs/superpowers/specs/2026-03-20-sentencepiece-tokenizer-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `Sources/EdgeRunnerIO/WeightMap.swift` | Modify | Add `floatArrayValue` to `MetadataValue` |
| `Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift` | Modify | Add `.llama` model case, `scores`, `unknownTokenID`, `addSpacePrefix`, `floatArrayValue` accessors |
| `Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift` | Modify | Add `shouldAddBOS` and `applyChatTemplate` to protocol |
| `Sources/EdgeRunnerCore/Tokenizer/SentencePieceTokenizer.swift` | Create | New tokenizer implementation |
| `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift` | Modify | Return `any Tokenizer`, dispatch `.llama`/`.sentencePiece` |
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | Change `BPETokenizer?` to `(any Tokenizer)?` |
| `Tests/EdgeRunnerIOTests/GGUFTokenizerMetadataTests.swift` | Modify | Add tests for `.llama` model and scores parsing |
| `Tests/EdgeRunnerCoreTests/SentencePieceTokenizerTests.swift` | Create | Unit tests |
| `Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift` | Modify | Add SentencePiece factory tests |
| `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift` | Create | Parity validation against HuggingFace |

---

### Task 1: Add `floatArrayValue` to MetadataValue and GGUFMetadataValue

**Files:**
- Modify: `Sources/EdgeRunnerIO/WeightMap.swift`
- Modify: `Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift`

- [ ] **Step 1: Add `floatArrayValue` to `MetadataValue`**

In `Sources/EdgeRunnerIO/WeightMap.swift`, add after the existing `arrayValue` property:

```swift
public var floatArrayValue: [Float]? {
    guard let values = arrayValue else { return nil }
    let floats = values.compactMap(\.floatValue)
    return floats.count == values.count ? floats : nil
}
```

- [ ] **Step 2: Add `floatArrayValue` to `GGUFMetadataValue`**

In `Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift`, add to both the `GGUFMetadataValue` and `MetadataValue` extensions (where `stringArrayValue` and `intArrayValue` already exist):

```swift
var floatArrayValue: [Float]? {
    guard let values = arrayValue else { return nil }
    let floats = values.compactMap(\.floatValue)
    return floats.count == values.count ? floats : nil
}
```

Note: `GGUFMetadataValue` uses `.float32Value` while `MetadataValue` uses `.floatValue`. Check which accessor name is correct for each extension.

- [ ] **Step 3: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/EdgeRunnerIO/
git commit -m "feat: add floatArrayValue accessor for GGUF scores metadata"
```

---

### Task 2: Extend GGUFTokenizerMetadata with `.llama`, scores, and new properties

**Files:**
- Modify: `Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift`
- Modify: `Tests/EdgeRunnerIOTests/GGUFTokenizerMetadataTests.swift`

- [ ] **Step 1: Write tests for new metadata features**

Add to `Tests/EdgeRunnerIOTests/GGUFTokenizerMetadataTests.swift`:

```swift
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
        "tokenizer.ggml.scores": .array([.float32(-1.0)]),  // 1 score for 2 tokens
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
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter GGUFTokenizerMetadataTests 2>&1 | tail -10`

- [ ] **Step 3: Add `.llama` case to `GGUFTokenizerModel`**

In `GGUFTokenizerModel.init(rawValue:)`, add before the `default` case:

```swift
case "llama":
    self = .llama
```

Add the enum case:
```swift
case llama
```

Add to `rawValue` computed property:
```swift
case .llama:
    return "llama"
```

- [ ] **Step 4: Add new properties to `GGUFTokenizerMetadata`**

Add stored properties:
```swift
public let scores: [Float]?
public let unknownTokenID: Int?
public let addSpacePrefix: Bool?
```

In the private `init(stringValueForKey:intValueForKey:boolValueForKey:stringArrayForKey:intArrayForKey:)`, add a `floatArrayForKey` parameter:

```swift
floatArrayForKey: (String) -> [Float]?
```

Add parsing logic after the existing `tokenTypes` validation:

```swift
let scores: [Float]?
if let rawScores = floatArrayForKey("tokenizer.ggml.scores") {
    guard rawScores.count == tokens.count else {
        throw GGUFTokenizerMetadataError.invalidValue(
            key: "tokenizer.ggml.scores",
            description: "Expected \(tokens.count) scores, found \(rawScores.count)"
        )
    }
    scores = rawScores
} else {
    scores = nil
}
```

Add at the end of the init:
```swift
self.scores = scores
self.unknownTokenID = intValueForKey("tokenizer.ggml.unknown_token_id")
self.addSpacePrefix = boolValueForKey("tokenizer.ggml.add_space_prefix")
```

Update both public init overloads to pass the new `floatArrayForKey` parameter:
- For `GGUFMetadataValue`: `{ metadata[$0]?.floatArrayValue }`
- For `MetadataValue`: `{ metadata[$0]?.floatArrayValue }`

- [ ] **Step 5: Run tests — verify they PASS**

Run: `swift test --filter GGUFTokenizerMetadataTests 2>&1 | tail -15`
Expected: All tests pass (existing + 4 new)

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunnerIO/ Tests/EdgeRunnerIOTests/
git commit -m "feat: add .llama model case, scores, unknownTokenID, addSpacePrefix to GGUFTokenizerMetadata"
```

---

### Task 3: Add `shouldAddBOS` and `applyChatTemplate` to Tokenizer protocol

**Files:**
- Modify: `Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift`
- Modify: `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift` (ensure conformance)

- [ ] **Step 1: Extend Tokenizer protocol**

Add to the protocol requirements:
```swift
var shouldAddBOS: Bool { get }
func applyChatTemplate(
    messages: [ChatMessage],
    addGenerationPrompt: Bool
) throws -> String?
```

Add default implementations:
```swift
extension Tokenizer {
    public var shouldAddBOS: Bool { false }
    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) throws -> String? { nil }
}
```

- [ ] **Step 2: Verify BPETokenizer still conforms**

`BPETokenizer` already has `shouldAddBOS` and `applyChatTemplate` — verify the signatures match. The protocol's `applyChatTemplate` has 2 params (messages, addGenerationPrompt); BPETokenizer's has 3 (+ tools). The protocol version is the subset — BPETokenizer's method satisfies the requirement because Swift allows extra defaulted params.

Actually, check if this is true. If not, add a separate conformance method on BPETokenizer that delegates to the 3-param version.

Run: `swift build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/
git commit -m "feat: add shouldAddBOS and applyChatTemplate to Tokenizer protocol"
```

---

### Task 4: Create SentencePieceTokenizer

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/SentencePieceTokenizer.swift`
- Create: `Tests/EdgeRunnerCoreTests/SentencePieceTokenizerTests.swift`

- [ ] **Step 1: Write SentencePieceTokenizer tests**

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("SentencePieceTokenizer")
struct SentencePieceTokenizerTests {

    /// Build a small SPM tokenizer with known scores.
    /// Higher score = merge first.
    private func makeTestTokenizer() -> SentencePieceTokenizer {
        // Vocabulary: individual chars + merged tokens
        // SentencePiece uses ▁ (U+2581) for word boundaries
        let tokens = [
            "<unk>",    // 0: unknown
            "<s>",      // 1: BOS
            "</s>",     // 2: EOS
            "▁",        // 3
            "h",        // 4
            "e",        // 5
            "l",        // 6
            "o",        // 7
            "▁h",       // 8
            "he",       // 9
            "ll",       // 10
            "lo",       // 11
            "hel",      // 12
            "llo",      // 13
            "hello",    // 14
            "▁hello",   // 15
            "▁w",       // 16
            "or",       // 17
            "ld",       // 18
            "orl",      // 19
            "orld",     // 20
            "world",    // 21
            "▁world",   // 22
        ]
        // Scores: higher = preferred merge
        let scores: [Float] = [
            0, 0, 0,           // special tokens
            -1, -2, -2, -2, -2,  // single chars
            -0.5,              // ▁h
            -0.4,              // he
            -0.4,              // ll
            -0.45,             // lo
            -0.3,              // hel
            -0.35,             // llo
            -0.1,              // hello
            -0.05,             // ▁hello (very high priority)
            -0.5,              // ▁w
            -0.4,              // or
            -0.4,              // ld
            -0.3,              // orl
            -0.2,              // orld
            -0.1,              // world
            -0.05,             // ▁world (very high priority)
        ]

        let vocab = TokenizerVocabulary(tokens: tokens)
        let special = SpecialTokens(
            bosToken: ("<s>", 1),
            eosToken: ("</s>", 2),
            padToken: nil
        )

        return SentencePieceTokenizer(
            vocabulary: vocab,
            specialTokens: special,
            tokenScores: Dictionary(
                uniqueKeysWithValues: tokens.enumerated().map { ($0.element, scores[$0.offset]) }
            ),
            addSpacePrefix: true
        )
    }

    @Test func encodeSimpleWord() {
        let tok = makeTestTokenizer()
        let ids = tok.encode("hello")
        // "hello" -> prepend ▁ -> "▁hello" -> split chars -> greedy merge -> [▁hello] -> [15]
        #expect(ids == [15])
    }

    @Test func encodeTwoWords() {
        let tok = makeTestTokenizer()
        let ids = tok.encode("hello world")
        // "hello world" -> "▁hello▁world" -> merge -> [▁hello, ▁world] -> [15, 22]
        #expect(ids == [15, 22])
    }

    @Test func roundTripEncodeDecode() {
        let tok = makeTestTokenizer()
        let original = "hello world"
        let decoded = tok.decode(tok.encode(original))
        #expect(decoded == original)
    }

    @Test func encodeWithBOS() {
        let tok = makeTestTokenizer()
        let ids = tok.encode("hello", addBOS: true)
        #expect(ids.first == 1) // BOS
        #expect(ids.last == 15) // ▁hello
    }

    @Test func decodeSkipsSpecialTokens() {
        let tok = makeTestTokenizer()
        let decoded = tok.decode([1, 15, 2], skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    @Test func unknownTokenIDProducesReplacementChar() {
        let tok = makeTestTokenizer()
        let decoded = tok.decode([99999])
        #expect(decoded == "\u{FFFD}")
    }

    @Test func emptyStringReturnsEmpty() {
        let tok = makeTestTokenizer()
        let ids = tok.encode("")
        #expect(ids.isEmpty)
    }

    @Test func vocabularySizeIsTokenCount() {
        let tok = makeTestTokenizer()
        #expect(tok.vocabularySize == 23)
    }

    @Test func mergeOrderFollowsScore() {
        // Build a tokenizer where we can verify merge order matters
        let tokens = ["▁", "a", "b", "ab", "▁a", "▁ab"]
        let scores: [Float] = [-1.0, -2.0, -2.0, -0.5, -0.8, -0.1]
        let vocab = TokenizerVocabulary(tokens: tokens)
        let special = SpecialTokens(bosToken: nil, eosToken: nil, padToken: nil)
        let tok = SentencePieceTokenizer(
            vocabulary: vocab,
            specialTokens: special,
            tokenScores: Dictionary(
                uniqueKeysWithValues: tokens.enumerated().map { ($0.element, scores[$0.offset]) }
            ),
            addSpacePrefix: true
        )
        let ids = tok.encode("ab")
        // "ab" -> "▁ab" -> should merge to ▁ab (score -0.1) as it's highest
        #expect(ids == [5]) // ▁ab
    }

    @Test func noSpacePrefixWhenDisabled() {
        let tokens = ["a", "b", "ab"]
        let scores: [Float] = [-2.0, -2.0, -0.5]
        let vocab = TokenizerVocabulary(tokens: tokens)
        let special = SpecialTokens(bosToken: nil, eosToken: nil, padToken: nil)
        let tok = SentencePieceTokenizer(
            vocabulary: vocab,
            specialTokens: special,
            tokenScores: Dictionary(
                uniqueKeysWithValues: tokens.enumerated().map { ($0.element, scores[$0.offset]) }
            ),
            addSpacePrefix: false  // disabled
        )
        let ids = tok.encode("ab")
        #expect(ids == [2]) // ab (no ▁ prefix)
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter SentencePieceTokenizerTests 2>&1 | tail -10`
Expected: compilation error — `SentencePieceTokenizer` not defined

- [ ] **Step 3: Implement SentencePieceTokenizer**

Create `Sources/EdgeRunnerCore/Tokenizer/SentencePieceTokenizer.swift`:

```swift
import Foundation

/// SentencePiece (Unigram) tokenizer using greedy bigram merging by score.
///
/// Matches llama.cpp's `LLAMA_VOCAB_TYPE_SPM` algorithm. Uses float scores
/// from GGUF metadata instead of BPE merge ranks. Space boundaries are
/// represented by `▁` (U+2581) instead of GPT-2 byte encoding.
public struct SentencePieceTokenizer: Tokenizer, Sendable {
    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let tokenScores: [String: Float]
    private let addSpacePrefix: Bool
    private let byteFallbackTable: [UInt8: Int]

    /// Whether to prepend BOS token during encoding.
    public let shouldAddBOS: Bool

    /// Optional chat template engine.
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int { vocabulary.count }
    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    /// The SentencePiece space character (U+2581).
    private static let sentencePieceSpace: Character = "\u{2581}"
    private static let sentencePieceSpaceString: String = "\u{2581}"

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        tokenScores: [String: Float],
        addSpacePrefix: Bool = true,
        shouldAddBOS: Bool = false,
        byteFallbackTable: [UInt8: Int] = [:],
        chatTemplateEngine: ChatTemplateEngine? = nil
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.tokenScores = tokenScores
        self.addSpacePrefix = addSpacePrefix
        self.shouldAddBOS = shouldAddBOS
        self.byteFallbackTable = byteFallbackTable
        self.chatTemplateEngine = chatTemplateEngine
    }

    // MARK: - Encode

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        var ids = [Int]()
        if addBOS || shouldAddBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        // Step 1: Split around special tokens
        let segments = splitAroundSpecialTokens(text)

        for segment in segments {
            if let specialID = specialTokens.specialTokenMap[segment] {
                ids.append(specialID)
            } else {
                // Step 2: Apply space prefix and replace spaces with ▁
                var processed = segment.replacingOccurrences(of: " ", with: Self.sentencePieceSpaceString)
                if addSpacePrefix && segment == segments.first(where: { specialTokens.specialTokenMap[$0] == nil }) {
                    // Only prepend ▁ to the very first non-special segment
                    if !processed.hasPrefix(Self.sentencePieceSpaceString) {
                        processed = Self.sentencePieceSpaceString + processed
                    }
                }

                // Step 3: Split into individual characters
                var tokens = processed.map { String($0) }

                // Step 4: Greedy bigram merge by score
                tokens = applyMerges(tokens)

                // Step 5: Vocabulary lookup with byte fallback
                for token in tokens {
                    if let id = vocabulary.tokenToID(token) {
                        ids.append(id)
                    } else {
                        appendByteFallback(token, to: &ids)
                    }
                }
            }
        }

        return ids
    }

    // MARK: - Decode

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var result = ""
        for id in ids {
            if skipSpecialTokens && specialTokens.specialTokenIDs.contains(id) {
                continue
            }
            if let token = vocabulary.idToToken(id) {
                result += token
            } else {
                result += "\u{FFFD}"
            }
        }

        // Replace ▁ back to spaces
        result = result.replacingOccurrences(of: Self.sentencePieceSpaceString, with: " ")

        // Strip leading space if we added one during encode
        if addSpacePrefix && result.hasPrefix(" ") {
            result = String(result.dropFirst())
        }

        return result
    }

    // MARK: - Chat Template

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        tools: [ToolDefinition]? = nil
    ) throws -> String? {
        guard let engine = chatTemplateEngine else { return nil }
        let bosToken = specialTokens.bosTokenID.flatMap { vocabulary.idToToken($0) }
        let eosToken = vocabulary.idToToken(eosTokenID)
        return try engine.apply(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            bosToken: bosToken,
            eosToken: eosToken,
            tools: tools
        )
    }

    // MARK: - Private

    private func splitAroundSpecialTokens(_ text: String) -> [String] {
        guard !specialTokens.specialTokenMap.isEmpty else { return [text] }
        let sortedSpecials = specialTokens.specialTokenMap.keys.sorted { $0.count > $1.count }
        var segments = [String]()
        var remaining = text

        while !remaining.isEmpty {
            var earliestRange: Range<String.Index>?
            var earliestToken: String?
            for special in sortedSpecials {
                if let range = remaining.range(of: special) {
                    if earliestRange == nil || range.lowerBound < earliestRange!.lowerBound {
                        earliestRange = range
                        earliestToken = special
                    }
                }
            }
            guard let range = earliestRange, let token = earliestToken else {
                segments.append(remaining)
                break
            }
            let prefix = String(remaining[remaining.startIndex..<range.lowerBound])
            if !prefix.isEmpty { segments.append(prefix) }
            segments.append(token)
            remaining = String(remaining[range.upperBound...])
        }
        return segments
    }

    private func appendByteFallback(_ token: String, to ids: inout [Int]) {
        for byte in Array(token.utf8) {
            if let id = byteFallbackTable[byte] {
                ids.append(id)
            }
        }
    }

    /// Greedy bigram merging by float score (highest score wins).
    private func applyMerges(_ initialTokens: [String]) -> [String] {
        var tokens = initialTokens
        while tokens.count >= 2 {
            var bestScore: Float = -.infinity
            var bestIndex = -1
            for i in 0..<(tokens.count - 1) {
                let merged = tokens[i] + tokens[i + 1]
                if let score = tokenScores[merged], score > bestScore {
                    bestScore = score
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let merged = tokens[bestIndex] + tokens[bestIndex + 1]
            var newTokens = [String]()
            newTokens.reserveCapacity(tokens.count - 1)
            for i in 0..<tokens.count {
                if i == bestIndex {
                    newTokens.append(merged)
                } else if i == bestIndex + 1 {
                    continue
                } else {
                    newTokens.append(tokens[i])
                }
            }
            tokens = newTokens
        }
        return tokens
    }
}
```

- [ ] **Step 4: Run tests — verify they PASS**

Run: `swift test --filter SentencePieceTokenizerTests 2>&1 | tail -15`
Expected: All 10 tests pass

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/SentencePieceTokenizer.swift Tests/EdgeRunnerCoreTests/SentencePieceTokenizerTests.swift
git commit -m "feat: add SentencePieceTokenizer with greedy bigram merge by score"
```

---

### Task 5: Update TokenizerFactory to dispatch SentencePiece models

**Files:**
- Modify: `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift`
- Modify: `Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift`

- [ ] **Step 1: Write factory tests for SentencePiece**

Add to `Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift`:

```swift
@Test func createFromLlamaModelReturnsSentencePiece() throws {
    let metadata = try GGUFTokenizerMetadata(metadata: [
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": .array(["<unk>", "<s>", "</s>", "▁", "a", "b", "▁a", "▁ab"]),
        "tokenizer.ggml.scores": .array([0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -0.5, -0.1].map { MetadataValue.float(Float($0)) }),
        "tokenizer.ggml.token_type": .array([2, 3, 3, 1, 1, 1, 1, 1].map { MetadataValue.int($0) }),
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    ])
    let tokenizer = try TokenizerFactory.create(from: metadata)
    #expect(tokenizer is SentencePieceTokenizer)
    #expect(tokenizer.eosTokenID == 2)
}

@Test func llamaModelScoresMissingThrows() {
    // .llama model without scores should throw
    let metadata = try! GGUFTokenizerMetadata(metadata: [
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": .array(["a", "b"]),
        "tokenizer.ggml.eos_token_id": 1,
    ])
    #expect(throws: TokenizerFactoryError.self) {
        _ = try TokenizerFactory.create(from: metadata)
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter TokenizerFactoryTests 2>&1 | tail -10`

- [ ] **Step 3: Update TokenizerFactory**

Change the return type of `create(from:)` to `any Tokenizer` and add SentencePiece dispatch:

```swift
public static func create(from metadata: GGUFTokenizerMetadata) throws -> any Tokenizer {
    switch metadata.model {
    case .gpt2, .llamaBPE:
        return try createBPE(from: metadata)
    case .llama, .sentencePiece:
        return try createSentencePiece(from: metadata)
    default:
        throw TokenizerFactoryError.unsupportedModel(metadata.model.rawValue)
    }
}
```

Extract existing BPE creation into `private static func createBPE(from:) throws -> BPETokenizer`.

Add `private static func createSentencePiece(from:) throws -> SentencePieceTokenizer`:

```swift
private static func createSentencePiece(from metadata: GGUFTokenizerMetadata) throws -> SentencePieceTokenizer {
    // Validate EOS
    guard let eosID = metadata.eosTokenID, eosID >= 0, eosID < metadata.tokens.count else {
        throw TokenizerFactoryError.missingRequiredToken("EOS")
    }

    // Validate scores exist
    guard let scores = metadata.scores else {
        throw TokenizerFactoryError.missingRequiredToken("scores")
    }

    // Build vocabulary
    let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)

    // Build token scores lookup
    var tokenScores = [String: Float](minimumCapacity: metadata.tokens.count)
    for (index, token) in metadata.tokens.enumerated() {
        tokenScores[token] = scores[index]
    }

    // Collect special tokens (same as BPE)
    var additionalSpecialTokens = [String: Int]()
    if let tokenTypes = metadata.tokenTypes {
        for (index, tokenType) in tokenTypes.enumerated() {
            if tokenType == .control || tokenType == .userDefined {
                additionalSpecialTokens[metadata.tokens[index]] = index
            }
        }
    }

    // Build SpecialTokens
    let bosToken: (String, Int)? = metadata.bosTokenID.flatMap { id in
        guard id >= 0, id < metadata.tokens.count else { return nil }
        additionalSpecialTokens.removeValue(forKey: metadata.tokens[id])
        return (metadata.tokens[id], id)
    }

    let eosString = metadata.tokens[eosID]
    additionalSpecialTokens.removeValue(forKey: eosString)

    let padToken: (String, Int)? = metadata.paddingTokenID.flatMap { id in
        guard id >= 0, id < metadata.tokens.count else { return nil }
        additionalSpecialTokens.removeValue(forKey: metadata.tokens[id])
        return (metadata.tokens[id], id)
    }

    let specialTokens = SpecialTokens(
        bosToken: bosToken,
        eosToken: (eosString, eosID),
        padToken: padToken,
        additionalSpecialTokens: additionalSpecialTokens
    )

    // Byte fallback
    var byteFallbackTable = [UInt8: Int]()
    if let tokenTypes = metadata.tokenTypes {
        for (index, tokenType) in tokenTypes.enumerated() {
            guard tokenType == .byte else { continue }
            if let byte = parseByteToken(metadata.tokens[index]) {
                byteFallbackTable[byte] = index
            }
        }
    }

    // Chat template
    let chatEngine: ChatTemplateEngine? = metadata.chatTemplate.flatMap { try? ChatTemplateEngine(template: $0) }

    return SentencePieceTokenizer(
        vocabulary: vocabulary,
        specialTokens: specialTokens,
        tokenScores: tokenScores,
        addSpacePrefix: metadata.addSpacePrefix ?? true,
        shouldAddBOS: metadata.shouldAddBOS ?? true,
        byteFallbackTable: byteFallbackTable,
        chatTemplateEngine: chatEngine
    )
}
```

- [ ] **Step 4: Run ALL factory tests**

Run: `swift test --filter TokenizerFactoryTests 2>&1 | tail -15`
Expected: All tests pass (existing BPE tests + new SentencePiece tests)

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift
git commit -m "feat: dispatch .llama/.sentencePiece models to SentencePieceTokenizer in factory"
```

---

### Task 6: Update LlamaLanguageModel to use `any Tokenizer`

**Files:**
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`

- [ ] **Step 1: Change stored property type**

Change:
```swift
private let tokenizer: BPETokenizer?
```
To:
```swift
private let tokenizer: (any Tokenizer)?
```

- [ ] **Step 2: Update `tokenize()` to use protocol method**

The existing code uses `tokenizer.shouldAddBOS` — this now comes from the protocol. And `applyChatTemplate` uses `try? tokenizer?.applyChatTemplate(...)` which also comes from the protocol. Both should work without changes since `shouldAddBOS` and `applyChatTemplate` are now on the `Tokenizer` protocol.

- [ ] **Step 3: Update `load()` factory call**

The `TokenizerFactory.create(from:)` now returns `any Tokenizer` instead of `BPETokenizer`. The existing code assigns to `bpeTokenizer: BPETokenizer?` — change the variable name and type:

```swift
let loadedTokenizer: (any Tokenizer)?
do {
    let tokenizerMetadata = try loader.modelConfig.tokenizerMetadata()
    loadedTokenizer = try TokenizerFactory.create(from: tokenizerMetadata)
} catch {
    loadedTokenizer = nil
}
```

Pass `tokenizer: loadedTokenizer` to the init.

- [ ] **Step 4: Build and verify**

Run: `swift build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 5: Run full test suite**

Run: `swift test 2>&1 | grep "Test run with" | tail -1`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunner/Models/LlamaLanguageModel.swift
git commit -m "refactor: change LlamaLanguageModel tokenizer type to any Tokenizer"
```

---

### Task 7: Gemma Parity Validation

**Files:**
- Create: `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift`

- [ ] **Step 1: Create parity test**

```swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore

@Suite("Gemma Tokenizer Parity")
struct GemmaTokenizerParityTest {

    static let modelPath = "/tmp/edgerunner-models/gemma-3-1b-it-Q4_K_M.gguf"

    /// Reference token IDs from:
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

    @Test func tokenizerLoadsAsSpentencePiece() throws {
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
}
```

- [ ] **Step 2: Run parity tests (if model available)**

Run: `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1 swift test --filter GemmaTokenizerParityTest 2>&1 | tail -15`

If tests fail, debug mismatches — the space prefix logic is the most likely source of divergence. Adjust the `SentencePieceTokenizer.encode()` space handling until parity is achieved.

- [ ] **Step 3: Commit**

```bash
git add Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift
git commit -m "test: add Gemma 3 1B tokenizer parity validation against HuggingFace reference"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `swift test 2>&1 | grep "Test run with" | tail -1`
Expected: All tests pass

- [ ] **Step 2: Build in release mode**

Run: `swift build -c release 2>&1 | tail -5`
Expected: Clean build

- [ ] **Step 3: Verify both tokenizer types work end-to-end**

Run: `swift test --filter "QwenTokenizerParity|GemmaTokenizerParity" 2>&1 | tail -10`
(with both env vars set)

- [ ] **Step 4: Commit any remaining fixes**

```bash
git status
```
