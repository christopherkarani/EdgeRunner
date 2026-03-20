# BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the byte-level placeholder tokenizer in LlamaLanguageModel with a production-grade BPE tokenizer that works across Qwen, Llama, Mistral, Granite, and other model families.

**Architecture:** Layered pipeline — PreTokenizer (regex) -> ByteEncoder (GPT-2 mapping) -> BPE merges -> vocab lookup. Chat template engine (minimal Jinja2) as a separate component. Factory bridges GGUF metadata to configured tokenizer.

**Tech Stack:** Swift 6.2, Swift Regex, Swift Testing framework

**Spec:** `docs/superpowers/specs/2026-03-20-bpe-tokenizer-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `Sources/EdgeRunnerCore/Tokenizer/ByteEncoder.swift` | Create | GPT-2 byte↔unicode 256-entry mapping |
| `Sources/EdgeRunnerCore/Tokenizer/PreTokenizer.swift` | Create | Protocol + RegexPreTokenizer |
| `Sources/EdgeRunnerCore/Tokenizer/PreTokenizerPattern.swift` | Create | Registry: GGUF `pre` string -> regex |
| `Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift` | Modify | Add `additionalSpecialTokens` parameter |
| `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift` | Rewrite | Full pipeline: special scan -> pre-tokenize -> byte-encode -> merge -> lookup |
| `Sources/EdgeRunnerCore/Tokenizer/ChatMessage.swift` | Create | ChatMessage + ToolDefinition types |
| `Sources/EdgeRunnerCore/Tokenizer/ChatTemplateEngine.swift` | Create | Minimal Jinja2 interpreter (Tier 1) |
| `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift` | Create | GGUFTokenizerMetadata -> BPETokenizer |
| `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift` | Modify | Add applyChatTemplate protocol requirement |
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | Wire tokenizer via factory, remove placeholder |
| `Package.swift` | Modify | Add `EdgeRunnerIO` to `EdgeRunnerCoreTests` dependencies |
| `Tests/EdgeRunnerCoreTests/ByteEncoderTests.swift` | Create | ByteEncoder unit tests |
| `Tests/EdgeRunnerCoreTests/PreTokenizerTests.swift` | Create | PreTokenizer unit tests |
| `Tests/EdgeRunnerCoreTests/TokenizerTests.swift` | Modify | Update existing tests for new BPETokenizer API |
| `Tests/EdgeRunnerCoreTests/ChatTemplateEngineTests.swift` | Create | Chat template unit tests |
| `Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift` | Create | Factory integration tests |

### Spec Deviations

- **`ChatTemplateEngine.apply()` is throwing:** The spec's API signature shows `apply() -> String` (non-throwing), but the spec's prose says it throws for runtime evaluation errors. The plan follows the prose — `apply()` is `throws`. This is the correct engineering decision since template evaluation can fail at runtime (undefined variables, type mismatches).
- **Additional pre-tokenizer patterns:** `codeshell` and `exaone` are added to the StarCoder case beyond what the spec lists, as llama.cpp groups them together.

---

### Task 1: ByteEncoder

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/ByteEncoder.swift`
- Create: `Tests/EdgeRunnerCoreTests/ByteEncoderTests.swift`

- [ ] **Step 1: Write ByteEncoder tests**

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("ByteEncoder")
struct ByteEncoderTests {
    @Test func roundTripAllBytes() {
        for byte in UInt8.min...UInt8.max {
            let encoded = ByteEncoder.encode(byte)
            let decoded = ByteEncoder.decode(encoded)
            #expect(decoded == byte, "Byte \(byte) failed round-trip")
        }
    }

    @Test func spaceMapsToDotAboveG() {
        let encoded = ByteEncoder.encode(0x20)
        #expect(encoded == Character("\u{0120}"))  // Ġ
    }

    @Test func newlineMapsToCorrectChar() {
        let encoded = ByteEncoder.encode(0x0A)
        #expect(encoded == Character("\u{010A}"))  // Ċ
    }

    @Test func tabMapsToCorrectChar() {
        let encoded = ByteEncoder.encode(0x09)
        #expect(encoded == Character("\u{0109}"))  // ĉ
    }

    @Test func printableASCIIMapsToItself() {
        // '!' (0x21) through '~' (0x7E)
        for byte: UInt8 in 0x21...0x7E {
            let encoded = ByteEncoder.encode(byte)
            #expect(encoded == Character(UnicodeScalar(byte)), "Byte \(byte) should map to itself")
        }
    }

    @Test func encodeStringConvertsAllBytes() {
        let result = ByteEncoder.encodeString(" hi")
        // space -> Ġ, 'h' -> 'h', 'i' -> 'i'
        #expect(result == "\u{0120}hi")
    }

    @Test func decodeStringReversesEncoding() {
        let encoded = ByteEncoder.encodeString("Hello world")
        let decoded = ByteEncoder.decodeString(encoded)
        #expect(decoded == "Hello world")
    }

    @Test func decodeStringHandlesMultibyteUTF8() {
        // Encode a string with non-ASCII: "café"
        let original = "café"
        let encoded = ByteEncoder.encodeString(original)
        let decoded = ByteEncoder.decodeString(encoded)
        #expect(decoded == original)
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter ByteEncoderTests 2>&1 | tail -20`
Expected: compilation error — `ByteEncoder` not defined

- [ ] **Step 3: Implement ByteEncoder**

```swift
import Foundation

/// GPT-2 byte-to-unicode mapping table.
///
/// Maps all 256 byte values to visible Unicode characters so BPE merge tables
/// can be expressed as plain strings. 188 "nice" bytes (printable ASCII + Latin-1)
/// map to themselves; 68 "ugly" bytes (control chars, space, DEL) map to U+0100–U+0143.
public enum ByteEncoder: Sendable {
    // MARK: - Static tables

    /// Byte -> Unicode character mapping (encode direction)
    private static let byteToChar: [Character] = {
        var table = [Character](repeating: "\0", count: 256)

        // 188 "nice" bytes that map to themselves
        // Printable ASCII: 0x21 ('!') through 0x7E ('~')
        for b in 0x21...0x7E { table[b] = Character(UnicodeScalar(b)!) }
        // Latin-1 supplement: 0xA1 through 0xAC
        for b in 0xA1...0xAC { table[b] = Character(UnicodeScalar(b)!) }
        // Latin-1 supplement: 0xAE through 0xFF
        for b in 0xAE...0xFF { table[b] = Character(UnicodeScalar(b)!) }

        // 68 "ugly" bytes remapped to U+0100+
        var offset = 0
        for b in 0...255 {
            let isNice = (0x21...0x7E).contains(b)
                || (0xA1...0xAC).contains(b)
                || (0xAE...0xFF).contains(b)
            if !isNice {
                table[b] = Character(UnicodeScalar(256 + offset)!)
                offset += 1
            }
        }
        return table
    }()

    /// Unicode character -> Byte mapping (decode direction)
    private static let charToByte: [Character: UInt8] = {
        var map = [Character: UInt8](minimumCapacity: 256)
        for b in 0..<256 {
            map[byteToChar[b]] = UInt8(b)
        }
        return map
    }()

    // MARK: - Public API

    /// Encode a single byte to its GPT-2 unicode character.
    public static func encode(_ byte: UInt8) -> Character {
        byteToChar[Int(byte)]
    }

    /// Decode a GPT-2 unicode character back to its byte value.
    public static func decode(_ char: Character) -> UInt8? {
        charToByte[char]
    }

    /// Encode a string's UTF-8 bytes to GPT-2 unicode representation.
    public static func encodeString(_ text: String) -> String {
        String(Array(text.utf8).map { byteToChar[Int($0)] })
    }

    /// Decode a GPT-2 unicode string back to the original UTF-8 string.
    public static func decodeString(_ encoded: String) -> String? {
        var bytes = [UInt8]()
        bytes.reserveCapacity(encoded.count)
        for char in encoded {
            guard let byte = charToByte[char] else { return nil }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8)
    }
}
```

- [ ] **Step 4: Run tests — verify they PASS**

Run: `swift test --filter ByteEncoderTests 2>&1 | tail -20`
Expected: All 8 tests pass

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/ByteEncoder.swift Tests/EdgeRunnerCoreTests/ByteEncoderTests.swift
git commit -m "feat: add GPT-2 byte-to-unicode encoder for BPE tokenization"
```

---

### Task 2: PreTokenizer Protocol + GPT-2 Pattern

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/PreTokenizer.swift`
- Create: `Sources/EdgeRunnerCore/Tokenizer/PreTokenizerPattern.swift`
- Create: `Tests/EdgeRunnerCoreTests/PreTokenizerTests.swift`

- [ ] **Step 1: Write PreTokenizer tests**

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("PreTokenizer")
struct PreTokenizerTests {
    // MARK: - GPT-2 pattern

    @Test func gpt2SplitsWordsWithLeadingSpaces() {
        let pre = PreTokenizerPattern.resolve(nil)  // default = GPT-2
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }

    @Test func gpt2SplitsContractions() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("I'm don't")
        #expect(result.contains("'m"))
        #expect(result.contains("'t"))
    }

    @Test func gpt2SplitsDigits() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("test 123 hello")
        #expect(result.contains(" 123"))
    }

    @Test func gpt2SplitsPunctuation() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("hello, world!")
        // Punctuation gets its own group
        #expect(result.contains(","))
    }

    @Test func emptyStringReturnsEmpty() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("")
        #expect(result.isEmpty)
    }

    // MARK: - Qwen2 pattern

    @Test func qwen2SplitsDigitsIndividually() {
        let pre = PreTokenizerPattern.resolve("qwen2")
        let result = pre.split("abc123")
        // Qwen2 matches individual digits: \p{N} not \p{N}+
        #expect(result.contains("1"))
        #expect(result.contains("2"))
        #expect(result.contains("3"))
    }

    @Test func qwen2CaseInsensitiveContractions() {
        let pre = PreTokenizerPattern.resolve("qwen2")
        let result = pre.split("I'M DON'T")
        #expect(result.contains("'M"))
        #expect(result.contains("'T"))
    }

    // MARK: - Llama3 pattern

    @Test func llama3GroupsDigitsUpToThree() {
        let pre = PreTokenizerPattern.resolve("llama3")
        let result = pre.split("price 123456")
        // Llama3 uses \p{N}{1,3}: splits "123456" into "123" and "456"
        #expect(result.contains("123"))
        #expect(result.contains("456"))
    }

    // MARK: - Fallback

    @Test func unknownPreValueFallsBackToGPT2() {
        let pre = PreTokenizerPattern.resolve("some-unknown-model")
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }

    @Test func defaultPatternUsedWhenNil() {
        let pre = PreTokenizerPattern.resolve(nil)
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter PreTokenizerTests 2>&1 | tail -20`
Expected: compilation error — `PreTokenizerPattern` not defined

- [ ] **Step 3: Implement PreTokenizer protocol**

Create `Sources/EdgeRunnerCore/Tokenizer/PreTokenizer.swift`:

```swift
import Foundation

/// Protocol for pre-tokenization — splitting text into word-level chunks before BPE.
public protocol PreTokenizer: Sendable {
    /// Split input text into word-level chunks using model-specific rules.
    func split(_ text: String) -> [String]
}

/// Pre-tokenizer that uses a compiled regex pattern to split text.
public struct RegexPreTokenizer: PreTokenizer, Sendable {
    private let patterns: [Regex<Substring>]

    public init(pattern: Regex<Substring>) {
        self.patterns = [pattern]
    }

    public init(patterns: [Regex<Substring>]) {
        self.patterns = patterns
    }

    public func split(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var results = [String]()
        for pattern in patterns {
            let source = results.isEmpty ? [text] : results
            if !results.isEmpty {
                results = []
            }
            for chunk in source {
                results.append(contentsOf: chunk.matches(of: pattern).map { String($0.output) })
            }
        }
        return results
    }
}
```

- [ ] **Step 4: Implement PreTokenizerPattern registry**

Create `Sources/EdgeRunnerCore/Tokenizer/PreTokenizerPattern.swift`:

```swift
import Foundation

/// Registry that maps GGUF `tokenizer.ggml.pre` strings to pre-tokenizer instances.
public enum PreTokenizerPattern: Sendable {
    /// Resolve a GGUF `tokenizer.ggml.pre` value to a PreTokenizer instance.
    /// Falls back to GPT-2 pattern for unknown values.
    public static func resolve(_ preTokenizerName: String?) -> PreTokenizer {
        switch preTokenizerName?.lowercased() {
        case nil, "default", "gpt-2", "granite-docling":
            return gpt2()
        case "qwen2":
            return qwen2()
        case "llama3", "llama-v3", "llama4":
            return llama3()
        case "tekken":
            return tekken()
        case "starcoder", "command-r", "refact", "smollm", "codeshell", "exaone":
            return starcoder()
        case "deepseek-llm":
            return deepseekLLM()
        case "deepseek-coder":
            return deepseekCoder()
        case "chatglm-bpe":
            return gpt2()  // ChatGLM uses GPT-2-like pattern
        case "viking":
            return gpt2()  // Viking/NorwAI uses GPT-2-like pattern
        default:
            return gpt2()  // fallback
        }
    }

    // MARK: - Pattern definitions

    private static func gpt2() -> RegexPreTokenizer {
        // Original GPT-2 pre-tokenizer
        RegexPreTokenizer(pattern: try! Regex(
            #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func qwen2() -> RegexPreTokenizer {
        // Case-insensitive contractions + individual digits + newline handling
        RegexPreTokenizer(pattern: try! Regex(
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func llama3() -> RegexPreTokenizer {
        // Like Qwen2 but digits in groups of 1-3
        RegexPreTokenizer(pattern: try! Regex(
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func tekken() -> RegexPreTokenizer {
        // CamelCase-aware splitting (Mistral)
        RegexPreTokenizer(pattern: try! Regex(
            #"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func starcoder() -> RegexPreTokenizer {
        // Digit-split first, then GPT-2-like
        RegexPreTokenizer(pattern: try! Regex(
            #"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func deepseekLLM() -> RegexPreTokenizer {
        // Multi-pattern: newlines, words, CJK, digits
        RegexPreTokenizer(patterns: [
            try! Regex(#"[\r\n]|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|[一-龥ࠀ-一가-퟿]+"#)
        ])
    }

    private static func deepseekCoder() -> RegexPreTokenizer {
        RegexPreTokenizer(patterns: [
            try! Regex(#"[\r\n]|\s?\p{L}+|\s?\p{P}+|[一-龥ࠀ-一가-퟿]+|\p{N}"#)
        ])
    }
}
```

- [ ] **Step 5: Run tests — verify they PASS**

Run: `swift test --filter PreTokenizerTests 2>&1 | tail -20`
Expected: All 10 tests pass

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/PreTokenizer.swift Sources/EdgeRunnerCore/Tokenizer/PreTokenizerPattern.swift Tests/EdgeRunnerCoreTests/PreTokenizerTests.swift
git commit -m "feat: add pre-tokenizer protocol with GPT-2, Qwen2, Llama3, Tekken patterns"
```

---

### Task 3: Enhance SpecialTokens with Additional Tokens

**Files:**
- Modify: `Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift`
- Modify: `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`

- [ ] **Step 1: Write tests for enhanced SpecialTokens**

Add to `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`:

```swift
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
    #expect(special.specialTokenIDs.count == 5)  // bos + eos + 3 additional
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
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter SpecialTokensTests 2>&1 | tail -20`
Expected: compilation error — init doesn't accept `additionalSpecialTokens`

- [ ] **Step 3: Update SpecialTokens.init**

Modify `Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift` — add `additionalSpecialTokens` parameter with default empty value:

```swift
public init(
    bosToken: (String, Int)?,
    eosToken: (String, Int)?,
    padToken: (String, Int)?,
    additionalSpecialTokens: [String: Int] = [:]
) {
    self._bosString = bosToken?.0
    self._bosID = bosToken?.1
    self._eosString = eosToken?.0
    self._eosID = eosToken?.1
    self._padString = padToken?.0
    self._padID = padToken?.1

    var ids = Set<Int>(minimumCapacity: 3 + additionalSpecialTokens.count)
    var map = [String: Int](minimumCapacity: 3 + additionalSpecialTokens.count)
    if let bos = bosToken { ids.insert(bos.1); map[bos.0] = bos.1 }
    if let eos = eosToken { ids.insert(eos.1); map[eos.0] = eos.1 }
    if let pad = padToken { ids.insert(pad.1); map[pad.0] = pad.1 }
    for (token, id) in additionalSpecialTokens {
        ids.insert(id)
        map[token] = id
    }
    self.specialTokenIDs = ids
    self.specialTokenMap = map
}
```

- [ ] **Step 4: Run ALL tokenizer tests — verify they PASS**

Run: `swift test --filter "SpecialTokensTests|TokenizerProtocolTests|BPETokenizerTests" 2>&1 | tail -20`
Expected: All tests pass (existing tests unaffected because new param has default value)

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift Tests/EdgeRunnerCoreTests/TokenizerTests.swift
git commit -m "feat: extend SpecialTokens with additionalSpecialTokens parameter"
```

---

### Task 4: Rewrite BPETokenizer with Full Pipeline

**Files:**
- Rewrite: `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift`
- Modify: `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`

- [ ] **Step 1: Write new BPETokenizer tests for the full pipeline**

Add new tests to `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`:

```swift
@Suite("BPETokenizer Pipeline")
struct BPETokenizerPipelineTests {
    /// Builds a tokenizer that mimics GPT-2 style with byte encoding.
    /// Vocabulary uses GPT-2 unicode: "Ġ" = space, raw ASCII for printable.
    private func makeByteEncodedTokenizer() -> BPETokenizer {
        // Token list: individual byte-encoded chars + merged tokens
        let tokens = [
            "H", "e", "l", "o", "\u{0120}", "w", "r", "d",  // 0-7: individual chars (Ġ = space)
            "he", "ll", "lo",                                  // 8-10: first merges
            "hel", "llo",                                      // 11-12
            "hello",                                           // 13
            "\u{0120}w", "\u{0120}wo", "\u{0120}wor",         // 14-16
            "\u{0120}worl", "\u{0120}world",                   // 17-18
            "<|im_start|>", "<|im_end|>",                      // 19-20: special tokens
        ]
        let vocab = TokenizerVocabulary(tokens: tokens)
        let merges: [(String, String)] = [
            ("h", "e"), ("l", "l"), ("l", "o"),
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
        let tok = makeByteEncodedTokenizer()
        let ids = tok.encode("hello")
        #expect(ids == [13])  // "hello" -> merged to single token
    }

    @Test func encodeWithLeadingSpace() {
        let tok = makeByteEncodedTokenizer()
        let ids = tok.encode(" world")
        #expect(ids == [18])  // " world" -> "Ġworld" -> merged
    }

    @Test func encodeTwoWords() {
        let tok = makeByteEncodedTokenizer()
        let ids = tok.encode("hello world")
        // GPT-2 pre-tokenizer splits: ["hello", " world"]
        // "hello" -> byte-encode -> "hello" -> merge -> [13]
        // " world" -> byte-encode -> "Ġworld" -> merge -> [18]
        #expect(ids == [13, 18])
    }

    @Test func roundTripEncodeDecode() {
        let tok = makeByteEncodedTokenizer()
        let original = "hello world"
        let decoded = tok.decode(tok.encode(original))
        #expect(decoded == original)
    }

    @Test func specialTokensPreservedDuringEncode() {
        let tok = makeByteEncodedTokenizer()
        let ids = tok.encode("<|im_start|>hello<|im_end|>")
        // Should NOT split "<|im_start|>" into characters
        #expect(ids.first == 19)  // <|im_start|>
        #expect(ids.last == 20)   // <|im_end|>
        #expect(ids.contains(13)) // hello
    }

    @Test func decodeSkipsSpecialTokens() {
        let tok = makeByteEncodedTokenizer()
        let decoded = tok.decode([19, 13, 20], skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    @Test func unknownTokenIDProducesReplacementChar() {
        let tok = makeByteEncodedTokenizer()
        let decoded = tok.decode([99999])
        #expect(decoded == "\u{FFFD}")
    }

    @Test func vocabularySizeIsTokenCount() {
        let tok = makeByteEncodedTokenizer()
        // 22 tokens total (0-21 including </s>)
        // vocabularySize = vocabulary.count, NOT vocabulary.count + specialTokenIDs.count
        #expect(tok.vocabularySize == 21)  // vocab has 21 entries (indices 0-20)
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter BPETokenizerPipelineTests 2>&1 | tail -20`
Expected: compilation error — `BPETokenizer` init doesn't accept `preTokenizer`

- [ ] **Step 3: Rewrite BPETokenizer with full pipeline**

Replace `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift` entirely:

```swift
import Foundation

/// Production-grade Byte-Pair Encoding tokenizer.
///
/// Implements the full BPE pipeline:
/// 1. Special token scan — split text around literal special token strings
/// 2. Pre-tokenization — regex-based word splitting (model-specific)
/// 3. Byte encoding — GPT-2 byte-to-unicode mapping
/// 4. BPE merges — iterative pair merging by rank
/// 5. Vocabulary lookup — token string to ID with byte-level fallback
public struct BPETokenizer: Tokenizer, Sendable {
    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let mergeRanks: [String: Int]
    private let mergeResults: [String: String]
    private let preTokenizer: PreTokenizer
    private let byteFallbackTable: [UInt8: Int]

    /// Whether to prepend BOS token during encoding (from GGUF metadata).
    public let shouldAddBOS: Bool

    /// Optional chat template engine for formatting conversation messages.
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int { vocabulary.count }

    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        merges: [(String, String)],
        preTokenizer: PreTokenizer,
        shouldAddBOS: Bool = false,
        byteFallbackTable: [UInt8: Int] = [:],
        chatTemplateEngine: ChatTemplateEngine? = nil
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.preTokenizer = preTokenizer
        self.shouldAddBOS = shouldAddBOS
        self.byteFallbackTable = byteFallbackTable
        self.chatTemplateEngine = chatTemplateEngine

        var ranks = [String: Int](minimumCapacity: merges.count)
        var results = [String: String](minimumCapacity: merges.count)
        for (index, merge) in merges.enumerated() {
            let key = "\(merge.0)\t\(merge.1)"
            ranks[key] = index
            results[key] = merge.0 + merge.1
        }
        self.mergeRanks = ranks
        self.mergeResults = results
    }

    // MARK: - Encode

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        var ids = [Int]()
        if addBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        // Step 1: Split around special tokens
        let segments = splitAroundSpecialTokens(text)

        for segment in segments {
            if let specialID = specialTokens.specialTokenMap[segment] {
                // This segment is a special token — map directly
                ids.append(specialID)
            } else {
                // Step 2: Pre-tokenize
                let words = preTokenizer.split(segment)

                for word in words {
                    // Step 3: Byte-encode
                    let byteEncoded = ByteEncoder.encodeString(word)

                    // Step 4: Split into characters and apply BPE merges
                    var tokens = byteEncoded.map { String($0) }
                    tokens = applyMerges(tokens)

                    // Step 5: Vocabulary lookup with byte fallback
                    for token in tokens {
                        if let id = vocabulary.tokenToID(token) {
                            ids.append(id)
                        } else {
                            // Byte-level fallback: decompose into individual byte tokens
                            appendByteFallback(for: token, to: &ids)
                        }
                    }
                }
            }
        }

        return ids
    }

    // MARK: - Decode

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var encoded = ""
        for id in ids {
            if skipSpecialTokens && specialTokens.specialTokenIDs.contains(id) {
                continue
            }
            if let token = vocabulary.idToToken(id) {
                encoded += token
            } else {
                encoded += "\u{FFFD}"  // replacement character for unknown IDs
            }
        }
        return ByteEncoder.decodeString(encoded) ?? encoded
    }

    // MARK: - Chat Template

    /// Apply the chat template to format conversation messages.
    /// Returns nil if no chat template is available.
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

    // MARK: - Private helpers

    /// Split text into segments of [text, special_token, text, special_token, ...]
    private func splitAroundSpecialTokens(_ text: String) -> [String] {
        let specialStrings = specialTokens.specialTokenMap.keys.sorted { $0.count > $1.count }
        guard !specialStrings.isEmpty else { return [text] }

        var segments = [String]()
        var remaining = text[...]

        while !remaining.isEmpty {
            var found = false
            for special in specialStrings {
                if let range = remaining.range(of: special) {
                    // Add text before the special token (if any)
                    if range.lowerBound > remaining.startIndex {
                        segments.append(String(remaining[remaining.startIndex..<range.lowerBound]))
                    }
                    segments.append(special)
                    remaining = remaining[range.upperBound...]
                    found = true
                    break
                }
            }
            if !found {
                segments.append(String(remaining))
                break
            }
        }
        return segments
    }

    /// Decompose an unrecognized token into individual byte tokens via fallback.
    private func appendByteFallback(for token: String, to ids: inout [Int]) {
        for char in token {
            let singleChar = String(char)
            if let id = vocabulary.tokenToID(singleChar) {
                ids.append(id)
            } else if let byte = ByteEncoder.decode(char), let id = byteFallbackTable[byte] {
                ids.append(id)
            }
            // If neither lookup succeeds, skip (should not happen in well-formed models)
        }
    }

    private func applyMerges(_ initialTokens: [String]) -> [String] {
        var tokens = initialTokens
        while tokens.count >= 2 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(tokens.count - 1) {
                let key = "\(tokens[i])\t\(tokens[i + 1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let key = "\(tokens[bestIndex])\t\(tokens[bestIndex + 1])"
            let merged = mergeResults[key]!
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

- [ ] **Step 4: Update existing tests for new BPETokenizer API**

The existing tests need several updates:

**A. Add `PassthroughPreTokenizer` helper** (for legacy tests that don't need pre-tokenization):

```swift
/// Simple pre-tokenizer that returns the whole text as one chunk (for legacy tests).
struct PassthroughPreTokenizer: PreTokenizer {
    func split(_ text: String) -> [String] { [text] }
}
```

**B. Update `makeTestTokenizer()`** — the vocabulary must use byte-encoded characters. The space character `" "` (0x20) maps to `"\u{0120}"` (Ġ) in GPT-2 byte encoding, so we must update the vocabulary token from `" "` to `"\u{0120}"`:

```swift
private func makeTestTokenizer() -> BPETokenizer {
    let tokens = [
        "h", "e", "l", "o", "\u{0120}", "w", "r", "d",  // index 4: Ġ (byte-encoded space)
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
```

**C. Update `vocabularySizeIncludesSpecial` test** — `vocabularySize` is now `vocabulary.count` only:

```swift
@Test func vocabularySizeIsVocabCount() {
    let tokenizer = makeTestTokenizer()
    #expect(tokenizer.vocabularySize == 18)  // vocabulary.count, NOT + specialTokenIDs.count
}
```

**D. Update `TokenizerProtocolTests.protocolConformance`** — same fix:

```swift
@Test func protocolConformance() {
    let vocab = TokenizerVocabulary(tokens: ["h", "e", "l", "lo", "hello", "\u{0120}", "world"])
    let special = SpecialTokens(bosToken: ("<s>", 7), eosToken: ("</s>", 8), padToken: ("<pad>", 9))
    let tokenizer = BPETokenizer(
        vocabulary: vocab, specialTokens: special, merges: [],
        preTokenizer: PassthroughPreTokenizer()
    )
    #expect(tokenizer.vocabularySize == 7)  // vocabulary.count only
    #expect(tokenizer.eosTokenID == 8)
    #expect(tokenizer.bosTokenID == 7)
    #expect(tokenizer.padTokenID == 9)
}
```

**E. Update `encodeTwoWords` test** — space is now byte-encoded so it still maps to token index 4 (Ġ):

```swift
@Test func encodeTwoWords() {
    let tokenizer = makeTestTokenizer()
    let ids = tokenizer.encode("hello world")
    // PassthroughPreTokenizer returns ["hello world"] as one chunk
    // ByteEncoder converts: "hello world" -> "hello\u{0120}world"
    // Merges: h+e->he, l+l->ll, ll+o->llo, he+llo->hello, w+o->wo, wo+r->wor, wor+l->worl, worl+d->world
    // After merges: ["hello", "\u{0120}", "world"] -> IDs [13, 4, 17]
    #expect(ids == [13, 4, 17])
}
```

- [ ] **Step 5: Run ALL tokenizer tests**

Run: `swift test --filter "BPETokenizer|TokenizerProtocol" 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift Tests/EdgeRunnerCoreTests/TokenizerTests.swift
git commit -m "feat: rewrite BPETokenizer with full production pipeline

Replaces character-level encode with:
- special token scanning
- regex pre-tokenization
- GPT-2 byte encoding
- BPE merge application
- byte-level fallback for unknowns"
```

---

### Task 5: ChatMessage + ToolDefinition Types

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/ChatMessage.swift`

- [ ] **Step 1: Create ChatMessage.swift**

```swift
import Foundation

/// A message in a chat conversation for template formatting.
public struct ChatMessage: Sendable, Equatable {
    /// The role of the message sender (e.g., "system", "user", "assistant", "tool").
    public let role: String
    /// The text content of the message.
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// A tool definition for function-calling chat templates.
public struct ToolDefinition: Sendable, Equatable {
    /// The tool function name.
    public let name: String
    /// A description of what the tool does.
    public let description: String
    /// JSON string describing the tool's parameters schema.
    public let parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `swift build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/ChatMessage.swift
git commit -m "feat: add ChatMessage and ToolDefinition types for chat templates"
```

---

### Task 6: ChatTemplateEngine (Tier 1)

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/ChatTemplateEngine.swift`
- Create: `Tests/EdgeRunnerCoreTests/ChatTemplateEngineTests.swift`

- [ ] **Step 1: Write ChatTemplateEngine tests**

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("ChatTemplateEngine")
struct ChatTemplateEngineTests {
    // MARK: - ChatML format (Qwen, many others)

    @Test func chatMLBasicFormat() throws {
        let template = """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
        """
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "user", content: "Hello"),
            ],
            addGenerationPrompt: true
        )
        #expect(result.contains("<|im_start|>user"))
        #expect(result.contains("Hello"))
        #expect(result.contains("<|im_end|>"))
        #expect(result.contains("<|im_start|>assistant"))
    }

    @Test func chatMLMultipleMessages() throws {
        let template = """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}
        """
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "system", content: "You are helpful."),
                ChatMessage(role: "user", content: "Hi"),
            ],
            addGenerationPrompt: false
        )
        #expect(result.contains("system"))
        #expect(result.contains("You are helpful."))
        #expect(result.contains("user"))
        #expect(result.contains("Hi"))
    }

    // MARK: - Conditionals

    @Test func ifElseConditional() throws {
        let template = "{% if add_generation_prompt %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)

        let yes = try engine.apply(messages: [], addGenerationPrompt: true)
        #expect(yes == "YES")

        let no = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(no == "NO")
    }

    // MARK: - Loop variables

    @Test func loopIndexVariables() throws {
        let template = "{% for m in messages %}{{ loop.index }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
                ChatMessage(role: "c", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "123")
    }

    @Test func loopFirstLast() throws {
        let template = "{% for m in messages %}{% if loop.first %}F{% endif %}{% if loop.last %}L{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "FL")
    }

    // MARK: - Whitespace control

    @Test func whitespaceStripping() throws {
        let template = "A {%- if true %} B {%- endif %} C"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "A B C")
    }

    // MARK: - Set statement

    @Test func setVariable() throws {
        let template = "{% set name = 'world' %}Hello {{ name }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "Hello world")
    }

    // MARK: - Trim filter

    @Test func trimFilter() throws {
        let template = "{{ '  hello  ' | trim }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "hello")
    }

    // MARK: - Error handling

    @Test func unsupportedFeatureThrows() throws {
        // Macros are not in Tier 1
        let template = "{% macro test() %}{% endmacro %}"
        #expect(throws: ChatTemplateError.self) {
            _ = try ChatTemplateEngine(template: template)
        }
    }

    // MARK: - Comparison operators

    @Test func equalityComparison() throws {
        let template = "{% for m in messages %}{% if m['role'] == 'user' %}U{% else %}O{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "system", content: ""),
                ChatMessage(role: "user", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "OU")
    }

    // MARK: - String concatenation

    @Test func stringConcatenation() throws {
        let template = "{{ 'Hello' + ' ' + 'World' }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "Hello World")
    }

    // MARK: - Thread safety (Sendable conformance)

    @Test func engineIsSendable() throws {
        let engine = try ChatTemplateEngine(template: "{{ 'test' }}")
        let sendableRef: any Sendable = engine
        #expect(sendableRef is ChatTemplateEngine)
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter ChatTemplateEngineTests 2>&1 | tail -20`
Expected: compilation error — `ChatTemplateEngine` not defined

- [ ] **Step 3: Implement ChatTemplateEngine**

Create `Sources/EdgeRunnerCore/Tokenizer/ChatTemplateEngine.swift`. This is the largest file — approximately 400-600 lines implementing:
- `ChatTemplateError` enum with `.parseError`, `.unsupportedFeature`, `.evaluationError`
- Lexer: splits template into `Token` types (text, expression `{{ }}`, statement `{% %}`, comment `{# #}`) with whitespace control (`{%-`, `-%}`)
- Parser: builds AST nodes (`TextNode`, `OutputNode`, `ForNode`, `IfNode`, `SetNode`)
- Expression parser: handles string literals, variable access (dot and bracket notation), string concatenation (`+`), comparisons (`==`, `!=`, `and`, `or`, `not`, `in`), filter pipe (`| trim`)
- Evaluator: walks AST with a context dictionary containing `messages`, `add_generation_prompt`, `bos_token`, `eos_token`. For-loops provide `loop.index`, `loop.index0`, `loop.first`, `loop.last`.
- `ChatTemplateEngine` struct stores the parsed AST (immutable after init) and exposes `apply()` that creates a fresh context per call

This file is substantial but well-contained. The implementer should reference the spec for the exact feature set and build incrementally — get text output working first, then expressions, then for-loops, then if-blocks.

- [ ] **Step 4: Run tests — verify they PASS**

Run: `swift test --filter ChatTemplateEngineTests 2>&1 | tail -30`
Expected: All 12 tests pass

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/ChatTemplateEngine.swift Tests/EdgeRunnerCoreTests/ChatTemplateEngineTests.swift
git commit -m "feat: add Tier 1 chat template engine (minimal Jinja2 interpreter)

Supports: for loops with loop vars, if/elif/else, set, string concat,
comparisons (==, !=, and, or, not, in), whitespace control, trim filter.
Covers ChatML, Llama 3, Mistral, Phi template formats."
```

---

### Task 7: TokenizerFactory

**Files:**
- Modify: `Package.swift` (add `EdgeRunnerIO` to `EdgeRunnerCoreTests` dependencies)
- Create: `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift`
- Create: `Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift`

- [ ] **Step 0: Add EdgeRunnerIO to EdgeRunnerCoreTests dependencies**

In `Package.swift`, the `EdgeRunnerCoreTests` target (line 57-60) needs `EdgeRunnerIO` so tests can import `GGUFTokenizerMetadata` and `MetadataValue`:

```swift
.testTarget(
    name: "EdgeRunnerCoreTests",
    dependencies: ["EdgeRunnerCore", "EdgeRunnerMetal", "EdgeRunnerSharedTypes", "EdgeRunnerIO"]
),
```

- [ ] **Step 1: Write TokenizerFactory tests**

```swift
import Testing
@testable import EdgeRunnerCore
@testable import EdgeRunnerIO

@Suite("TokenizerFactory")
struct TokenizerFactoryTests {
    private func makeQwenMetadata(
        eosTokenID: Int? = 151645,
        model: String = "gpt2"
    ) throws -> GGUFTokenizerMetadata {
        var metadata: [String: MetadataValue] = [
            "tokenizer.ggml.model": .string(model),
            "tokenizer.ggml.pre": .string("qwen2"),
            "tokenizer.ggml.tokens": .array(
                (0..<256).map { .string(String(ByteEncoder.encode(UInt8($0)))) }
                + [.string("hello"), .string("\u{0120}world")]
                + [.string("<|im_start|>"), .string("<|im_end|>")]
            ),
            "tokenizer.ggml.merges": .array([.string("h e")]),
            "tokenizer.ggml.token_type": .array(
                (0..<256).map { _ in MetadataValue.int(1) }
                + [.int(1), .int(1)]
                + [.int(3), .int(3)]  // control tokens
            ),
        ]
        if let eos = eosTokenID {
            metadata["tokenizer.ggml.eos_token_id"] = .int(eos)
        }
        metadata["tokenizer.ggml.bos_token_id"] = .int(151643)
        return try GGUFTokenizerMetadata(metadata: metadata)
    }

    @Test func createsTokenizerFromValidMetadata() throws {
        let metadata = try makeQwenMetadata()
        let tokenizer = try TokenizerFactory.create(from: metadata)
        #expect(tokenizer.eosTokenID == 151645)
        #expect(tokenizer.vocabularySize == 260)  // 256 byte tokens + 2 merged + 2 special
    }

    @Test func throwsForSentencePieceModel() throws {
        #expect(throws: TokenizerFactoryError.self) {
            let metadata = try makeQwenMetadata(model: "sentencepiece")
            _ = try TokenizerFactory.create(from: metadata)
        }
    }

    @Test func throwsForMissingEOS() throws {
        #expect(throws: TokenizerFactoryError.self) {
            let metadata = try makeQwenMetadata(eosTokenID: nil)
            _ = try TokenizerFactory.create(from: metadata)
        }
    }

    @Test func specialTokensIncludeControlTokens() throws {
        let metadata = try makeQwenMetadata()
        let tokenizer = try TokenizerFactory.create(from: metadata)
        // Control tokens (type 3) at indices 258 and 259 should be in special token set
        let ids = tokenizer.encode("<|im_start|>")
        #expect(ids == [258])  // resolved as special token, not split
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter TokenizerFactoryTests 2>&1 | tail -20`
Expected: compilation error — `TokenizerFactory` not defined

- [ ] **Step 3: Implement TokenizerFactory**

Create `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift`:

```swift
import Foundation
import EdgeRunnerIO

/// Errors that occur during tokenizer creation from GGUF metadata.
public enum TokenizerFactoryError: Error, Sendable {
    case unsupportedModel(String)
    case missingRequiredToken(String)
}

/// Creates tokenizer instances from GGUF model metadata.
public enum TokenizerFactory: Sendable {
    /// Create a BPETokenizer from parsed GGUF tokenizer metadata.
    ///
    /// - Parameter metadata: Parsed tokenizer metadata from a GGUF file.
    /// - Returns: A configured BPETokenizer ready for encoding/decoding.
    /// - Throws: `TokenizerFactoryError` if the model type is unsupported or required data is missing.
    public static func create(from metadata: GGUFTokenizerMetadata) throws -> BPETokenizer {
        // Step 1: Validate model type
        switch metadata.model {
        case .gpt2, .llamaBPE:
            break  // supported
        case .sentencePiece:
            throw TokenizerFactoryError.unsupportedModel(
                "SentencePiece tokenizer not yet supported. Model uses '\(metadata.model.rawValue)'"
            )
        case .wordPiece:
            throw TokenizerFactoryError.unsupportedModel(
                "WordPiece tokenizer not yet supported. Model uses '\(metadata.model.rawValue)'"
            )
        case .unknown(let name):
            throw TokenizerFactoryError.unsupportedModel(
                "Unknown tokenizer model '\(name)'"
            )
        }

        // Step 2: Validate EOS exists
        guard let eosID = metadata.eosTokenID else {
            throw TokenizerFactoryError.missingRequiredToken("eos")
        }

        // Step 3: Build vocabulary
        let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)

        // Step 4: Collect special tokens from tokenTypes
        var additionalSpecialTokens = [String: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                if tokenType == .control || tokenType == .userDefined {
                    additionalSpecialTokens[metadata.tokens[index]] = index
                }
            }
        }

        // Step 5: Build SpecialTokens
        let bosToken: (String, Int)? = metadata.bosTokenID.map { id in
            (metadata.tokens.indices.contains(id) ? metadata.tokens[id] : "<s>", id)
        }
        let eosToken: (String, Int) = (
            metadata.tokens.indices.contains(eosID) ? metadata.tokens[eosID] : "</s>",
            eosID
        )
        let padToken: (String, Int)? = metadata.paddingTokenID.map { id in
            (metadata.tokens.indices.contains(id) ? metadata.tokens[id] : "<pad>", id)
        }
        let specialTokens = SpecialTokens(
            bosToken: bosToken,
            eosToken: eosToken,
            padToken: padToken,
            additionalSpecialTokens: additionalSpecialTokens
        )

        // Step 6: Build byte fallback table
        var byteFallbackTable = [UInt8: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                if tokenType == .byte {
                    // Parse byte value from token string (format varies)
                    let token = metadata.tokens[index]
                    if let byte = parseByteFallbackToken(token) {
                        byteFallbackTable[byte] = index
                    }
                }
            }
        }

        // Step 7: Convert merges
        let merges = metadata.merges.map { ($0.left, $0.right) }

        // Step 8: Resolve pre-tokenizer
        let preTokenizer = PreTokenizerPattern.resolve(metadata.preTokenizer)

        // Step 9: Create chat template engine (if template available)
        let chatEngine: ChatTemplateEngine?
        if let template = metadata.chatTemplate {
            chatEngine = try? ChatTemplateEngine(template: template)
        } else {
            chatEngine = nil
        }

        // Step 10: Read shouldAddBOS from metadata
        let shouldAddBOS = metadata.shouldAddBOS ?? false

        // Step 11: Assemble
        return BPETokenizer(
            vocabulary: vocabulary,
            specialTokens: specialTokens,
            merges: merges,
            preTokenizer: preTokenizer,
            shouldAddBOS: shouldAddBOS,
            byteFallbackTable: byteFallbackTable,
            chatTemplateEngine: chatEngine
        )
    }

    /// Parse a byte fallback token string to its byte value.
    /// Handles formats like "<0x0A>" and single byte-encoded unicode chars.
    private static func parseByteFallbackToken(_ token: String) -> UInt8? {
        // Format: <0xHH>
        if token.hasPrefix("<0x") && token.hasSuffix(">") {
            let hex = String(token.dropFirst(3).dropLast(1))
            return UInt8(hex, radix: 16)
        }
        // Single character — decode through ByteEncoder
        if token.count == 1, let char = token.first {
            return ByteEncoder.decode(char)
        }
        return nil
    }
}
```

- [ ] **Step 4: Run tests — verify they PASS**

Run: `swift test --filter TokenizerFactoryTests 2>&1 | tail -20`
Expected: All 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift Tests/EdgeRunnerCoreTests/TokenizerFactoryTests.swift
git commit -m "feat: add TokenizerFactory to bridge GGUF metadata to BPETokenizer"
```

---

### Task 8: EdgeRunnerLanguageModel Protocol Update

**Files:**
- Modify: `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift`

- [ ] **Step 1: Add applyChatTemplate to protocol**

Add to the `EdgeRunnerLanguageModel` protocol definition (after `var vocabularySize: Int { get }`):

```swift
/// Formats conversation messages using the model's chat template.
/// Returns nil if no chat template is available.
func applyChatTemplate(
    messages: [ChatMessage],
    addGenerationPrompt: Bool
) -> String?
```

Add default implementation:

```swift
extension EdgeRunnerLanguageModel {
    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String? { nil }
}
```

- [ ] **Step 2: Verify build succeeds**

Run: `swift build 2>&1 | tail -10`
Expected: Build succeeds (default implementation satisfies conformance for all existing types)

- [ ] **Step 3: Commit**

```bash
git add Sources/EdgeRunner/EdgeRunnerLanguageModel.swift
git commit -m "feat: add applyChatTemplate requirement to EdgeRunnerLanguageModel protocol"
```

---

### Task 9: LlamaLanguageModel Integration

**Files:**
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`

- [ ] **Step 1: Add tokenizer stored property**

Add to the struct's stored properties (around line 26, after `private let config: LlamaConfig`):

```swift
private let tokenizer: BPETokenizer
```

- [ ] **Step 2: Update the initializer to accept tokenizer**

Add `tokenizer: BPETokenizer` parameter to the private `init` (around line 120). Store it:

```swift
self.tokenizer = tokenizer
```

- [ ] **Step 3: Update `load()` to create tokenizer from GGUF**

In the `load(from:configuration:)` method (around line 238-256), after `let ggufConfig = ...` and before constructing `LlamaLanguageModel`, add:

```swift
let tokenizerMetadata = try loader.modelConfig.tokenizerMetadata()
let tokenizer = try TokenizerFactory.create(from: tokenizerMetadata)
```

Pass `tokenizer` to the `LlamaLanguageModel` init.

- [ ] **Step 4: Replace placeholder tokenize/detokenize**

Replace lines 258-288:

```swift
public func tokenize(_ text: String) -> [Int] {
    tokenizer.encode(text, addBOS: tokenizer.shouldAddBOS)
}

public func detokenize(_ ids: [Int]) -> String {
    tokenizer.decode(ids, skipSpecialTokens: true)
}

public var eosTokenID: Int { tokenizer.eosTokenID }
public var bosTokenID: Int? { tokenizer.bosTokenID }
public var vocabularySize: Int { tokenizer.vocabularySize }

public func applyChatTemplate(
    messages: [ChatMessage],
    addGenerationPrompt: Bool
) -> String? {
    try? tokenizer.applyChatTemplate(
        messages: messages,
        addGenerationPrompt: addGenerationPrompt
    )
}
```

- [ ] **Step 5: Build the project**

Run: `swift build 2>&1 | tail -20`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunner/Models/LlamaLanguageModel.swift
git commit -m "feat: wire BPETokenizer into LlamaLanguageModel, remove placeholder

Replaces byte-level tokenize/detokenize with production BPE pipeline
loaded from GGUF metadata. Removes hardcoded Qwen3 token IDs."
```

---

### Task 10: Run Full Test Suite + Fix Regressions

**Files:**
- Potentially modify any file that has test failures

- [ ] **Step 1: Run entire test suite**

Run: `swift test 2>&1 | tail -40`

- [ ] **Step 2: Fix any compilation errors or test failures**

Common issues to watch for:
- Existing tests that construct `LlamaLanguageModel` directly may need the new `tokenizer` parameter
- Qwen parity tests may need updating since tokenization output will change from byte-level to proper BPE
- `vocabularySizeIncludesSpecial` test expectation needs updating

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test regressions from BPE tokenizer integration"
```

---

### Task 11: Parity Validation

**Files:**
- No new files — uses existing parity test infrastructure

- [ ] **Step 1: Generate reference token IDs**

Use Python with HuggingFace tokenizers to generate known-good token IDs for test strings against the Qwen3 0.6B model. Save these as test fixtures.

Run (if Python + transformers available):
```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
tests = ['Hello, world!', 'The quick brown fox', '1+1=2', '你好世界', 'def foo():\n    return 42']
for t in tests:
    ids = tok.encode(t, add_special_tokens=False)
    print(f'{repr(t)}: {ids}')
"
```

- [ ] **Step 2: Add parity assertions to QwenTokenizerParityTest**

If a real GGUF model is available in the test fixtures, load it and compare. Otherwise, use the reference IDs as hardcoded expectations in unit tests.

- [ ] **Step 3: Run parity tests**

Run: `swift test --filter QwenTokenizerParity 2>&1 | tail -20`
Expected: All pass with byte-perfect match

- [ ] **Step 4: Commit**

```bash
git add Tests/
git commit -m "test: add Qwen3 tokenizer parity validation against HuggingFace reference"
```

---

### Task 12: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `swift test 2>&1`
Expected: All tests pass

- [ ] **Step 2: Check test coverage**

Run: `swift test --enable-code-coverage 2>&1 | tail -5`
Verify 80%+ coverage on new files.

- [ ] **Step 3: Build in release mode**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Clean build with no warnings

- [ ] **Step 4: Final commit if any remaining changes**

```bash
git status
# Only commit if there are meaningful changes
```
