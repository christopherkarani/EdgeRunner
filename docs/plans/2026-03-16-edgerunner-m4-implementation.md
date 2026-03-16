# EdgeRunner Milestone 4: High-Level API & Developer Experience — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Abstract low-level tensor details behind an ergonomic API: streaming token output, structured generation via @Generable, sampling strategies, tool calling, Foundation Models backend swapping, and documentation.

**Architecture:** Type-erased `any EdgeRunnerLanguageModel` protocol with streaming via AsyncThrowingStream. Sampling pipeline as composable transforms. Tool calling via EdgeRunnerTool protocol with JSON schema. Foundation Models parity for backend swapping.

**Tech Stack:** Swift 6.2, Metal Shading Language 4.0, Swift Testing, DocC

**Depends on:** Milestone 3 (docs/plans/2026-03-16-edgerunner-m3-implementation.md)

---

## Task 1: EdgeRunnerLanguageModel Protocol

**Files:**
- Create: `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift`
- Create: `Sources/EdgeRunner/ModelConfiguration.swift`
- Create: `Sources/EdgeRunner/GenerationError.swift`
- Test: `Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift
import Testing
@testable import EdgeRunner

// MARK: - Mock model for protocol conformance testing

struct MockLanguageModel: EdgeRunnerLanguageModel {
    static let modelIdentifier = "mock-test-v1"

    let vocabulary: [String] = ["Hello", " world", "!", "<eos>"]
    let fixedTokenIDs: [Int]

    init(fixedTokenIDs: [Int] = [0, 1, 2, 3]) {
        self.fixedTokenIDs = fixedTokenIDs
    }

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockLanguageModel {
        return MockLanguageModel()
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        // Return a fixed logits vector of vocabulary size
        var result = [Float](repeating: -100.0, count: vocabulary.count)
        let nextIndex = min(tokenIDs.count, fixedTokenIDs.count - 1)
        result[fixedTokenIDs[nextIndex]] = 10.0
        return result
    }

    func tokenize(_ text: String) -> [Int] {
        // Trivial: map known words to indices
        var ids: [Int] = []
        var remaining = text
        for (i, word) in vocabulary.enumerated() {
            while remaining.hasPrefix(word) {
                ids.append(i)
                remaining.removeFirst(word.count)
            }
        }
        return ids
    }

    func detokenize(_ ids: [Int]) -> String {
        ids.map { id in
            guard id >= 0, id < vocabulary.count else { return "" }
            return vocabulary[id]
        }.joined()
    }

    var eosTokenID: Int { 3 }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocabulary.count }
}

@Suite("EdgeRunnerLanguageModel Protocol")
struct EdgeRunnerLanguageModelProtocolTests {

    @Test func protocolConformance() async throws {
        let model = MockLanguageModel()
        #expect(MockLanguageModel.modelIdentifier == "mock-test-v1")
        #expect(model.vocabularySize == 4)
        #expect(model.eosTokenID == 3)
    }

    @Test func loadFromURL() async throws {
        let url = URL(fileURLWithPath: "/tmp/fake-model.gguf")
        let config = ModelConfiguration()
        let model = try await MockLanguageModel.load(from: url, configuration: config)
        #expect(model.vocabularySize == 4)
    }

    @Test func logitsReturnCorrectSize() async throws {
        let model = MockLanguageModel()
        let logits = try await model.logits(for: [0])
        #expect(logits.count == model.vocabularySize)
    }

    @Test func logitsHighestAtExpectedToken() async throws {
        let model = MockLanguageModel(fixedTokenIDs: [0, 1, 2, 3])
        let logits = try await model.logits(for: [0]) // next should be index 1
        let maxIndex = logits.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(maxIndex == 1)
    }

    @Test func tokenizeRoundTrip() {
        let model = MockLanguageModel()
        let text = "Hello world!"
        let ids = model.tokenize(text)
        let decoded = model.detokenize(ids)
        #expect(decoded == text)
    }

    @Test func modelConfigurationDefaults() {
        let config = ModelConfiguration()
        #expect(config.maxTokens == 2048)
        #expect(config.contextWindowSize == 4096)
    }

    @Test func modelConfigurationCustom() {
        let config = ModelConfiguration(
            maxTokens: 512,
            contextWindowSize: 8192
        )
        #expect(config.maxTokens == 512)
        #expect(config.contextWindowSize == 8192)
    }

    @Test func generationErrorDescriptions() {
        let error1 = GenerationError.modelLoadFailed(reason: "File not found")
        let error2 = GenerationError.contextWindowExceeded(requested: 5000, maximum: 4096)
        let error3 = GenerationError.cancelled

        #expect("\(error1)".contains("File not found"))
        #expect("\(error2)".contains("5000"))
        #expect("\(error3)".contains("cancelled"))
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter EdgeRunnerLanguageModelProtocolTests 2>&1`
Expected: FAIL — `EdgeRunnerLanguageModel` protocol not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/GenerationError.swift
import Foundation

/// Errors that can occur during model loading and text generation.
public enum GenerationError: Error, Sendable, CustomStringConvertible {
    case modelLoadFailed(reason: String)
    case contextWindowExceeded(requested: Int, maximum: Int)
    case invalidTokenID(Int)
    case decodingFailed(String)
    case cancelled
    case samplingFailed(String)
    case toolCallFailed(name: String, reason: String)
    case structuredOutputFailed(reason: String)

    public var description: String {
        switch self {
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .contextWindowExceeded(let requested, let maximum):
            return "Context window exceeded: requested \(requested) tokens, maximum \(maximum)"
        case .invalidTokenID(let id):
            return "Invalid token ID: \(id)"
        case .decodingFailed(let reason):
            return "Decoding failed: \(reason)"
        case .cancelled:
            return "Generation cancelled"
        case .samplingFailed(let reason):
            return "Sampling failed: \(reason)"
        case .toolCallFailed(let name, let reason):
            return "Tool call '\(name)' failed: \(reason)"
        case .structuredOutputFailed(let reason):
            return "Structured output failed: \(reason)"
        }
    }
}
```

```swift
// Sources/EdgeRunner/ModelConfiguration.swift
import Foundation

/// Configuration for model loading and generation behavior.
public struct ModelConfiguration: Sendable {
    /// Maximum number of tokens to generate in a single call.
    public var maxTokens: Int

    /// Maximum context window size in tokens.
    public var contextWindowSize: Int

    /// Whether to use memory-mapped IO for weight loading.
    public var useMemoryMapping: Bool

    /// Optional path to a tokenizer file (if separate from model).
    public var tokenizerURL: URL?

    public init(
        maxTokens: Int = 2048,
        contextWindowSize: Int = 4096,
        useMemoryMapping: Bool = true,
        tokenizerURL: URL? = nil
    ) {
        self.maxTokens = maxTokens
        self.contextWindowSize = contextWindowSize
        self.useMemoryMapping = useMemoryMapping
        self.tokenizerURL = tokenizerURL
    }
}
```

```swift
// Sources/EdgeRunner/EdgeRunnerLanguageModel.swift
import Foundation

/// Protocol that all EdgeRunner language models must conform to.
///
/// Provides a uniform interface for local Metal-accelerated models and
/// system Foundation Models backends. Designed for type-erased usage
/// via `any EdgeRunnerLanguageModel`.
public protocol EdgeRunnerLanguageModel: Sendable {
    /// A unique identifier for this model architecture (e.g., "llama-3-8b-q4_0").
    static var modelIdentifier: String { get }

    /// Load a model from a file URL (GGUF, SafeTensor, etc.).
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self

    /// Convert a string to a sequence of token IDs.
    func tokenize(_ text: String) -> [Int]

    /// Convert a sequence of token IDs back to a string.
    func detokenize(_ ids: [Int]) -> String

    /// The token ID that signals end-of-sequence.
    var eosTokenID: Int { get }

    /// Optional beginning-of-sequence token ID.
    var bosTokenID: Int? { get }

    /// The total vocabulary size.
    var vocabularySize: Int { get }

    /// Generate the next token given context. Returns a token ID.
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int

    /// Stream tokens for a prompt.
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}

/// Sub-protocol for local Metal-accelerated models that expose raw logits.
/// Foundation Models backends do NOT conform to this — they use `nextToken` instead.
public protocol LogitsModel: EdgeRunnerLanguageModel {
    /// Compute raw logits (unnormalized log-probabilities) for next-token prediction.
    ///
    /// - Parameter tokenIDs: The input token sequence.
    /// - Returns: A Float array of size `vocabularySize` containing raw logits.
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter EdgeRunnerLanguageModelProtocolTests 2>&1`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/EdgeRunnerLanguageModel.swift \
      Sources/EdgeRunner/ModelConfiguration.swift \
      Sources/EdgeRunner/GenerationError.swift \
      Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift
git commit -m "feat(m4): add EdgeRunnerLanguageModel protocol, ModelConfiguration, and GenerationError

Type-erased protocol for uniform model interface across local Metal models
and Foundation Models backends. Includes generation configuration and
comprehensive error types."
```

---

## Task 2: Tokenizer

**Files:**
- Create: `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift`
- Create: `Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift`
- Create: `Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift`
- Create: `Sources/EdgeRunnerCore/Tokenizer/TokenizerVocabulary.swift`
- Test: `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/TokenizerTests.swift
import Testing
@testable import EdgeRunnerCore

@Suite("Tokenizer Protocol")
struct TokenizerProtocolTests {

    @Test func protocolConformance() {
        let vocab = TokenizerVocabulary(tokens: ["h", "e", "l", "lo", "hello", " ", "world"])
        let special = SpecialTokens(
            bosToken: ("<s>", 7),
            eosToken: ("</s>", 8),
            padToken: ("<pad>", 9)
        )
        let tokenizer = BPETokenizer(vocabulary: vocab, specialTokens: special, merges: [])
        #expect(tokenizer.vocabularySize == 10) // 7 tokens + 3 special
        #expect(tokenizer.eosTokenID == 8)
        #expect(tokenizer.bosTokenID == 7)
        #expect(tokenizer.padTokenID == 9)
    }
}

@Suite("SpecialTokens")
struct SpecialTokensTests {

    @Test func specialTokenIDs() {
        let special = SpecialTokens(
            bosToken: ("<s>", 0),
            eosToken: ("</s>", 1),
            padToken: ("<pad>", 2)
        )
        #expect(special.bosTokenID == 0)
        #expect(special.eosTokenID == 1)
        #expect(special.padTokenID == 2)
        #expect(special.bosTokenString == "<s>")
    }

    @Test func optionalTokens() {
        let special = SpecialTokens(
            bosToken: nil,
            eosToken: ("</s>", 1),
            padToken: nil
        )
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

    /// Build a minimal BPE tokenizer with known merges.
    private func makeTestTokenizer() -> BPETokenizer {
        // Vocabulary: individual bytes + merged tokens
        let tokens = [
            "h", "e", "l", "o", " ", "w", "r", "d",  // 0-7
            "he", "ll", "lo",                          // 8-10
            "hel", "llo",                              // 11-12
            "hello",                                   // 13
            "wo", "wor", "worl", "world",              // 14-17
        ]
        let vocab = TokenizerVocabulary(tokens: tokens)

        // Merges in priority order (most frequent first)
        let merges: [(String, String)] = [
            ("h", "e"),       // h + e -> he (8)
            ("l", "l"),       // l + l -> ll (9)
            ("l", "o"),       // l + o -> lo (10)
            ("he", "l"),      // he + l -> hel (11)
            ("l", "lo"),      // l + lo -> llo (12) — note: alternate path
            ("hel", "lo"),    // hel + lo -> hello (13)
            ("w", "o"),       // w + o -> wo (14)
            ("wo", "r"),      // wo + r -> wor (15)
            ("wor", "l"),     // wor + l -> worl (16)
            ("worl", "d"),    // worl + d -> world (17)
        ]

        let special = SpecialTokens(
            bosToken: ("<s>", 18),
            eosToken: ("</s>", 19),
            padToken: nil
        )

        return BPETokenizer(vocabulary: vocab, specialTokens: special, merges: merges)
    }

    @Test func encodeSimpleWord() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello")
        #expect(ids == [13]) // "hello" is a single merged token
    }

    @Test func encodeTwoWords() {
        let tokenizer = makeTestTokenizer()
        let ids = tokenizer.encode("hello world")
        #expect(ids == [13, 4, 17]) // "hello" + " " + "world"
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
        #expect(ids.first == 18) // BOS token
        #expect(ids.last == 13)  // "hello"
    }

    @Test func decodeSkipsSpecialTokens() {
        let tokenizer = makeTestTokenizer()
        let ids = [18, 13, 19] // BOS + "hello" + EOS
        let decoded = tokenizer.decode(ids, skipSpecialTokens: true)
        #expect(decoded == "hello")
    }

    @Test func encodeUnknownFallsBackToBytes() {
        let tokenizer = makeTestTokenizer()
        // "her" has 'h','e','r' as individual bytes — 'he' merges, 'r' stays
        let ids = tokenizer.encode("her")
        #expect(ids == [8, 6]) // "he" + "r"
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
        // 18 base tokens + 2 special (BOS, EOS)
        #expect(tokenizer.vocabularySize == 20)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "TokenizerTests|BPETokenizerTests|SpecialTokensTests|TokenizerVocabularyTests" 2>&1`
Expected: FAIL — types not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift
import Foundation

/// Protocol for all tokenizer implementations.
public protocol Tokenizer: Sendable {
    /// Encode a string into token IDs.
    func encode(_ text: String, addBOS: Bool) -> [Int]

    /// Decode token IDs back to a string.
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String

    /// The total vocabulary size (including special tokens).
    var vocabularySize: Int { get }

    /// End-of-sequence token ID.
    var eosTokenID: Int { get }

    /// Beginning-of-sequence token ID (optional).
    var bosTokenID: Int? { get }

    /// Padding token ID (optional).
    var padTokenID: Int? { get }
}

extension Tokenizer {
    public func encode(_ text: String) -> [Int] {
        encode(text, addBOS: false)
    }

    public func decode(_ ids: [Int]) -> String {
        decode(ids, skipSpecialTokens: false)
    }
}
```

```swift
// Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift
import Foundation

/// Container for special token definitions (BOS, EOS, PAD, etc.).
public struct SpecialTokens: Sendable {
    public let bosToken: (string: String, id: Int)?
    public let eosToken: (string: String, id: Int)?
    public let padToken: (string: String, id: Int)?

    // Workaround: tuples are not Sendable by default in Swift 6.2,
    // so we store backing storage as Sendable types.
    private let _bosString: String?
    private let _bosID: Int?
    private let _eosString: String?
    private let _eosID: Int?
    private let _padString: String?
    private let _padID: Int?

    public var bosTokenID: Int? { _bosID }
    public var eosTokenID: Int? { _eosID }
    public var padTokenID: Int? { _padID }
    public var bosTokenString: String? { _bosString }
    public var eosTokenString: String? { _eosString }
    public var padTokenString: String? { _padString }

    /// Set of all special token IDs for quick lookup.
    public let specialTokenIDs: Set<Int>

    /// Mapping from special token string to ID.
    public let specialTokenMap: [String: Int]

    public init(
        bosToken: (String, Int)?,
        eosToken: (String, Int)?,
        padToken: (String, Int)?
    ) {
        self._bosString = bosToken?.0
        self._bosID = bosToken?.1
        self._eosString = eosToken?.0
        self._eosID = eosToken?.1
        self._padString = padToken?.0
        self._padID = padToken?.1

        // Build tuple accessors
        self.bosToken = bosToken.map { (string: $0.0, id: $0.1) }
        self.eosToken = eosToken.map { (string: $0.0, id: $0.1) }
        self.padToken = padToken.map { (string: $0.0, id: $0.1) }

        var ids = Set<Int>()
        var map = [String: Int]()
        if let bos = bosToken { ids.insert(bos.1); map[bos.0] = bos.1 }
        if let eos = eosToken { ids.insert(eos.1); map[eos.0] = eos.1 }
        if let pad = padToken { ids.insert(pad.1); map[pad.0] = pad.1 }
        self.specialTokenIDs = ids
        self.specialTokenMap = map
    }
}
```

```swift
// Sources/EdgeRunnerCore/Tokenizer/TokenizerVocabulary.swift
import Foundation

/// Bidirectional mapping between token strings and their integer IDs.
public struct TokenizerVocabulary: Sendable {
    private let tokenToIDMap: [String: Int]
    private let idToTokenMap: [Int: String]
    public let count: Int
    public let offset: Int

    /// Initialize from an ordered array of token strings.
    /// Token at index `i` gets ID `offset + i`.
    public init(tokens: [String], offset: Int = 0) {
        self.offset = offset
        self.count = tokens.count
        var t2i = [String: Int](minimumCapacity: tokens.count)
        var i2t = [Int: String](minimumCapacity: tokens.count)
        for (index, token) in tokens.enumerated() {
            let id = offset + index
            t2i[token] = id
            i2t[id] = token
        }
        self.tokenToIDMap = t2i
        self.idToTokenMap = i2t
    }

    /// Look up the ID for a token string.
    public func tokenToID(_ token: String) -> Int? {
        tokenToIDMap[token]
    }

    /// Look up the token string for an ID.
    public func idToToken(_ id: Int) -> String? {
        idToTokenMap[id]
    }

    /// Check if a token string exists in the vocabulary.
    public func contains(_ token: String) -> Bool {
        tokenToIDMap[token] != nil
    }
}
```

```swift
// Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift
import Foundation

/// Byte-Pair Encoding tokenizer compatible with Llama/GPT tokenizer formats.
///
/// Implements the standard BPE algorithm:
/// 1. Split input into individual characters (or bytes)
/// 2. Iteratively merge the highest-priority pair
/// 3. Map merged tokens to vocabulary IDs
public struct BPETokenizer: Tokenizer, Sendable {
    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens

    /// Ordered merge rules: (left, right) -> merged token.
    /// Index = priority (lower index = higher priority = merge first).
    private let mergeRanks: [String: Int]
    private let mergeResults: [String: String]

    public var vocabularySize: Int {
        vocabulary.count + specialTokens.specialTokenIDs.count
    }

    public var eosTokenID: Int {
        specialTokens.eosTokenID!
    }

    public var bosTokenID: Int? {
        specialTokens.bosTokenID
    }

    public var padTokenID: Int? {
        specialTokens.padTokenID
    }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        merges: [(String, String)]
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens

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

    // MARK: - Tokenizer Protocol

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        var ids = [Int]()
        if addBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        // Split into characters as initial tokens
        var tokens = text.map { String($0) }

        // Iteratively apply BPE merges
        tokens = applyMerges(tokens)

        // Convert token strings to IDs
        for token in tokens {
            if let id = vocabulary.tokenToID(token) {
                ids.append(id)
            } else if let id = specialTokens.specialTokenMap[token] {
                ids.append(id)
            }
            // Unknown tokens are silently dropped (should not happen with byte fallback)
        }

        return ids
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var result = ""
        for id in ids {
            if skipSpecialTokens && specialTokens.specialTokenIDs.contains(id) {
                continue
            }
            if let token = vocabulary.idToToken(id) {
                result += token
            } else if let token = specialTokenStringForID(id) {
                if !skipSpecialTokens {
                    result += token
                }
            }
        }
        return result
    }

    // MARK: - BPE Algorithm

    /// Apply BPE merge rules iteratively until no more merges are possible.
    private func applyMerges(_ initialTokens: [String]) -> [String] {
        var tokens = initialTokens

        while tokens.count >= 2 {
            // Find the highest-priority (lowest rank) mergeable pair
            var bestRank = Int.max
            var bestIndex = -1

            for i in 0..<(tokens.count - 1) {
                let key = "\(tokens[i])\t\(tokens[i + 1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }

            // No more merges possible
            guard bestIndex >= 0 else { break }

            // Apply the merge
            let key = "\(tokens[bestIndex])\t\(tokens[bestIndex + 1])"
            let merged = mergeResults[key]!
            var newTokens = [String]()
            newTokens.reserveCapacity(tokens.count - 1)

            for i in 0..<tokens.count {
                if i == bestIndex {
                    newTokens.append(merged)
                } else if i == bestIndex + 1 {
                    continue // skip — already merged
                } else {
                    newTokens.append(tokens[i])
                }
            }

            tokens = newTokens
        }

        return tokens
    }

    /// Look up a special token string by ID.
    private func specialTokenStringForID(_ id: Int) -> String? {
        if id == specialTokens.bosTokenID { return specialTokens.bosTokenString }
        if id == specialTokens.eosTokenID { return specialTokens.eosTokenString }
        if id == specialTokens.padTokenID { return specialTokens.padTokenString }
        return nil
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "BPETokenizerTests|SpecialTokensTests|TokenizerVocabularyTests|TokenizerProtocolTests" 2>&1`
Expected: All 15 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift \
      Sources/EdgeRunnerCore/Tokenizer/SpecialTokens.swift \
      Sources/EdgeRunnerCore/Tokenizer/TokenizerVocabulary.swift \
      Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift \
      Tests/EdgeRunnerCoreTests/TokenizerTests.swift
git commit -m "feat(m4): add BPE tokenizer with vocabulary and special token support

Standard BPE encoding/decoding with priority-ranked merge rules.
Bidirectional vocabulary lookup with offset support.
Special tokens (BOS, EOS, PAD) with configurable IDs."
```

---

## Task 3: Sampling Pipeline

**Files:**
- Create: `Sources/EdgeRunnerCore/Sampling/SamplingStrategy.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/GreedySampler.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/TemperatureSampler.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/TopKSampler.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/TopPSampler.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/MinPSampler.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/RepetitionPenalty.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift`
- Create: `Sources/EdgeRunnerCore/Sampling/SeededRandomSource.swift`
- Test: `Tests/EdgeRunnerCoreTests/SamplingTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/SamplingTests.swift
import Testing
@testable import EdgeRunnerCore

// MARK: - Test Helpers

/// Softmax for verifying probability distributions.
private func softmax(_ logits: [Float]) -> [Float] {
    let maxLogit = logits.max() ?? 0
    let exps = logits.map { exp($0 - maxLogit) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

@Suite("GreedySampler")
struct GreedySamplerTests {

    @Test func selectsHighestLogit() {
        let sampler = GreedySampler()
        let logits: [Float] = [1.0, 3.0, 2.0, 0.5]
        let tokenID = sampler.sample(logits: logits)
        #expect(tokenID == 1) // index of 3.0
    }

    @Test func handlesNegativeLogits() {
        let sampler = GreedySampler()
        let logits: [Float] = [-5.0, -1.0, -3.0, -2.0]
        let tokenID = sampler.sample(logits: logits)
        #expect(tokenID == 1) // index of -1.0 (highest)
    }

    @Test func handlesSingleElement() {
        let sampler = GreedySampler()
        let logits: [Float] = [42.0]
        let tokenID = sampler.sample(logits: logits)
        #expect(tokenID == 0)
    }

    @Test func tieBreaksToFirst() {
        let sampler = GreedySampler()
        let logits: [Float] = [5.0, 5.0, 5.0]
        let tokenID = sampler.sample(logits: logits)
        #expect(tokenID == 0) // first occurrence
    }
}

@Suite("TemperatureSampler")
struct TemperatureSamplerTests {

    @Test func temperatureZeroIsGreedy() {
        let sampler = TemperatureSampler(temperature: 0.0)
        let logits: [Float] = [1.0, 5.0, 2.0]
        let result = sampler.transformLogits(logits)
        // At temperature 0, should return logits unchanged (pipeline will use greedy)
        let maxIndex = result.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(maxIndex == 1)
    }

    @Test func temperatureOnePreservesDistribution() {
        let sampler = TemperatureSampler(temperature: 1.0)
        let logits: [Float] = [1.0, 2.0, 3.0]
        let result = sampler.transformLogits(logits)
        #expect(result.count == 3)
        // At temperature 1.0, logits should be unchanged
        for i in 0..<3 {
            #expect(abs(result[i] - logits[i]) < 1e-6)
        }
    }

    @Test func highTemperatureFlattensDistribution() {
        let sampler = TemperatureSampler(temperature: 100.0)
        let logits: [Float] = [1.0, 10.0, 1.0]
        let result = sampler.transformLogits(logits)
        let probs = softmax(result)
        // With very high temperature, probabilities should be nearly uniform
        for p in probs {
            #expect(abs(p - 1.0 / 3.0) < 0.05)
        }
    }

    @Test func lowTemperatureSharpenDistribution() {
        let sampler = TemperatureSampler(temperature: 0.01)
        let logits: [Float] = [1.0, 3.0, 2.0]
        let result = sampler.transformLogits(logits)
        let probs = softmax(result)
        // With very low temperature, probability should concentrate on max
        #expect(probs[1] > 0.99)
    }
}

@Suite("TopKSampler")
struct TopKSamplerTests {

    @Test func filtersToTopK() {
        let sampler = TopKSampler(k: 2)
        let logits: [Float] = [1.0, 5.0, 3.0, 2.0]
        let result = sampler.transformLogits(logits)
        // Only top-2 (indices 1 and 2) should remain; others set to -inf
        #expect(result[1] == 5.0)
        #expect(result[2] == 3.0)
        #expect(result[0] == -.infinity)
        #expect(result[3] == -.infinity)
    }

    @Test func kGreaterThanVocabKeepsAll() {
        let sampler = TopKSampler(k: 100)
        let logits: [Float] = [1.0, 2.0, 3.0]
        let result = sampler.transformLogits(logits)
        #expect(result == logits)
    }

    @Test func kOfOneIsGreedy() {
        let sampler = TopKSampler(k: 1)
        let logits: [Float] = [1.0, 5.0, 3.0]
        let result = sampler.transformLogits(logits)
        #expect(result[1] == 5.0)
        #expect(result[0] == -.infinity)
        #expect(result[2] == -.infinity)
    }
}

@Suite("TopPSampler")
struct TopPSamplerTests {

    @Test func pOfOneKeepsAll() {
        let sampler = TopPSampler(p: 1.0)
        let logits: [Float] = [1.0, 2.0, 3.0]
        let result = sampler.transformLogits(logits)
        // All tokens should remain
        for i in 0..<3 {
            #expect(result[i] > -.infinity)
        }
    }

    @Test func lowPKeepsFewTokens() {
        let sampler = TopPSampler(p: 0.1)
        // Logits designed so softmax gives ~[0.01, 0.97, 0.02]
        let logits: [Float] = [0.0, 5.0, 0.5]
        let result = sampler.transformLogits(logits)
        // Only the dominant token (index 1) should remain
        #expect(result[1] == 5.0)
        #expect(result[0] == -.infinity)
        #expect(result[2] == -.infinity)
    }

    @Test func moderatePKeepsTopTokens() {
        let sampler = TopPSampler(p: 0.9)
        // More spread logits
        let logits: [Float] = [2.0, 2.1, 2.2, -10.0, -10.0]
        let result = sampler.transformLogits(logits)
        // Top 3 should remain, bottom 2 should be -inf
        #expect(result[0] > -.infinity)
        #expect(result[1] > -.infinity)
        #expect(result[2] > -.infinity)
        #expect(result[3] == -.infinity)
        #expect(result[4] == -.infinity)
    }
}

@Suite("MinPSampler")
struct MinPSamplerTests {

    @Test func filtersTokensBelowMinProbability() {
        let sampler = MinPSampler(minP: 0.1)
        // Logits: softmax([10, 1, 1, 1]) ≈ [0.9994, 0.0002, 0.0002, 0.0002]
        let logits: [Float] = [10.0, 1.0, 1.0, 1.0]
        let result = sampler.transformLogits(logits)
        // minP threshold = 0.1 * maxProb ≈ 0.1 * 0.9994 ≈ 0.0999
        // Only index 0 passes
        #expect(result[0] == 10.0)
        #expect(result[1] == -.infinity)
        #expect(result[2] == -.infinity)
        #expect(result[3] == -.infinity)
    }

    @Test func minPZeroKeepsAll() {
        let sampler = MinPSampler(minP: 0.0)
        let logits: [Float] = [10.0, 1.0, 0.5]
        let result = sampler.transformLogits(logits)
        for i in 0..<3 {
            #expect(result[i] > -.infinity)
        }
    }
}

@Suite("RepetitionPenalty")
struct RepetitionPenaltyTests {

    @Test func penalizesRepeatedTokens() {
        let penalty = RepetitionPenalty(penalty: 1.5)
        let logits: [Float] = [2.0, 3.0, 1.0, 4.0]
        let previousTokens = [1, 3] // penalize indices 1 and 3
        let result = penalty.apply(logits: logits, previousTokens: previousTokens)
        // Positive logits are divided by penalty
        #expect(result[0] == 2.0) // unchanged
        #expect(abs(result[1] - 3.0 / 1.5) < 1e-6) // penalized
        #expect(result[2] == 1.0) // unchanged
        #expect(abs(result[3] - 4.0 / 1.5) < 1e-6) // penalized
    }

    @Test func penalizesNegativeLogitsByMultiplying() {
        let penalty = RepetitionPenalty(penalty: 2.0)
        let logits: [Float] = [-1.0, -2.0, 3.0]
        let previousTokens = [0, 1]
        let result = penalty.apply(logits: logits, previousTokens: previousTokens)
        // Negative logits are multiplied by penalty (making them more negative)
        #expect(abs(result[0] - (-1.0 * 2.0)) < 1e-6)
        #expect(abs(result[1] - (-2.0 * 2.0)) < 1e-6)
        #expect(result[2] == 3.0) // unchanged
    }

    @Test func penaltyOfOneNoChange() {
        let penalty = RepetitionPenalty(penalty: 1.0)
        let logits: [Float] = [2.0, 3.0, 1.0]
        let previousTokens = [0, 1, 2]
        let result = penalty.apply(logits: logits, previousTokens: previousTokens)
        for i in 0..<3 {
            #expect(abs(result[i] - logits[i]) < 1e-6)
        }
    }

    @Test func frequencyPenalty() {
        let penalty = RepetitionPenalty(penalty: 1.0, frequencyPenalty: 0.5)
        let logits: [Float] = [2.0, 3.0, 1.0]
        // Token 1 appears twice
        let previousTokens = [1, 1, 0]
        let result = penalty.apply(logits: logits, previousTokens: previousTokens)
        // Token 1 penalized by 2 * 0.5 = 1.0
        #expect(abs(result[1] - (3.0 - 1.0)) < 1e-6)
        // Token 0 penalized by 1 * 0.5 = 0.5
        #expect(abs(result[0] - (2.0 - 0.5)) < 1e-6)
        // Token 2 unchanged
        #expect(abs(result[2] - 1.0) < 1e-6)
    }
}

@Suite("SeededRandomSource")
struct SeededRandomSourceTests {

    @Test func deterministicWithSameSeed() {
        var rng1 = SeededRandomSource(seed: 42)
        var rng2 = SeededRandomSource(seed: 42)
        let values1 = (0..<10).map { _ in Float.random(in: 0...1, using: &rng1) }
        let values2 = (0..<10).map { _ in Float.random(in: 0...1, using: &rng2) }
        #expect(values1 == values2)
    }

    @Test func differentSeedsProduceDifferentValues() {
        var rng1 = SeededRandomSource(seed: 42)
        var rng2 = SeededRandomSource(seed: 99)
        let v1 = Float.random(in: 0...1, using: &rng1)
        let v2 = Float.random(in: 0...1, using: &rng2)
        #expect(v1 != v2)
    }
}

@Suite("SamplingPipeline")
struct SamplingPipelineTests {

    @Test func greedyPipelineSelectsMax() {
        let pipeline = SamplingPipeline(
            transforms: [],
            selector: GreedySampler()
        )
        let logits: [Float] = [1.0, 5.0, 3.0]
        let tokenID = pipeline.sample(logits: logits)
        #expect(tokenID == 1)
    }

    @Test func temperaturePlusSampling() {
        var rng = SeededRandomSource(seed: 12345)
        let pipeline = SamplingPipeline(
            transforms: [TemperatureSampler(temperature: 0.001)],
            selector: StochasticSampler(randomSource: &rng)
        )
        let logits: [Float] = [1.0, 10.0, 2.0]
        let tokenID = pipeline.sample(logits: logits)
        // Very low temperature — should almost always pick index 1
        #expect(tokenID == 1)
    }

    @Test func topKPlusTopPPipeline() {
        let pipeline = SamplingPipeline(
            transforms: [
                TopKSampler(k: 2),
                TopPSampler(p: 0.9),
            ],
            selector: GreedySampler()
        )
        let logits: [Float] = [1.0, 5.0, 3.0, 0.5]
        let tokenID = pipeline.sample(logits: logits)
        #expect(tokenID == 1)
    }

    @Test func fullPipelineWithRepetitionPenalty() {
        let pipeline = SamplingPipeline(
            transforms: [
                TemperatureSampler(temperature: 1.0),
                TopKSampler(k: 3),
            ],
            selector: GreedySampler(),
            repetitionPenalty: RepetitionPenalty(penalty: 100.0)
        )
        let logits: [Float] = [5.0, 4.9, 4.8, 1.0]
        // Token 0 was already generated — heavily penalized
        let previousTokens = [0]
        let tokenID = pipeline.sample(logits: logits, previousTokens: previousTokens)
        // Token 0 should be demoted, token 1 should win
        #expect(tokenID == 1)
    }

    @Test func defaultPipelineIsGreedy() {
        let pipeline = SamplingPipeline.greedy
        let logits: [Float] = [1.0, 9.0, 3.0]
        #expect(pipeline.sample(logits: logits) == 1)
    }

    @Test func topPWithTemperature() {
        let pipeline = SamplingPipeline.nucleus(temperature: 0.001, topP: 0.9)
        let logits: [Float] = [1.0, 10.0, 2.0, 0.5]
        let tokenID = pipeline.sample(logits: logits)
        #expect(tokenID == 1)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "GreedySamplerTests|TemperatureSamplerTests|TopKSamplerTests|TopPSamplerTests|MinPSamplerTests|RepetitionPenaltyTests|SeededRandomSourceTests|SamplingPipelineTests" 2>&1`
Expected: FAIL — types not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunnerCore/Sampling/SamplingStrategy.swift
import Foundation

/// A transform that modifies logits before token selection.
public protocol LogitsTransform: Sendable {
    /// Transform the logits array (e.g., apply temperature, mask tokens).
    func transformLogits(_ logits: [Float]) -> [Float]
}

/// A token selector that picks a token ID from a (possibly transformed) logits array.
public protocol TokenSelector: Sendable {
    /// Select a single token ID from the logits.
    func sample(logits: [Float]) -> Int
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/GreedySampler.swift
import Foundation

/// Always selects the token with the highest logit (argmax).
public struct GreedySampler: TokenSelector, Sendable {
    public init() {}

    public func sample(logits: [Float]) -> Int {
        var maxValue: Float = -.infinity
        var maxIndex = 0
        for (index, value) in logits.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }
        return maxIndex
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/TemperatureSampler.swift
import Foundation

/// Scales logits by 1/temperature before sampling.
///
/// - temperature = 1.0: no change
/// - temperature < 1.0: sharper distribution (more confident)
/// - temperature > 1.0: flatter distribution (more random)
/// - temperature = 0.0: equivalent to greedy (logits unchanged, rely on greedy selector)
public struct TemperatureSampler: LogitsTransform, Sendable {
    public let temperature: Float

    public init(temperature: Float) {
        precondition(temperature >= 0, "Temperature must be non-negative")
        self.temperature = temperature
    }

    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard temperature > 0, temperature != 1.0 else {
            return logits
        }
        return logits.map { $0 / temperature }
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/TopKSampler.swift
import Foundation

/// Keeps only the top-k tokens by logit value; sets all others to -infinity.
public struct TopKSampler: LogitsTransform, Sendable {
    public let k: Int

    public init(k: Int) {
        precondition(k >= 1, "k must be at least 1")
        self.k = k
    }

    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard k < logits.count else { return logits }

        // Find the k-th largest value
        let sorted = logits.sorted(by: >)
        let threshold = sorted[k - 1]

        var result = [Float](repeating: -.infinity, count: logits.count)
        var kept = 0
        for (i, logit) in logits.enumerated() {
            if logit >= threshold && kept < k {
                result[i] = logit
                kept += 1
            }
        }
        return result
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/TopPSampler.swift
import Foundation

/// Nucleus sampling: keeps the smallest set of tokens whose cumulative
/// probability exceeds `p`. All other tokens are set to -infinity.
public struct TopPSampler: LogitsTransform, Sendable {
    public let p: Float

    public init(p: Float) {
        precondition(p > 0 && p <= 1.0, "p must be in (0, 1]")
        self.p = p
    }

    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard p < 1.0 else { return logits }

        // Compute softmax probabilities
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }

        // Sort indices by probability descending
        let sortedIndices = probs.enumerated()
            .sorted { $0.element > $1.element }
            .map(\.offset)

        // Find cutoff: smallest set whose cumulative probability >= p
        var cumulative: Float = 0
        var keepSet = Set<Int>()
        for index in sortedIndices {
            keepSet.insert(index)
            cumulative += probs[index]
            if cumulative >= p {
                break
            }
        }

        // Mask tokens not in the keep set
        var result = logits
        for i in 0..<result.count {
            if !keepSet.contains(i) {
                result[i] = -.infinity
            }
        }
        return result
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/MinPSampler.swift
import Foundation

/// Min-p sampling: filters out tokens whose probability is below
/// `minP * max_probability`. Adaptive threshold that scales with confidence.
public struct MinPSampler: LogitsTransform, Sendable {
    public let minP: Float

    public init(minP: Float) {
        precondition(minP >= 0 && minP <= 1.0, "minP must be in [0, 1]")
        self.minP = minP
    }

    public func transformLogits(_ logits: [Float]) -> [Float] {
        guard minP > 0 else { return logits }

        // Compute softmax probabilities
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }

        let maxProb = probs.max() ?? 0
        let threshold = minP * maxProb

        var result = logits
        for i in 0..<result.count {
            if probs[i] < threshold {
                result[i] = -.infinity
            }
        }
        return result
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/RepetitionPenalty.swift
import Foundation

/// Applies repetition and frequency penalties to discourage repeated tokens.
///
/// - Repetition penalty: positive logits are divided by `penalty`,
///   negative logits are multiplied by `penalty`.
/// - Frequency penalty: subtracts `frequencyPenalty * count` from each
///   token's logit, where `count` is how many times it appeared.
public struct RepetitionPenalty: Sendable {
    public let penalty: Float
    public let frequencyPenalty: Float

    public init(penalty: Float = 1.0, frequencyPenalty: Float = 0.0) {
        precondition(penalty >= 1.0, "Repetition penalty must be >= 1.0")
        precondition(frequencyPenalty >= 0, "Frequency penalty must be non-negative")
        self.penalty = penalty
        self.frequencyPenalty = frequencyPenalty
    }

    /// Apply penalties based on previously generated tokens.
    public func apply(logits: [Float], previousTokens: [Int]) -> [Float] {
        guard !previousTokens.isEmpty else { return logits }

        // Count occurrences
        var counts = [Int: Int]()
        for token in previousTokens {
            counts[token, default: 0] += 1
        }

        var result = logits
        for (tokenID, count) in counts {
            guard tokenID >= 0, tokenID < result.count else { continue }

            // Repetition penalty
            if penalty != 1.0 {
                if result[tokenID] > 0 {
                    result[tokenID] /= penalty
                } else {
                    result[tokenID] *= penalty
                }
            }

            // Frequency penalty
            if frequencyPenalty > 0 {
                result[tokenID] -= frequencyPenalty * Float(count)
            }
        }

        return result
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/SeededRandomSource.swift
import Foundation

/// A deterministic random number generator using xoshiro256** algorithm.
/// Suitable for reproducible sampling in tests and experiments.
public struct SeededRandomSource: RandomNumberGenerator, Sendable {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    public init(seed: UInt64) {
        // SplitMix64 to initialize state from a single seed
        var s = seed
        func next() -> UInt64 {
            s &+= 0x9e3779b97f4a7c15
            var z = s
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return z ^ (z >> 31)
        }
        state = (next(), next(), next(), next())
    }

    public mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17

        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3

        state.2 ^= t
        state.3 = rotl(state.3, 45)

        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}
```

```swift
// Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift
import Foundation

/// Stochastic token selector that samples from the probability distribution.
public struct StochasticSampler<RNG: RandomNumberGenerator & Sendable>: TokenSelector, Sendable {
    private var rng: RNG

    public init(randomSource: inout RNG) {
        self.rng = randomSource
    }

    public func sample(logits: [Float]) -> Int {
        // Compute softmax
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)

        guard sumExps > 0 else {
            // All -infinity — fall back to first non-infinite or 0
            return logits.firstIndex(where: { $0 > -.infinity }) ?? 0
        }

        let probs = exps.map { $0 / sumExps }

        // Sample from cumulative distribution
        var mutableRng = rng
        let r = Float.random(in: 0..<1, using: &mutableRng)
        var cumulative: Float = 0
        for (i, p) in probs.enumerated() {
            cumulative += p
            if r < cumulative {
                return i
            }
        }
        return probs.count - 1
    }
}

/// Composable sampling pipeline: applies a chain of logit transforms,
/// then selects a token using the configured selector.
public struct SamplingPipeline: Sendable {
    private let transforms: [any LogitsTransform]
    private let selector: any TokenSelector
    private let repetitionPenalty: RepetitionPenalty?

    public init(
        transforms: [any LogitsTransform],
        selector: any TokenSelector,
        repetitionPenalty: RepetitionPenalty? = nil
    ) {
        self.transforms = transforms
        self.selector = selector
        self.repetitionPenalty = repetitionPenalty
    }

    /// Sample a single token from the logits.
    public func sample(
        logits: [Float],
        previousTokens: [Int] = []
    ) -> Int {
        var currentLogits = logits

        // Apply repetition penalty first
        if let penalty = repetitionPenalty {
            currentLogits = penalty.apply(logits: currentLogits, previousTokens: previousTokens)
        }

        // Apply transforms in order
        for transform in transforms {
            currentLogits = transform.transformLogits(currentLogits)
        }

        // Select token
        return selector.sample(logits: currentLogits)
    }

    // MARK: - Factory Methods

    /// Greedy decoding (always pick the highest-logit token).
    public static var greedy: SamplingPipeline {
        SamplingPipeline(transforms: [], selector: GreedySampler())
    }

    /// Top-p (nucleus) sampling with temperature.
    public static func nucleus(
        temperature: Float = 0.8,
        topP: Float = 0.9,
        seed: UInt64 = 0
    ) -> SamplingPipeline {
        var rng = SeededRandomSource(seed: seed)
        return SamplingPipeline(
            transforms: [
                TemperatureSampler(temperature: temperature),
                TopPSampler(p: topP),
            ],
            selector: StochasticSampler(randomSource: &rng)
        )
    }

    /// Top-k sampling with temperature.
    public static func topK(
        k: Int = 40,
        temperature: Float = 0.8,
        seed: UInt64 = 0
    ) -> SamplingPipeline {
        var rng = SeededRandomSource(seed: seed)
        return SamplingPipeline(
            transforms: [
                TemperatureSampler(temperature: temperature),
                TopKSampler(k: k),
            ],
            selector: StochasticSampler(randomSource: &rng)
        )
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "GreedySamplerTests|TemperatureSamplerTests|TopKSamplerTests|TopPSamplerTests|MinPSamplerTests|RepetitionPenaltyTests|SeededRandomSourceTests|SamplingPipelineTests" 2>&1`
Expected: All 25 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Sampling/SamplingStrategy.swift \
      Sources/EdgeRunnerCore/Sampling/GreedySampler.swift \
      Sources/EdgeRunnerCore/Sampling/TemperatureSampler.swift \
      Sources/EdgeRunnerCore/Sampling/TopKSampler.swift \
      Sources/EdgeRunnerCore/Sampling/TopPSampler.swift \
      Sources/EdgeRunnerCore/Sampling/MinPSampler.swift \
      Sources/EdgeRunnerCore/Sampling/RepetitionPenalty.swift \
      Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift \
      Sources/EdgeRunnerCore/Sampling/SeededRandomSource.swift \
      Tests/EdgeRunnerCoreTests/SamplingTests.swift
git commit -m "feat(m4): add composable sampling pipeline with greedy, top-k, top-p, min-p, temperature, and repetition penalty

Composable LogitsTransform chain with pluggable TokenSelector.
Deterministic SeededRandomSource (xoshiro256**) for reproducible tests.
Factory methods for common configurations: greedy, nucleus, top-k."
```

---

## Task 4: Speculative Decoding

**Files:**
- Create: `Sources/EdgeRunnerCore/Generation/SpeculativeDecoder.swift`
- Test: `Tests/EdgeRunnerCoreTests/SpeculativeDecodingTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/SpeculativeDecodingTests.swift
import Testing
@testable import EdgeRunnerCore

// MARK: - Mock Draft & Verification Models

/// A mock model that returns predictable logits for speculative decoding tests.
private struct MockSpecModel: SpeculativeModel {
    let fixedLogitsSequence: [[Float]]
    let vocabSize: Int

    init(vocabSize: Int, fixedLogitsSequence: [[Float]]) {
        self.vocabSize = vocabSize
        self.fixedLogitsSequence = fixedLogitsSequence
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let step = tokenIDs.count - 1
        if step < fixedLogitsSequence.count {
            return fixedLogitsSequence[step]
        }
        // Default: uniform logits
        return [Float](repeating: 0.0, count: vocabSize)
    }

    func batchLogits(for sequences: [[Int]]) async throws -> [[Float]] {
        // Parallel verification: return logits for each sequence
        try await sequences.asyncMap { try await logits(for: $0) }
    }
}

/// Helper: create logits where a specific token has the highest value.
private func makeLogits(vocabSize: Int, peakAt index: Int, peakValue: Float = 10.0) -> [Float] {
    var logits = [Float](repeating: -10.0, count: vocabSize)
    logits[index] = peakValue
    return logits
}

@Suite("SpeculativeDecoder")
struct SpeculativeDecodingTests {

    @Test func allDraftTokensAccepted() async throws {
        let vocabSize = 5
        // Draft model predicts tokens [1, 2, 3]
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        // Verification model agrees: same peaks
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
            makeLogits(vocabSize: vocabSize, peakAt: 4), // bonus token
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)

        let decoder = SpeculativeDecoder(
            draftModel: draft,
            verificationModel: verifier,
            draftTokenCount: 3,
            samplingPipeline: .greedy
        )

        let result = try await decoder.decodeStep(inputTokens: [0])
        // All 3 draft tokens accepted + 1 bonus from verifier
        #expect(result.acceptedTokens == [1, 2, 3, 4])
        #expect(result.acceptanceRate == 1.0)
    }

    @Test func firstDraftTokenRejected() async throws {
        let vocabSize = 5
        // Draft predicts [1, 2, 3]
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        // Verifier disagrees on first token: prefers token 4
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 4), // rejects draft token 1
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)

        let decoder = SpeculativeDecoder(
            draftModel: draft,
            verificationModel: verifier,
            draftTokenCount: 3,
            samplingPipeline: .greedy
        )

        let result = try await decoder.decodeStep(inputTokens: [0])
        // First token rejected — verifier's choice used instead
        #expect(result.acceptedTokens == [4])
        #expect(result.acceptanceRate == 0.0)
    }

    @Test func partialAcceptance() async throws {
        let vocabSize = 5
        // Draft predicts [1, 2, 3]
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        // Verifier agrees on first two, disagrees on third
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1), // accept
            makeLogits(vocabSize: vocabSize, peakAt: 2), // accept
            makeLogits(vocabSize: vocabSize, peakAt: 0), // reject token 3, pick 0
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)

        let decoder = SpeculativeDecoder(
            draftModel: draft,
            verificationModel: verifier,
            draftTokenCount: 3,
            samplingPipeline: .greedy
        )

        let result = try await decoder.decodeStep(inputTokens: [0])
        // First 2 accepted, third rejected — verifier's token used
        #expect(result.acceptedTokens == [1, 2, 0])
        #expect(abs(result.acceptanceRate - 2.0 / 3.0) < 1e-6)
    }

    @Test func draftTokenCountRespected() async throws {
        let vocabSize = 3
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
        ]
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 0),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)

        let decoder = SpeculativeDecoder(
            draftModel: draft,
            verificationModel: verifier,
            draftTokenCount: 2,
            samplingPipeline: .greedy
        )

        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.acceptedTokens.count <= 3) // at most draft + 1
    }

    @Test func decodingResultProperties() async throws {
        let vocabSize = 3
        let draftLogits = [makeLogits(vocabSize: vocabSize, peakAt: 1)]
        let verifyLogits = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)

        let decoder = SpeculativeDecoder(
            draftModel: draft,
            verificationModel: verifier,
            draftTokenCount: 1,
            samplingPipeline: .greedy
        )

        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.draftTokenCount == 1)
        #expect(result.acceptedTokens.count >= 1)
    }
}

// Helper for async map
extension Array {
    fileprivate func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var results = [T]()
        results.reserveCapacity(count)
        for element in self {
            try await results.append(transform(element))
        }
        return results
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter SpeculativeDecodingTests 2>&1`
Expected: FAIL — `SpeculativeDecoder`, `SpeculativeModel` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunnerCore/Generation/SpeculativeDecoder.swift
import Foundation

/// Protocol for models that can participate in speculative decoding.
public protocol SpeculativeModel: Sendable {
    /// Compute logits for the next token given a sequence.
    func logits(for tokenIDs: [Int]) async throws -> [Float]

    /// Batch verification: compute logits for multiple sequences in parallel.
    /// Default implementation calls `logits(for:)` sequentially.
    func batchLogits(for sequences: [[Int]]) async throws -> [[Float]]
}

extension SpeculativeModel {
    public func batchLogits(for sequences: [[Int]]) async throws -> [[Float]] {
        var results = [[Float]]()
        results.reserveCapacity(sequences.count)
        for seq in sequences {
            try await results.append(logits(for: seq))
        }
        return results
    }
}

/// Result of a single speculative decoding step.
public struct SpeculativeDecodingResult: Sendable {
    /// Tokens accepted in this step (includes verified draft tokens + optional bonus token).
    public let acceptedTokens: [Int]

    /// Number of tokens the draft model proposed.
    public let draftTokenCount: Int

    /// Fraction of draft tokens that were accepted (0.0 to 1.0).
    public var acceptanceRate: Float {
        guard draftTokenCount > 0 else { return 0.0 }
        let accepted = min(acceptedTokens.count - 1, draftTokenCount) // -1 for bonus token
        return Float(max(accepted, 0)) / Float(draftTokenCount)
    }
}

/// Speculative decoding: uses a fast draft model to propose N token candidates,
/// then verifies them in parallel with the main model.
///
/// Algorithm:
/// 1. Draft model generates N candidate tokens autoregressively.
/// 2. Verification model scores all N+1 prefixes in a single forward pass.
/// 3. Accept tokens left-to-right while draft and verifier agree.
/// 4. On first disagreement, use verifier's token. On full agreement, emit bonus token.
public struct SpeculativeDecoder: Sendable {
    private let draftModel: any SpeculativeModel
    private let verificationModel: any SpeculativeModel
    private let draftTokenCount: Int
    private let samplingPipeline: SamplingPipeline

    public init(
        draftModel: any SpeculativeModel,
        verificationModel: any SpeculativeModel,
        draftTokenCount: Int = 4,
        samplingPipeline: SamplingPipeline = .greedy
    ) {
        precondition(draftTokenCount >= 1, "Must draft at least 1 token")
        self.draftModel = draftModel
        self.verificationModel = verificationModel
        self.draftTokenCount = draftTokenCount
        self.samplingPipeline = samplingPipeline
    }

    /// Run one speculative decoding step.
    ///
    /// - Parameter inputTokens: The current token sequence (context).
    /// - Returns: A result containing accepted tokens and statistics.
    public func decodeStep(inputTokens: [Int]) async throws -> SpeculativeDecodingResult {
        // Step 1: Draft model generates N candidate tokens
        var draftTokens = [Int]()
        var currentSequence = inputTokens
        for _ in 0..<draftTokenCount {
            let logits = try await draftModel.logits(for: currentSequence)
            let token = samplingPipeline.sample(logits: logits)
            draftTokens.append(token)
            currentSequence.append(token)
        }

        // Step 2: Build verification sequences (each prefix of draft tokens)
        var verificationSequences = [[Int]]()
        for i in 0...draftTokenCount {
            var seq = inputTokens
            seq.append(contentsOf: draftTokens.prefix(i))
            verificationSequences.append(seq)
        }

        // Step 3: Verify all prefixes
        let allVerifyLogits = try await verificationModel.batchLogits(for: verificationSequences)

        // Step 4: Accept/reject left-to-right
        var acceptedTokens = [Int]()
        for i in 0..<draftTokenCount {
            let verifierLogits = allVerifyLogits[i]
            let verifierToken = samplingPipeline.sample(logits: verifierLogits)

            if verifierToken == draftTokens[i] {
                // Draft token accepted
                acceptedTokens.append(draftTokens[i])
            } else {
                // Draft token rejected — use verifier's token instead
                acceptedTokens.append(verifierToken)
                return SpeculativeDecodingResult(
                    acceptedTokens: acceptedTokens,
                    draftTokenCount: draftTokenCount
                )
            }
        }

        // All draft tokens accepted — get bonus token from last verification logits
        if allVerifyLogits.count > draftTokenCount {
            let bonusLogits = allVerifyLogits[draftTokenCount]
            let bonusToken = samplingPipeline.sample(logits: bonusLogits)
            acceptedTokens.append(bonusToken)
        }

        return SpeculativeDecodingResult(
            acceptedTokens: acceptedTokens,
            draftTokenCount: draftTokenCount
        )
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter SpeculativeDecodingTests 2>&1`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Generation/SpeculativeDecoder.swift \
      Tests/EdgeRunnerCoreTests/SpeculativeDecodingTests.swift
git commit -m "feat(m4): add speculative decoding with draft-verify pipeline

Draft model proposes N candidates, verification model accepts/rejects
left-to-right. Bonus token on full acceptance. SpeculativeModel protocol
with batch verification support."
```

---

## Task 5: Structured Generation (Codable Schema Extraction)

**Files:**
- Create: `Sources/EdgeRunnerCore/StructuredGeneration/JSONSchemaExtractor.swift`
- Create: `Sources/EdgeRunnerCore/StructuredGeneration/ConstrainedDecoder.swift`
- Create: `Sources/EdgeRunnerCore/StructuredGeneration/GrammarState.swift`
- Create: `Sources/EdgeRunnerCore/StructuredGeneration/StructuredGenerator.swift`
- Test: `Tests/EdgeRunnerCoreTests/StructuredGenerationTests.swift`

> **Note:** `@Generable` is Apple's proprietary macro available in the Foundation Models framework.
> EdgeRunner provides equivalent functionality via `JSONSchemaExtractor` working with standard
> `Codable` types, enabling the same structured output pattern without requiring Apple's macro.

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/StructuredGenerationTests.swift
import Testing
import Foundation
@testable import EdgeRunnerCore

// MARK: - Test types for structured generation

struct PersonInfo: Codable, Equatable, Sendable {
    let name: String
    let age: Int
}

struct WeatherReport: Codable, Equatable, Sendable {
    let city: String
    let temperature: Double
    let isRaining: Bool
}

struct NestedType: Codable, Equatable, Sendable {
    struct Address: Codable, Equatable, Sendable {
        let street: String
        let zip: String
    }
    let name: String
    let address: Address
}

struct ArrayType: Codable, Equatable, Sendable {
    let tags: [String]
    let scores: [Int]
}

struct OptionalType: Codable, Equatable, Sendable {
    let required: String
    let optional: String?
}

enum Status: String, Codable, Sendable {
    case active
    case inactive
    case pending
}

struct EnumType: Codable, Equatable, Sendable {
    let name: String
    let status: Status
}

@Suite("JSONSchemaExtractor")
struct JSONSchemaExtractorTests {

    @Test func extractSimpleStruct() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: PersonInfo.self)
        #expect(schema.type == .object)
        #expect(schema.properties?.count == 2)
        #expect(schema.properties?["name"]?.type == .string)
        #expect(schema.properties?["age"]?.type == .integer)
        #expect(schema.required?.contains("name") == true)
        #expect(schema.required?.contains("age") == true)
    }

    @Test func extractWithBoolAndDouble() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: WeatherReport.self)
        #expect(schema.properties?["city"]?.type == .string)
        #expect(schema.properties?["temperature"]?.type == .number)
        #expect(schema.properties?["isRaining"]?.type == .boolean)
    }

    @Test func extractNestedObject() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: NestedType.self)
        #expect(schema.properties?["name"]?.type == .string)
        let addressSchema = schema.properties?["address"]
        #expect(addressSchema?.type == .object)
        #expect(addressSchema?.properties?["street"]?.type == .string)
        #expect(addressSchema?.properties?["zip"]?.type == .string)
    }

    @Test func extractArrayTypes() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: ArrayType.self)
        #expect(schema.properties?["tags"]?.type == .array)
        #expect(schema.properties?["tags"]?.items?.type == .string)
        #expect(schema.properties?["scores"]?.type == .array)
        #expect(schema.properties?["scores"]?.items?.type == .integer)
    }

    @Test func extractOptionalFields() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: OptionalType.self)
        #expect(schema.required?.contains("required") == true)
        #expect(schema.required?.contains("optional") == false)
    }

    @Test func schemaToJSON() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: PersonInfo.self)
        let json = try schema.toJSON()
        #expect(json.contains("\"type\":\"object\"") || json.contains("\"type\": \"object\""))
        #expect(json.contains("name"))
        #expect(json.contains("age"))
    }
}

@Suite("GrammarState")
struct GrammarStateTests {

    @Test func initialStateExpectsOpenBrace() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        let state = GrammarState(schema: schema)
        let allowed = state.allowedNextCharacters()
        #expect(allowed.contains("{"))
        #expect(!allowed.contains("}"))
    }

    @Test func afterOpenBraceExpectsKey() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        var state = GrammarState(schema: schema)
        state.advance(with: "{")
        let allowed = state.allowedNextCharacters()
        #expect(allowed.contains("\""))
    }

    @Test func validateCompleteJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
            "age": JSONSchema(type: .integer),
        ], required: ["name", "age"])
        let json = #"{"name":"Alice","age":30}"#
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(isValid)
    }

    @Test func rejectInvalidJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        let json = #"{"name": 42}"# // name should be string
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(!isValid)
    }

    @Test func validateNestedJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
            "address": JSONSchema(type: .object, properties: [
                "street": JSONSchema(type: .string),
            ], required: ["street"]),
        ], required: ["name", "address"])
        let json = #"{"name":"Bob","address":{"street":"123 Main"}}"#
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(isValid)
    }
}

@Suite("ConstrainedDecoder")
struct ConstrainedDecoderTests {

    @Test func maskDisallowedTokens() {
        let vocab = ["a", "b", "{", "}", "\"", ":", ",", "1", " ", "\n"]
        let schema = JSONSchema(type: .object, properties: [
            "x": JSONSchema(type: .string),
        ], required: ["x"])
        let state = GrammarState(schema: schema)

        let decoder = ConstrainedDecoder(vocabulary: vocab)
        let mask = decoder.computeMask(for: state)

        #expect(mask.count == vocab.count)
        // At start, only "{" should be allowed
        #expect(mask[2] == true)  // "{"
        #expect(mask[0] == false) // "a"
    }

    @Test func applyMaskToLogits() {
        let vocab = ["{", "}", "a"]
        let decoder = ConstrainedDecoder(vocabulary: vocab)
        let mask = [true, false, false]
        let logits: [Float] = [5.0, 10.0, 3.0]
        let masked = decoder.applyMask(mask, to: logits)
        #expect(masked[0] == 5.0)
        #expect(masked[1] == -.infinity)
        #expect(masked[2] == -.infinity)
    }
}

@Suite("StructuredGenerator")
struct StructuredGeneratorTests {

    @Test func parseValidJSON() throws {
        let json = #"{"name":"Alice","age":30}"#
        let result: PersonInfo = try StructuredGenerator.parse(json: json)
        #expect(result.name == "Alice")
        #expect(result.age == 30)
    }

    @Test func parseNestedJSON() throws {
        let json = #"{"name":"Bob","address":{"street":"123 Main","zip":"12345"}}"#
        let result: NestedType = try StructuredGenerator.parse(json: json)
        #expect(result.name == "Bob")
        #expect(result.address.street == "123 Main")
        #expect(result.address.zip == "12345")
    }

    @Test func parseWithArrays() throws {
        let json = #"{"tags":["swift","metal"],"scores":[95,87]}"#
        let result: ArrayType = try StructuredGenerator.parse(json: json)
        #expect(result.tags == ["swift", "metal"])
        #expect(result.scores == [95, 87])
    }

    @Test func invalidJSONThrows() throws {
        let json = "not valid json"
        #expect(throws: (any Error).self) {
            let _: PersonInfo = try StructuredGenerator.parse(json: json)
        }
    }

    @Test func extractJSONFromModelOutput() throws {
        let output = """
        Here is the result:
        ```json
        {"name":"Alice","age":25}
        ```
        That's the answer.
        """
        let json = try StructuredGenerator.extractJSON(from: output)
        #expect(json == #"{"name":"Alice","age":25}"#)
    }

    @Test func extractJSONBracketMatching() throws {
        let output = #"Some text {"name":"Bob","age":30} more text"#
        let json = try StructuredGenerator.extractJSON(from: output)
        #expect(json.contains("Bob"))
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "JSONSchemaExtractorTests|GrammarStateTests|ConstrainedDecoderTests|StructuredGeneratorTests" 2>&1`
Expected: FAIL — types not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunnerCore/StructuredGeneration/JSONSchemaExtractor.swift
import Foundation

/// JSON Schema type identifiers.
public enum JSONSchemaType: String, Codable, Sendable {
    case object
    case array
    case string
    case integer
    case number
    case boolean
    case null
}

/// A simplified JSON Schema representation for constrained decoding.
public struct JSONSchema: Sendable {
    public let type: JSONSchemaType
    public let properties: [String: JSONSchema]?
    public let required: [String]?
    public let items: JSONSchema?
    public let enumValues: [String]?
    public let description: String?

    public init(
        type: JSONSchemaType,
        properties: [String: JSONSchema]? = nil,
        required: [String]? = nil,
        items: JSONSchema? = nil,
        enumValues: [String]? = nil,
        description: String? = nil
    ) {
        self.type = type
        self.properties = properties
        self.required = required
        self.items = items
        self.enumValues = enumValues
        self.description = description
    }

    /// Serialize to JSON string.
    public func toJSON() throws -> String {
        let dict = toDictionary()
        let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        guard let str = String(data: data, encoding: .utf8) else {
            throw GenerationError.structuredOutputFailed(reason: "Failed to serialize schema")
        }
        return str
    }

    private func toDictionary() -> [String: Any] {
        var dict = [String: Any]()
        dict["type"] = type.rawValue

        if let props = properties {
            var propsDict = [String: Any]()
            for (key, schema) in props {
                propsDict[key] = schema.toDictionary()
            }
            dict["properties"] = propsDict
        }

        if let req = required {
            dict["required"] = req
        }

        if let items = items {
            dict["items"] = items.toDictionary()
        }

        if let enums = enumValues {
            dict["enum"] = enums
        }

        return dict
    }
}

/// Extracts JSON Schema from Swift Decodable types using Mirror-based reflection.
public enum JSONSchemaExtractor {

    /// Extract a JSON Schema from a Decodable type.
    ///
    /// Uses a sentinel decoder to introspect the type's coding keys and value types.
    public static func extractSchema<T: Decodable>(for type: T.Type) throws -> JSONSchema {
        let decoder = SchemaIntrospectionDecoder()
        _ = try? T(from: decoder)
        return decoder.buildSchema()
    }
}

// MARK: - Schema Introspection Decoder

/// A decoder that doesn't actually decode data — it records the structure
/// of the type being decoded to build a JSON Schema.
private final class SchemaIntrospectionDecoder: Decoder {
    var codingPath: [CodingKey] = []
    var userInfo: [CodingUserInfoKey: Any] = [:]
    var discoveredProperties: [(key: String, schema: JSONSchema, isOptional: Bool)] = []

    func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
        return KeyedDecodingContainer(SchemaKeyedContainer<Key>(decoder: self))
    }

    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: self)
    }

    func singleValueContainer() throws -> SingleValueDecodingContainer {
        SchemaSingleValueContainer(decoder: self)
    }

    func buildSchema() -> JSONSchema {
        let properties = Dictionary(
            uniqueKeysWithValues: discoveredProperties.map { ($0.key, $0.schema) }
        )
        let required = discoveredProperties.filter { !$0.isOptional }.map(\.key)
        return JSONSchema(
            type: .object,
            properties: properties,
            required: required.isEmpty ? nil : required
        )
    }
}

private struct SchemaKeyedContainer<Key: CodingKey>: KeyedDecodingContainerProtocol {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }
    var allKeys: [Key] { [] }

    func contains(_ key: Key) -> Bool { true }

    func decodeNil(forKey key: Key) throws -> Bool {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .null), true))
        return true
    }

    func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .boolean), false))
        return false
    }

    func decode(_ type: String.Type, forKey key: Key) throws -> String {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .string), false))
        return ""
    }

    func decode(_ type: Double.Type, forKey key: Key) throws -> Double {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .number), false))
        return 0.0
    }

    func decode(_ type: Float.Type, forKey key: Key) throws -> Float {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .number), false))
        return 0.0
    }

    func decode(_ type: Int.Type, forKey key: Key) throws -> Int {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false))
        return 0
    }

    func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
        // Try to determine the schema for nested types
        let schema = inferSchema(for: T.self)
        let isOptional = isOptionalType(T.self)
        decoder.discoveredProperties.append((key.stringValue, schema, isOptional))
        throw SchemaExtractionComplete()
    }

    func decodeIfPresent<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T? {
        let schema = inferSchema(for: T.self)
        decoder.discoveredProperties.append((key.stringValue, schema, true))
        return nil
    }

    func nestedContainer<NestedKey: CodingKey>(
        keyedBy type: NestedKey.Type, forKey key: Key
    ) throws -> KeyedDecodingContainer<NestedKey> {
        KeyedDecodingContainer(SchemaKeyedContainer<NestedKey>(decoder: decoder))
    }

    func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: decoder)
    }

    func superDecoder() throws -> Decoder { decoder }
    func superDecoder(forKey key: Key) throws -> Decoder { decoder }
}

private struct SchemaUnkeyedContainer: UnkeyedDecodingContainer {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }
    var count: Int? { 0 }
    var isAtEnd: Bool { true }
    var currentIndex: Int { 0 }

    mutating func decodeNil() throws -> Bool { true }
    mutating func decode(_ type: Bool.Type) throws -> Bool { false }
    mutating func decode(_ type: String.Type) throws -> String { "" }
    mutating func decode(_ type: Double.Type) throws -> Double { 0 }
    mutating func decode(_ type: Float.Type) throws -> Float { 0 }
    mutating func decode(_ type: Int.Type) throws -> Int { 0 }
    mutating func decode(_ type: Int8.Type) throws -> Int8 { 0 }
    mutating func decode(_ type: Int16.Type) throws -> Int16 { 0 }
    mutating func decode(_ type: Int32.Type) throws -> Int32 { 0 }
    mutating func decode(_ type: Int64.Type) throws -> Int64 { 0 }
    mutating func decode(_ type: UInt.Type) throws -> UInt { 0 }
    mutating func decode(_ type: UInt8.Type) throws -> UInt8 { 0 }
    mutating func decode(_ type: UInt16.Type) throws -> UInt16 { 0 }
    mutating func decode(_ type: UInt32.Type) throws -> UInt32 { 0 }
    mutating func decode(_ type: UInt64.Type) throws -> UInt64 { 0 }
    mutating func decode<T: Decodable>(_ type: T.Type) throws -> T {
        throw SchemaExtractionComplete()
    }
    mutating func nestedContainer<NestedKey: CodingKey>(
        keyedBy type: NestedKey.Type
    ) throws -> KeyedDecodingContainer<NestedKey> {
        KeyedDecodingContainer(SchemaKeyedContainer<NestedKey>(decoder: decoder))
    }
    mutating func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: decoder)
    }
    mutating func superDecoder() throws -> Decoder { decoder }
}

private struct SchemaSingleValueContainer: SingleValueDecodingContainer {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }

    func decodeNil() -> Bool { true }
    func decode(_ type: Bool.Type) throws -> Bool { false }
    func decode(_ type: String.Type) throws -> String { "" }
    func decode(_ type: Double.Type) throws -> Double { 0 }
    func decode(_ type: Float.Type) throws -> Float { 0 }
    func decode(_ type: Int.Type) throws -> Int { 0 }
    func decode(_ type: Int8.Type) throws -> Int8 { 0 }
    func decode(_ type: Int16.Type) throws -> Int16 { 0 }
    func decode(_ type: Int32.Type) throws -> Int32 { 0 }
    func decode(_ type: Int64.Type) throws -> Int64 { 0 }
    func decode(_ type: UInt.Type) throws -> UInt { 0 }
    func decode(_ type: UInt8.Type) throws -> UInt8 { 0 }
    func decode(_ type: UInt16.Type) throws -> UInt16 { 0 }
    func decode(_ type: UInt32.Type) throws -> UInt32 { 0 }
    func decode(_ type: UInt64.Type) throws -> UInt64 { 0 }
    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        throw SchemaExtractionComplete()
    }
}

private struct SchemaExtractionComplete: Error {}

// MARK: - Type Inference Helpers

private func inferSchema(for type: Any.Type) -> JSONSchema {
    switch type {
    case is String.Type:
        return JSONSchema(type: .string)
    case is Int.Type, is Int8.Type, is Int16.Type, is Int32.Type, is Int64.Type,
         is UInt.Type, is UInt8.Type, is UInt16.Type, is UInt32.Type, is UInt64.Type:
        return JSONSchema(type: .integer)
    case is Double.Type, is Float.Type:
        return JSONSchema(type: .number)
    case is Bool.Type:
        return JSONSchema(type: .boolean)
    default:
        // Check for arrays
        if let arrayElementType = arrayElementType(type) {
            let itemSchema = inferSchema(for: arrayElementType)
            return JSONSchema(type: .array, items: itemSchema)
        }

        // Check for optionals
        if let wrappedType = optionalWrappedType(type) {
            return inferSchema(for: wrappedType)
        }

        // Assume nested object — try to extract its schema
        if let decodableType = type as? Decodable.Type {
            let decoder = SchemaIntrospectionDecoder()
            _ = try? decodableType.init(from: decoder)
            return decoder.buildSchema()
        }

        return JSONSchema(type: .object)
    }
}

private func isOptionalType(_ type: Any.Type) -> Bool {
    let mirror = Mirror(reflecting: type)
    return String(describing: type).hasPrefix("Optional<")
}

private func optionalWrappedType(_ type: Any.Type) -> Any.Type? {
    let description = String(describing: type)
    guard description.hasPrefix("Optional<") else { return nil }
    // Cannot reliably extract the wrapped type at runtime, return nil
    return nil
}

private func arrayElementType(_ type: Any.Type) -> Any.Type? {
    let description = String(describing: type)
    if description.hasPrefix("Array<String>") { return String.self }
    if description.hasPrefix("Array<Int>") { return Int.self }
    if description.hasPrefix("Array<Double>") { return Double.self }
    if description.hasPrefix("Array<Float>") { return Float.self }
    if description.hasPrefix("Array<Bool>") { return Bool.self }
    return nil
}
```

```swift
// Sources/EdgeRunnerCore/StructuredGeneration/GrammarState.swift
import Foundation

/// Tracks the state of JSON generation for grammar-guided constrained decoding.
///
/// Implements a simple state machine that determines which characters/tokens
/// are valid at each position in a JSON document conforming to a schema.
public struct GrammarState: Sendable {
    public enum ParseState: Sendable {
        case expectObjectOpen
        case expectKeyOrClose
        case expectColon
        case expectValue
        case expectCommaOrClose
        case expectArrayOpen
        case expectArrayValueOrClose
        case expectArrayCommaOrClose
        case complete
    }

    public let schema: JSONSchema
    public private(set) var state: ParseState
    public private(set) var buffer: String
    public private(set) var depth: Int

    public init(schema: JSONSchema) {
        self.schema = schema
        self.state = schema.type == .array ? .expectArrayOpen : .expectObjectOpen
        self.buffer = ""
        self.depth = 0
    }

    /// Returns the set of characters that are valid at the current position.
    public func allowedNextCharacters() -> Set<String> {
        switch state {
        case .expectObjectOpen:
            return ["{"]
        case .expectKeyOrClose:
            return ["\"", "}"]
        case .expectColon:
            return [":"]
        case .expectValue:
            // Allow any value start: string, number, bool, null, object, array
            return ["\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "t", "f", "n", "{", "["]
        case .expectCommaOrClose:
            return [",", "}"]
        case .expectArrayOpen:
            return ["["]
        case .expectArrayValueOrClose:
            return ["\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "t", "f", "n", "{", "[", "]"]
        case .expectArrayCommaOrClose:
            return [",", "]"]
        case .complete:
            return []
        }
    }

    /// Advance the state machine by one character.
    public mutating func advance(with character: String) {
        buffer += character

        switch (state, character) {
        case (.expectObjectOpen, "{"):
            depth += 1
            state = .expectKeyOrClose
        case (.expectKeyOrClose, "\""):
            state = .expectColon // simplified: skip key content
        case (.expectKeyOrClose, "}"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectColon, ":"):
            state = .expectValue
        case (.expectValue, "\""), (.expectValue, _) where character.first?.isNumber == true
             || character == "-" || character == "t" || character == "f" || character == "n":
            state = .expectCommaOrClose // simplified: skip value content
        case (.expectValue, "{"):
            depth += 1
            state = .expectKeyOrClose
        case (.expectValue, "["):
            depth += 1
            state = .expectArrayValueOrClose
        case (.expectCommaOrClose, ","):
            state = .expectKeyOrClose
        case (.expectCommaOrClose, "}"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectArrayOpen, "["):
            depth += 1
            state = .expectArrayValueOrClose
        case (.expectArrayValueOrClose, "]"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectArrayValueOrClose, _):
            state = .expectArrayCommaOrClose
        case (.expectArrayCommaOrClose, ","):
            state = .expectArrayValueOrClose
        case (.expectArrayCommaOrClose, "]"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        default:
            break // stay in current state for content characters
        }
    }

    /// Validate that a complete JSON string conforms to the schema.
    public static func validate(json: String, against schema: JSONSchema) -> Bool {
        guard let data = json.data(using: .utf8) else { return false }
        guard let parsed = try? JSONSerialization.jsonObject(with: data) else { return false }
        return validateValue(parsed, against: schema)
    }

    private static func validateValue(_ value: Any, against schema: JSONSchema) -> Bool {
        switch schema.type {
        case .object:
            guard let dict = value as? [String: Any] else { return false }
            // Check required fields
            if let required = schema.required {
                for key in required {
                    guard dict[key] != nil else { return false }
                }
            }
            // Check property types
            if let properties = schema.properties {
                for (key, propSchema) in properties {
                    if let propValue = dict[key] {
                        if !validateValue(propValue, against: propSchema) {
                            return false
                        }
                    }
                }
            }
            return true

        case .array:
            guard let array = value as? [Any] else { return false }
            if let itemSchema = schema.items {
                for item in array {
                    if !validateValue(item, against: itemSchema) {
                        return false
                    }
                }
            }
            return true

        case .string:
            return value is String

        case .integer:
            if value is Int { return true }
            if let num = value as? NSNumber {
                return CFNumberIsFloatType(num as CFNumber) == false
            }
            return false

        case .number:
            return value is Double || value is Float || value is Int || value is NSNumber

        case .boolean:
            if let num = value as? NSNumber {
                return CFGetTypeID(num) == CFBooleanGetTypeID()
            }
            return false

        case .null:
            return value is NSNull
        }
    }
}
```

```swift
// Sources/EdgeRunnerCore/StructuredGeneration/ConstrainedDecoder.swift
import Foundation

/// Applies grammar constraints to logits during generation,
/// ensuring the model only produces tokens that are valid JSON
/// at the current position.
public struct ConstrainedDecoder: Sendable {
    private let vocabulary: [String]

    public init(vocabulary: [String]) {
        self.vocabulary = vocabulary
    }

    /// Compute a boolean mask indicating which tokens are valid
    /// given the current grammar state.
    public func computeMask(for state: GrammarState) -> [Bool] {
        let allowed = state.allowedNextCharacters()
        return vocabulary.map { token in
            guard let firstChar = token.first else { return false }
            return allowed.contains(String(firstChar))
        }
    }

    /// Apply a boolean mask to logits: set disallowed tokens to -infinity.
    public func applyMask(_ mask: [Bool], to logits: [Float]) -> [Float] {
        precondition(mask.count == logits.count, "Mask and logits must have same length")
        var result = logits
        for i in 0..<result.count {
            if !mask[i] {
                result[i] = -.infinity
            }
        }
        return result
    }
}
```

```swift
// Sources/EdgeRunnerCore/StructuredGeneration/StructuredGenerator.swift
import Foundation

/// Utilities for structured (typed) generation from model output.
public enum StructuredGenerator {

    /// Parse a JSON string into a Decodable type.
    public static func parse<T: Decodable>(json: String) throws -> T {
        guard let data = json.data(using: .utf8) else {
            throw GenerationError.structuredOutputFailed(reason: "Invalid UTF-8 in JSON string")
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw GenerationError.structuredOutputFailed(
                reason: "JSON decode failed: \(error.localizedDescription)"
            )
        }
    }

    /// Extract a JSON object or array from model output text.
    ///
    /// Handles common patterns:
    /// - JSON in a ```json code block
    /// - Raw JSON starting with { or [
    /// - JSON embedded in surrounding text
    public static func extractJSON(from text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Try code block extraction first
        if let codeBlockJSON = extractFromCodeBlock(trimmed) {
            return codeBlockJSON
        }

        // Try bracket matching
        if let bracketJSON = extractByBracketMatching(trimmed) {
            return bracketJSON
        }

        throw GenerationError.structuredOutputFailed(
            reason: "No valid JSON found in model output"
        )
    }

    private static func extractFromCodeBlock(_ text: String) -> String? {
        // Match ```json ... ``` or ``` ... ```
        let patterns = [
            "```json\\s*\\n([\\s\\S]*?)\\n\\s*```",
            "```\\s*\\n([\\s\\S]*?)\\n\\s*```",
        ]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(
                   in: text,
                   range: NSRange(text.startIndex..., in: text)
               ),
               let range = Range(match.range(at: 1), in: text) {
                return String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return nil
    }

    private static func extractByBracketMatching(_ text: String) -> String? {
        // Find first { or [
        guard let startIndex = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            return nil
        }

        let openBracket: Character = text[startIndex]
        let closeBracket: Character = openBracket == "{" ? "}" : "]"

        var depth = 0
        var inString = false
        var escaped = false

        for index in text[startIndex...].indices {
            let char = text[index]

            if escaped {
                escaped = false
                continue
            }

            if char == "\\" && inString {
                escaped = true
                continue
            }

            if char == "\"" {
                inString.toggle()
                continue
            }

            if !inString {
                if char == openBracket {
                    depth += 1
                } else if char == closeBracket {
                    depth -= 1
                    if depth == 0 {
                        return String(text[startIndex...index])
                    }
                }
            }
        }

        return nil
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "JSONSchemaExtractorTests|GrammarStateTests|ConstrainedDecoderTests|StructuredGeneratorTests" 2>&1`
Expected: All 17 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/StructuredGeneration/JSONSchemaExtractor.swift \
      Sources/EdgeRunnerCore/StructuredGeneration/ConstrainedDecoder.swift \
      Sources/EdgeRunnerCore/StructuredGeneration/GrammarState.swift \
      Sources/EdgeRunnerCore/StructuredGeneration/StructuredGenerator.swift \
      Tests/EdgeRunnerCoreTests/StructuredGenerationTests.swift
git commit -m "feat(m4): add structured generation with JSON schema extraction and constrained decoding

Mirror-based JSON schema extraction from Decodable types.
Grammar-guided state machine for token-by-token validation.
ConstrainedDecoder masks logits to enforce valid JSON output.
StructuredGenerator parses and extracts JSON from model output."
```

---

## Task 6: Tool Calling

**Files:**
- Create: `Sources/EdgeRunner/ToolCalling/EdgeRunnerTool.swift`
- Create: `Sources/EdgeRunner/ToolCalling/ToolChoice.swift`
- Create: `Sources/EdgeRunner/ToolCalling/ToolCallParser.swift`
- Create: `Sources/EdgeRunner/ToolCalling/ToolExecutor.swift`
- Test: `Tests/EdgeRunnerTests/ToolCallingTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/ToolCallingTests.swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Mock Tools

struct WeatherTool: EdgeRunnerTool {
    static let name = "get_weather"
    static let description = "Get the current weather for a city."
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "city", type: .string, description: "City name", required: true),
        ToolParameter(name: "units", type: .string, description: "Temperature units", required: false),
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        guard let city = arguments["city"] as? String else {
            throw GenerationError.toolCallFailed(name: Self.name, reason: "Missing city parameter")
        }
        let units = arguments["units"] as? String ?? "celsius"
        return "{\"city\": \"\(city)\", \"temperature\": 22, \"units\": \"\(units)\"}"
    }
}

struct CalculatorTool: EdgeRunnerTool {
    static let name = "calculate"
    static let description = "Evaluate a math expression."
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "expression", type: .string, description: "Math expression", required: true),
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        guard let expr = arguments["expression"] as? String else {
            throw GenerationError.toolCallFailed(name: Self.name, reason: "Missing expression")
        }
        return "{\"result\": \"\(expr) = 42\"}"
    }
}

@Suite("EdgeRunnerTool Protocol")
struct EdgeRunnerToolProtocolTests {

    @Test func toolMetadata() {
        #expect(WeatherTool.name == "get_weather")
        #expect(WeatherTool.description == "Get the current weather for a city.")
        #expect(WeatherTool.parameters.count == 2)
        #expect(WeatherTool.parameters[0].name == "city")
        #expect(WeatherTool.parameters[0].required == true)
        #expect(WeatherTool.parameters[1].required == false)
    }

    @Test func toolParameterTypes() {
        let param = ToolParameter(name: "count", type: .integer, description: "Number of items", required: true)
        #expect(param.type == .integer)
        #expect(param.name == "count")
    }

    @Test func toolInvocation() async throws {
        let tool = WeatherTool()
        let result = try await tool.invoke(arguments: ["city": "London"])
        #expect(result.contains("London"))
        #expect(result.contains("22"))
    }

    @Test func toolInvocationWithOptionalParam() async throws {
        let tool = WeatherTool()
        let result = try await tool.invoke(arguments: ["city": "Tokyo", "units": "fahrenheit"])
        #expect(result.contains("Tokyo"))
        #expect(result.contains("fahrenheit"))
    }

    @Test func toolInvocationMissingRequiredParam() async throws {
        let tool = WeatherTool()
        await #expect(throws: GenerationError.self) {
            _ = try await tool.invoke(arguments: [:])
        }
    }

    @Test func toolJSONSchema() {
        let schema = WeatherTool.jsonSchema
        #expect(schema.contains("get_weather"))
        #expect(schema.contains("city"))
        #expect(schema.contains("string"))
    }
}

@Suite("ToolChoice")
struct ToolChoiceTests {

    @Test func autoChoice() {
        let choice = ToolChoice.auto
        #expect(choice == .auto)
    }

    @Test func requiredChoice() {
        let choice = ToolChoice.required
        #expect(choice == .required)
    }

    @Test func noneChoice() {
        let choice = ToolChoice.none
        #expect(choice == .none)
    }

    @Test func specificToolChoice() {
        let choice = ToolChoice.specific("get_weather")
        if case .specific(let name) = choice {
            #expect(name == "get_weather")
        } else {
            Issue.record("Expected .specific")
        }
    }
}

@Suite("ToolCallParser")
struct ToolCallParserTests {

    @Test func parseStandardFormat() throws {
        let output = """
        <tool_call>
        {"name": "get_weather", "arguments": {"city": "London"}}
        </tool_call>
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["city"] as? String == "London")
    }

    @Test func parseMultipleToolCalls() throws {
        let output = """
        <tool_call>
        {"name": "get_weather", "arguments": {"city": "London"}}
        </tool_call>
        <tool_call>
        {"name": "calculate", "arguments": {"expression": "2+2"}}
        </tool_call>
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 2)
        #expect(calls[0].name == "get_weather")
        #expect(calls[1].name == "calculate")
    }

    @Test func parseNoToolCalls() throws {
        let output = "Just a regular response with no tool calls."
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.isEmpty)
    }

    @Test func parseFunctionCallFormat() throws {
        let output = """
        {"function_call": {"name": "get_weather", "arguments": "{\\\"city\\\": \\\"Paris\\\"}"}}
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
    }

    @Test func containsToolCall() {
        #expect(ToolCallParser.containsToolCall(in: "<tool_call>{}</tool_call>"))
        #expect(!ToolCallParser.containsToolCall(in: "Just text"))
    }
}

@Suite("ToolExecutor")
struct ToolExecutorTests {

    @Test func executeRegisteredTool() async throws {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let call = ToolCall(name: "get_weather", arguments: ["city": "Berlin"])
        let result = try await executor.execute(call)
        #expect(result.contains("Berlin"))
    }

    @Test func executeUnknownToolThrows() async throws {
        let executor = ToolExecutor(tools: [WeatherTool()])
        let call = ToolCall(name: "unknown_tool", arguments: [:])
        await #expect(throws: GenerationError.self) {
            _ = try await executor.execute(call)
        }
    }

    @Test func executeMultipleCalls() async throws {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let calls = [
            ToolCall(name: "get_weather", arguments: ["city": "NYC"]),
            ToolCall(name: "calculate", arguments: ["expression": "1+1"]),
        ]
        let results = try await executor.executeAll(calls)
        #expect(results.count == 2)
        #expect(results[0].contains("NYC"))
        #expect(results[1].contains("1+1"))
    }

    @Test func toolListGeneration() {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let toolList = executor.toolDescriptions()
        #expect(toolList.contains("get_weather"))
        #expect(toolList.contains("calculate"))
    }

    @Test func shouldCallToolWithAutoChoice() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .auto, modelOutput: "<tool_call>") == true)
        #expect(executor.shouldAttemptToolCall(choice: .none, modelOutput: "<tool_call>") == false)
        #expect(executor.shouldAttemptToolCall(choice: .required, modelOutput: "anything") == true)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "EdgeRunnerToolProtocolTests|ToolChoiceTests|ToolCallParserTests|ToolExecutorTests" 2>&1`
Expected: FAIL — types not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/ToolCalling/EdgeRunnerTool.swift
import Foundation

/// Type of a tool parameter.
public enum ToolParameterType: String, Sendable {
    case string
    case integer
    case number
    case boolean
    case array
    case object
}

/// Describes a single parameter of a tool.
public struct ToolParameter: Sendable {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let required: Bool

    public init(name: String, type: ToolParameterType, description: String, required: Bool) {
        self.name = name
        self.type = type
        self.description = description
        self.required = required
    }
}

/// A parsed tool call extracted from model output.
public struct ToolCall: Sendable {
    public let name: String
    public let arguments: [String: any Sendable]

    public init(name: String, arguments: [String: any Sendable]) {
        self.name = name
        self.arguments = arguments
    }
}

/// Protocol for tools that can be called by EdgeRunner models.
///
/// Each tool declares its name, description, parameters, and an
/// invoke method. The tool's schema is automatically generated
/// for inclusion in the model prompt.
public protocol EdgeRunnerTool: Sendable {
    /// Unique name for this tool.
    static var name: String { get }

    /// Human-readable description of what the tool does.
    static var description: String { get }

    /// Parameter definitions for the tool.
    static var parameters: [ToolParameter] { get }

    /// Execute the tool with the given arguments.
    func invoke(arguments: [String: Any]) async throws -> String
}

extension EdgeRunnerTool {
    /// Generate a JSON Schema description of this tool.
    public static var jsonSchema: String {
        var properties = [String: [String: String]]()
        var required = [String]()

        for param in parameters {
            properties[param.name] = [
                "type": param.type.rawValue,
                "description": param.description,
            ]
            if param.required {
                required.append(param.name)
            }
        }

        let schema: [String: Any] = [
            "name": name,
            "description": description,
            "parameters": [
                "type": "object",
                "properties": properties,
                "required": required,
            ] as [String: Any],
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys]),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }
}
```

```swift
// Sources/EdgeRunner/ToolCalling/ToolChoice.swift
import Foundation

/// Controls whether and how the model should use tools.
public enum ToolChoice: Sendable, Equatable {
    /// Model decides whether to call a tool.
    case auto

    /// Model must call at least one tool.
    case required

    /// Model must not call any tools.
    case none

    /// Model must call the specified tool.
    case specific(String)
}
```

```swift
// Sources/EdgeRunner/ToolCalling/ToolCallParser.swift
import Foundation

/// Parses tool call invocations from model text output.
///
/// Supports multiple formats:
/// - XML-style: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
/// - OpenAI-style: `{"function_call": {"name": "...", "arguments": "..."}}`
public enum ToolCallParser {

    /// Parse all tool calls from model output text.
    public static func parse(modelOutput: String) throws -> [ToolCall] {
        var calls = [ToolCall]()

        // Try XML-style parsing
        calls.append(contentsOf: parseXMLStyle(modelOutput))

        // Try function_call style if no XML-style found
        if calls.isEmpty {
            calls.append(contentsOf: parseFunctionCallStyle(modelOutput))
        }

        return calls
    }

    /// Quick check whether the output likely contains a tool call.
    public static func containsToolCall(in text: String) -> Bool {
        text.contains("<tool_call>") || text.contains("\"function_call\"")
    }

    // MARK: - XML Style

    private static func parseXMLStyle(_ text: String) -> [ToolCall] {
        var calls = [ToolCall]()
        let pattern = "<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }

        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches {
            guard let range = Range(match.range(at: 1), in: text) else { continue }
            let jsonStr = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = jsonStr.data(using: .utf8),
                  let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = dict["name"] as? String else { continue }

            let arguments = dict["arguments"] as? [String: Any] ?? [:]
            calls.append(ToolCall(name: name, arguments: arguments))
        }

        return calls
    }

    // MARK: - Function Call Style

    private static func parseFunctionCallStyle(_ text: String) -> [ToolCall] {
        guard let data = text.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let functionCall = dict["function_call"] as? [String: Any],
              let name = functionCall["name"] as? String else {
            return []
        }

        var arguments = [String: Any]()
        if let argsString = functionCall["arguments"] as? String,
           let argsData = argsString.data(using: .utf8),
           let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
            arguments = argsDict
        } else if let argsDict = functionCall["arguments"] as? [String: Any] {
            arguments = argsDict
        }

        return [ToolCall(name: name, arguments: arguments)]
    }
}
```

```swift
// Sources/EdgeRunner/ToolCalling/ToolExecutor.swift
import Foundation

/// Manages tool registration and execution.
///
/// Holds a registry of available tools and handles dispatching
/// tool calls parsed from model output.
public struct ToolExecutor: Sendable {
    private let tools: [String: any EdgeRunnerTool]

    public init(tools: [any EdgeRunnerTool]) {
        var registry = [String: any EdgeRunnerTool]()
        for tool in tools {
            registry[type(of: tool).name] = tool
        }
        self.tools = registry
    }

    /// Execute a single tool call.
    public func execute(_ call: ToolCall) async throws -> String {
        guard let tool = tools[call.name] else {
            throw GenerationError.toolCallFailed(
                name: call.name,
                reason: "Tool '\(call.name)' not found. Available tools: \(tools.keys.sorted().joined(separator: ", "))"
            )
        }
        return try await tool.invoke(arguments: call.arguments)
    }

    /// Execute multiple tool calls sequentially.
    public func executeAll(_ calls: [ToolCall]) async throws -> [String] {
        var results = [String]()
        results.reserveCapacity(calls.count)
        for call in calls {
            let result = try await execute(call)
            results.append(result)
        }
        return results
    }

    /// Generate a text description of all available tools for inclusion in prompts.
    public func toolDescriptions() -> String {
        tools.values.map { tool in
            type(of: tool).jsonSchema
        }.joined(separator: "\n")
    }

    /// Determine whether to attempt tool calling based on choice and model output.
    public func shouldAttemptToolCall(choice: ToolChoice, modelOutput: String) -> Bool {
        switch choice {
        case .none:
            return false
        case .required:
            return true
        case .auto:
            return ToolCallParser.containsToolCall(in: modelOutput)
        case .specific(let name):
            return tools[name] != nil
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "EdgeRunnerToolProtocolTests|ToolChoiceTests|ToolCallParserTests|ToolExecutorTests" 2>&1`
Expected: All 16 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/ToolCalling/EdgeRunnerTool.swift \
      Sources/EdgeRunner/ToolCalling/ToolChoice.swift \
      Sources/EdgeRunner/ToolCalling/ToolCallParser.swift \
      Sources/EdgeRunner/ToolCalling/ToolExecutor.swift \
      Tests/EdgeRunnerTests/ToolCallingTests.swift
git commit -m "feat(m4): add tool calling with EdgeRunnerTool protocol, parser, and executor

EdgeRunnerTool protocol with JSON Schema generation.
ToolCallParser supports XML-style and function_call formats.
ToolExecutor manages tool registry with auto/required/none/specific choice modes."
```

---

## Task 7: Foundation Models Backend

**Files:**
- Create: `Sources/EdgeRunner/Backends/FoundationModelsBackend.swift`
- Create: `Sources/EdgeRunner/Backends/EdgeRunnerLocalBackend.swift`
- Create: `Sources/EdgeRunner/Backends/BackendFactory.swift`
- Test: `Tests/EdgeRunnerTests/BackendTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/BackendTests.swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Mock backends for testing without Foundation Models framework

struct MockFoundationModelsBackend: EdgeRunnerLanguageModel {
    static let modelIdentifier = "foundation-models-mock"

    let isAvailable: Bool
    let fixedResponse: String

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockFoundationModelsBackend {
        return MockFoundationModelsBackend(isAvailable: true, fixedResponse: "Hello from Foundation Models")
    }

    // Foundation Models does NOT expose logits — use nextToken directly.
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        // Simulate Foundation Models returning a fixed token
        return 42
    }

    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(fixedResponse)
            continuation.finish()
        }
    }

    func tokenize(_ text: String) -> [Int] {
        // Delegate to system tokenizer
        Array(text.utf8).map { Int($0) }
    }

    func detokenize(_ ids: [Int]) -> String {
        String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    var eosTokenID: Int { 2 }
    var bosTokenID: Int? { 1 }
    var vocabularySize: Int { 32000 }
}

struct MockLocalBackend: LogitsModel {
    static let modelIdentifier = "local-mock-q4_0"

    let modelURL: URL

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> MockLocalBackend {
        return MockLocalBackend(modelURL: url)
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        var result = [Float](repeating: -10.0, count: 100)
        result[42] = 10.0
        return result
    }

    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        // Use logits + sampling pipeline for local models
        let l = try await logits(for: tokenIDs)
        return l.enumerated().max(by: { $0.element < $1.element })!.offset
    }

    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield("mock output")
            continuation.finish()
        }
    }

    func tokenize(_ text: String) -> [Int] { [1, 2, 3] }
    func detokenize(_ ids: [Int]) -> String { "mock output" }

    var eosTokenID: Int { 0 }
    var bosTokenID: Int? { 1 }
    var vocabularySize: Int { 100 }
}

@Suite("BackendFactory")
struct BackendFactoryTests {

    @Test func createLocalBackend() async throws {
        let url = URL(fileURLWithPath: "/tmp/model.gguf")
        let config = ModelConfiguration()
        let model = try await MockLocalBackend.load(from: url, configuration: config)
        #expect(MockLocalBackend.modelIdentifier == "local-mock-q4_0")
        #expect(model.vocabularySize == 100)
    }

    @Test func createFoundationModelsBackend() async throws {
        let url = URL(fileURLWithPath: "/tmp/system-model")
        let config = ModelConfiguration()
        let model = try await MockFoundationModelsBackend.load(from: url, configuration: config)
        #expect(MockFoundationModelsBackend.modelIdentifier == "foundation-models-mock")
        #expect(model.vocabularySize == 32000)
    }

    @Test func typeErasedUsage() async throws {
        // Demonstrate backend swapping via any EdgeRunnerLanguageModel (base protocol).
        // Both backends support nextToken — no logits required for uniform usage.
        let url = URL(fileURLWithPath: "/tmp/model.gguf")
        let config = ModelConfiguration()

        let useSystem = false
        let model: any EdgeRunnerLanguageModel
        if useSystem {
            model = try await MockFoundationModelsBackend.load(from: url, configuration: config)
        } else {
            model = try await MockLocalBackend.load(from: url, configuration: config)
        }

        let sampling = SamplingConfiguration()
        let token = try await model.nextToken(for: [1, 2], sampling: sampling)
        #expect(token >= 0)
    }

    @Test func backendSwapAtRuntime() async throws {
        let url = URL(fileURLWithPath: "/tmp/model.gguf")
        let config = ModelConfiguration()

        // Start with local, switch to system
        var model: any EdgeRunnerLanguageModel = try await MockLocalBackend.load(from: url, configuration: config)
        #expect(model.vocabularySize == 100)

        model = try await MockFoundationModelsBackend.load(from: url, configuration: config)
        #expect(model.vocabularySize == 32000)
    }

    @Test func backendRegistryLookup() {
        let registry = BackendRegistry()
        registry.register(MockLocalBackend.self, for: "gguf")
        registry.register(MockFoundationModelsBackend.self, for: "foundation")

        #expect(registry.availableBackends.count == 2)
        #expect(registry.availableBackends.contains("gguf"))
        #expect(registry.availableBackends.contains("foundation"))
    }
}

@Suite("EdgeRunnerLocalBackend")
struct EdgeRunnerLocalBackendTests {

    @Test func localBackendConformsToProtocol() async throws {
        let model = try await MockLocalBackend.load(
            from: URL(fileURLWithPath: "/tmp/test.gguf"),
            configuration: ModelConfiguration()
        )
        let tokens = model.tokenize("test")
        #expect(!tokens.isEmpty)

        let logits = try await model.logits(for: tokens)
        #expect(logits.count == model.vocabularySize)

        let text = model.detokenize([1, 2])
        #expect(!text.isEmpty)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "BackendFactoryTests|EdgeRunnerLocalBackendTests" 2>&1`
Expected: FAIL — `BackendRegistry` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Backends/BackendFactory.swift
import Foundation

/// Registry for available model backends.
///
/// Allows dynamic backend selection based on model format or user preference.
/// Supports runtime backend swapping via the type-erased `any EdgeRunnerLanguageModel`.
/// Thread-safe via Mutex from the Synchronization framework (consistent with MetalBackend pattern).
public final class BackendRegistry: Sendable {
    private let state: Mutex<[String: any EdgeRunnerLanguageModel.Type]>

    public init() {
        self.state = Mutex([:])
    }

    /// Register a backend for a given format identifier.
    public func register<T: EdgeRunnerLanguageModel>(_ type: T.Type, for format: String) {
        state.withLock { $0[format] = type }
    }

    /// Look up the backend type for a format.
    public func backend(for format: String) -> (any EdgeRunnerLanguageModel.Type)? {
        state.withLock { $0[format] }
    }

    /// List all registered format identifiers.
    public var availableBackends: Set<String> {
        state.withLock { Set($0.keys) }
    }

    /// Load a model using the appropriate backend for the given format.
    public func load(
        from url: URL,
        format: String,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws -> any EdgeRunnerLanguageModel {
        guard let backendType = backend(for: format) else {
            throw GenerationError.modelLoadFailed(
                reason: "No backend registered for format '\(format)'. Available: \(availableBackends.sorted().joined(separator: ", "))"
            )
        }
        return try await backendType.load(from: url, configuration: configuration)
    }
}
```

```swift
// Sources/EdgeRunner/Backends/EdgeRunnerLocalBackend.swift
import Foundation

/// Marker protocol for local Metal-accelerated model backends.
///
/// Conforming types run inference entirely on-device using Metal compute shaders.
/// They load weights from GGUF, SafeTensor, or NPZ files and execute the full
/// forward pass through the EdgeRunner tensor/graph pipeline.
public protocol LocalModelBackend: EdgeRunnerLanguageModel {
    /// The file format this backend supports (e.g., "gguf", "safetensors").
    static var supportedFormat: String { get }

    /// Memory estimate for loading this model (bytes).
    func estimatedMemoryUsage() -> Int
}
```

```swift
// Sources/EdgeRunner/Backends/FoundationModelsBackend.swift
import Foundation

/// Protocol for Foundation Models system backend integration.
///
/// When Apple's Foundation Models framework is available (iOS 26+/macOS 26+),
/// conforming types can delegate inference to the system LLM while maintaining
/// the same `EdgeRunnerLanguageModel` interface.
///
/// Usage:
/// ```swift
/// let model: any EdgeRunnerLanguageModel
/// if FoundationModelsAvailability.isAvailable {
///     model = try await SystemModelBackend.load(from: url, configuration: config)
/// } else {
///     model = try await EdgeRunnerGGUFModel.load(from: url, configuration: config)
/// }
/// ```
///
/// Note: The actual Foundation Models integration requires `import FoundationModels`
/// which is only available at runtime on supported devices. This file provides
/// the protocol and availability checking infrastructure.
public enum FoundationModelsAvailability {

    /// Check if Foundation Models is available.
    ///
    /// Uses compile-time `#if canImport` gating. For true runtime detection
    /// on devices that may have the framework but lack model support,
    /// use `LanguageModelSession.isAvailable` from Foundation Models directly.
    public static var isAvailable: Bool {
        #if canImport(FoundationModels)
        return true
        #else
        return false
        #endif
    }
}

/// Protocol that Foundation Models backends must conform to.
///
/// Extends EdgeRunnerLanguageModel with system-model-specific capabilities
/// like guided generation and session management.
public protocol SystemModelBackend: EdgeRunnerLanguageModel {
    /// Whether the system model supports structured/guided generation natively.
    var supportsGuidedGeneration: Bool { get }

    /// Generate text using the system model's native streaming API.
    func generateStream(prompt: String) -> AsyncThrowingStream<String, Error>
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "BackendFactoryTests|EdgeRunnerLocalBackendTests" 2>&1`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Backends/FoundationModelsBackend.swift \
      Sources/EdgeRunner/Backends/EdgeRunnerLocalBackend.swift \
      Sources/EdgeRunner/Backends/BackendFactory.swift \
      Tests/EdgeRunnerTests/BackendTests.swift
git commit -m "feat(m4): add Foundation Models backend swapping with BackendRegistry

BackendRegistry for dynamic model format routing.
LocalModelBackend protocol for Metal-accelerated inference.
SystemModelBackend protocol for Foundation Models integration.
Type-erased any EdgeRunnerLanguageModel for seamless backend swapping."
```

---

## Task 8: Streaming Token Output

**Files:**
- Create: `Sources/EdgeRunner/Streaming/TokenStream.swift`
- Create: `Sources/EdgeRunner/Streaming/GenerationSession.swift`
- Test: `Tests/EdgeRunnerTests/StreamingTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/StreamingTests.swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Mock model for streaming tests

private struct StreamingMockModel: EdgeRunnerLanguageModel {
    static let modelIdentifier = "streaming-mock"

    /// Sequence of token IDs the model will produce (deterministic).
    let tokenSequence: [Int]
    let vocab: [String]

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> StreamingMockModel {
        return StreamingMockModel(
            tokenSequence: [0, 1, 2, 3],
            vocab: ["Hello", " ", "world", "!"]
        )
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let step = tokenIDs.count
        var result = [Float](repeating: -100.0, count: vocab.count + 1) // +1 for EOS
        if step < tokenSequence.count {
            result[tokenSequence[step]] = 10.0
        } else {
            result[vocab.count] = 10.0 // EOS
        }
        return result
    }

    func tokenize(_ text: String) -> [Int] { [] }

    func detokenize(_ ids: [Int]) -> String {
        ids.compactMap { id in
            id < vocab.count ? vocab[id] : nil
        }.joined()
    }

    var eosTokenID: Int { vocab.count } // last index is EOS
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocab.count + 1 }
}

@Suite("TokenStream")
struct TokenStreamTests {

    @Test func basicStreaming() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0, 1, 2, 3],
            vocab: ["Hello", " ", "world", "!"]
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 10
        )

        var tokens = [String]()
        let stream = session.stream(prompt: "")
        for try await token in stream {
            tokens.append(token)
        }
        #expect(tokens == ["Hello", " ", "world", "!"])
    }

    @Test func streamStopsAtEOS() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0, 1], // Only 2 tokens before EOS
            vocab: ["Hi", "!"]
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 100
        )

        var count = 0
        let stream = session.stream(prompt: "")
        for try await _ in stream {
            count += 1
        }
        #expect(count == 2)
    }

    @Test func streamRespectsMaxTokens() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 10 tokens
            vocab: ["a"]
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 3
        )

        var count = 0
        let stream = session.stream(prompt: "")
        for try await _ in stream {
            count += 1
        }
        #expect(count == 3)
    }

    @Test func streamCancellation() async throws {
        let model = StreamingMockModel(
            tokenSequence: (0..<100).map { _ in 0 },
            vocab: ["token"]
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 100
        )

        var count = 0
        let stream = session.stream(prompt: "")
        for try await _ in stream {
            count += 1
            if count >= 5 {
                break // Cancel by breaking out
            }
        }
        #expect(count == 5)
    }

    @Test func tokenCallbackHook() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0, 1, 2],
            vocab: ["a", "b", "c"]
        )

        var callbackTokenIDs = [Int]()
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 10,
            onToken: { tokenID, tokenString in
                callbackTokenIDs.append(tokenID)
            }
        )

        let stream = session.stream(prompt: "")
        for try await _ in stream {}
        #expect(callbackTokenIDs == [0, 1, 2])
    }

    @Test func collectFullResponse() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0, 1, 2, 3],
            vocab: ["Hello", " ", "world", "!"]
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 10
        )

        let response = try await session.generate(prompt: "")
        #expect(response == "Hello world!")
    }

    @Test func generateWithStructuredOutput() async throws {
        // Model produces JSON token by token
        let jsonTokens = ["{", "\"", "n", "\"", ":", "1", "}"]
        let model = StreamingMockModel(
            tokenSequence: Array(0..<jsonTokens.count),
            vocab: jsonTokens
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 20
        )

        let response = try await session.generate(prompt: "")
        #expect(response.contains("{"))
        #expect(response.contains("}"))
    }
}

@Suite("GenerationSession")
struct GenerationSessionTests {

    @Test func sessionMetadata() {
        let model = StreamingMockModel(
            tokenSequence: [],
            vocab: []
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: .greedy,
            maxTokens: 512
        )
        #expect(session.maxTokens == 512)
    }

    @Test func sessionWithCustomSampling() async throws {
        let model = StreamingMockModel(
            tokenSequence: [0],
            vocab: ["test"]
        )
        let pipeline = SamplingPipeline(
            transforms: [TemperatureSampler(temperature: 0.001)],
            selector: GreedySampler()
        )
        let session = GenerationSession(
            model: model,
            samplingPipeline: pipeline,
            maxTokens: 5
        )

        let response = try await session.generate(prompt: "")
        #expect(response == "test")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "TokenStreamTests|GenerationSessionTests" 2>&1`
Expected: FAIL — `GenerationSession`, `TokenStream` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Streaming/TokenStream.swift
import Foundation

/// A token emitted during streaming generation.
public struct StreamToken: Sendable {
    /// The token ID.
    public let id: Int

    /// The decoded text for this token.
    public let text: String

    /// Whether this is the end-of-sequence token.
    public let isEOS: Bool

    public init(id: Int, text: String, isEOS: Bool = false) {
        self.id = id
        self.text = text
        self.isEOS = isEOS
    }
}

/// Statistics collected during a generation session.
public struct GenerationStats: Sendable {
    /// Total tokens generated.
    public var tokenCount: Int = 0

    /// Time to first token (seconds).
    public var timeToFirstToken: Double = 0

    /// Total generation time (seconds).
    public var totalTime: Double = 0

    /// Tokens per second.
    public var tokensPerSecond: Double {
        guard totalTime > 0 else { return 0 }
        return Double(tokenCount) / totalTime
    }
}
```

```swift
// Sources/EdgeRunner/Streaming/GenerationSession.swift
import Foundation
@preconcurrency import EdgeRunnerCore

/// Manages a single text generation session with streaming output.
///
/// Wraps an `EdgeRunnerLanguageModel` and provides:
/// - `AsyncThrowingStream`-based token streaming
/// - Backpressure handling (producer waits if consumer is slow)
/// - Cancellation support via task cancellation
/// - Token callback hooks for monitoring
/// - Full response collection via `generate(prompt:)`
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    private let model: Model
    private let samplingPipeline: SamplingPipeline
    public let maxTokens: Int
    private let onToken: (@Sendable (Int, String) -> Void)?

    public init(
        model: Model,
        samplingPipeline: SamplingPipeline = .greedy,
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    ) {
        self.model = model
        self.samplingPipeline = samplingPipeline
        self.maxTokens = maxTokens
        self.onToken = onToken
    }

    /// Stream generated tokens one at a time.
    ///
    /// Returns an `AsyncThrowingStream<String, Error>` that yields
    /// decoded text for each generated token. The stream terminates when:
    /// - The EOS token is generated
    /// - `maxTokens` is reached
    /// - The task is cancelled
    /// - An error occurs
    public func stream(prompt: String) -> AsyncThrowingStream<String, Error> {
        let model = self.model
        let pipeline = self.samplingPipeline
        let maxTokens = self.maxTokens
        let onToken = self.onToken

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    if let bos = model.bosTokenID, tokenIDs.first != bos {
                        tokenIDs.insert(bos, at: 0)
                    }

                    var generatedTokenIDs = [Int]()

                    for _ in 0..<maxTokens {
                        // Check for cancellation
                        try Task.checkCancellation()

                        let logits = try await model.logits(for: tokenIDs)
                        let tokenID = pipeline.sample(
                            logits: logits,
                            previousTokens: generatedTokenIDs
                        )

                        // Check for EOS
                        if tokenID == model.eosTokenID {
                            break
                        }

                        tokenIDs.append(tokenID)
                        generatedTokenIDs.append(tokenID)

                        let text = model.detokenize([tokenID])

                        // Invoke callback
                        onToken?(tokenID, text)

                        // Yield to stream (backpressure: if buffer is full, this waits)
                        continuation.yield(text)
                    }

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: GenerationError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Generate a complete response (non-streaming).
    ///
    /// Collects all tokens from the stream and concatenates them.
    public func generate(prompt: String) async throws -> String {
        var result = ""
        let stream = self.stream(prompt: prompt)
        for try await token in stream {
            result += token
        }
        return result
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "TokenStreamTests|GenerationSessionTests" 2>&1`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Streaming/TokenStream.swift \
      Sources/EdgeRunner/Streaming/GenerationSession.swift \
      Tests/EdgeRunnerTests/StreamingTests.swift
git commit -m "feat(m4): add AsyncThrowingStream-based token streaming with backpressure and cancellation

GenerationSession wraps any EdgeRunnerLanguageModel for streaming generation.
Supports EOS detection, max token limits, task cancellation, and token callbacks.
Non-streaming generate() convenience method collects full response."
```

---

## Task 9: DocC Documentation

**Files:**
- Create: `Sources/EdgeRunner/Documentation.docc/EdgeRunner.md`
- Create: `Sources/EdgeRunner/Documentation.docc/GettingStarted.md`
- Create: `Sources/EdgeRunner/Documentation.docc/Articles/ModelLoading.md`
- Create: `Sources/EdgeRunner/Documentation.docc/Articles/StreamingGeneration.md`
- Create: `Sources/EdgeRunner/Documentation.docc/Articles/StructuredOutput.md`
- Create: `Sources/EdgeRunner/Documentation.docc/Articles/ToolCalling.md`
- Create: `Sources/EdgeRunner/Documentation.docc/Articles/Sampling.md`

**Step 1: Write the failing tests**

DocC does not have runtime tests — verification is build-based.

```swift
// No test file needed. Verification:
// swift package generate-documentation --target EdgeRunner
```

**Step 2: Run tests to verify they fail**

Run: `swift package generate-documentation --target EdgeRunner 2>&1`
Expected: FAIL or WARN — documentation catalog not found

**Step 3: Implement**

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/EdgeRunner.md -->
# ``EdgeRunner``

A Metal-native Swift 6.2 inference engine for running large language models on Apple Silicon.

## Overview

EdgeRunner provides a high-performance, on-device inference engine for LLMs. It runs entirely on Apple Silicon GPUs using Metal 4 compute shaders, with no external dependencies.

Key capabilities:
- **Streaming generation** via `AsyncThrowingStream`
- **Structured output** with constrained decoding from `Decodable` types
- **Tool calling** with automatic schema generation
- **Composable sampling** with temperature, top-k, top-p, min-p, and repetition penalty
- **Backend swapping** between local Metal inference and Apple Foundation Models
- **Speculative decoding** for faster generation with draft models

## Topics

### Essentials
- <doc:GettingStarted>
- ``EdgeRunnerLanguageModel``
- ``ModelConfiguration``
- ``GenerationSession``

### Model Loading
- <doc:ModelLoading>
- ``BackendRegistry``
- ``LocalModelBackend``

### Generation
- <doc:StreamingGeneration>
- ``SamplingPipeline``
- ``GenerationError``

### Structured Output
- <doc:StructuredOutput>
- ``JSONSchemaExtractor``
- ``ConstrainedDecoder``
- ``StructuredGenerator``

### Tool Calling
- <doc:ToolCalling>
- ``EdgeRunnerTool``
- ``ToolExecutor``
- ``ToolChoice``

### Sampling
- <doc:Sampling>
- ``GreedySampler``
- ``TemperatureSampler``
- ``TopKSampler``
- ``TopPSampler``
- ``MinPSampler``
- ``RepetitionPenalty``

### Backend Swapping
- ``FoundationModelsAvailability``
- ``SystemModelBackend``
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/GettingStarted.md -->
# Getting Started with EdgeRunner

Load a model, configure sampling, and generate text in a few lines of Swift.

## Overview

EdgeRunner is designed for simplicity. Load a GGUF model, create a generation session, and start streaming tokens.

### Load a Model

```swift
import EdgeRunner

let model = try await MyModel.load(
    from: URL(fileURLWithPath: "path/to/model.gguf"),
    configuration: ModelConfiguration(
        maxTokens: 1024,
        contextWindowSize: 4096
    )
)
```

### Stream Tokens

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .nucleus(temperature: 0.7, topP: 0.9),
    maxTokens: 512
)

for try await token in session.stream(prompt: "Explain quantum computing:") {
    print(token, terminator: "")
}
```

### Generate a Complete Response

```swift
let response = try await session.generate(prompt: "What is Swift?")
print(response)
```

### Backend Swapping

```swift
let model: any EdgeRunnerLanguageModel
if FoundationModelsAvailability.isAvailable {
    model = try await SystemModel.load(from: url, configuration: config)
} else {
    model = try await LocalGGUFModel.load(from: url, configuration: config)
}
```
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/Articles/ModelLoading.md -->
# Model Loading

Load models from GGUF, SafeTensor, or NPZ formats.

## Overview

EdgeRunner supports multiple model formats through its backend system. The ``BackendRegistry`` maps file formats to the appropriate loader.

### Direct Loading

```swift
let model = try await LlamaModel.load(
    from: URL(fileURLWithPath: "llama-3-8b-q4_0.gguf"),
    configuration: ModelConfiguration(useMemoryMapping: true)
)
```

### Registry-Based Loading

```swift
let registry = BackendRegistry()
registry.register(LlamaModel.self, for: "gguf")
registry.register(GPT2Model.self, for: "safetensors")

let model = try await registry.load(
    from: modelURL,
    format: "gguf"
)
```

### Memory Configuration

For devices with limited memory (iPhone), configure appropriately:

```swift
let config = ModelConfiguration(
    maxTokens: 512,
    contextWindowSize: 2048,  // Smaller context for 8GB devices
    useMemoryMapping: true     // Memory-map weights instead of loading
)
```
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/Articles/StreamingGeneration.md -->
# Streaming Generation

Generate text token-by-token using Swift's structured concurrency.

## Overview

EdgeRunner uses `AsyncThrowingStream` for streaming token output with built-in backpressure handling and cancellation support.

### Basic Streaming

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .greedy,
    maxTokens: 256
)

for try await token in session.stream(prompt: "Once upon a time") {
    print(token, terminator: "")
}
```

### Token Callbacks

Monitor generation progress with callback hooks:

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .nucleus(temperature: 0.8, topP: 0.9),
    maxTokens: 512,
    onToken: { tokenID, text in
        // Update UI, log metrics, etc.
        print("Token \(tokenID): '\(text)'")
    }
)
```

### Cancellation

Streams respect Swift task cancellation:

```swift
let task = Task {
    for try await token in session.stream(prompt: "...") {
        updateUI(with: token)
    }
}

// Cancel generation at any time
task.cancel()
```
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/Articles/StructuredOutput.md -->
# Structured Output

Generate typed Swift objects from model output using constrained decoding.

## Overview

EdgeRunner can generate JSON that conforms to a specific `Decodable` type using grammar-guided sampling. The ``JSONSchemaExtractor`` introspects the type at runtime, and the ``ConstrainedDecoder`` masks invalid tokens during generation.

### Define Your Type

```swift
struct MovieReview: Codable {
    let title: String
    let rating: Int
    let summary: String
}
```

### Generate Structured Output

```swift
let schema = try JSONSchemaExtractor.extractSchema(for: MovieReview.self)
// Use schema to guide constrained decoding...
let json = try await session.generate(prompt: "Review the movie Inception:")
let review: MovieReview = try StructuredGenerator.parse(json: json)
print(review.title)  // "Inception"
print(review.rating) // 9
```

### Schema Extraction

The schema extractor supports:
- Primitive types: `String`, `Int`, `Double`, `Bool`
- Nested objects
- Arrays
- Optional fields
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/Articles/ToolCalling.md -->
# Tool Calling

Enable models to call Swift functions with typed parameters.

## Overview

EdgeRunner's tool calling system lets models invoke Swift functions. Define tools with the ``EdgeRunnerTool`` protocol, register them with a ``ToolExecutor``, and the system handles parsing, validation, and execution.

### Define a Tool

```swift
struct SearchTool: EdgeRunnerTool {
    static let name = "web_search"
    static let description = "Search the web for information."
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "query", type: .string, description: "Search query", required: true),
        ToolParameter(name: "limit", type: .integer, description: "Max results", required: false),
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let query = arguments["query"] as! String
        // Perform search...
        return "{\"results\": [\"Result for: \(query)\"]}"
    }
}
```

### Execute Tool Calls

```swift
let executor = ToolExecutor(tools: [SearchTool(), CalculatorTool()])

// Parse tool calls from model output
let calls = try ToolCallParser.parse(modelOutput: modelResponse)

// Execute all calls
let results = try await executor.executeAll(calls)
```

### Tool Choice

Control when tools are used:

```swift
// Model decides
executor.shouldAttemptToolCall(choice: .auto, modelOutput: output)

// Force tool use
executor.shouldAttemptToolCall(choice: .required, modelOutput: output)

// Disable tools
executor.shouldAttemptToolCall(choice: .none, modelOutput: output)
```
```

```markdown
<!-- Sources/EdgeRunner/Documentation.docc/Articles/Sampling.md -->
# Sampling Strategies

Configure how tokens are selected during generation.

## Overview

EdgeRunner uses a composable ``SamplingPipeline`` that chains logit transforms before token selection. This allows mixing strategies like temperature scaling, top-k filtering, and repetition penalties.

### Quick Start

```swift
// Greedy (deterministic)
let pipeline = SamplingPipeline.greedy

// Nucleus sampling
let pipeline = SamplingPipeline.nucleus(temperature: 0.8, topP: 0.9)

// Top-k sampling
let pipeline = SamplingPipeline.topK(k: 40, temperature: 0.7)
```

### Custom Pipeline

Build a custom pipeline with chained transforms:

```swift
let pipeline = SamplingPipeline(
    transforms: [
        TemperatureSampler(temperature: 0.7),
        TopKSampler(k: 50),
        TopPSampler(p: 0.9),
        MinPSampler(minP: 0.05),
    ],
    selector: StochasticSampler(randomSource: &myRNG),
    repetitionPenalty: RepetitionPenalty(penalty: 1.2, frequencyPenalty: 0.5)
)
```

### Available Transforms

| Transform | Effect |
|-----------|--------|
| ``TemperatureSampler`` | Scale logits by 1/T to control randomness |
| ``TopKSampler`` | Keep only the top K tokens |
| ``TopPSampler`` | Keep smallest set of tokens with cumulative probability >= P |
| ``MinPSampler`` | Filter tokens below `minP * max_probability` |
| ``RepetitionPenalty`` | Penalize previously generated tokens |

### Deterministic Output

For reproducible results, use ``SeededRandomSource``:

```swift
var rng = SeededRandomSource(seed: 42)
let pipeline = SamplingPipeline(
    transforms: [TemperatureSampler(temperature: 0.5)],
    selector: StochasticSampler(randomSource: &rng)
)
```
```

**Step 4: Run tests to verify they pass**

Run: `swift package generate-documentation --target EdgeRunner 2>&1`
Expected: Documentation generated successfully

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Documentation.docc/
git commit -m "docs(m4): add DocC documentation catalog with getting started guide and API articles

Documentation catalog with landing page, getting started tutorial,
and articles covering model loading, streaming, structured output,
tool calling, and sampling strategies."
```

---

## Task 10: EdgeRunnerChat Demo App (iOS)

**Files:**
- Create: `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift`
- Create: `Examples/EdgeRunnerChat/Views/ChatView.swift`
- Create: `Examples/EdgeRunnerChat/Views/MessageBubble.swift`
- Create: `Examples/EdgeRunnerChat/Views/ModelPickerView.swift`
- Create: `Examples/EdgeRunnerChat/Views/MemoryUsageView.swift`
- Create: `Examples/EdgeRunnerChat/ViewModels/ChatViewModel.swift`
- Create: `Examples/EdgeRunnerChat/Models/ChatMessage.swift`
- Create: `Examples/EdgeRunnerChat/Models/ModelInfo.swift`
- Test: `Tests/EdgeRunnerTests/ChatViewModelTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/ChatViewModelTests.swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Test types for ChatViewModel

struct TestChatMessage: Identifiable, Equatable, Sendable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date

    enum MessageRole: String, Sendable {
        case user
        case assistant
        case system
    }

    init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}

// Mock model for view model tests
private struct ChatMockModel: EdgeRunnerLanguageModel {
    static let modelIdentifier = "chat-mock"
    let responseTokens: [Int]
    let vocab: [String]

    static func load(from url: URL, configuration: ModelConfiguration) async throws -> ChatMockModel {
        return ChatMockModel(responseTokens: [0, 1], vocab: ["Hi", "!"])
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let step = tokenIDs.count
        var result = [Float](repeating: -100, count: vocab.count + 1)
        if step < responseTokens.count {
            result[responseTokens[step]] = 10
        } else {
            result[vocab.count] = 10 // EOS
        }
        return result
    }

    func tokenize(_ text: String) -> [Int] { [] }
    func detokenize(_ ids: [Int]) -> String {
        ids.compactMap { $0 < vocab.count ? vocab[$0] : nil }.joined()
    }
    var eosTokenID: Int { vocab.count }
    var bosTokenID: Int? { nil }
    var vocabularySize: Int { vocab.count + 1 }
}

@Suite("ChatMessage")
struct ChatMessageTests {

    @Test func messageCreation() {
        let msg = TestChatMessage(role: .user, content: "Hello")
        #expect(msg.role == .user)
        #expect(msg.content == "Hello")
    }

    @Test func messageRoles() {
        let user = TestChatMessage(role: .user, content: "")
        let assistant = TestChatMessage(role: .assistant, content: "")
        let system = TestChatMessage(role: .system, content: "")
        #expect(user.role == .user)
        #expect(assistant.role == .assistant)
        #expect(system.role == .system)
    }
}

@Suite("ModelInfo")
struct ModelInfoTests {

    @Test func modelInfoProperties() {
        let info = ModelInfo(
            name: "Llama 3 8B Q4_0",
            path: URL(fileURLWithPath: "/models/llama-3-8b-q4_0.gguf"),
            format: "gguf",
            parameterCount: "8B",
            quantization: "Q4_0",
            fileSizeBytes: 4_500_000_000
        )
        #expect(info.name == "Llama 3 8B Q4_0")
        #expect(info.format == "gguf")
        #expect(info.quantization == "Q4_0")
    }

    @Test func fileSizeFormatted() {
        let info = ModelInfo(
            name: "Test",
            path: URL(fileURLWithPath: "/test"),
            format: "gguf",
            parameterCount: "1B",
            quantization: "Q8_0",
            fileSizeBytes: 1_073_741_824
        )
        #expect(info.fileSizeFormatted == "1.0 GB")
    }

    @Test func smallFileSizeFormatted() {
        let info = ModelInfo(
            name: "Test",
            path: URL(fileURLWithPath: "/test"),
            format: "gguf",
            parameterCount: "100M",
            quantization: "Q4_0",
            fileSizeBytes: 52_428_800
        )
        #expect(info.fileSizeFormatted == "50.0 MB")
    }
}

@Suite("ChatViewModel Logic")
struct ChatViewModelLogicTests {

    @Test func initialState() {
        let vm = ChatViewModelState()
        #expect(vm.messages.isEmpty)
        #expect(vm.isGenerating == false)
        #expect(vm.currentInput == "")
    }

    @Test func addUserMessage() {
        var vm = ChatViewModelState()
        vm.addUserMessage("Hello there")
        #expect(vm.messages.count == 1)
        #expect(vm.messages[0].role == .user)
        #expect(vm.messages[0].content == "Hello there")
    }

    @Test func addAssistantMessage() {
        var vm = ChatViewModelState()
        vm.addAssistantMessage("")
        #expect(vm.messages.count == 1)
        #expect(vm.messages[0].role == .assistant)
    }

    @Test func appendToLastAssistantMessage() {
        var vm = ChatViewModelState()
        vm.addAssistantMessage("")
        vm.appendToLastMessage("Hello")
        vm.appendToLastMessage(" world")
        #expect(vm.messages.last?.content == "Hello world")
    }

    @Test func clearMessages() {
        var vm = ChatViewModelState()
        vm.addUserMessage("test")
        vm.addAssistantMessage("response")
        vm.clearMessages()
        #expect(vm.messages.isEmpty)
    }

    @Test func generatingState() {
        var vm = ChatViewModelState()
        vm.isGenerating = true
        #expect(vm.isGenerating)
        vm.isGenerating = false
        #expect(!vm.isGenerating)
    }

    @Test func memoryUsageTracking() {
        var vm = ChatViewModelState()
        vm.updateMemoryUsage(usedMB: 1024, totalMB: 8192)
        #expect(vm.memoryUsedMB == 1024)
        #expect(vm.memoryTotalMB == 8192)
        #expect(abs(vm.memoryUsagePercent - 12.5) < 0.1)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "ChatMessageTests|ModelInfoTests|ChatViewModelLogicTests" 2>&1`
Expected: FAIL — types not defined

**Step 3: Implement**

```swift
// Examples/EdgeRunnerChat/Models/ChatMessage.swift
import Foundation

/// A single message in a chat conversation.
public struct ChatMessage: Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public var content: String
    public let timestamp: Date

    public enum MessageRole: String, Sendable {
        case user
        case assistant
        case system
    }

    public init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}
```

```swift
// Examples/EdgeRunnerChat/Models/ModelInfo.swift
import Foundation

/// Metadata about an available model file.
public struct ModelInfo: Identifiable, Sendable {
    public let id = UUID()
    public let name: String
    public let path: URL
    public let format: String
    public let parameterCount: String
    public let quantization: String
    public let fileSizeBytes: Int64

    public init(
        name: String,
        path: URL,
        format: String,
        parameterCount: String,
        quantization: String,
        fileSizeBytes: Int64
    ) {
        self.name = name
        self.path = path
        self.format = format
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.fileSizeBytes = fileSizeBytes
    }

    /// Human-readable file size.
    public var fileSizeFormatted: String {
        let gb = Double(fileSizeBytes) / 1_073_741_824.0
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(fileSizeBytes) / 1_048_576.0
        return String(format: "%.1f MB", mb)
    }
}
```

```swift
// Examples/EdgeRunnerChat/ViewModels/ChatViewModel.swift
import Foundation
import EdgeRunner
import EdgeRunnerCore

/// Pure state container for the chat view model (testable without SwiftUI).
public struct ChatViewModelState: Sendable {
    public var messages: [ChatMessage] = []
    public var isGenerating: Bool = false
    public var currentInput: String = ""
    public var selectedModel: ModelInfo? = nil
    public var memoryUsedMB: Double = 0
    public var memoryTotalMB: Double = 0
    public var tokensPerSecond: Double = 0
    public var error: String? = nil

    public init() {}

    public var memoryUsagePercent: Double {
        guard memoryTotalMB > 0 else { return 0 }
        return (memoryUsedMB / memoryTotalMB) * 100
    }

    public mutating func addUserMessage(_ content: String) {
        messages.append(ChatMessage(role: .user, content: content))
    }

    public mutating func addAssistantMessage(_ content: String) {
        messages.append(ChatMessage(role: .assistant, content: content))
    }

    public mutating func appendToLastMessage(_ text: String) {
        guard !messages.isEmpty else { return }
        messages[messages.count - 1].content += text
    }

    public mutating func clearMessages() {
        messages.removeAll()
    }

    public mutating func updateMemoryUsage(usedMB: Double, totalMB: Double) {
        self.memoryUsedMB = usedMB
        self.memoryTotalMB = totalMB
    }
}

/// Observable view model for the chat interface.
@MainActor
@Observable
public final class ChatViewModel {
    public var state = ChatViewModelState()
    private var generationTask: Task<Void, Never>?

    public init() {}

    /// Send a user message and generate a response.
    public func send<M: EdgeRunnerLanguageModel>(
        message: String,
        model: M,
        samplingPipeline: SamplingPipeline = .nucleus(temperature: 0.7, topP: 0.9)
    ) {
        guard !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        state.addUserMessage(message)
        state.addAssistantMessage("")
        state.isGenerating = true
        state.currentInput = ""

        generationTask = Task {
            do {
                let session = GenerationSession(
                    model: model,
                    samplingPipeline: samplingPipeline,
                    maxTokens: 1024,
                    onToken: { [weak self] _, _ in
                        // Could track tokens/sec here
                    }
                )

                let prompt = buildPrompt()
                let stream = session.stream(prompt: prompt)

                for try await token in stream {
                    state.appendToLastMessage(token)
                }
            } catch {
                state.error = error.localizedDescription
            }
            state.isGenerating = false
        }
    }

    /// Cancel the current generation.
    public func cancel() {
        generationTask?.cancel()
        generationTask = nil
        state.isGenerating = false
    }

    /// Clear all messages.
    public func clear() {
        cancel()
        state.clearMessages()
    }

    /// Build the full prompt from message history.
    private func buildPrompt() -> String {
        state.messages.map { msg in
            switch msg.role {
            case .system: return "System: \(msg.content)"
            case .user: return "User: \(msg.content)"
            case .assistant: return "Assistant: \(msg.content)"
            }
        }.joined(separator: "\n")
    }
}
```

```swift
// Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift
import SwiftUI

@main
struct EdgeRunnerChatApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView()
        }
    }
}
```

```swift
// Examples/EdgeRunnerChat/Views/ChatView.swift
import SwiftUI
import EdgeRunner

struct ChatView: View {
    @State private var viewModel = ChatViewModel()
    @State private var showModelPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Memory usage bar
                MemoryUsageView(
                    usedMB: viewModel.state.memoryUsedMB,
                    totalMB: viewModel.state.memoryTotalMB
                )

                // Message list
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.state.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.state.messages.count) {
                        if let last = viewModel.state.messages.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }

                Divider()

                // Input bar
                HStack(spacing: 12) {
                    TextField("Message...", text: $viewModel.state.currentInput, axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...5)

                    if viewModel.state.isGenerating {
                        Button(action: { viewModel.cancel() }) {
                            Image(systemName: "stop.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                        }
                    } else {
                        Button(action: {
                            // In real app: viewModel.send(message:model:)
                        }) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.blue)
                        }
                        .disabled(viewModel.state.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
                .padding()
            }
            .navigationTitle("EdgeRunner Chat")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Model") {
                        showModelPicker = true
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Clear") {
                        viewModel.clear()
                    }
                }
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView(selectedModel: $viewModel.state.selectedModel)
            }
        }
    }
}
```

```swift
// Examples/EdgeRunnerChat/Views/MessageBubble.swift
import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 60)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : "Assistant")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Text(message.content)
                    .padding(12)
                    .background(
                        message.role == .user
                            ? Color.blue.opacity(0.2)
                            : Color.gray.opacity(0.15)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }

            if message.role == .assistant {
                Spacer(minLength: 60)
            }
        }
    }
}
```

```swift
// Examples/EdgeRunnerChat/Views/ModelPickerView.swift
import SwiftUI

struct ModelPickerView: View {
    @Binding var selectedModel: ModelInfo?
    @Environment(\.dismiss) private var dismiss
    @State private var models: [ModelInfo] = []

    var body: some View {
        NavigationStack {
            List(models) { model in
                Button(action: {
                    selectedModel = model
                    dismiss()
                }) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.name)
                            .font(.headline)
                        HStack {
                            Text(model.parameterCount)
                            Text(model.quantization)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.blue.opacity(0.15))
                                .clipShape(Capsule())
                            Spacer()
                            Text(model.fileSizeFormatted)
                                .foregroundStyle(.secondary)
                        }
                        .font(.caption)
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Select Model")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .task {
                models = scanForModels()
            }
        }
    }

    /// Scan the documents directory for GGUF files.
    private func scanForModels() -> [ModelInfo] {
        let documentsURL = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first!

        guard let files = try? FileManager.default.contentsOfDirectory(
            at: documentsURL,
            includingPropertiesForKeys: [.fileSizeKey]
        ) else { return [] }

        return files
            .filter { $0.pathExtension == "gguf" }
            .compactMap { url -> ModelInfo? in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return ModelInfo(
                    name: url.deletingPathExtension().lastPathComponent,
                    path: url,
                    format: "gguf",
                    parameterCount: "Unknown",
                    quantization: "Unknown",
                    fileSizeBytes: Int64(size)
                )
            }
    }
}
```

```swift
// Examples/EdgeRunnerChat/Views/MemoryUsageView.swift
import SwiftUI

struct MemoryUsageView: View {
    let usedMB: Double
    let totalMB: Double

    private var percent: Double {
        guard totalMB > 0 else { return 0 }
        return usedMB / totalMB
    }

    private var color: Color {
        switch percent {
        case 0..<0.5: return .green
        case 0.5..<0.8: return .yellow
        default: return .red
        }
    }

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                    Rectangle()
                        .fill(color)
                        .frame(width: geo.size.width * percent)
                }
            }
            .frame(height: 4)

            HStack {
                Text(String(format: "Memory: %.0f / %.0f MB", usedMB, totalMB))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f%%", percent * 100))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
        }
        .padding(.vertical, 4)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "ChatMessageTests|ModelInfoTests|ChatViewModelLogicTests" 2>&1`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add Examples/EdgeRunnerChat/ \
      Tests/EdgeRunnerTests/ChatViewModelTests.swift
git commit -m "feat(m4): add EdgeRunnerChat demo app with SwiftUI chat interface

Full chat UI with streaming token display, model picker, memory usage bar.
Testable ChatViewModelState with pure state management.
ModelInfo with file size formatting and GGUF file scanning."
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | EdgeRunnerLanguageModel Protocol | 3 created | 8 tests |
| 2 | BPE Tokenizer | 4 created | 15 tests |
| 3 | Sampling Pipeline | 9 created | 25 tests |
| 4 | Speculative Decoding | 1 created | 5 tests |
| 5 | Structured Generation | 4 created | 17 tests |
| 6 | Tool Calling | 4 created | 16 tests |
| 7 | Foundation Models Backend | 3 created | 5 tests |
| 8 | Streaming Token Output | 2 created | 9 tests |
| 9 | DocC Documentation | 7 created | build verification |
| 10 | EdgeRunnerChat Demo App | 8 created | 10 tests |

**Total: ~45 files, ~110 tests, 10 commits**

### File Tree (New in M4)

```
Sources/
├── EdgeRunner/
│   ├── EdgeRunnerLanguageModel.swift
│   ├── ModelConfiguration.swift
│   ├── GenerationError.swift
│   ├── Backends/
│   │   ├── BackendFactory.swift
│   │   ├── EdgeRunnerLocalBackend.swift
│   │   └── FoundationModelsBackend.swift
│   ├── ToolCalling/
│   │   ├── EdgeRunnerTool.swift
│   │   ├── ToolChoice.swift
│   │   ├── ToolCallParser.swift
│   │   └── ToolExecutor.swift
│   ├── Streaming/
│   │   ├── TokenStream.swift
│   │   └── GenerationSession.swift
│   └── Documentation.docc/
│       ├── EdgeRunner.md
│       ├── GettingStarted.md
│       └── Articles/
│           ├── ModelLoading.md
│           ├── StreamingGeneration.md
│           ├── StructuredOutput.md
│           ├── ToolCalling.md
│           └── Sampling.md
├── EdgeRunnerCore/
│   ├── Tokenizer/
│   │   ├── TokenizerProtocol.swift
│   │   ├── SpecialTokens.swift
│   │   ├── TokenizerVocabulary.swift
│   │   └── BPETokenizer.swift
│   ├── Sampling/
│   │   ├── SamplingStrategy.swift
│   │   ├── GreedySampler.swift
│   │   ├── TemperatureSampler.swift
│   │   ├── TopKSampler.swift
│   │   ├── TopPSampler.swift
│   │   ├── MinPSampler.swift
│   │   ├── RepetitionPenalty.swift
│   │   ├── SamplingPipeline.swift
│   │   └── SeededRandomSource.swift
│   ├── Generation/
│   │   └── SpeculativeDecoder.swift
│   └── StructuredGeneration/
│       ├── JSONSchemaExtractor.swift
│       ├── ConstrainedDecoder.swift
│       ├── GrammarState.swift
│       └── StructuredGenerator.swift
Tests/
├── EdgeRunnerTests/
│   ├── EdgeRunnerLanguageModelTests.swift
│   ├── BackendTests.swift
│   ├── ToolCallingTests.swift
│   ├── StreamingTests.swift
│   └── ChatViewModelTests.swift
└── EdgeRunnerCoreTests/
    ├── TokenizerTests.swift
    ├── SamplingTests.swift
    ├── SpeculativeDecodingTests.swift
    └── StructuredGenerationTests.swift
Examples/
└── EdgeRunnerChat/
    ├── EdgeRunnerChatApp.swift
    ├── Views/
    │   ├── ChatView.swift
    │   ├── MessageBubble.swift
    │   ├── ModelPickerView.swift
    │   └── MemoryUsageView.swift
    ├── ViewModels/
    │   └── ChatViewModel.swift
    └── Models/
        ├── ChatMessage.swift
        └── ModelInfo.swift
```
