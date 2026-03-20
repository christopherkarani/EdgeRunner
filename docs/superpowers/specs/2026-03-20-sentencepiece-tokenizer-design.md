# SentencePiece Tokenizer Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Add SentencePiece (Unigram) tokenizer support for Gemma, Phi-3, and similar models

## Requirements

- Support GGUF models with `tokenizer.ggml.model = "llama"` (SentencePiece)
- Greedy bigram merging by float score (matching llama.cpp's `LLAMA_VOCAB_TYPE_SPM`)
- `▁` (U+2581) space handling instead of GPT-2 byte encoding
- Byte fallback via `<0xHH>` tokens for unknown characters
- Parity validation against HuggingFace Gemma 3 1B tokenizer
- Plug into existing `Tokenizer` protocol and `TokenizerFactory`

## Architecture

```
Tokenizer (protocol)
  ├── BPETokenizer              (existing — GPT-2/Qwen/Llama3/Mistral)
  └── SentencePieceTokenizer    (new — Gemma/Phi-3)
```

`TokenizerFactory.create(from:)` dispatches based on `GGUFTokenizerModel`:
- `.gpt2`, `.llamaBPE` → `BPETokenizer`
- `.llama`, `.sentencePiece` → `SentencePieceTokenizer`

## Component Designs

### 1. GGUFTokenizerMetadata Changes

**New model case:**
```swift
case llama  // "llama" in GGUF → SentencePiece models (Gemma, Phi-3)
```

Mapping: `"llama"` → `.llama`. Existing `.sentencePiece` stays for literal `"sentencepiece"` strings.

**New properties:**
```swift
public let scores: [Float]?          // tokenizer.ggml.scores (one per token)
public let unknownTokenID: Int?      // tokenizer.ggml.unknown_token_id
public let addSpacePrefix: Bool?     // tokenizer.ggml.add_space_prefix
```

**New accessor on MetadataValue and GGUFMetadataValue:**
```swift
var floatArrayValue: [Float]?
```

**Validation:** If `scores` is present, its count must match `tokens.count`.

### 2. SentencePiece Encoding Pipeline

**Encode:**
1. If `addSpacePrefix` (default true): prepend `▁` to input
2. Replace all spaces with `▁` (U+2581)
3. Split into individual UTF-8 characters
4. Greedy bigram merge: for each adjacent pair, look up concatenation in vocabulary. If found, the pair's priority = token's float score. Merge the highest-scoring pair. Repeat until no mergeable pairs remain
5. Vocabulary lookup: token string → token ID
6. Byte fallback for unknowns: decompose to individual bytes, look up `<0xHH>` tokens (type 6)

**Decode:**
1. Vocabulary lookup: token ID → token string (U+FFFD for unknown IDs)
2. Skip special tokens if `skipSpecialTokens` is true
3. Concatenate all token strings
4. Replace all `▁` with spaces
5. Strip leading space (if `addSpacePrefix` was true)

### 3. Greedy Bigram Merge Algorithm

Same structure as `BPETokenizer.applyMerges()` but:
- Instead of `mergeRanks: [String: Int]` (lower rank = merge first), uses `tokenScores: [String: Float]` (higher score = merge first)
- Each iteration: scan all adjacent pairs, find the pair whose concatenation has the highest score in vocabulary, merge it
- Stop when no adjacent pair's concatenation exists in vocabulary

```swift
private func applyMerges(_ tokens: [String]) -> [String] {
    var tokens = tokens
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
        // merge tokens[bestIndex] and tokens[bestIndex + 1]
        ...
    }
    return tokens
}
```

### 4. SentencePieceTokenizer API

```swift
public struct SentencePieceTokenizer: Tokenizer, Sendable {
    public let shouldAddBOS: Bool
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int
    public var eosTokenID: Int
    public var bosTokenID: Int?
    public var padTokenID: Int?

    public func encode(_ text: String, addBOS: Bool) -> [Int]
    public func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool,
        tools: [ToolDefinition]?
    ) throws -> String?
}
```

### 5. TokenizerFactory Changes

`TokenizerFactory` gains a private `createSentencePiece(from:)` method:
1. Validate `scores` exists and `scores.count == tokens.count`
2. Build `TokenizerVocabulary` from `metadata.tokens`
3. Build `[String: Float]` score lookup from tokens + scores
4. Collect control/userDefined tokens as additional special tokens
5. Build `SpecialTokens` with BOS/EOS/PAD + additional
6. Build byte fallback table from type-6 tokens (`<0xHH>` format)
7. Read `addSpacePrefix` (default `true`)
8. Read `shouldAddBOS` (default `true` for SPM, unlike `false` for BPE)
9. Create `ChatTemplateEngine` from `chatTemplate` if present
10. Return `SentencePieceTokenizer`

`TokenizerFactory.create(from:)` dispatch:
```swift
case .gpt2, .llamaBPE:
    return try createBPE(from: metadata)
case .llama, .sentencePiece:
    return try createSentencePiece(from: metadata)
```

**Return type change:** `TokenizerFactory.create(from:)` must return `any Tokenizer` (not `BPETokenizer`) since it can now return either type.

### 6. LlamaLanguageModel Changes

The stored `tokenizer` property type changes from `BPETokenizer?` to `(any Tokenizer)?`. The `applyChatTemplate` method needs adjustment since `ChatTemplateEngine` access is tokenizer-type-specific.

Option: Add `applyChatTemplate` to the `Tokenizer` protocol, or use a `ChatCapable` protocol. Simplest: add optional `applyChatTemplate` directly to the `Tokenizer` protocol with a default nil implementation.

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `Sources/EdgeRunnerCore/Tokenizer/SentencePieceTokenizer.swift` | Create | New tokenizer implementation |
| `Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift` | Modify | Add `.llama`, `scores`, `unknownTokenID`, `addSpacePrefix` |
| `Sources/EdgeRunnerIO/WeightMap.swift` | Modify | Add `floatArrayValue` to `MetadataValue` |
| `Sources/EdgeRunnerCore/Tokenizer/TokenizerFactory.swift` | Modify | Dispatch `.llama`/`.sentencePiece` → `SentencePieceTokenizer`, return `any Tokenizer` |
| `Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift` | Modify | Add optional `applyChatTemplate` to protocol |
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | Change tokenizer type to `(any Tokenizer)?` |
| `Tests/EdgeRunnerCoreTests/SentencePieceTokenizerTests.swift` | Create | Unit tests |
| `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift` | Create | Parity validation |

## Testing Strategy

### Unit Tests — `SentencePieceTokenizerTests.swift`
- Greedy merge follows score order (higher score merges first)
- Space prefix: `▁` prepended and removed in round-trip
- Byte fallback for unknown characters via `<0xHH>`
- Round-trip encode/decode
- Empty string, single character edge cases
- BOS/EOS insertion

### Integration Tests — `TokenizerFactoryTests.swift`
- Factory creates `SentencePieceTokenizer` for `.llama` model
- Factory creates `SentencePieceTokenizer` for `.sentencePiece` model
- Scores validation (count mismatch throws)

### Parity Tests — `GemmaTokenizerParityTest.swift`
- Model: Gemma 3 1B IT Q4_K_M at `/tmp/edgerunner-models/gemma-3-1b-it-Q4_K_M.gguf`
- Environment-gated: `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1`
- 8 reference test strings with HuggingFace reference IDs:
  - `'Hello, world!'`: `[9259, 236764, 1902, 236888]`
  - `'The capital of France is'`: `[818, 5279, 529, 7001, 563]`
  - `'1+1=2'`: `[236770, 236862, 236770, 236784, 236778]`
  - `'def foo():\n    return 42'`: `[2063, 46293, 6141, 107, 140, 2060, 236743, 236812, 236778]`
  - `"I'm don't can't"`: `[236777, 236789, 236757, 1537, 236789, 236745, 740, 236789, 236745]`
  - `'Hello 你好'`: `[9259, 43758, 237389]`
  - `'  spaces  and\ttabs'`: `[138, 35220, 138, 624, 255968, 39218]`
  - `'emoji: 🎉🚀'`: `[67906, 236787, 204906, 242015]`
- Round-trip validation for all test strings

## Out of Scope
- Viterbi DP algorithm (T5/UGM models)
- `precompiled_charsmap` normalization (T5-specific)
- Training / vocabulary modification
