# Production BPE Tokenizer Integration

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Integrate a production-grade BPE tokenizer into EdgeRunner, replacing the byte-level placeholder in LlamaLanguageModel

## Requirements

- **Generalize across model families:** Qwen, Llama, Mistral, Granite, Phi-4, StarCoder, Command-R, and any future BPE-based models
- **Correctness + performance:** Byte-perfect parity with HuggingFace/llama.cpp tokenizers AND low-latency encoding for 4K+ token prompts
- **Byte-level fallback:** GPT-2 style byte-to-unicode mapping so no input is ever truly unknown
- **Chat template support:** Integrated `applyChatTemplate()` via a minimal Jinja2 engine
- **Extensible:** Protocol-based design allows future SentencePiece/WordPiece tokenizers to plug in

## Architecture

### Approach: Layered Pipeline (Approach A)

Composable pipeline: `PreTokenizer -> ByteEncoder -> BPE Merges -> Vocab Lookup`, with a chat template engine as a separate component.

```
ChatTemplateEngine (optional)
  | formatted text
  v
BPETokenizer
  |-- 1. Special Token Scan (split text around literal special tokens)
  |-- 2. PreTokenizer (regex split per text chunk)
  |-- 3. ByteEncoder (UTF-8 bytes -> GPT-2 unicode chars)
  |-- 4. BPE Merges (apply merge ranks iteratively)
  '-- 5. Vocabulary Lookup (token string -> ID, with byte fallback)
```

### Module Dependency Graph

```
EdgeRunnerSharedTypes
  ^
  |
EdgeRunnerMetal
  ^
  |
EdgeRunnerIO          (GGUF parsing, GGUFTokenizerMetadata)
  ^
  |
EdgeRunnerCore        (Tokenizer types, BPETokenizer, TokenizerFactory, ChatTemplateEngine)
  ^
  |
EdgeRunner            (LlamaLanguageModel integration)
```

`TokenizerFactory` lives in `EdgeRunnerCore` (not `EdgeRunnerIO`) because it needs to import both `EdgeRunnerIO` (for `GGUFTokenizerMetadata`) and `EdgeRunnerCore` types (for `BPETokenizer`, `SpecialTokens`, etc.). `EdgeRunnerCore` already depends on `EdgeRunnerIO` in `Package.swift`.

### File Layout

| File | Module | Purpose |
|------|--------|---------|
| `PreTokenizer.swift` | EdgeRunnerCore/Tokenizer | Protocol + `RegexPreTokenizer` implementation |
| `PreTokenizerPattern.swift` | EdgeRunnerCore/Tokenizer | Registry: `tokenizer.ggml.pre` string -> regex pattern |
| `ByteEncoder.swift` | EdgeRunnerCore/Tokenizer | GPT-2 byte<->unicode 256-entry bidirectional table |
| `BPETokenizer.swift` | EdgeRunnerCore/Tokenizer | Enhanced existing file — replaces current `encode()` with full pipeline |
| `ChatTemplateEngine.swift` | EdgeRunnerCore/Tokenizer | Minimal Jinja2 interpreter |
| `ChatMessage.swift` | EdgeRunnerCore/Tokenizer | Message type for chat formatting |
| `TokenizerFactory.swift` | EdgeRunnerCore/Tokenizer | `GGUFTokenizerMetadata` -> `BPETokenizer` bridge |

## Component Designs

### 1. PreTokenizer

```swift
public protocol PreTokenizer: Sendable {
    func split(_ text: String) -> [String]
}

public struct RegexPreTokenizer: PreTokenizer, Sendable {
    private let pattern: Regex<Substring>
    public func split(_ text: String) -> [String]
}
```

**Pattern registry** maps `tokenizer.ggml.pre` to regex:

| GGUF `pre` value | Pattern Family | Models |
|---|---|---|
| `"gpt-2"` / `"default"` / `nil` / `"granite-docling"` | GPT-2 | GPT-2, MPT, OLMo, JAIS, Granite, Phi-4 |
| `"qwen2"` | Qwen2 (case-insensitive contractions, individual digits) | Qwen2, Qwen3, StableLM2 |
| `"llama3"` / `"llama-v3"` / `"llama4"` | Llama3 (like Qwen2 but `\p{N}{1,3}`) | Llama 3/4, Falcon3, DBRX, Smaug |
| `"tekken"` | Tekken (camelCase-aware) | Mistral |
| `"starcoder"` / `"command-r"` / `"refact"` / `"smollm"` | StarCoder (digit-split first, then GPT-2) | StarCoder, Command-R, SmolLM |
| `"deepseek-llm"` | DeepSeek LLM (multi-regex with CJK support) | DeepSeek v1/v2 |
| `"deepseek-coder"` | DeepSeek Coder (code-optimized split) | DeepSeek Coder |
| `"chatglm-bpe"` | ChatGLM | ChatGLM 3/4 |
| `"viking"` | Viking | NorwAI/Viking models |

Unrecognized `pre` values fall back to GPT-2 pattern (the most common default, also what llama.cpp does).

### 2. ByteEncoder

Static, precomputed GPT-2 byte-to-unicode mapping:

- 188 "nice" bytes (printable ASCII 33-126, Latin-1 161-172, 174-255) map to themselves
- 68 "ugly" bytes (control chars, space, DEL, etc.) map to U+0100-U+0143
- Key mappings: space (0x20) -> `Ġ` (U+0120), newline (0x0A) -> `Ċ` (U+010A), tab (0x09) -> `ĉ` (U+0109)
- Two static dictionaries: `[UInt8: Character]` for encode, `[Character: UInt8]` for decode
- Computed once, shared across all tokenizer instances

### 3. BPE Encoding Pipeline

**Encode:**
1. Scan text for literal special token strings (all control/userDefined tokens from `tokenTypes`, plus BOS/EOS/PAD), split into `[text_chunk | special_token]` segments
2. For each text chunk: apply PreTokenizer regex -> word chunks
3. For each word chunk: convert UTF-8 bytes through ByteEncoder -> unicode string
4. Split unicode string into characters, apply BPE merges by rank (existing `applyMerges` algorithm)
5. Look up each merged token in vocabulary -> token ID
6. Byte-level fallback for unknowns (see below)
7. Special token segments map directly to their IDs

**Byte-level fallback (step 6):**
When a token string after BPE merges is not found in the vocabulary, decompose it into individual byte-encoded characters. Each byte has a corresponding token in GPT-2-style vocabularies. Lookup strategy:
- The byte-encoded unicode character itself IS the vocabulary key (e.g., `Ġ` for byte 0x20)
- Models also have `<0xHH>` format byte tokens (GGUF token type `.byte`, type 6). During factory initialization, build a `[UInt8: Int]` byte fallback table from tokens with `GGUFTokenType.byte`
- Try the unicode character lookup first; fall back to the byte token table
- If neither exists (should never happen in well-formed models), insert UNK token ID if available, otherwise skip

**Decode:**
1. Look up each token ID in vocabulary -> token string
2. If token ID is not found: insert Unicode replacement character U+FFFD (never silently drop)
3. Skip special tokens if `skipSpecialTokens` is true
4. Concatenate all token strings
5. Convert through ByteDecoder (reverse GPT-2 mapping) -> UTF-8 bytes -> String

**Note:** The current `BPETokenizer.encode()` splits text via `text.map { String($0) }` (Swift Characters). This will be **replaced** with the byte-encoded pipeline above, not extended. The merge algorithm (`applyMerges`) is preserved, but its input changes from Swift characters to byte-encoded unicode characters.

### 4. Special Token Handling

The existing `SpecialTokens` struct only handles BOS, EOS, and PAD. Real models have many more special/control tokens (e.g., `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, tool-call markers).

**Enhanced design:**
- `TokenizerFactory` scans `GGUFTokenizerMetadata.tokenTypes` for tokens marked as `.control` (type 3) or `.userDefined` (type 4)
- The existing `SpecialTokens.init` gains an `additionalSpecialTokens: [String: Int] = [:]` parameter
- The additional tokens are merged into the existing `specialTokenMap: [String: Int]` and `specialTokenIDs: Set<Int>` properties (keeping existing property names, not adding new ones)
- BOS, EOS, PAD remain as named convenience properties; they are also included in the shared map/set
- **Important:** All special tokens (including control/userDefined) are already present in the `TokenizerVocabulary` because `metadata.tokens` is the complete vocabulary list. During decode, special token strings are resolved via normal vocabulary lookup — no separate reverse-lookup is needed. The `specialTokenIDs` set is only used for the `skipSpecialTokens` filter and the encode-time literal scan

### 5. Chat Template Engine

Minimal Jinja2 interpreter in two tiers:

**Tier 1 (launch — covers ~80% of models: ChatML, Llama 3, Mistral, Phi, Gemma):**
- `{{ expression }}` output with string concatenation (`+`)
- `{% for item in list %}` with `loop.index`, `loop.index0`, `loop.first`, `loop.last`
- `{% if %}` / `{% elif %}` / `{% else %}`
- `{% set var = expr %}`
- Comparisons: `==`, `!=`, `and`, `or`, `not`, `in`
- Whitespace control: `{%-` / `-%}`
- `| trim` filter

**Tier 2 (fast follow — gets to ~98%: Qwen3 tool calling, DeepSeek R1):**
- `| tojson`, `| length`, `| join` filters
- `is defined`, `is string` tests
- `namespace()` for mutable loop state
- String methods: `.strip()`, `.split()`, `.startswith()`
- Array slicing: `messages[::-1]`

**API:**
```swift
public struct ChatMessage: Sendable, Equatable {
    public let role: String        // "system", "user", "assistant", "tool"
    public let content: String
}

public struct ToolDefinition: Sendable, Equatable {
    public let name: String
    public let description: String
    public let parametersJSON: String  // JSON string — avoids [String: Any] non-Sendable issue
}

public struct ChatTemplateEngine: Sendable {
    public init(template: String) throws  // parses template at init time

    public func apply(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        bosToken: String? = nil,
        eosToken: String? = nil,
        tools: [ToolDefinition]? = nil   // Tier 2 — Sendable-safe
    ) -> String
}
```

**Thread safety:** Template evaluation is purely functional — the AST is immutable after `init`, and `apply()` creates a fresh context dictionary per call. No internal mutable state, so `Sendable` conformance is valid without synchronization.

**Implementation:** Lexer -> Parser (AST) -> Evaluator. Three-stage pipeline. `init(template:)` throws for parse errors. `apply()` also throws for runtime evaluation errors (undefined variable access, type mismatches). Error types:
- `ChatTemplateError.parseError(String)` — malformed template syntax
- `ChatTemplateError.unsupportedFeature(String)` — valid Jinja2 but not in our subset
- `ChatTemplateError.evaluationError(String)` — runtime failure (e.g., accessing undefined variable)

### 6. Chat Template Access Surface

`BPETokenizer` gains an optional `chatTemplateEngine` property:

```swift
public struct BPETokenizer: Tokenizer, Sendable {
    // ... existing properties ...
    public let chatTemplateEngine: ChatTemplateEngine?

    /// vocabularySize is simply vocabulary.count (the total token list from GGUF metadata).
    /// Do NOT add specialTokenIDs.count — special tokens are already included in the vocabulary.
    public var vocabularySize: Int { vocabulary.count }

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        tools: [ToolDefinition]? = nil
    ) -> String?  // nil if no template available
}
```

The `Tokenizer` protocol is NOT extended with chat template methods — chat templating is BPE-specific (SentencePiece models may use different template formats). Consumers that need chat templates work with `BPETokenizer` directly or check at runtime.

`EdgeRunnerLanguageModel` gains `applyChatTemplate` as a **protocol requirement** with a default implementation (not a plain extension method). This ensures dynamic dispatch works correctly when consumers hold an existential `any EdgeRunnerLanguageModel`:

```swift
// In the protocol definition:
public protocol EdgeRunnerLanguageModel {
    // ... existing requirements ...
    func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool
    ) -> String?
}

// Default implementation:
extension EdgeRunnerLanguageModel {
    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String? { nil }
}

// LlamaLanguageModel provides its own conformance:
// delegates to tokenizer.applyChatTemplate(messages:addGenerationPrompt:)
```

### 7. TokenizerFactory

Lives in `EdgeRunnerCore/Tokenizer/TokenizerFactory.swift`.

```swift
public enum TokenizerFactory {
    public static func create(from metadata: GGUFTokenizerMetadata) throws -> BPETokenizer
}
```

**Steps:**
1. Validate `metadata.model` is `.gpt2` or `.llamaBPE`. Throw `TokenizerFactoryError.unsupportedModel(String)` for `.sentencePiece` / `.wordPiece` / `.unknown` with a clear message
2. **Validate EOS exists:** `guard let eosID = metadata.eosTokenID` else throw `TokenizerFactoryError.missingRequiredToken("eos")`. All BPE models have EOS; this is a data integrity check
3. Build `TokenizerVocabulary` from `metadata.tokens` (index = token ID)
4. Scan `metadata.tokenTypes` to collect all control (type 3) and userDefined (type 4) tokens into the special token map
5. Build `SpecialTokens` with BOS/EOS/PAD IDs + the full special token map
6. Build byte fallback table from tokens with `GGUFTokenType.byte` (type 6)
7. Convert `[GGUFTokenizerMerge]` -> `[(String, String)]` tuples
8. Resolve `PreTokenizer` from `metadata.preTokenizer` via pattern registry
9. Create `ChatTemplateEngine` from `metadata.chatTemplate` (if present, wrapped in try?)
10. Return configured `BPETokenizer`

### 8. LlamaLanguageModel Integration

**Before:** Placeholder `Array(text.utf8).map { Int($0) }` with hardcoded Qwen3 token IDs (151645).

**After:**
```swift
private let tokenizer: BPETokenizer  // stored property, created during load()

public func tokenize(_ text: String) -> [Int] {
    tokenizer.encode(text, addBOS: shouldAddBOS)
}

public func detokenize(_ ids: [Int]) -> String {
    tokenizer.decode(ids, skipSpecialTokens: true)
}

public var eosTokenID: Int { tokenizer.eosTokenID }
public var bosTokenID: Int? { tokenizer.bosTokenID }
public var vocabularySize: Int { tokenizer.vocabularySize }
```

**`addBOS` behavior:** The factory reads `metadata.shouldAddBOS` and stores it. `LlamaLanguageModel` passes this value when calling `encode()`. This matches the GGUF model's intent.

**In `LlamaLanguageModel.load()`:**
```swift
let tokenizerMetadata = try modelConfig.tokenizerMetadata()
let tokenizer = try TokenizerFactory.create(from: tokenizerMetadata)
```

If tokenizer creation fails, `load()` throws. No model with a broken tokenizer.

## Out of Scope

- **SentencePiece tokenizer:** Gemma, Nemotron, Phi-3 use SentencePiece. The `Tokenizer` protocol supports adding this later without changes to existing code. `TokenizerFactory` will throw a clear error for these models.
- **WordPiece tokenizer:** BERT-family models. Same extensibility path.
- **External tokenizer files:** `ModelConfiguration.tokenizerURL` for loading tokenizer.json. Future enhancement.

## Testing Strategy

### Unit Tests

- **ByteEncoderTests:** Round-trip all 256 bytes, known mappings (space/newline/tab), printable ASCII maps to itself
- **PreTokenizerTests:** Each pattern family with representative inputs (contractions, digits, CJK, punctuation, whitespace, newlines), unknown `pre` value falls back to GPT-2, empty string returns empty array
- **BPETokenizerTests (enhanced):** Full pipeline encode/decode, special token preservation (`<|im_start|>user` doesn't split the special token), byte fallback for unknown sequences, round-trip `decode(encode(text)) == text` for ASCII/Unicode/emoji/CJK, BOS/EOS insertion, unknown token ID produces replacement character on decode
- **ChatTemplateEngineTests:** ChatML format (Qwen), Llama 3 format, Mistral format, `addGenerationPrompt` behavior, whitespace control, unsupported features throw `ChatTemplateError`
- **SpecialTokensTests:** Extended special token map includes control/userDefined tokens from `tokenTypes`

### Integration Tests

- **TokenizerFactoryTests:** Creates working tokenizer from realistic `GGUFTokenizerMetadata`, throws for SentencePiece model with clear error, throws for missing EOS token, resolves correct pre-tokenizer for each `pre` string value

### Parity Tests

- **QwenTokenizerParityTest:** Load actual Qwen3 0.6B GGUF, create tokenizer via factory, compare encode output against known-good token IDs from HuggingFace/llama.cpp for a set of test strings covering: ASCII, Unicode, mixed scripts, emoji, code, special characters

### Coverage Target

80%+ across all new files, with parity tests as the ultimate correctness gate.

## Performance Considerations

- Pre-tokenizer regex compiled once at init, reused for all calls
- ByteEncoder tables are static constants — zero allocation per call
- Merge rank lookup is O(1) via `[String: Int]` dictionary
- Vocabulary lookup is O(1) via `TokenizerVocabulary`'s bidirectional dictionaries
- Special token scan uses `Set<String>` for O(1) membership testing
- ChatTemplateEngine AST is immutable after init — `apply()` is allocation-light
- Benchmark target: tokenize a 4K-token prompt in under 10ms on Apple Silicon (M-series)
