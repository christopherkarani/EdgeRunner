# Multi-Turn Conversation & Model Auto-Detection Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Add KV cache prefix matching for multi-turn efficiency, Conversation helper struct, and ModelLoader auto-detection

## 1. KV Cache Prefix Matching

### Problem
Currently `LlamaLanguageModel.forwardLogitsBuffer()` only optimizes the single-token append case (decode mode). If a new prompt shares a long prefix with the previous one (typical in multi-turn chat), the entire prompt is recomputed from scratch.

### Solution
Detect common prefix between new token sequence and cached state. Skip recomputation for matching prefix positions.

```
New tokens:  [A, B, C, D, E, F, G]
Cached:      [A, B, C, D]
Common prefix length: 4
→ Rewind KV cache to position 4
→ Only process [E, F, G] through transformer
→ KV positions 0-3 already valid
```

### Logic in `forwardLogitsBuffer()`:
```swift
let commonPrefixLen = zip(tokenIDs, previousTokenIDs).prefix(while: ==).count

if commonPrefixLen == previousTokenIDs.count && tokenIDs.count == commonPrefixLen + 1 {
    // Single-token decode (existing fast path, unchanged)
} else if commonPrefixLen > 0 && commonPrefixLen < tokenIDs.count {
    // Prefix reuse: rewind KV cache, prefill only new suffix
    kvCache.setPosition(commonPrefixLen)
    let newTokens = Array(tokenIDs[commonPrefixLen...])
    // Prefill newTokens starting at KV position commonPrefixLen
} else {
    // No common prefix: reset KV cache, full prefill
}
```

### Integration
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` — `forwardLogitsBuffer()` method
- The KV cache already supports `setPosition()` and `reset()`
- The prefill path already handles arbitrary-length token sequences
- This is a ~20 line change in the existing forward pass logic

## 2. Conversation Struct

Lightweight message history manager. No KV cache logic — that's handled transparently by prefix matching.

```swift
public struct Conversation: Sendable {
    public private(set) var messages: [ChatMessage]

    public init(systemPrompt: String? = nil)
    public mutating func addUser(_ content: String)
    public mutating func addAssistant(_ content: String)
    public mutating func addSystem(_ content: String)
    public var messageCount: Int { messages.count }
    public mutating func reset(keepSystem: Bool = true)
}
```

**Usage pattern** (matches llama.cpp/MLX/ollama):
```swift
var convo = Conversation(systemPrompt: "You are helpful.")
convo.addUser("What is 2+2?")

let prompt = model.applyChatTemplate(messages: convo.messages, addGenerationPrompt: true)!
let tokens = model.tokenize(prompt)
// generate...
convo.addAssistant(response)

// Next turn — KV cache prefix reuse is automatic
convo.addUser("And 3+3?")
let prompt2 = model.applyChatTemplate(messages: convo.messages, addGenerationPrompt: true)!
let tokens2 = model.tokenize(prompt2)
// tokens2 shares long prefix with tokens — only new suffix computed
```

**File:** `Sources/EdgeRunner/Conversation.swift`

## 3. ModelLoader Auto-Detection

Reads `general.architecture` from GGUF metadata and dispatches to the correct model class.

```swift
public enum ModelLoader {
    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> any EdgeRunnerLanguageModel
}
```

Currently all supported architectures (llama, qwen, gemma, phi, mistral) use the same Llama transformer pattern, so they all dispatch to `LlamaLanguageModel`. When new architectures are added, new cases dispatch to different classes.

Recognized architectures: `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`, `mistral`, `starcoder`

**File:** `Sources/EdgeRunner/ModelLoader.swift`

## Files

| File | Action | Purpose |
|------|--------|---------|
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | KV cache prefix matching in `forwardLogitsBuffer()` |
| `Sources/EdgeRunner/Conversation.swift` | Create | Message history manager |
| `Sources/EdgeRunner/ModelLoader.swift` | Create | GGUF architecture auto-detection |
| `Tests/EdgeRunnerTests/ConversationTests.swift` | Create | Conversation unit tests |
| `Tests/EdgeRunnerTests/ModelLoaderTests.swift` | Create | Auto-detection tests |

## Testing

### Conversation Tests
- Add messages, verify history
- Reset with keepSystem=true preserves system prompt
- Reset with keepSystem=false clears everything
- Empty conversation

### ModelLoader Tests
- Load Qwen GGUF → returns LlamaLanguageModel (if model available)
- Unknown architecture string → throws with clear error

### KV Cache Prefix Reuse (integration)
- Two-turn conversation: measure that turn 2 is faster than turn 1 (prefix reuse working)
- Verify output correctness matches full-recompute baseline

## Out of Scope
- Conversation persistence / serialization to disk
- Token counting / context window management
- Automatic truncation of long conversations
- Parallel conversations (multiple KV cache slots)
