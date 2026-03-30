# Streaming API Polish

**Date:** 2026-03-19  
**Status:** Planning  
**Owner:** TBD  
**Effort:** 2-3 days  

---

## Goal

Transform EdgeRunner's primitive streaming into a production-ready API that developers love. The current implementation works but lacks:

1. **Chat completion interface** - No structured conversation support
2. **Progress callbacks** - Can't track generation progress
3. **Rich stream events** - Only string chunks, no metadata
4. **Synchronous convenience methods** - Must use async/await for everything

Target API:

```swift
// Simple completion
let response = try model.complete("What is Swift?")

// Streaming with progress
for await event in model.stream("Tell me a story") {
    switch event {
    case .token(let text): print(text, terminator: "")
    case .stats(let s): print("\n\(s.tokensPerSecond) tok/s")
    }
}

// Chat completion
let chat = ChatSession(model: model)
let reply = try await chat.send(user: "Hello!")
```

---

## Current State

| Component | Status | Issues |
|-----------|--------|--------|
| `GenerationSession` | ✅ Basic | No progress tracking |
| `StreamToken` | ✅ Exists | Not used in streaming |
| `GenerationStats` | ✅ Exists | Not populated |
| `ChatMessage` | ✅ Exists | No chat session management |
| `stream()` default | ⚠️ Basic | Returns `AsyncThrowingStream<String, Error>` |

---

## Implementation Plan

### Task 1: Rich Stream Events (4 hours)

**Current:**
```swift
for try await token: String in model.stream("...")
```

**Target:**
```swift
for try await event: StreamEvent in model.stream("...")
```

**Implementation:**

```swift
/// Events emitted during streaming generation.
public enum StreamEvent: Sendable {
    /// A generated token as text.
    case token(String)
    
    /// Token with full metadata.
    case tokenWithMetadata(StreamToken)
    
    /// Generation statistics update (after first token, periodically).
    case stats(GenerationStats)
    
    /// Generation complete with final stats.
    case complete(GenerationStats)
    
    /// An error occurred during generation.
    case error(GenerationError)
}

/// Options for streaming behavior.
public struct StreamOptions: Sendable {
    /// Include metadata with each token (token ID, logprobs).
    public var includeMetadata: Bool
    
    /// Report statistics every N tokens (0 = only at end).
    public var statsInterval: Int
    
    /// Callback for token-level access (synchronous, for logging).
    public var onToken: (@Sendable (StreamToken) -> Void)?
    
    public init(
        includeMetadata: Bool = false,
        statsInterval: Int = 0,
        onToken: (@Sendable (StreamToken) -> Void)? = nil
    ) {
        self.includeMetadata = includeMetadata
        self.statsInterval = statsInterval
        self.onToken = onToken
    }
}
```

**Files to modify:**
- `Sources/EdgeRunner/Streaming/TokenStream.swift` - Add `StreamEvent` enum
- `Sources/EdgeRunner/Streaming/GenerationSession.swift` - Add `streamEvents()` method

**Tests:**
- Stream produces correct event sequence
- Stats are calculated accurately
- Metadata is included when requested

---

### Task 2: Progress Tracking (3 hours)

**Goal:** Enable progress bars and real-time throughput monitoring.

**Implementation:**

```swift
extension GenerationSession {
    /// Stream with progress tracking.
    ///
    /// - Parameters:
    ///   - prompt: The input prompt.
    ///   - options: Streaming options.
    ///   - onProgress: Called periodically with generation stats.
    /// - Returns: An async stream of generated text.
    public func stream(
        prompt: String,
        options: StreamOptions = .init(),
        onProgress: (@Sendable (GenerationStats) -> Void)? = nil
    ) -> AsyncThrowingStream<String, Error>
}

/// Enhanced stats with detailed metrics.
public struct GenerationStats: Sendable {
    public var tokenCount: Int
    public var timeToFirstToken: Double
    public var totalTime: Double
    public var tokensPerSecond: Double { ... }
    
    // NEW:
    /// Estimated completion percentage (0.0 - 1.0).
    public var progress: Double { Double(tokenCount) / Double(maxTokens) }
    
    /// Input token count.
    public var inputTokens: Int
    
    /// Output token count.
    public var outputTokens: Int
    
    /// Wall clock time for the most recent token.
    public var lastTokenLatency: Double
    
    /// Average latency over the last N tokens.
    public var recentLatency: Double
}
```

**Files to modify:**
- `Sources/EdgeRunner/Streaming/GenerationSession.swift`
- `Sources/EdgeRunner/Streaming/TokenStream.swift`

**Tests:**
- Progress increases monotonically
- TTFT is accurate (within 10ms)
- Throughput calculation matches actual rate

---

### Task 3: Chat Completion API (6 hours)

**Goal:** OpenAI-compatible chat interface with conversation history.

**Implementation:**

```swift
/// A chat completion request.
public struct ChatCompletionRequest: Sendable {
    public var messages: [ChatMessage]
    public var model: String?
    public var temperature: Double?
    public var maxTokens: Int?
    public var stream: Bool
    
    public init(
        messages: [ChatMessage],
        model: String? = nil,
        temperature: Double? = nil,
        maxTokens: Int? = 2048,
        stream: Bool = false
    ) {
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.stream = stream
    }
}

/// A chat completion response.
public struct ChatCompletion: Sendable {
    public let id: String
    public let message: ChatMessage
    public let usage: UsageStats
    public let finishReason: FinishReason
    
    public struct UsageStats: Sendable {
        public let promptTokens: Int
        public let completionTokens: Int
        public let totalTokens: Int
    }
    
    public enum FinishReason: String, Sendable {
        case stop, length, error
    }
}

/// Manages a chat conversation with a model.
public actor ChatSession {
    private let model: any EdgeRunnerLanguageModel
    private let sampling: SamplingConfiguration
    private var history: [ChatMessage]
    private let systemPrompt: String?
    
    public init(
        model: any EdgeRunnerLanguageModel,
        systemPrompt: String? = nil,
        sampling: SamplingConfiguration = .init()
    ) {
        self.model = model
        self.sampling = sampling
        self.systemPrompt = systemPrompt
        self.history = []
    }
    
    /// Send a message and get a response (non-streaming).
    public func send(user message: String) async throws -> ChatMessage {
        // 1. Format conversation with chat template
        // 2. Generate response
        // 3. Update history
        // 4. Return assistant message
    }
    
    /// Send a message with streaming response.
    public func sendStream(
        user message: String
    ) -> AsyncThrowingStream<ChatStreamEvent, Error>
    
    /// Clear conversation history.
    public func clearHistory() {
        history.removeAll()
    }
    
    /// Get current conversation history.
    public func getHistory() -> [ChatMessage] {
        history
    }
}

/// Events from a chat streaming session.
public enum ChatStreamEvent: Sendable {
    case content(String)
    case message(ChatMessage)
    case stats(GenerationStats)
    case complete(FinishReason)
}
```

**Chat Template Support:**

```swift
/// Loads chat template from GGUF metadata and applies it.
struct ChatTemplate {
    let template: String  // Jinja2-style template from GGUF
    
    /// Apply template to messages.
    func apply(messages: [ChatMessage]) -> String
}

// Example templates from GGUF:
// Qwen: "<|im_start|>user\n{{message}}<|im_end|>\n<|im_start|>assistant\n"
// Llama3: "<|start_header_id|>user<|end_header_id|>\n\n{{message}}<|eot_id|>..."
```

**Files to create:**
- `Sources/EdgeRunner/Chat/ChatSession.swift`
- `Sources/EdgeRunner/Chat/ChatTemplate.swift`
- `Sources/EdgeRunner/Chat/ChatCompletion.swift`

**Files to modify:**
- `Sources/EdgeRunner/Chat/ChatMessage.swift` - Add `usage` and metadata

**Tests:**
- Chat history is maintained correctly
- Template formatting matches reference
- Streaming chat produces correct events
- System prompts work

---

### Task 4: Synchronous Convenience API (2 hours)

**Goal:** Simple blocking API for scripts and quick usage.

**Implementation:**

```swift
extension EdgeRunnerLanguageModel {
    /// Generate a completion (blocking, non-streaming).
    ///
    /// ```swift
    /// let response = try model.complete("What is Swift?")
    /// print(response)
    /// ```
    public func complete(
        _ prompt: String,
        maxTokens: Int = 256,
        sampling: SamplingConfiguration = .init()
    ) async throws -> String {
        let session = GenerationSession(
            model: self,
            samplingPipeline: SamplingPipeline(configuration: sampling),
            maxTokens: maxTokens
        )
        return try await session.generate(prompt: prompt)
    }
    
    /// Generate with progress callback.
    public func complete(
        _ prompt: String,
        maxTokens: Int = 256,
        sampling: SamplingConfiguration = .init(),
        onProgress: @escaping (GenerationStats) -> Void
    ) async throws -> String {
        var result = ""
        let stream = self.stream(prompt, maxTokens: maxTokens, sampling: sampling)
        for try await event in stream {
            switch event {
            case .token(let text):
                result += text
            case .stats(let stats):
                onProgress(stats)
            default: break
            }
        }
        return result
    }
}

// MARK: - Fluent API

extension EdgeRunnerLanguageModel {
    /// Start building a generation request.
    public func generate(_ prompt: String) -> GenerationRequestBuilder {
        GenerationRequestBuilder(model: self, prompt: prompt)
    }
}

/// Builder for fluent API.
public struct GenerationRequestBuilder {
    private let model: any EdgeRunnerLanguageModel
    private let prompt: String
    private var maxTokens: Int = 256
    private var sampling: SamplingConfiguration = .init()
    private var stream: Bool = false
    
    public func maxTokens(_ count: Int) -> Self {
        var copy = self
        copy.maxTokens = count
        return copy
    }
    
    public func temperature(_ value: Double) -> Self {
        var copy = self
        copy.sampling = SamplingConfiguration(temperature: value)
        return copy
    }
    
    public func stream() -> Self {
        var copy = self
        copy.stream = true
        return copy
    }
    
    /// Execute the request.
    public func execute() async throws -> String {
        try await model.complete(
            prompt,
            maxTokens: maxTokens,
            sampling: sampling
        )
    }
    
    /// Execute with streaming.
    public func executeStream() -> AsyncThrowingStream<String, Error> {
        // Return streaming sequence
    }
}
```

**Usage:**

```swift
// Simple
let response = try await model.complete("Explain Swift concurrency")

// With options
let response = try await model
    .generate("Write a poem")
    .maxTokens(500)
    .temperature(0.8)
    .execute()

// Streaming
for try await token in model
    .generate("Tell me a story")
    .stream() {
    print(token, terminator: "")
}
```

**Files to create:**
- `Sources/EdgeRunner/API/ConvenienceAPI.swift` - Extension on `EdgeRunnerLanguageModel`

**Tests:**
- Blocking API returns correct result
- Builder pattern chains correctly
- Streaming builder returns AsyncStream

---

### Task 5: Documentation & Examples (3 hours)

**DocC Articles:**

1. **StreamingGeneration.md** (update existing)
   - Rich stream events
   - Progress tracking
   - Cancellation patterns

2. **ChatCompletion.md** (new)
   - Chat session basics
   - Conversation history
   - System prompts
   - Template customization

3. **ConvenienceAPI.md** (new)
   - One-shot completion
   - Builder pattern
   - When to use async vs sync

**Code Examples:**

```swift
// Examples/EdgeRunnerExamples/StreamingExample.swift
// Examples/EdgeRunnerExamples/ChatExample.swift
// Examples/EdgeRunnerExamples/ProgressTrackingExample.swift
```

**Files to modify:**
- Update `Sources/EdgeRunner/Documentation.docc/Articles/StreamingGeneration.md`
- Create `Sources/EdgeRunner/Documentation.docc/Articles/ChatCompletion.md`
- Update `README.md` with new API examples

---

## Verification Checklist

- [ ] `streamEvents()` produces correct `StreamEvent` sequence
- [ ] Progress tracking shows increasing percentages
- [ ] TTFT and throughput are accurate
- [ ] ChatSession maintains conversation history
- [ ] Chat template produces correct formatting
- [ ] Synchronous `complete()` works without explicit Task
- [ ] Builder API chains correctly
- [ ] All examples compile and run
- [ ] Documentation renders correctly in Xcode

---

## Dependencies

**Soft Dependencies** (nice to have but not blocking):
- BPE Tokenizer - chat templates work better with real tokenization
- Chat template from GGUF - can hardcode Qwen3 template initially

**None** - This work is independent of tokenizer implementation.

---

## Success Criteria

1. Developer can write complete chat app in <20 lines:
   ```swift
   let model = try await LlamaLanguageModel.load(...)
   let chat = ChatSession(model: model)
   for try await event in chat.sendStream(user: "Hello!") {
       if case .content(let text) = event { print(text) }
   }
   ```

2. Progress tracking enables UI progress bars

3. API feels "Swifty" - uses async/await, result builders, etc.

4. Performance overhead < 5% vs raw `logits()` calls
