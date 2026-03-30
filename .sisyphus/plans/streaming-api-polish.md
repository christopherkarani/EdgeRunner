# Streaming API Polish

**Date:** 2026-03-20  
**Status:** Ready for Execution  
**Owner:** Sisyphus  
**Effort:** 2-3 days

---

## TL;DR

Transform EdgeRunner's primitive streaming into a production-ready API with:
1. Rich stream events (`StreamEvent` enum with tokens, stats, completion)
2. Progress tracking (TTFT, TPS, progress percentage)
3. Chat completion API (`ChatSession` with conversation history)
4. Synchronous convenience methods (`complete()`, builder pattern)
5. Documentation and examples

**Estimated Effort:** Medium (~15-20 hours)  
**Parallel Execution:** YES - Tasks 1, 2, 4 can proceed in parallel after foundation  
**Critical Path:** Task 1 → Task 3 (Chat depends on stream events)

---

## Context

### Current State Analysis

| Component | File | Status | What's Missing |
|-----------|------|--------|----------------|
| `GenerationSession` | `Sources/EdgeRunner/Streaming/GenerationSession.swift` | Basic streaming | No stats tracking, no events, no progress |
| `StreamToken` | `Sources/EdgeRunner/Streaming/TokenStream.swift` | Exists | Not used in streaming flow |
| `GenerationStats` | `Sources/EdgeRunner/Streaming/TokenStream.swift` | Basic | Missing TTFT, progress %, input/output token counts |
| `ChatMessage` | `Sources/EdgeRunner/Chat/ChatMessage.swift` | Exists | No session management, no templates |
| `stream()` | `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift` | Returns `AsyncThrowingStream<String, Error>` | No events, no metadata, no stats |

### Key Findings

1. **GenerationSession** (82 lines): Simple loop yielding strings, no timing, no stats collection
2. **StreamToken**: Has `id`, `text`, `isEOS` but is NOT used in the actual streaming path
3. **GenerationStats**: Only has `tokenCount`, `timeToFirstToken`, `totalTime`, `tokensPerSecond` - missing progress metrics
4. **ChatMessage**: Basic struct with `role`, `content`, `timestamp` - no conversation management
5. **No ChatSession**: No actor-based conversation state management
6. **No convenience API**: Must use `GenerationSession` directly for everything

---

## Work Objectives

### Core Objective
Create a developer-friendly streaming API with rich events, progress tracking, chat sessions, and synchronous convenience methods.

### Concrete Deliverables

1. **StreamEvent enum** with `.token`, `.tokenWithMetadata`, `.stats`, `.complete`, `.error` cases
2. **Enhanced GenerationStats** with progress %, input/output tokens, latency tracking
3. **ChatSession actor** with conversation history and system prompts
4. **ChatTemplate** for applying GGUF chat templates (start with Qwen3)
5. **Convenience API** with `complete()` and `GenerationRequestBuilder`
6. **Documentation** - DocC articles and code examples

### Definition of Done
- All verification commands in TODOs pass
- Examples compile and run
- API feels "Swifty" (async/await, result builders where appropriate)

### Must Have
- Rich stream events working
- Progress tracking accurate (within 10ms for TTFT)
- Chat session maintains history correctly
- Synchronous API works without explicit Task

### Must NOT Have
- Breaking changes to existing `stream()` method
- Complex chat template parsing (Jinja2) - hardcode Qwen3 initially
- BPE tokenizer dependency - use existing tokenization

---

## Verification Strategy

### Test Infrastructure
- **Framework**: Swift Testing (`import Testing`)
- **Existing tests**: Check `Tests/EdgeRunnerTests/` for patterns
- **New tests needed**: Stream event sequence, stats accuracy, chat history

### QA Policy
Every task includes agent-executed QA scenarios:
- **Unit tests**: Swift Testing with `@Test` macros
- **Integration**: Run examples, verify output
- **Evidence**: Test results captured in `.sisyphus/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - can start immediately):
├── Task 1: Rich Stream Events
│   └── Creates StreamEvent enum, streamEvents() method
├── Task 2: Progress Tracking  
│   └── Enhances GenerationStats, adds progress callbacks
└── Task 4: Synchronous Convenience API (partial)
    └── Creates complete() method, basic builder

Wave 2 (Core Features - after Wave 1):
├── Task 3: Chat Completion API
│   └── ChatSession actor, ChatTemplate, ChatCompletion structs
│   └── Depends on: Task 1 (streamEvents), Task 2 (stats)
└── Task 4: Builder pattern completion
    └── Finish GenerationRequestBuilder with streaming support

Wave 3 (Documentation & Polish):
├── Task 5: Documentation & Examples
│   └── DocC articles, README updates, code examples
└── Final verification and integration tests

Wave FINAL (Verification):
├── F1: Plan compliance audit
├── F2: Code quality review
├── F3: Real manual QA (run examples)
└── F4: Scope fidelity check
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 (Stream Events) | - | 3 |
| 2 (Progress) | - | 3, 4 |
| 3 (Chat) | 1, 2 | - |
| 4 (Convenience) | 2 | - |
| 5 (Docs) | 1, 2, 3, 4 | - |

---

## TODOs

### Task 1: Rich Stream Events

- [ ] 1.1 Create `StreamEvent` enum in `Sources/EdgeRunner/Streaming/StreamEvent.swift`

  **What to do:**
  - Create enum with cases: `.token(String)`, `.tokenWithMetadata(StreamToken)`, `.stats(GenerationStats)`, `.complete(GenerationStats)`, `.error(GenerationError)`
  - Make it `Sendable`
  - Add documentation comments for each case

  **Must NOT do:**
  - Change existing `stream()` method signature (keep for backward compat)
  - Add complex event types beyond what's specified

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization:**
  - **Can Run In Parallel**: YES (with Task 2, 4)
  - **Blocks**: Task 3 (Chat needs stream events)

  **References:**
  - Pattern: `Sources/EdgeRunner/Streaming/TokenStream.swift` - existing `StreamToken` struct
  - Pattern: `Sources/EdgeRunnerCore/GenerationError.swift` - error enum pattern

  **Acceptance Criteria:**
  - [ ] `StreamEvent` enum compiles
  - [ ] All cases have documentation
  - [ ] `swift build` passes

  **QA Scenarios:**
  ```
  Scenario: StreamEvent enum compiles
    Tool: Bash
    Steps:
      1. swift build 2>&1 | grep -i error
    Expected Result: No errors related to StreamEvent
    Evidence: .sisyphus/evidence/task-1-1-build.log
  ```

  **Commit**: YES
  - Message: `feat(streaming): add StreamEvent enum for rich stream events`
  - Files: `Sources/EdgeRunner/Streaming/StreamEvent.swift`

- [ ] 1.2 Create `StreamOptions` struct

  **What to do:**
  - Create `StreamOptions` with `includeMetadata: Bool`, `statsInterval: Int`, `onToken: ((StreamToken) -> Void)?`
  - Make it `Sendable`
  - Add sensible defaults

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: YES

  **References:**
  - Pattern: `Sources/EdgeRunner/ModelConfiguration.swift` - configuration struct pattern

  **Acceptance Criteria:**
  - [ ] `StreamOptions` struct compiles with default init
  - [ ] All properties have public access

  **Commit**: YES (group with 1.1)

- [ ] 1.3 Add `streamEvents()` method to `GenerationSession`

  **What to do:**
  - Add new method `streamEvents(prompt: String, options: StreamOptions = .init()) -> AsyncThrowingStream<StreamEvent, Error>`
  - Track timing (start time, first token time)
  - Emit `.token`, `.stats` (periodically), `.complete` events
  - Handle errors with `.error` event

  **Must NOT do:**
  - Remove or modify existing `stream()` method
  - Change internal generation logic (keep using `nextToken`)

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - **Skills**: None

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 1.1, 1.2)

  **References:**
  - Implementation base: `Sources/EdgeRunner/Streaming/GenerationSession.swift` lines 24-71
  - Timing pattern: Use `CFAbsoluteTimeGetCurrent()` or `DispatchTime.now()`

  **Acceptance Criteria:**
  - [ ] `streamEvents()` method exists and compiles
  - [ ] Returns `AsyncThrowingStream<StreamEvent, Error>`
  - [ ] Emits at least: `.token` for each token, `.complete` at end

  **QA Scenarios:**
  ```
  Scenario: streamEvents produces correct sequence
    Tool: Bash (swift test)
    Steps:
      1. Create test that collects events from streamEvents()
      2. Verify sequence: [.token, .token, ..., .complete]
    Expected Result: Events in correct order, .complete at end
    Evidence: .sisyphus/evidence/task-1-3-events.log
  ```

  **Commit**: YES
  - Message: `feat(streaming): add streamEvents() method with rich events`
  - Files: `Sources/EdgeRunner/Streaming/GenerationSession.swift`

### Task 2: Progress Tracking

- [ ] 2.1 Enhance `GenerationStats` struct

  **What to do:**
  - Add fields: `inputTokens: Int`, `outputTokens: Int`, `lastTokenLatency: Double`, `recentLatency: Double`, `maxTokens: Int`
  - Add computed property: `progress: Double { Double(outputTokens) / Double(maxTokens) }`
  - Keep existing fields: `tokenCount`, `timeToFirstToken`, `totalTime`, `tokensPerSecond`

  **Must NOT do:**
  - Remove existing fields (backward compatibility)
  - Make fields optional if they weren't before

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: YES (with Task 1)

  **References:**
  - Current: `Sources/EdgeRunner/Streaming/TokenStream.swift` lines 14-22

  **Acceptance Criteria:**
  - [ ] All new fields added
  - [ ] `progress` computed property works
  - [ ] `swift build` passes

  **Commit**: YES
  - Message: `feat(streaming): enhance GenerationStats with progress tracking`
  - Files: `Sources/EdgeRunner/Streaming/TokenStream.swift`

- [ ] 2.2 Update `GenerationSession` to populate stats

  **What to do:**
  - Track start time, first token time
  - Count input tokens (from tokenized prompt)
  - Count output tokens (generated)
  - Calculate latencies
  - Pass stats to callbacks and events

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 2.1)

  **References:**
  - Timing: `CFAbsoluteTimeGetCurrent()` for high precision
  - Token counting: `model.tokenize(prompt).count`

  **Acceptance Criteria:**
  - [ ] TTFT is accurate (within 10ms)
  - [ ] Token counts are correct
  - [ ] Progress percentage increases monotonically

  **QA Scenarios:**
  ```
  Scenario: Stats accuracy verification
    Tool: Swift Testing
    Steps:
      1. Generate with known token count
      2. Verify outputTokens matches actual generated
      3. Verify TTFT > 0 and < totalTime
    Expected Result: All stats within expected ranges
    Evidence: .sisyphus/evidence/task-2-2-stats.log
  ```

  **Commit**: YES (group with 2.1)

- [ ] 2.3 Add progress callback to `stream()` method

  **What to do:**
  - Add `onProgress: ((GenerationStats) -> Void)?` parameter to `stream()`
  - Call callback when stats interval is reached
  - Default to no-op if not provided

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 2.2)

  **Acceptance Criteria:**
  - [ ] Callback is called periodically during generation
  - [ ] Callback receives accurate stats

  **Commit**: YES
  - Message: `feat(streaming): add progress callback to stream()`
  - Files: `Sources/EdgeRunner/Streaming/GenerationSession.swift`

### Task 3: Chat Completion API

- [ ] 3.1 Create `ChatCompletionRequest` struct

  **What to do:**
  - Create in `Sources/EdgeRunner/Chat/ChatCompletion.swift`
  - Fields: `messages: [ChatMessage]`, `model: String?`, `temperature: Double?`, `maxTokens: Int?`, `stream: Bool`
  - Make `Sendable`

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Task 1, 2)

  **References:**
  - Pattern: OpenAI API structure (for familiarity)

  **Acceptance Criteria:**
  - [ ] Struct compiles with default init

  **Commit**: YES
  - Message: `feat(chat): add ChatCompletionRequest struct`
  - Files: `Sources/EdgeRunner/Chat/ChatCompletion.swift`

- [ ] 3.2 Create `ChatCompletion` response struct

  **What to do:**
  - Fields: `id: String`, `message: ChatMessage`, `usage: UsageStats`, `finishReason: FinishReason`
  - Nested `UsageStats` with `promptTokens`, `completionTokens`, `totalTokens`
  - `FinishReason` enum: `.stop`, `.length`, `.error`

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: YES (with 3.1)

  **Acceptance Criteria:**
  - [ ] All types compile

  **Commit**: YES (group with 3.1)

- [ ] 3.3 Create `ChatTemplate` struct

  **What to do:**
  - Create in `Sources/EdgeRunner/Chat/ChatTemplate.swift`
  - Hardcode Qwen3 template initially: `"<|im_start|>{role}\n{content}<|im_end|>\n"`
  - Method: `apply(messages: [ChatMessage]) -> String`
  - Support roles: user, assistant, system

  **Must NOT do:**
  - Parse Jinja2 templates (out of scope)
  - Support all possible templates (just Qwen3 for now)

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 3.1, 3.2)

  **References:**
  - Qwen3 format: See plan document for template format

  **Acceptance Criteria:**
  - [ ] Applies template correctly to message list
  - [ ] Produces valid Qwen3-formatted prompt

  **QA Scenarios:**
  ```
  Scenario: Chat template produces correct format
    Tool: Swift Testing
    Steps:
      1. Create messages: [system, user, assistant]
      2. Apply template
      3. Verify output contains <|im_start|> markers
    Expected Result: Valid Qwen3 chat format
    Evidence: .sisyphus/evidence/task-3-3-template.log
  ```

  **Commit**: YES
  - Message: `feat(chat): add ChatTemplate with Qwen3 support`
  - Files: `Sources/EdgeRunner/Chat/ChatTemplate.swift`

- [ ] 3.4 Create `ChatSession` actor

  **What to do:**
  - Create in `Sources/EdgeRunner/Chat/ChatSession.swift`
  - Properties: `model`, `sampling`, `history: [ChatMessage]`, `systemPrompt: String?`
  - Methods:
    - `send(user: String) async throws -> ChatMessage` (non-streaming)
    - `sendStream(user: String) -> AsyncThrowingStream<ChatStreamEvent, Error>`
    - `clearHistory()`
    - `getHistory() -> [ChatMessage]`

  **Must NOT do:**
  - Use class instead of actor (must be thread-safe)
  - Store state outside actor isolation

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - **Skills**: None

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 3.3)

  **References:**
  - Actor pattern: Swift rules for actor-based persistence
  - Generation: `GenerationSession` for streaming pattern

  **Acceptance Criteria:**
  - [ ] Actor compiles with all methods
  - [ ] History is maintained correctly across sends
  - [ ] System prompt is included in first message

  **QA Scenarios:**
  ```
  Scenario: Chat session maintains history
    Tool: Swift Testing
    Steps:
      1. Create ChatSession
      2. Send "Hello" -> get response
      3. Send "What's my name?" 
      4. Check history has 4 messages (user1, assistant1, user2, assistant2)
    Expected Result: History correctly maintained
    Evidence: .sisyphus/evidence/task-3-4-history.log
  ```

  **Commit**: YES
  - Message: `feat(chat): add ChatSession actor for conversation management`
  - Files: `Sources/EdgeRunner/Chat/ChatSession.swift`

- [ ] 3.5 Create `ChatStreamEvent` enum

  **What to do:**
  - Cases: `.content(String)`, `.message(ChatMessage)`, `.stats(GenerationStats)`, `.complete(FinishReason)`
  - Make `Sendable`

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: YES (with 3.4)

  **Acceptance Criteria:**
  - [ ] Enum compiles

  **Commit**: YES (group with 3.4)

### Task 4: Synchronous Convenience API

- [ ] 4.1 Create `ConvenienceAPI.swift` with `complete()` method

  **What to do:**
  - Create in `Sources/EdgeRunner/API/ConvenienceAPI.swift`
  - Extension on `EdgeRunnerLanguageModel`
  - Method: `complete(_ prompt: String, maxTokens: Int = 256, sampling: SamplingConfiguration = .init()) async throws -> String`
  - Uses `GenerationSession` internally

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: YES (with Task 1, 2)

  **References:**
  - Pattern: `GenerationSession.generate()` method

  **Acceptance Criteria:**
  - [ ] Extension compiles
  - [ ] Method returns complete generated text

  **QA Scenarios:**
  ```
  Scenario: complete() returns full response
    Tool: Swift Testing
    Steps:
      1. Call model.complete("Hello")
      2. Verify result is non-empty string
    Expected Result: Complete generated text returned
    Evidence: .sisyphus/evidence/task-4-1-complete.log
  ```

  **Commit**: YES
  - Message: `feat(api): add complete() convenience method`
  - Files: `Sources/EdgeRunner/API/ConvenienceAPI.swift`

- [ ] 4.2 Add `complete()` with progress callback

  **What to do:**
  - Overload with `onProgress: (GenerationStats) -> Void` parameter
  - Internally uses `streamEvents()` and collects tokens

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 4.1, Task 2)

  **Acceptance Criteria:**
  - [ ] Callback is called during generation
  - [ ] Final result is still returned

  **Commit**: YES (group with 4.1)

- [ ] 4.3 Create `GenerationRequestBuilder`

  **What to do:**
  - Struct with fluent API: `.maxTokens()`, `.temperature()`, `.stream()`
  - Methods: `execute() async throws -> String`, `executeStream() -> AsyncThrowingStream<String, Error>`

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on 4.2)

  **Acceptance Criteria:**
  - [ ] Builder chains correctly
  - [ ] execute() returns complete text
  - [ ] executeStream() returns stream

  **QA Scenarios:**
  ```
  Scenario: Builder pattern works
    Tool: Swift Testing
    Steps:
      1. model.generate("Hello").maxTokens(100).temperature(0.8).execute()
      2. Verify it compiles and runs
    Expected Result: Fluent API works correctly
    Evidence: .sisyphus/evidence/task-4-3-builder.log
  ```

  **Commit**: YES
  - Message: `feat(api): add GenerationRequestBuilder for fluent API`
  - Files: `Sources/EdgeRunner/API/ConvenienceAPI.swift`

### Task 5: Documentation & Examples

- [ ] 5.1 Update `StreamingGeneration.md` DocC article

  **What to do:**
  - Update existing article in `Sources/EdgeRunner/Documentation.docc/Articles/`
  - Document rich stream events
  - Document progress tracking
  - Add cancellation patterns

  **Recommended Agent Profile:**
  - **Category**: `writing`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Tasks 1, 2)

  **Acceptance Criteria:**
  - [ ] Article renders correctly in Xcode
  - [ ] Covers all new streaming features

  **Commit**: YES
  - Message: `docs: update StreamingGeneration article with new features`
  - Files: `Sources/EdgeRunner/Documentation.docc/Articles/StreamingGeneration.md`

- [ ] 5.2 Create `ChatCompletion.md` DocC article

  **What to do:**
  - Create new article
  - Document ChatSession basics
  - Document conversation history
  - Document system prompts
  - Document template customization

  **Recommended Agent Profile:**
  - **Category**: `writing`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Task 3)

  **Acceptance Criteria:**
  - [ ] Article renders correctly
  - [ ] Includes code examples

  **Commit**: YES
  - Message: `docs: add ChatCompletion article`
  - Files: `Sources/EdgeRunner/Documentation.docc/Articles/ChatCompletion.md`

- [ ] 5.3 Create `ConvenienceAPI.md` DocC article

  **What to do:**
  - Document one-shot completion
  - Document builder pattern
  - Explain when to use async vs sync

  **Recommended Agent Profile:**
  - **Category**: `writing`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Task 4)

  **Commit**: YES
  - Message: `docs: add ConvenienceAPI article`
  - Files: `Sources/EdgeRunner/Documentation.docc/Articles/ConvenienceAPI.md`

- [ ] 5.4 Create code examples

  **What to do:**
  - Create `Examples/EdgeRunnerExamples/StreamingExample.swift`
  - Create `Examples/EdgeRunnerExamples/ChatExample.swift`
  - Create `Examples/EdgeRunnerExamples/ProgressTrackingExample.swift`

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on all previous tasks)

  **Acceptance Criteria:**
  - [ ] All examples compile
  - [ ] Examples demonstrate key features

  **Commit**: YES
  - Message: `docs: add code examples for new APIs`
  - Files: `Examples/EdgeRunnerExamples/*.swift`

- [ ] 5.5 Update README.md

  **What to do:**
  - Add new API examples to README
  - Show streaming with events
  - Show chat completion
  - Show convenience methods

  **Recommended Agent Profile:**
  - **Category**: `writing`

  **Parallelization:**
  - **Can Run In Parallel**: YES (with 5.4)

  **Commit**: YES
  - Message: `docs: update README with new API examples`
  - Files: `README.md`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify all "Must Have" items are implemented, all "Must NOT Have" items are absent.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `swift build`, check for warnings, verify Sendable conformance.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run examples, verify they produce expected output.

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify no scope creep - only planned features were added.

---

## Commit Strategy

- Task 1: 2 commits (StreamEvent, streamEvents)
- Task 2: 1 commit (GenerationStats + progress)
- Task 3: 2 commits (Chat structs, ChatSession)
- Task 4: 2 commits (complete(), builder)
- Task 5: 5 commits (3 articles, examples, README)

Total: ~12 commits

---

## Success Criteria

### Verification Commands
```bash
# Build passes
swift build

# Tests pass
swift test

# Examples compile
swift build --target EdgeRunnerExamples 2>/dev/null || echo "No examples target"

# DocC builds
swift build --target EdgeRunner 2>&1 | grep -i "documentation" || echo "Check Xcode for DocC"
```

### Final Checklist
- [ ] StreamEvent enum with all cases
- [ ] streamEvents() method working
- [ ] GenerationStats with progress tracking
- [ ] ChatSession actor with history
- [ ] ChatTemplate with Qwen3 support
- [ ] complete() convenience method
- [ ] GenerationRequestBuilder fluent API
- [ ] All DocC articles updated/created
- [ ] Examples compile and run
- [ ] README updated
- [ ] No breaking changes to existing API
