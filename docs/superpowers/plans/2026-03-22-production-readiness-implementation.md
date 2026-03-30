# EdgeRunner Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the production-readiness spec into an implementation sequence that first re-establishes trustworthy benchmarks, then hardens correctness and regression coverage, and only then performs memory optimization work.

**Architecture:** Execute in phases, not parallel streams. Phase 1 establishes a single benchmark source of truth and fixes the spec drift. Phases 2-4 harden error paths, streaming behavior, tokenizer coverage, and regression protection. Phases 5-6 add memory instrumentation and only then decide whether the Q8 dual-copy optimization is real, measurable, and worth landing.

**Tech Stack:** Swift 6.2, Swift Testing, Metal, GGUF loader stack, EdgeRunner / EdgeRunnerCore / EdgeRunnerIO / EdgeRunnerMetal

**Spec:** `docs/superpowers/specs/2026-03-22-production-readiness-plan.md`

---

## Scope Split

The spec covers multiple subsystems. This plan keeps them in one document for coordination, but implementation should proceed in this order:

1. Re-baseline benchmark reality
2. Harden loader, generation, and tokenizer correctness
3. Add regression gates
4. Add memory instrumentation
5. Investigate and optimize memory only if instrumentation confirms the target
6. Re-tune publishable benchmark methodology last

Do **not** start the optimization tasks before the benchmark and regression tasks are green.

## Pinned Model Prerequisites

The full verification path depends on two local GGUFs:

- Qwen benchmark/parity model: `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
- Gemma parity model: `/tmp/edgerunner-models/gemma-3-1b-it-Q4_K_M.gguf`

If either file is missing, production-readiness verification is blocked until the environment is repaired.

## CI Prerequisites

Because the success criteria require PR-enforced regression protection, the plan also depends on a provisioned CI runner:

- self-hosted Apple Silicon macOS runner
- pinned Qwen GGUF present at `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
- workflow registered as a required pull-request check before the CI stream is considered complete

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `docs/superpowers/specs/2026-03-22-production-readiness-plan.md` | Modify | Update stale benchmark claims and stream ordering after Phase 1 evidence exists |
| `docs/superpowers/plans/2026-03-22-production-readiness-implementation.md` | Modify | Execution plan and handoff document |
| `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift` | Modify | Add context-window checks and keep default stream behavior aligned with tests |
| `Sources/EdgeRunner/Streaming/GenerationSession.swift` | Modify | Tighten cancellation semantics and preserve model usability after termination |
| `Sources/EdgeRunner/ModelConfiguration.swift` | Modify | Keep any generation-related limits/config explicit if new knobs are required |
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | Add context overflow handling, memory instrumentation hooks, and Q8 memory tracing |
| `Sources/EdgeRunnerCore/GenerationError.swift` | Modify | Extend or clarify generation failures only if tests prove missing cases |
| `Sources/EdgeRunnerCore/Sampling/*.swift` | Modify | Define behavior for NaN / Inf / degenerate logits only where current behavior is undefined |
| `Sources/EdgeRunnerIO/GGUF/GGUFParser.swift` | Modify | Harden malformed GGUF rejection paths |
| `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift` | Modify | Surface load failures deterministically and expose metrics for mapped tensor usage |
| `Sources/EdgeRunnerIO/WeightLoaderError.swift` | Modify | Add/clarify specific parse failures only if current errors are too coarse for tests |
| `Sources/EdgeRunnerIO/MemoryPressureHandler.swift` | Modify | Add threshold-driven observation API only after instrumentation exists |
| `Sources/EdgeRunnerIO/EdgeRunnerMemoryPolicy.swift` | Modify | Support threshold configuration for memory pressure policy |
| `Sources/EdgeRunnerIO/LlamaModel.swift` | Modify | Trace whether Float32 copies are created during weight binding |
| `Sources/EdgeRunnerIO/DequantQ8_0Kernel.swift` | Modify | Confirm whether Q8 kernels require persistent Float32 materialization |
| `Sources/EdgeRunnerMetal/MetalBackend.swift` | Modify | Add memory instrumentation and benchmark-facing reporting hooks |
| `Sources/EdgeRunnerMetal/BufferCache.swift` | Modify | Expose cache-size metrics and optional eviction hooks |
| `Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift` | Modify | Add malformed header and truncated-data cases |
| `Tests/EdgeRunnerIOTests/GGUFTensorTableTests.swift` | Modify | Add corrupt tensor table and bounds cases |
| `Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift` | Modify | Add threshold and callback tests for pressure policy |
| `Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift` | Modify | Add context overflow tests at protocol/default-behavior level |
| `Tests/EdgeRunnerTests/StreamingTests.swift` | Modify | Add task cancellation, partial-output, and clean-reuse tests |
| `Tests/EdgeRunnerCoreTests/SamplingTests.swift` | Modify | Define finite-logit behavior and all-zero behavior explicitly |
| `Tests/EdgeRunnerCoreTests/TokenizerTests.swift` | Modify | Add Unicode, RTL, emoji, long-text, and round-trip coverage |
| `Tests/EdgeRunnerTests/QwenTokenizerParityTest.swift` | Modify | Expand existing parity corpus instead of creating a duplicate parity suite |
| `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift` | Modify | Keep parity structure consistent across model-backed integration suites |
| `Tests/EdgeRunnerTests/QwenBenchmark.swift` | Modify | Make this the short correctness/perf gate or explicitly demote it |
| `Tests/EdgeRunnerTests/PublishableBenchmark.swift` | Modify | Rework warmup, variance reporting, and deterministic-mode warnings |
| `Tests/EdgeRunnerTests/GoldenOutputTests.swift` | Create | Pinned-token correctness regression suite |
| `Tests/EdgeRunnerTests/PerformanceRegressionTests.swift` | Create | Fast benchmark gate with explicit thresholds |
| `Tests/EdgeRunnerTests/MemoryLeakTests.swift` | Create | Repeated-generate leak detection once memory counters exist |
| `.github/workflows/performance-regression.yml` | Create | Run the short regression gate in CI on pull requests |
| `benchmarks/baseline.json` | Auto-generated | Canonical short benchmark result after Phase 1 |
| `benchmarks/publishable_benchmark.json` | Auto-generated | Long benchmark artifact after Phase 6 |
| `benchmarks/memory_profile.json` | Create/auto-generate | Memory breakdown for load, prefill, decode, and cache growth |
| `benchmarks/variance_tracking.json` | Create/auto-generate | Historical variance report keyed by pinned model hash and commit |
| `benchmarks/thermal_profile.md` | Create | Machine-specific benchmark methodology notes |

---

## Task 1: Re-Baseline the Source of Truth

**Files:**
- Modify: `Tests/EdgeRunnerTests/QwenBenchmark.swift`
- Modify: `Tests/EdgeRunnerTests/PublishableBenchmark.swift`
- Modify: `docs/superpowers/specs/2026-03-22-production-readiness-plan.md`
- Auto-generated: `benchmarks/baseline.json`
- Auto-generated: `benchmarks/publishable_benchmark.json`

- [ ] **Step 1: Pick one pinned benchmark input and document it**

Use exactly one expected file size and one greedy prefix across both benchmark files. If `QwenBenchmark` remains the short gate, it must use the same pinned artifact as `PublishableBenchmark`.

Before treating benchmark commands as verification gates, replace the current silent `SKIP:` preflight behavior with an explicit failure when the pinned model is missing.

```swift
static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
static let expectedModelFileSizeBytes: Int64 = <resolved pinned artifact size>
static let expectedGreedyPrefix = [1, 14582, 25]

guard FileManager.default.fileExists(atPath: Self.modelPath) else {
    Issue.record("Missing pinned model at \(Self.modelPath)")
    return
}
```

- [ ] **Step 2: Run the short benchmark and capture the real current baseline**

Run: `swift test --filter "QwenBenchmark/decodeBenchmark"`

Expected:
- test completes against the pinned model
- if the pinned model is missing, stop and treat that as an environment blocker for this task
- `benchmarks/baseline.json` is rewritten with the same `model_file_size_bytes` as the test constant

- [ ] **Step 3: Run the publishable benchmark and capture the long-form artifact**

Run: `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`

Expected:
- test completes against the pinned model
- if the pinned model is missing, stop and treat that as an environment blocker for this task
- `benchmarks/publishable_benchmark.json` and `benchmarks/baseline.json` no longer disagree on pinned input identity

- [ ] **Step 4: Update the spec so its acceptance criteria match the executable plan**

Replace the speculative benchmark table and “all streams are independent” claim in `docs/superpowers/specs/2026-03-22-production-readiness-plan.md` with measured values and this order:

```markdown
1. Benchmark source of truth
2. Error handling + tokenizer + streaming hardening
3. Regression gates
4. Memory instrumentation
5. Memory optimization
6. Benchmark stabilization
```

Also reconcile these spec sections while editing the file:

- tokenizer acceptance criteria:
  use the existing model-backed suites `Tests/EdgeRunnerTests/QwenTokenizerParityTest.swift` and `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift`, expanded to the required corpus size, instead of creating a new `Tests/EdgeRunnerCoreTests/TokenizerParityTests.swift`
- memory acceptance criteria:
  define memory work as `instrument -> confirm redundant persistent copies -> optimize if confirmed`; if instrumentation disproves the original dual-copy assumption, update the target and rationale in the spec instead of forcing a fake optimization
- verification gates:
  replace generic filters like `ErrorHandling` and `TokenizerParity` with the exact runnable commands from this implementation plan

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerTests/QwenBenchmark.swift Tests/EdgeRunnerTests/PublishableBenchmark.swift docs/superpowers/specs/2026-03-22-production-readiness-plan.md benchmarks/baseline.json benchmarks/publishable_benchmark.json
git commit -m "docs: re-baseline production readiness metrics"
```

---

## Task 2: Harden GGUF Loader Failure Paths

**Files:**
- Modify: `Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift`
- Modify: `Tests/EdgeRunnerIOTests/GGUFTensorTableTests.swift`
- Modify: `Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift`
- Modify: `Sources/EdgeRunnerIO/GGUF/GGUFParser.swift`
- Modify: `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift`
- Modify: `Sources/EdgeRunnerIO/WeightLoaderError.swift`

- [ ] **Step 1: Add failing parser tests for malformed and truncated GGUF input**

Extend the existing builder-based tests with explicit error assertions:

```swift
@Test func rejectTruncatedTensorTable() {
    let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 1, metadataKVCount: 0)
    #expect(throws: WeightLoaderError.invalidFormat("Unexpected end of GGUF data at offset 24")) {
        let reader = GGUFReader(data: data)
        _ = try reader.readHeader()
        _ = try reader.readTensorInfos(count: 1)
    }
}

@Test func rejectMissingRequiredArchitectureMetadata() throws {
    let data = GGUFBuilder.minimalHeader(metadataKVCount: 0)
    let reader = GGUFReader(data: data)
    let header = try reader.readHeader()
    let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
    #expect(throws: WeightLoaderError.missingMetadata("general.architecture")) {
        _ = try ModelConfig.from(ggufMetadata: metadata)
    }
}
```

Add an early memory-pressure/OOM regression in `Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift` so Stream A covers the priority-1 failure path:

```swift
@Test func modelLoadFailsGracefullyUnderSimulatedPressure() async {
    let loader = OOMInjectingModelLoader()
    await #expect(throws: GenerationError.modelLoadFailed(reason: "Simulated memory pressure")) {
        _ = try await loader.loadPinnedModelUnderPressure()
    }
}
```

Keep the existing quantization-fallback tests, but do not treat them as sufficient for A1 by themselves. Stream A is not complete until a model-load-under-pressure path fails gracefully and predictably.

- [ ] **Step 2: Run the targeted IO suites and verify they fail for the new cases**

Run: `swift test --filter "GGUFHeaderTests|GGUFTensorTableTests|Memory Pressure Handler Tests"`

Expected:
- new tests fail because error specificity is missing or assertions do not yet match

- [ ] **Step 3: Implement the minimal parser/loader changes**

Keep changes small:
- preserve `invalidFormat` for truncation and bounds failures
- preserve `unsupportedVersion` for unsupported versions
- surface `missingMetadata` for required metadata absence
- do not create new error cases unless current ones are demonstrably insufficient

- [ ] **Step 4: Re-run the targeted suites**

Run: `swift test --filter "GGUFHeaderTests|GGUFTensorTableTests|Memory Pressure Handler Tests"`

Expected:
- all malformed-file tests pass
- existing happy-path GGUF tests stay green

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift Tests/EdgeRunnerIOTests/GGUFTensorTableTests.swift Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift Sources/EdgeRunnerIO/GGUF/GGUFParser.swift Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift Sources/EdgeRunnerIO/WeightLoaderError.swift
git commit -m "test: harden GGUF parser error coverage"
```

---

## Task 3: Harden Generation Cancellation and Context Limits

**Files:**
- Modify: `Tests/EdgeRunnerTests/StreamingTests.swift`
- Modify: `Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift`
- Modify: `Sources/EdgeRunner/Streaming/GenerationSession.swift`
- Modify: `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift`
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`
- Modify: `Sources/EdgeRunnerCore/GenerationError.swift`

- [ ] **Step 1: Add failing cancellation and overflow tests**

Add tests that reflect current API behavior, not speculative backpressure features:

```swift
@Test func generateThrowsCancelledWhenTaskIsCancelled() async {
    let model = StreamingMockModel(tokenSequence: Array(repeating: 0, count: 1000), vocab: ["x"])
    let session = GenerationSession(model: model, samplingPipeline: .greedy, maxTokens: 1000)

    let task = Task { try await session.generate(prompt: "") }
    task.cancel()

    await #expect(throws: GenerationError.cancelled) {
        _ = try await task.value
    }
}

@Test func mockModelRejectsPromptBeyondConfiguredContextWindow() async throws {
    let model = MockLanguageModel(fixedTokenIDs: [0, 1, 2, 3], contextWindowSize: 2)
    await #expect(throws: GenerationError.contextWindowExceeded(requested: 3, maximum: 2)) {
        _ = try await model.nextToken(for: [0, 1, 2], sampling: SamplingConfiguration())
    }
}
```

- [ ] **Step 2: Run the targeted suites and verify failure**

Run: `swift test --filter "TokenStream|GenerationSession|EdgeRunnerLanguageModel Protocol"`

Expected:
- cancellation test fails because the path returns partial output or no `GenerationError.cancelled`
- overflow test fails to compile until the mock stores `contextWindowSize`, or fails at runtime until validation is implemented

- [ ] **Step 3: Implement the smallest behavior changes**

Implement three rules:

```swift
guard tokenIDs.count <= configuration.contextWindowSize else {
    throw GenerationError.contextWindowExceeded(
        requested: tokenIDs.count,
        maximum: configuration.contextWindowSize
    )
}
```

- add `contextWindowSize` storage to `MockLanguageModel` in the test file so the protocol-level regression can actually go red first
- check cancellation before each token step
- finish the async stream with `GenerationError.cancelled` on task cancellation
- make sure a cancelled session does not poison the next `generate()` call

- [ ] **Step 4: Re-run the targeted suites**

Run: `swift test --filter "TokenStream|GenerationSession|EdgeRunnerLanguageModel Protocol"`

Expected:
- cancellation tests pass
- overflow tests pass
- existing basic streaming tests still pass

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerTests/StreamingTests.swift Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift Sources/EdgeRunner/Streaming/GenerationSession.swift Sources/EdgeRunner/EdgeRunnerLanguageModel.swift Sources/EdgeRunner/Models/LlamaLanguageModel.swift Sources/EdgeRunnerCore/GenerationError.swift
git commit -m "test: harden cancellation and context overflow handling"
```

---

## Task 4: Lock in Sampling Behavior and Expand Tokenizer Coverage

**Files:**
- Modify: `Tests/EdgeRunnerCoreTests/SamplingTests.swift`
- Modify: `Tests/EdgeRunnerCoreTests/TokenizerTests.swift`
- Modify: `Tests/EdgeRunnerTests/QwenTokenizerParityTest.swift`
- Modify: `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift`
- Modify: `Sources/EdgeRunnerCore/Sampling/*.swift`

- [ ] **Step 1: Add characterization tests for sampler edge behavior**

The NaN and all-zero greedy cases already pass today. Capture that behavior first and only change sampler code if a characterization test exposes a real undefined path:

```swift
@Test func greedySamplerIgnoresNaNWhenFiniteCandidateExists() {
    let logits: [Float] = [.nan, 1.0, -.infinity]
    #expect(GreedySampler().sample(logits: logits) == 1)
}

@Test func greedySamplerReturnsZeroForAllZeroLogits() {
    #expect(GreedySampler().sample(logits: [0, 0, 0]) == 0)
}
```

If these tests pass immediately, do **not** force a sampler implementation change in this task.

- [ ] **Step 2: Add Unicode and round-trip tokenizer coverage**

Extend `TokenizerTests.swift` with:

```swift
@Test func roundTripMixedScripts() {
    let tokenizer = makeByteEncodedTokenizer()
    let text = "Hello世界🎉"
    #expect(tokenizer.decode(tokenizer.encode(text)) == text)
}

@Test func roundTripRightToLeftText() {
    let tokenizer = makeByteEncodedTokenizer()
    let text = "שלום עולם"
    #expect(tokenizer.decode(tokenizer.encode(text)) == text)
}
```

- [ ] **Step 3: Expand the existing model-backed parity corpus instead of creating a second parity suite**

Add more prompts to `QwenTokenizerParityTest.swift` and `GemmaTokenizerParityTest.swift` until the required 100-case corpus is met. Keep the existing gates:

- Qwen: `EDGERUNNER_RUN_TOKENIZER_PARITY=1`
- Gemma: `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1`

Before using these suites as verification gates, replace silent skip-on-missing-env/model behavior with explicit failure:

```swift
guard ProcessInfo.processInfo.environment["EDGERUNNER_RUN_TOKENIZER_PARITY"] == "1" else {
    Issue.record("Set EDGERUNNER_RUN_TOKENIZER_PARITY=1 to run tokenizer parity tests")
    return false
}
guard FileManager.default.fileExists(atPath: Self.modelPath) else {
    Issue.record("Missing pinned model at \(Self.modelPath)")
    return false
}
```

Do not create `Tests/EdgeRunnerCoreTests/TokenizerParityTests.swift`; that would duplicate the existing integration seam while testing the wrong layer.

- [ ] **Step 4: Run targeted suites**

Run:
- `swift test --filter "GreedySampler|TemperatureSampler|TopKSampler|TopPSampler|MinPSampler|RepetitionPenalty|SamplingPipeline|Tokenizer Protocol|SpecialTokens|TokenizerVocabulary|BPETokenizer|BPETokenizer Pipeline"`
- `EDGERUNNER_RUN_TOKENIZER_PARITY=1 swift test --filter "QwenTokenizerParityTest"`
- `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1 swift test --filter "GemmaTokenizerParityTest"`

Expected:
- core suites pass locally
- model-backed parity suites run against the pinned models
- if either model file is missing, stop and treat that as an environment blocker for production-readiness verification

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerCoreTests/SamplingTests.swift Tests/EdgeRunnerCoreTests/TokenizerTests.swift Tests/EdgeRunnerTests/QwenTokenizerParityTest.swift Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift Sources/EdgeRunnerCore/Sampling
git commit -m "test: expand sampling and tokenizer edge coverage"
```

---

## Task 5: Add Regression Gates Before Optimization

**Files:**
- Create: `Tests/EdgeRunnerTests/GoldenOutputTests.swift`
- Create: `Tests/EdgeRunnerTests/PerformanceRegressionTests.swift`
- Create: `.github/workflows/performance-regression.yml`
- Modify: `Tests/EdgeRunnerTests/QwenBenchmark.swift`
- Modify: `Tests/EdgeRunnerTests/PublishableBenchmark.swift`

- [ ] **Step 1: Add a pinned-output regression suite**

Use the Phase 1 pinned model identity and pinned golden token sequence as the only source of truth:

```swift
@Suite("Golden Output")
struct GoldenOutputTests {
    @Test func qwenGreedyDecodeMatchesPinnedGoldenSequence() async throws {
        let model = try await loadPinnedQwen()
        let expectedTokens = loadPhase1PinnedGoldenTokens()
        var tokenIDs = [1]
        for _ in 0..<50 {
            tokenIDs.append(argmax(try await model.logits(for: tokenIDs)))
        }
        #expect(Array(tokenIDs.prefix(expectedTokens.count)) == expectedTokens)
    }
}
```

- [ ] **Step 2: Add a short performance gate**

Create a fast regression test that uses the short benchmark path and fails only on a materially bad drop relative to the Phase 1 pinned baseline:

```swift
@Suite("Performance Regression")
struct PerformanceRegressionTests {
    @Test func qwenShortDecodeDoesNotRegressBeyondThreshold() async throws {
        let throughput = try await runShortQwenDecodeBenchmark()
        #expect(throughput >= baselineThroughput * 0.85)
    }
}
```

- [ ] **Step 3: Wire the short regression gate into CI**

Create `.github/workflows/performance-regression.yml` so the short gate runs on pull requests:

```yaml
name: Performance Regression

on:
  pull_request:
  workflow_dispatch:

jobs:
  qwen-short-regression:
    runs-on: [self-hosted, macOS, arm64]
    steps:
      - uses: actions/checkout@v4
      - name: Select Xcode
        run: sudo xcode-select -s /Applications/Xcode.app
      - name: Verify pinned model is present
        run: test -f /tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf
      - name: Run short regression gate
        run: swift test --filter "Golden Output|Performance Regression"
```

This workflow is intentionally self-hosted because the pinned GGUF path, Apple Silicon hardware class, and absolute throughput threshold are part of the acceptance criteria. Do not claim the CI stream complete until this job is provisioned and required on pull requests.

- [ ] **Step 4: Run the new regression suites locally**

Run: `swift test --filter "Golden Output|Performance Regression"`

Expected:
- suites pass against the pinned model
- if the pinned model is unavailable, stop and treat that as an environment blocker
- no new suite depends on unpublished benchmark numbers from the stale spec

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerTests/GoldenOutputTests.swift Tests/EdgeRunnerTests/PerformanceRegressionTests.swift Tests/EdgeRunnerTests/QwenBenchmark.swift Tests/EdgeRunnerTests/PublishableBenchmark.swift .github/workflows/performance-regression.yml
git commit -m "test: add production-readiness regression gates"
```

---

## Task 6: Add Memory Instrumentation

**Files:**
- Modify: `Sources/EdgeRunnerMetal/MetalBackend.swift`
- Modify: `Sources/EdgeRunnerMetal/BufferCache.swift`
- Modify: `Sources/EdgeRunnerIO/MemoryPressureHandler.swift`
- Modify: `Sources/EdgeRunnerIO/EdgeRunnerMemoryPolicy.swift`
- Modify: `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift`
- Modify: `Sources/EdgeRunnerIO/LlamaModel.swift`
- Modify: `Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift`
- Modify: `Tests/EdgeRunnerTests/PublishableBenchmark.swift`
- Create: `Tests/EdgeRunnerTests/MemoryLeakTests.swift`
- Create/auto-generate: `benchmarks/memory_profile.json`

- [ ] **Step 1: Add failing tests around threshold observation and callback wiring**

```swift
@Test func handlerRecordsThresholdCrossingWithoutChangingQuantisation() async {
    let handler = MemoryPressureHandler(
        policy: EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0, .q4_k_m, .q4_0],
            evictBufferCacheOnPressure: true,
            maxMemoryBytes: 1024
        )
    )
    await handler.observeMemoryUsage(bytes: 2048)
    #expect(handler.currentQuantisation == .q8_0)
    #expect(handler.isAboveMemoryThreshold == true)
}
```

- [ ] **Step 2: Expose memory counters without changing runtime behavior**

Add read-only metrics first:

```swift
public struct MemoryProfileSnapshot: Sendable, Codable {
    let bufferCacheBytes: Int
    let recommendedMaxWorkingSetBytes: Int
    let rssBytes: UInt64
}
```

The first version should only report numbers; do not evict KV cache or rewrite allocation policy yet.

- [ ] **Step 3: Plumb the snapshot into the load and benchmark paths**

Write `benchmarks/memory_profile.json` with:
- model file hash or size
- load snapshot
- prefill snapshot
- decode snapshot
- buffer-cache bytes

Collect the load-side numbers in the actual load/binding path:
- `GGUFLoader` for mapped tensor loading
- `LlamaModel` for weight binding and any persistent copies
- `MetalBackend` / `BufferCache` for backend allocation state

- [ ] **Step 4: Add the repeated-generate memory leak suite now that counters exist**

```swift
@Suite("Memory Leak")
struct MemoryLeakTests {
    @Test func repeatedGenerateStabilizesAfterWarmup() async throws {
        let samples = try await runRepeatedGenerationAndCollectMemorySamples(iterations: 25)
        #expect(samples.last! <= samples[5] * 1.05)
    }
}
```

- [ ] **Step 5: Run the targeted suites**

Run:
- `swift test --filter "Memory Pressure Handler Tests"`
- `swift test --filter "Memory Leak"`
- `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- `test -f benchmarks/memory_profile.json`

Expected:
- pressure-handler unit tests pass
- publishable benchmark still works and now emits memory breakdown data
- `benchmarks/memory_profile.json` exists after the benchmark run

- [ ] **Step 6: Commit**

```bash
git add Sources/EdgeRunnerMetal/MetalBackend.swift Sources/EdgeRunnerMetal/BufferCache.swift Sources/EdgeRunnerIO/MemoryPressureHandler.swift Sources/EdgeRunnerIO/EdgeRunnerMemoryPolicy.swift Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift Sources/EdgeRunnerIO/LlamaModel.swift Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift Tests/EdgeRunnerTests/PublishableBenchmark.swift Tests/EdgeRunnerTests/MemoryLeakTests.swift benchmarks/memory_profile.json
git commit -m "feat: add memory instrumentation for production benchmarks"
```

---

## Task 7: Trace the Q8 Float32 Copy Before Optimizing

**Files:**
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`
- Modify: `Sources/EdgeRunnerIO/LlamaModel.swift`
- Modify: `Sources/EdgeRunnerIO/DequantQ8_0Kernel.swift`
- Modify: `Tests/EdgeRunnerTests/PublishableBenchmark.swift`
- Modify: `docs/superpowers/specs/2026-03-22-production-readiness-plan.md`
- Auto-generated: `benchmarks/memory_profile.json`

- [ ] **Step 1: Add instrumentation points around weight binding and Q8 dequant setup**

Capture whether a Float32 copy is created:
- once at model load
- once per layer
- once per decode

Log counters, not strings, so they can be compared in tests/benchmarks.

- [ ] **Step 2: Run the publishable benchmark and inspect the emitted counters**

Run: `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`

Expected:
- benchmark artifact includes enough evidence to answer:
  - whether Float32 copies exist
  - when they are created
  - whether they are persistent or per-step

- [ ] **Step 3: Only if the evidence shows redundant persistent copies, implement the minimal optimization**

Candidate changes:
- lazy-create the Float32 copy only for CPU fallback
- keep quantized storage as the primary representation
- avoid duplicate persistent buffers for tied weights

Do **not** attempt dequant in-place and policy-driven eviction in the same commit.

- [ ] **Step 4: Re-run short + publishable benchmarks**

Run:
- `swift test --filter "QwenBenchmark/decodeBenchmark"`
- `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`

Expected:
- greedy prefix remains unchanged
- memory profile improves measurably
- throughput does not regress past the agreed threshold

- [ ] **Step 5: If the hypothesis is disproved, update the spec instead of forcing an optimization**

If the instrumentation shows that redundant persistent Float32 copies are **not** the real memory driver:
- update `docs/superpowers/specs/2026-03-22-production-readiness-plan.md`
- record the disproved hypothesis
- revise the memory target/rationale
- keep `benchmarks/memory_profile.json` as the evidence artifact

- [ ] **Step 6: Commit the correct outcome**

Optimization-kept path:

```bash
git add Sources/EdgeRunner/Models/LlamaLanguageModel.swift Sources/EdgeRunnerIO/LlamaModel.swift Sources/EdgeRunnerIO/DequantQ8_0Kernel.swift Tests/EdgeRunnerTests/PublishableBenchmark.swift benchmarks/memory_profile.json
git commit -m "perf: remove redundant Q8 Float32 weight copies"
```

Hypothesis-disproved path:

```bash
git add docs/superpowers/specs/2026-03-22-production-readiness-plan.md benchmarks/memory_profile.json Tests/EdgeRunnerTests/PublishableBenchmark.swift
git commit -m "docs: record disproved Q8 dual-copy memory hypothesis"
```

---

## Task 8: Stabilize the Publishable Benchmark Methodology

**Files:**
- Modify: `Tests/EdgeRunnerTests/QwenBenchmark.swift`
- Modify: `Tests/EdgeRunnerTests/PublishableBenchmark.swift`
- Create/auto-generate: `benchmarks/variance_tracking.json`
- Create: `benchmarks/thermal_profile.md`

- [ ] **Step 1: Document the machine-specific benchmark protocol**

Create `benchmarks/thermal_profile.md` with:
- pinned machine identity
- ambient/pre-run thermal notes
- warmup count
- release-build command
- when to discard runs
- how `EDGERUNNER_DETERMINISTIC=1` is invoked after Task 8 lands
- whether CPU-frequency pinning is available on the current machine; if not, document that explicitly instead of silently skipping it

- [ ] **Step 2: Make benchmark variance reporting explicit**

Add fields for:

```json
{
  "deterministic_mode": true,
  "variance_pct": 0.0,
  "thermal_notes": "cold-start|warm|throttled",
  "thermal_delta_c": 0.0,
  "model_file_size_bytes": 804753504
}
```

Apply the same stabilization rules to `QwenBenchmark.swift` where the short gate is still used:
- increase warmup as required by the spec
- record thermal start/end
- make the short gate’s role explicit as a fast regression check, not the only publishable benchmark artifact

- [ ] **Step 3: Run five consecutive publishable benchmarks and write the tracking file**

Run five times:

```bash
EDGERUNNER_DETERMINISTIC=1 swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```

Then write `benchmarks/variance_tracking.json` with per-run medians, min/max, and greedy prefix.

- [ ] **Step 4: Tighten thresholds only after five stable runs exist**

Acceptable exit criteria:
- publishable runs are within 5% variance across five consecutive runs on the pinned machine
- greedy prefix is stable
- artifacts agree on model identity
- thermal delta during the measurement window is <= 10°C, or the run is rejected
- deterministic mode is implemented and enforced for the five-run publishable-benchmark verification path

- [ ] **Step 5: Commit**

```bash
git add Tests/EdgeRunnerTests/QwenBenchmark.swift Tests/EdgeRunnerTests/PublishableBenchmark.swift benchmarks/variance_tracking.json benchmarks/thermal_profile.md
git commit -m "test: stabilize publishable benchmark methodology"
```

---

## Final Verification Sequence

Run this sequence only after Tasks 1-8 are complete. Steps 9-10 assume Task 8 has already landed the deterministic-mode support in `PublishableBenchmark.swift`.

Run these in order before claiming production-readiness progress:

1. `swift test --filter "GGUFHeaderTests|GGUFTensorTableTests|Memory Pressure Handler Tests"`
2. `swift test --filter "TokenStream|GenerationSession|EdgeRunnerLanguageModel Protocol"`
3. `swift test --filter "GreedySampler|TemperatureSampler|TopKSampler|TopPSampler|MinPSampler|RepetitionPenalty|SamplingPipeline|Tokenizer Protocol|SpecialTokens|TokenizerVocabulary|BPETokenizer|BPETokenizer Pipeline"`
4. `EDGERUNNER_RUN_TOKENIZER_PARITY=1 swift test --filter "QwenTokenizerParityTest"`
5. `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1 swift test --filter "GemmaTokenizerParityTest"`
6. `swift test --filter "Golden Output|Performance Regression"`
7. `swift test --filter "QwenBenchmark/decodeBenchmark"`
8. `swift test --filter "Memory Leak"`
9. `EDGERUNNER_DETERMINISTIC=1 swift test -c release --filter "PublishableBenchmark/fullBenchmark"` and verify `benchmarks/memory_profile.json` exists
10. `EDGERUNNER_DETERMINISTIC=1 swift test -c release --filter "PublishableBenchmark/fullBenchmark"` run 5 times total, then verify `benchmarks/variance_tracking.json`

If any model-backed suite cannot run because the pinned GGUF is missing, treat that as an environment blocker, not a passing result.

---

## Execution Notes

- Keep benchmark metric updates in their own commits. Do not mix them with logic changes.
- Do not create a second tokenizer parity stack in `EdgeRunnerCoreTests`; extend the existing model-backed parity suites.
- Do not implement backpressure semantics unless the API is first extended to express them. Current production-readiness scope only requires correct cancellation and bounded behavior of the existing stream implementation.
- Do not land memory-pressure-driven KV eviction until memory instrumentation exists and a specific pressure threshold is defined.
- If the Q8 memory profile does **not** show redundant Float32 persistence, stop after Task 7 evidence collection and update the spec instead of forcing an optimization.
