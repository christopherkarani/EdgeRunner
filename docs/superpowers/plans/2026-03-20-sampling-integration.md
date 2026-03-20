# Sampling Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `SamplingPipeline` into `nextToken()` so `SamplingConfiguration` (temperature, top-p, top-k, repetition penalty) actually works instead of being silently ignored.

**Architecture:** Add a `SamplingConfiguration.toPipeline()` converter, change `LlamaLanguageModel.nextToken()` to use it, fix `GenerationSession` to use its stored pipeline, and add a sampling-aware `stream()` overload.

**Tech Stack:** Swift 6.2, Swift Testing, EdgeRunnerCore (SamplingPipeline), EdgeRunner (LlamaLanguageModel)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `Sources/EdgeRunner/SamplingConfiguration.swift` | Modify | Add `toPipeline()` method |
| `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | Modify | Wire sampling into `nextToken()` |
| `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift` | Modify | Fix default `nextToken()` and add `stream(_:sampling:)` |
| `Sources/EdgeRunner/Streaming/GenerationSession.swift` | Modify | Use stored `samplingPipeline` |
| `Tests/EdgeRunnerCoreTests/SamplingPipelineTests.swift` | Create | Test pipeline construction from config |
| `Tests/EdgeRunnerTests/SamplingIntegrationTests.swift` | Create | End-to-end sampling tests |

---

### Task 1: SamplingConfiguration → SamplingPipeline converter

**Files:**
- Modify: `Sources/EdgeRunner/SamplingConfiguration.swift`
- Create: `Tests/EdgeRunnerCoreTests/SamplingPipelineTests.swift`

- [ ] **Step 1: Write tests for the converter**

```swift
import Testing
@testable import EdgeRunnerCore
@testable import EdgeRunner

@Suite("SamplingPipeline")
struct SamplingPipelineTests {
    @Test func defaultConfigProducesGreedyPipeline() {
        let config = SamplingConfiguration()
        let pipeline = config.toPipeline()
        // Greedy should always pick the highest logit
        let logits: [Float] = [0.1, 0.5, 0.3, 0.9, 0.2]
        let token = pipeline.sample(logits: logits)
        #expect(token == 3)  // index of 0.9
    }

    @Test func zeroTemperatureIsGreedy() {
        let config = SamplingConfiguration(temperature: 0.0)
        let pipeline = config.toPipeline()
        let logits: [Float] = [0.1, 0.5, 0.3, 0.9, 0.2]
        let token = pipeline.sample(logits: logits)
        #expect(token == 3)
    }

    @Test func temperatureSamplingIsStochastic() {
        let config = SamplingConfiguration(temperature: 1.0, topK: 50, topP: 1.0, seed: 42)
        let pipeline = config.toPipeline()
        // With uniform-ish logits and temperature, should not always pick the same token
        let logits: [Float] = [1.0, 1.0, 1.0, 1.01, 1.0]
        var tokens = Set<Int>()
        for _ in 0..<50 {
            tokens.insert(pipeline.sample(logits: logits))
        }
        #expect(tokens.count > 1, "Stochastic sampling should produce varied output")
    }

    @Test func repetitionPenaltyReducesRepeats() {
        let config = SamplingConfiguration(temperature: 0.0, repetitionPenalty: 2.0)
        let pipeline = config.toPipeline()
        let logits: [Float] = [0.5, 0.49, 0.3, 0.2, 0.1]
        // Without penalty, picks token 0 (0.5)
        let first = pipeline.sample(logits: logits, previousTokens: [])
        #expect(first == 0)
        // With token 0 in history and penalty=2.0, token 0's logit is halved → token 1 wins
        let second = pipeline.sample(logits: logits, previousTokens: [0])
        #expect(second == 1)
    }

    @Test func seededSamplingIsDeterministic() {
        let config = SamplingConfiguration(temperature: 0.8, topP: 0.9, seed: 12345)
        let logits: [Float] = [1.0, 2.0, 3.0, 2.5, 1.5]
        let pipeline1 = config.toPipeline()
        let pipeline2 = config.toPipeline()
        let t1 = pipeline1.sample(logits: logits)
        let t2 = pipeline2.sample(logits: logits)
        #expect(t1 == t2, "Same seed should produce same result")
    }
}
```

- [ ] **Step 2: Run tests — verify they FAIL**

Run: `swift test --filter SamplingPipelineTests 2>&1 | tail -10`
Expected: compilation error — `toPipeline()` not defined

- [ ] **Step 3: Implement `toPipeline()`**

Add to `Sources/EdgeRunner/SamplingConfiguration.swift`:

```swift
import EdgeRunnerCore

extension SamplingConfiguration {
    /// Convert this configuration into a composable SamplingPipeline.
    ///
    /// - Temperature ≤ 0 or defaults with no stochastic params → greedy
    /// - Otherwise → temperature + topK/topP transforms + stochastic selector
    public func toPipeline() -> SamplingPipeline {
        // Greedy if temperature is 0 or effectively disabled
        if temperature <= 0 {
            return .greedy
        }

        // Check if this is effectively the "don't sample" config
        // (temperature=1.0, no top-k/top-p filtering, no repetition penalty)
        let isDefaultGreedy = temperature == 1.0
            && topK == 0
            && topP >= 1.0
            && repetitionPenalty <= 1.0
            && seed == nil

        if isDefaultGreedy {
            return .greedy
        }

        // Build transforms
        var transforms: [any LogitsTransform] = []

        if temperature != 1.0 {
            transforms.append(TemperatureSampler(temperature: temperature))
        }

        if topK > 0 {
            transforms.append(TopKSampler(k: topK))
        }

        if topP < 1.0 {
            transforms.append(TopPSampler(p: topP))
        }

        // Build selector
        let selector: any TokenSelector
        if let seed {
            var rng = SeededRandomSource(seed: seed)
            selector = StochasticSampler(randomSource: &rng)
        } else {
            var rng = SeededRandomSource(seed: UInt64.random(in: 0...UInt64.max))
            selector = StochasticSampler(randomSource: &rng)
        }

        // Build repetition penalty
        let penalty: RepetitionPenalty?
        if repetitionPenalty > 1.0 {
            penalty = RepetitionPenalty(penalty: repetitionPenalty)
        } else {
            penalty = nil
        }

        return SamplingPipeline(
            transforms: transforms,
            selector: selector,
            repetitionPenalty: penalty
        )
    }
}
```

**Note:** The default `SamplingConfiguration()` has `temperature: 1.0, topK: 40, topP: 0.9` — this is NOT greedy, so it will produce stochastic sampling. This is correct behavior.

- [ ] **Step 4: Run tests — verify they PASS**

Run: `swift test --filter SamplingPipelineTests 2>&1 | tail -15`
Expected: All 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunner/SamplingConfiguration.swift Tests/EdgeRunnerCoreTests/SamplingPipelineTests.swift
git commit -m "feat: add SamplingConfiguration.toPipeline() converter"
```

---

### Task 2: Wire sampling into LlamaLanguageModel.nextToken()

**Files:**
- Modify: `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`

- [ ] **Step 1: Change `nextToken()` to use sampling**

Replace the current implementation (around line 323):

```swift
public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
    // Get logits (using cache if available)
    let logits: [Float]
    if tokenIDs == decoderState.cachedLogitsInput, let cached = decoderState.cachedLogits {
        logits = cached
    } else {
        logits = try await self.logits(for: tokenIDs)
    }

    // Apply sampling pipeline
    let pipeline = sampling.toPipeline()
    return pipeline.sample(logits: logits, previousTokens: tokenIDs)
}
```

**Key change:** Instead of `greedyArgmax()`, we now call `sampling.toPipeline().sample()`. The default `SamplingConfiguration()` with `temperature: 1.0, topK: 40, topP: 0.9` will produce stochastic output. Users who want greedy must pass `SamplingConfiguration(temperature: 0)`.

- [ ] **Step 2: Fix the `tokenIDs.last!` force unwrap**

Search for `tokenIDs.last!` in the file. If found, replace with safe access:
```swift
guard let lastToken = tokenIDs.last else {
    throw GenerationError.decodingFailed(reason: "Empty token sequence")
}
```

- [ ] **Step 3: Build and verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/EdgeRunner/Models/LlamaLanguageModel.swift
git commit -m "feat: wire SamplingConfiguration into LlamaLanguageModel.nextToken()"
```

---

### Task 3: Fix default protocol implementation and streaming

**Files:**
- Modify: `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift`
- Modify: `Sources/EdgeRunner/Streaming/GenerationSession.swift`

- [ ] **Step 1: Fix LogitsModel.nextToken() default**

In `EdgeRunnerLanguageModel.swift`, update the `LogitsModel` extension (around line 83):

```swift
extension LogitsModel {
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let logitsArray = try await logits(for: tokenIDs)
        let pipeline = sampling.toPipeline()
        return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
    }
}
```

- [ ] **Step 2: Add `stream(_:sampling:)` overload**

Add a new stream method that accepts sampling config:

```swift
extension EdgeRunnerLanguageModel {
    public func stream(
        _ prompt: String,
        sampling: SamplingConfiguration = SamplingConfiguration()
    ) -> AsyncThrowingStream<String, Error> {
        let model = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var tokenIDs = model.tokenize(prompt)
                    for _ in 0..<2048 {
                        try Task.checkCancellation()
                        let tokenID = try await model.nextToken(
                            for: tokenIDs,
                            sampling: sampling
                        )
                        if tokenID == model.eosTokenID { break }
                        tokenIDs.append(tokenID)
                        let text = model.detokenize([tokenID])
                        continuation.yield(text)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: GenerationError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
```

- [ ] **Step 3: Fix GenerationSession to use its stored pipeline**

In `GenerationSession.swift`, change line 43-45 to use the stored `samplingPipeline`:

The challenge: `nextToken()` takes `SamplingConfiguration`, not `SamplingPipeline`. Two options:
- Option A: Call `model.logits(for:)` directly and use `samplingPipeline.sample()` (only works for `LogitsModel`)
- Option B: Keep using `nextToken()` but convert the pipeline back to config (lossy)

**Best approach:** Since `GenerationSession` is generic over `EdgeRunnerLanguageModel` (not `LogitsModel`), it must use `nextToken()`. Add a `SamplingConfiguration` stored property alongside the pipeline:

```swift
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    private let model: Model
    private let sampling: SamplingConfiguration
    public let maxTokens: Int
    private let onToken: (@Sendable (Int, String) -> Void)?

    public init(
        model: Model,
        sampling: SamplingConfiguration = SamplingConfiguration(),
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    ) {
        self.model = model
        self.sampling = sampling
        self.maxTokens = maxTokens
        self.onToken = onToken
    }
```

And in `stream()`, use `self.sampling`:
```swift
let tokenID = try await model.nextToken(
    for: tokenIDs,
    sampling: self.sampling  // was: SamplingConfiguration()
)
```

Keep the `SamplingPipeline` init for backward compatibility:
```swift
    public init(
        model: Model,
        samplingPipeline: SamplingPipeline = .greedy,
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    ) {
        self.model = model
        self.sampling = SamplingConfiguration()  // default
        self.maxTokens = maxTokens
        self.onToken = onToken
    }
```

- [ ] **Step 4: Build and verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add Sources/EdgeRunner/EdgeRunnerLanguageModel.swift Sources/EdgeRunner/Streaming/GenerationSession.swift
git commit -m "feat: fix default nextToken(), add stream(_:sampling:), fix GenerationSession"
```

---

### Task 4: End-to-end sampling integration test

**Files:**
- Create: `Tests/EdgeRunnerTests/SamplingIntegrationTests.swift`

- [ ] **Step 1: Write integration test**

```swift
import Testing
import Foundation
@testable import EdgeRunner
@testable import EdgeRunnerCore

@Suite("Sampling Integration")
struct SamplingIntegrationTests {
    static let modelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

    private func shouldRun() -> Bool {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            print("SKIP: Model not found at \(Self.modelPath)")
            return false
        }
        return true
    }

    @Test func greedyProducesDeterministicOutput() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("Hello")
        let greedy = SamplingConfiguration(temperature: 0)

        var run1 = prompt
        var run2 = prompt
        for _ in 0..<10 {
            run1.append(try await model.nextToken(for: run1, sampling: greedy))
            run2.append(try await model.nextToken(for: run2, sampling: greedy))
        }
        #expect(run1 == run2, "Greedy sampling should be deterministic")
    }

    @Test func temperatureSamplingProducesVariedOutput() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("Once upon")
        let sampling = SamplingConfiguration(temperature: 1.0, topK: 50, topP: 0.95)

        var outputs = Set<[Int]>()
        for _ in 0..<3 {
            var tokens = prompt
            for _ in 0..<5 {
                tokens.append(try await model.nextToken(for: tokens, sampling: sampling))
            }
            outputs.insert(Array(tokens.dropFirst(prompt.count)))
        }
        #expect(outputs.count > 1, "Temperature sampling should produce varied output across runs")
    }

    @Test func seededSamplingIsDeterministic() async throws {
        guard shouldRun() else { return }
        let model = try await LlamaLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 512)
        )
        let prompt = model.tokenize("The")
        let sampling = SamplingConfiguration(temperature: 0.8, topP: 0.9, seed: 42)

        var run1 = prompt
        var run2 = prompt
        for _ in 0..<10 {
            run1.append(try await model.nextToken(for: run1, sampling: sampling))
        }
        for _ in 0..<10 {
            run2.append(try await model.nextToken(for: run2, sampling: sampling))
        }
        #expect(run1 == run2, "Same seed should produce same sequence")
    }
}
```

- [ ] **Step 2: Run integration tests (if model available)**

Run: `swift test --filter SamplingIntegrationTests 2>&1 | tail -15`
Expected: All 3 tests pass

- [ ] **Step 3: Commit**

```bash
git add Tests/EdgeRunnerTests/SamplingIntegrationTests.swift
git commit -m "test: add end-to-end sampling integration tests"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `swift test 2>&1 | grep "Test run with" | tail -1`
Expected: All tests pass (except pre-existing benchmark flakes)

- [ ] **Step 2: Run coherence test with temperature**

Quick manual test to verify stochastic output is coherent:
```bash
swift test --filter "storyAt200TokPerSec" 2>&1 | tail -20
```
(If this test exists, or create a temporary one)

- [ ] **Step 3: Commit any remaining fixes**
