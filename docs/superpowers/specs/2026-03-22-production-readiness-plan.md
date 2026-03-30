# EdgeRunner Production Readiness Plan

**Date:** 2026-03-22
**Status:** Draft — execution sequencing aligned; benchmark metrics must be re-baselined in Phase 1
**Goal:** Close the gap between current "beta" state and production-ready for Qwen/Llama workloads on Apple Silicon

---

## Context

EdgeRunner is a high-performance local LLM inference framework for Apple Silicon, built in Swift/Metal with GGUF model support. Current benchmark state:

| Metric | Value |
|--------|-------|
| Decode throughput | To be re-baselined in Phase 1 |
| TTFT | To be re-baselined in Phase 1 |
| Memory | To be re-baselined in Phase 1 |
| vs llama.cpp | To be re-baselined in Phase 1 |
| vs MLX | To be re-baselined in Phase 1 |

**Current state:** Beta. Core inference works for Qwen3-0.6B. Multiple model architectures untested. Benchmark variance still exists. Error paths less validated.

---

## Execution Preconditions

The full production-readiness verification path assumes these local GGUFs are present:

- Qwen benchmark/parity model: `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
- Gemma parity model: `/tmp/edgerunner-models/gemma-3-1b-it-Q4_K_M.gguf`

The tokenizer parity suites are environment-gated:

- Qwen: `EDGERUNNER_RUN_TOKENIZER_PARITY=1`
- Gemma: `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1`

If those artifacts are missing, production-readiness verification is blocked.

As part of production-readiness hardening, the model-backed benchmark and parity
suites should fail explicitly on missing env/model prerequisites instead of
printing `SKIP:` and returning success.

---

## Workstreams

The five streams should be executed in phases, not treated as fully independent parallel work:

1. Re-baseline benchmark source of truth
2. Error handling + tokenizer + streaming hardening
3. Regression gates
4. Memory instrumentation
5. Memory optimization only if instrumentation confirms the dual-copy hypothesis
6. Benchmark stabilization last

---

### Stream A: Error Handling Hardening

**Owner:** `EdgeRunnerCore` + `EdgeRunnerIO`
**Phase:** 2 — early hardening

#### A1: OOM Injection Tests
```
File: Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift
Add:
- Test model load under simulated memory pressure
- Verify graceful failure with GenerationError.modelLoadFailed
- Verify no use-after-free or double-free on OOM
```

#### A2: Malformed GGUF Rejection
```
File: Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift
Add:
- Corrupt tensor table length → WeightLoaderError.invalidFormat
- Wrong magic number → WeightLoaderError.invalidFormat
- Truncated file → WeightLoaderError.invalidFormat
- Unsupported GGUF version → WeightLoaderError.unsupportedVersion
- Missing required metadata keys → WeightLoaderError.missingMetadata
```

#### A3: Cancellation Mid-Generate
```
File: Tests/EdgeRunnerTests/StreamingTests.swift
Add:
- Task cancellation during generate() → GenerationError.cancelled
- AsyncStream backpressure + cancel → clean termination
- Partial output before cancel → output preserved or error reported
```

#### A4: Context Window Overflow
```
File: Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift
Add:
- Prompt exceeding maxSeqLen → GenerationError.contextWindowExceeded
- Verify error message includes actual vs max
- Verify no crash or corrupted state after error
```

#### A5: Invalid Token Recovery
```
File: Tests/EdgeRunnerCoreTests/SamplingTests.swift
Add:
- NaN logits → graceful handling
- Inf logits → graceful handling
- All-zeros logits → defined behavior (currently undefined)
```

---

### Stream B: Memory Optimization

**Owner:** `EdgeRunnerIO` + `EdgeRunnerMetal`
**Phase:** 4-5 — instrumentation first, optimization only if confirmed

#### B1: Profile VRAM Allocation
```
Add to MetalBackend:
- Track currentAllocatedSize before/after each major operation
- Log allocation breakdown: weights, KV cache, intermediate buffers
- Output: memory profile report in benchmarks/memory_profile.json

Operations to instrument:
1. Model load (GGUF → Metal buffers)
2. Prefill phase
3. Each decode step
4. KV cache growth
```

#### B2: Trace F32 Copy Origin
```
Files: Sources/EdgeRunnerIO/LlamaModel.swift, Sources/EdgeRunnerIO/DequantQ8_0Kernel.swift
Trace:
- Where does the F32 copy get created? (likely in weight binding)
- Is it created for every projection or only some?
- Can we elide it entirely by keeping only quantized form?

Expected finding: Float32 copy is created during weight binding for dequantization kernels
```

#### B3: Eliminate Dual-Copy for Q8 (Conditional)
```
Based on B2 findings:
- If F32 copy is for dequant math, can we dequant in-place?
- If F32 copy is for CPU fallback, can we lazy-allocate it?
- If F32 copy is redundant, remove it and measure memory delta

If instrumentation disproves the redundant-persistent-copy hypothesis, stop here,
document the finding, and revise the target instead of forcing an optimization.
```

#### B4: Memory Pressure Handler Improvements
```
File: Sources/EdgeRunnerIO/MemoryPressureHandler.swift
Improve:
- Add explicit threshold configuration
- Add memory pressure callback to MetalBackend
- Define the callback boundary for future KV cache pressure responses

Note: do not land KV cache eviction until instrumentation proves it is needed and
the runtime semantics are fully specified.
```

---

### Stream C: Streaming & Regression Suite

**Owner:** `EdgeRunner` + `EdgeRunnerCore`
**Phase:** 3 — regression protection

#### C1: Streaming Backpressure
```
File: Tests/EdgeRunnerTests/StreamingTests.swift
Add:
- Task cancellation during generate() → GenerationError.cancelled
- Partial output before cancel → output preserved or explicit cancellation
- Clean termination → no dangling state for the next generate() call

Note: explicit producer backpressure should only be tested if the streaming API is
extended to express bounded buffering or timeout semantics.
```

#### C2: Golden Output Tests
```
File: Tests/EdgeRunnerTests/GoldenOutputTests.swift (new)
Add:
- Fixed prompt → fixed token sequence (greedy argmax)
- Pinned model file → generate 50 tokens with seed=X → same output every time
- Compare against previously verified good outputs

Purpose: catch silent correctness regressions
```

#### C3: Performance Regression CI Gate
```
File: Tests/EdgeRunnerTests/PerformanceRegressionTests.swift (new)
Add:
- Run 4-token benchmark after every commit
- If tok/s drops >15% vs the Phase 1 pinned baseline → fail CI
- If greedy prefix diverges → fail CI
- Baseline: whatever Phase 1 pins in `QwenBenchmark.swift` / `PublishableBenchmark.swift`

Integration: run in CI on every PR
```

#### C4: Memory Leak Detection
```
File: Tests/EdgeRunnerTests/MemoryLeakTests.swift (new)
Add:
- Loop generate 100 times
- Measure VRAM before/after each iteration
- Assert VRAM stable (within 5% after warmup)
- Detect: buffer leaks, tensor accumulation, Metal resource leaks

Execution note: this suite depends on the memory counters added in the memory
instrumentation phase, so it should be landed and verified alongside Stream B
even though it protects regression behavior globally.
```

#### C5: Cancellation Clean Termination
```
File: Tests/EdgeRunnerTests/StreamingTests.swift
Verify:
- Task.cancel() during generate → clean async iteration termination
- No dangling Metal resources
- Model state valid for next generate() call
```

---

### Stream D: Tokenizer Audit

**Owner:** `EdgeRunnerCore`
**Phase:** 2 — tokenizer hardening alongside error-handling work

#### D1: Unicode Edge Cases
```
File: Tests/EdgeRunnerCoreTests/TokenizerTests.swift
Add test corpus:
- Mixed scripts: "Hello世界🎉"
- Emoji sequences: "👨‍👩‍👧‍👦" (ZWJ sequence)
- Right-to-left: "שלום עולם"
- Control characters: tab, newline, null
- Very long texts: 10K+ characters
- Empty strings
```

#### D2: Compare Against HuggingFace Reference
```
Files:
- Tests/EdgeRunnerTests/QwenTokenizerParityTest.swift
- Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift
Add:
- Expand the existing model-backed parity corpus to 100 diverse prompts
- Encode prompts and compare token IDs against previously verified HuggingFace references
- Decode token IDs and compare output strings
- Report any mismatches with the first divergence point

Purpose: validate EdgeRunner's tokenizer matches reference implementations without
creating a second parity stack in a lower-level test target
```

#### D3: Character-Level Debugging
```
File: Tests/EdgeRunnerCoreTests/TokenizerTests.swift
Add:
- encode("a") → should return single token
- decode([id]) → should return single character where applicable
- Round-trip: decode(encode(x)) == x for printable ASCII
- Find the first mismatch point for any failing case
```

---

### Stream E: Benchmark Stability

**Owner:** `EdgeRunnerTests`
**Phase:** 6 — final benchmark stabilization

#### E1: Thermal Profiling
```
File: benchmarks/thermal_profile.md (new)
Document:
- Measure CPU/GPU temp before benchmark run
- Establish thermal baseline (cold start, mid-run, throttled)
- Find the warmup period needed to reach steady state
- Record: thermal throttling threshold for this machine

Purpose: understand if 10x variance is thermal
```

#### E2: Proper Warmup Protocol
```
File: Tests/EdgeRunnerTests/QwenBenchmark.swift
Update benchmark methodology:
1. 10 decode warmup iterations (not 3)
2. Discard first 2 and last 2 timing samples (outliers)
3. Report median of remaining samples
4. Log thermal state at start/end of measurement window
5. Fail benchmark if thermal delta > 10°C during measurement
```

#### E3: Deterministic Benchmark Mode
```
File: Tests/EdgeRunnerTests/PublishableBenchmark.swift
Add:
- EDGERUNNER_DETERMINISTIC=1 mode: single-thread, no OS interrupts
- Pin CPU frequency if possible (disable turbo boost); if not possible on the current machine, document that explicitly
- Run 5 consecutive full benchmarks
- If variance > 5% between runs → surface warning in JSON
- If variance > 15% → fail with explanation
- Record thermal delta and reject runs that exceed the documented measurement bound
```

#### E4: Benchmark Variance Tracking
```
File: benchmarks/variance_tracking.json (auto-generated)
Track across runs:
- median tok/s
- min/max tok/s
- thermal delta
- model file hash
- commit SHA

Purpose: build variance baseline over 30+ runs
```

---

## Verification Gates

Each stream has a verification gate before it is marked complete.

| Stream | Gate | Command |
|--------|------|---------|
| A: Error Handling | GGUF parser, memory-pressure/OOM coverage, generation hardening, and sampling edge-case tests pass | `swift test --filter "GGUFHeaderTests|GGUFTensorTableTests|Memory Pressure Handler Tests"` then `swift test --filter "TokenStream|GenerationSession|EdgeRunnerLanguageModel Protocol"` then `swift test --filter "GreedySampler|TemperatureSampler|TopKSampler|TopPSampler|MinPSampler|RepetitionPenalty|SamplingPipeline"` |
| B: Memory | Memory profile emitted, leak suite green, and optimization only required if redundant persistent copies are confirmed | `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` then verify `benchmarks/memory_profile.json` exists, then `swift test --filter "Memory Leak"` |
| C: Streaming + Regression | Golden output stable and perf regression gate green | `swift test --filter "Golden Output|Performance Regression"` |
| D: Tokenizer | Core tokenizer coverage and expanded Qwen/Gemma parity suites pass against pinned models | `swift test --filter "Tokenizer Protocol|SpecialTokens|TokenizerVocabulary|BPETokenizer|BPETokenizer Pipeline"` then `EDGERUNNER_RUN_TOKENIZER_PARITY=1 swift test --filter "QwenTokenizerParityTest"` and `EDGERUNNER_RUN_GEMMA_TOKENIZER_PARITY=1 swift test --filter "GemmaTokenizerParityTest"` |
| E: Benchmark | 5 consecutive deterministic publishable runs within 5% variance and recorded in `benchmarks/variance_tracking.json` | `EDGERUNNER_DETERMINISTIC=1 swift test -c release --filter "PublishableBenchmark/fullBenchmark"` run 5 times, then verify `benchmarks/variance_tracking.json` |

---

## Success Criteria

After all streams complete, EdgeRunner should meet:

1. **Correctness:** 100% deterministic greedy decode for pinned model
2. **Memory:** produce a measured memory profile for load, prefill, decode, and cache growth; if redundant persistent Float32 copies are confirmed, remove them and re-baseline the target, otherwise document the disproved hypothesis and revise the target before calling the stream complete
3. **Stability:** Benchmark variance <5% across 5 consecutive runs
4. **Error handling:** All error paths tested and graceful
5. **Regression protection:** CI enforces 15% perf regression threshold
6. **Tokenizer:** core tokenizer coverage passes for Unicode/RTL/round-trip behavior, and Qwen/Gemma model-backed parity suites pass on the expanded 100-case corpus derived from verified HuggingFace references

---

## Timeline Estimate

| Stream | Estimated Time | Dependencies |
|--------|---------------|--------------|
| A: Error Handling | 1 week | None |
| B: Memory | 1-2 weeks | A1 (OOM tests first) |
| C: Streaming + Regression | 1 week | None |
| D: Tokenizer | 3-5 days | None |
| E: Benchmark | 1 week | C (needs stable code) |

**Total: ~3-4 weeks** of focused work

---

## Out of Scope (for this plan)

- Model architecture validation (Llama 3, Mistral, Phi-3) — do after framework is stable
- Concurrent inference support — not needed for client-side use case
- Server mode — not needed for target audience
- iOS deployment — separate effort after macOS is production-ready
- New quantization types (IQ variants) — existing K-suffix types are sufficient

---

## Review History

| Date | Reviewer | Changes |
|------|----------|---------|
| 2026-03-22 | Initial | Draft created |
