# EdgeRunner Agent Instructions

## Autoresearch: Inference Optimization Agent

Autonomous optimization loop targeting maximum autoregressive decode tokens/sec
on Qwen 3 0.6B Q8_0 (pinned GGUF size: 639,446,688 bytes). Follows the Karpathy autoresearch pattern:
RESEARCH → MODIFY → BUILD → BENCHMARK → KEEP/ROLLBACK → REPEAT.

### Current Benchmark Ground Rules

- Primary metric: **Publishable benchmark** (128-token greedy decode, TTFT separated, release build)
- Command: `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- Model: Qwen 3 0.6B Q8_0 at `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf` (expected size: **639,446,688** bytes)
- Contract source of truth: `benchmarks/pinned_qwen3_0.6b_q8_0.json`
- Benchmark harnesses pin the safe decode path with the mega fused GQA kernel disabled until that kernel regains deterministic correctness on the pinned artifact.
- `QwenBenchmark/decodeBenchmark` is **smoke/regression only** (4 tokens, not apples-to-apples)
- Cached JSON artifacts in `benchmarks/` are for record-keeping only; always rerun benchmarks for truth
- Metal 4: available on macOS 26+, but the optimized Metal 3 decode path remains the default. Use `EDGERUNNER_DECODE_PREFER_METAL4=1` to compare Metal 4 against the default path.

### The Loop

```
1. READ latest benchmark results (rerun if stale) → publishable benchmark JSON
2. RESEARCH → find optimization (see Optimization Patterns below)
3. MODIFY → edit Sources/EdgeRunner/Models/LlamaLanguageModel.swift
4. BUILD → swift build 2>&1 | tail -5
   - If fails → FIX or ROLLBACK: git checkout -- <files>
5. BENCHMARK → swift test -c release --filter "PublishableBenchmark/fullBenchmark" 2>&1 | tail -15
   - Parse: publishable decode metrics (p50 decode tok/s is canonical)
6. EVALUATE:
   - IMPROVED + correctness PASSED → git add + commit
   - REGRESSED or correctness FAILED → git checkout -- <files>
7. LOG → append to benchmarks/experiment_log.md
8. REPEAT
```

### What You Can Modify

- `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` — primary target
- `Sources/EdgeRunnerMetal/*.swift` — Metal kernel wrappers
- `Sources/EdgeRunnerIO/Dequant*.swift` — dequantization kernels
- `Sources/EdgeRunnerCore/Sampling/*.swift` — sampling pipeline

### What You MUST NOT Modify

- Benchmark harness semantics (`Tests/EdgeRunnerTests/PublishableBenchmark.swift`, `Tests/EdgeRunnerTests/QwenBenchmark.swift`) without intentional, documented changes
- `Package.swift` — no dependency changes

### Correctness Guard

Canonical publishable benchmark guard: greedy prefix must start with `[1, 1479, 35]`.
Canonical publishable runs also enforce the pinned full-token hash for the 128-token harness.
If the publishable benchmark loses determinism, the prefix changes, or the canonical token hash changes on the pinned GGUF, treat it as a correctness regression.

### Commit Protocol

```bash
git commit -m "perf: <what changed> — <old> → <new> tok/s (+<pct>%)"
```

### Experiment Log Format

Append to `benchmarks/experiment_log.md`:

```markdown
### Experiment N: <title>
- **Hypothesis:** <what you expect to improve>
- **Change:** <description>
- **Result:** <before> → <after> tok/s (+/- pct%)
- **Status:** KEPT / ROLLED BACK
```

---

## Optimization Patterns (from llama.cpp + MLX)

### Pattern 1: Single-Token Decode Path (5-100x)

EdgeRunner recomputes the FULL sequence every decode step. llama.cpp and MLX
separate prefill (n_tokens > 1) from decode (n_tokens == 1).

For decode: only process the NEW token. Use KV cache for all previous K/V.

```swift
func logits(for tokenIDs: [Int]) async throws -> [Float] {
    if tokenIDs.count == cachedPosition + 1 {
        // DECODE: only process last token
        return try await decodeSingleToken(tokenID: tokenIDs.last!, position: cachedPosition)
    } else {
        // PREFILL: process all, populate KV cache
        kvCache.reset()
        return try await prefillTokens(tokenIDs)
    }
}
```

### Pattern 2: Batched Projections via GEMM (10-50x)

Replace per-token GEMV loop with single GEMM call:

```swift
// BEFORE: seqLen × command buffers
for t in 0..<seqLen {
    let q = try await gemvKernel.execute(a: wq, x: token[t], M: qDim, K: dim, ...)
}

// AFTER: 1 command buffer
// C[qDim, seqLen] = A[qDim, dim] × B[dim, seqLen]
```

### Pattern 3: Single Command Buffer (2-5x)

Each kernel call creates its own MTLCommandBuffer. llama.cpp encodes ALL ops
into a single command buffer and commits once.

```swift
let commandBuffer = commandQueue.makeCommandBuffer()!
// Encode ALL layer ops into this buffer
for layer in 0..<layerCount {
    // RMSNorm, Q/K/V, RoPE, Attention, FFN — all same buffer
}
commandBuffer.commit()
await commandBuffer.completed()  // ONE GPU round-trip
```

### Pattern 4: Fused Dequant+GEMV (2-3x)

EdgeRunner already uses fused Q8_0 decode kernels in the hot path.
The remaining work is to profile and tighten the highest-cost fused kernels rather than re-introducing float materialization.

### Pattern 5: GPU LM Head (2-5x for 151K vocab)

The active path already keeps the LM head on GPU. Current work should focus on:
- avoiding host logits materialization for greedy decode
- profiling the fused final norm + LM head cost
- improving kernel occupancy if LM head remains dominant

### Pattern 6: KV Cache Usage

EdgeRunner already uses KV cache in decode and prefix-reuse paths.
Current optimization work is about reducing cache/barrier overhead and measuring context-length scaling.

### Priority Order

| # | Optimization | Expected | Complexity |
|---|-------------|----------|------------|
| 1 | Single-token decode + KV cache | 5-100x | Medium |
| 2 | Batched GEMM for prefill | 10-50x | Medium |
| 3 | Single command buffer | 2-5x | Low |
| 4 | Fused dequant+GEMV Q8_0 | 2-3x | Medium |
| 5 | GPU LM head | 2-5x | Low-Medium |

---

## Model Config (Qwen 3 0.6B)

```
embeddingDim: 1024, layerCount: 28, headCount: 16, kvHeadCount: 8
headDim: 128, intermediateDim: 3072, vocabSize: 151936
ropeFreqBase: 1000000.0, rmsNormEpsilon: 1e-6
```

## Running the Benchmark

```bash
swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```
