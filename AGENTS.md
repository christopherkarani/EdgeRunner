# EdgeRunner Agent Instructions

## Autoresearch: Inference Optimization Agent

Autonomous optimization loop targeting maximum autoregressive decode tokens/sec
on Qwen 3 0.6B Q8_0 (610 MB GGUF). Follows the Karpathy autoresearch pattern:
RESEARCH → MODIFY → BUILD → BENCHMARK → KEEP/ROLLBACK → REPEAT.

### Current Baseline

- **0.058 tok/s** (17.3s per token)
- Model: Qwen 3 0.6B Q8_0 at `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
- Baseline file: `benchmarks/baseline.json`
- Target: **15-30 tok/s**

### The Loop

```
1. READ baseline → benchmarks/baseline.json
2. RESEARCH → find optimization (see Optimization Patterns below)
3. MODIFY → edit Sources/EdgeRunner/Models/LlamaLanguageModel.swift
4. BUILD → swift build 2>&1 | tail -5
   - If fails → FIX or ROLLBACK: git checkout -- <files>
5. BENCHMARK → swift test --filter "QwenBenchmark/decodeBenchmark" 2>&1 | tail -15
   - Parse: BENCHMARK: qwen_decode_throughput <value> tokens/sec
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

- `Tests/EdgeRunnerTests/QwenBenchmark.swift` — benchmark is ground truth
- `benchmarks/baseline.json` — written by benchmark only
- `Package.swift` — no dependency changes

### Correctness Guard

Expected greedy tokens: `[1, 1479, 21456, 96793, 15859]`.
If your optimization changes these, it broke correctness. **ROLLBACK IMMEDIATELY.**

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

EdgeRunner dequantizes Q8_0 → Float32, then does GEMV. This doubles memory bandwidth.
llama.cpp and MLX operate directly on quantized data.

EdgeRunner already has `dequantQ4_0.fusedDequantGEMV()`. Need Q8_0 equivalent:
```swift
dequantQ8_0.fusedDequantGEMV(quantisedRows: rawBytes, x: hidden, rows: M, cols: K, ...)
```

### Pattern 5: GPU LM Head (2-5x for 151K vocab)

`computeTiedLMHead()` runs 151K dot products on CPU. Move to GPU:
```swift
// Use fused dequant+GEMV for the tied embedding LM head
let logits = try await dequantQ8_0.fusedDequantGEMV(
    quantisedRows: embeddingRawBytes, x: hidden,
    rows: vocabSize, cols: dim, commandQueue: queue
)
```

### Pattern 6: KV Cache Usage

EdgeRunner has KVCache but doesn't use it in the forward pass. For single-token decode:
```swift
// Project new K, V
let newK = try await gemvKernel.execute(a: wk, x: hidden, M: kvDim, K: dim, ...)
// Append to cache
try kvCache.append(layer: layerIndex, keys: newK, values: newV)
// Attend new Q against ALL cached K/V
let (allK, allV) = try kvCache.retrieve(layer: layerIndex, asType: Float.self)
```

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
swift test --filter "QwenBenchmark/decodeBenchmark" 2>&1 | grep -E "BENCHMARK:|BASELINE:"
```
