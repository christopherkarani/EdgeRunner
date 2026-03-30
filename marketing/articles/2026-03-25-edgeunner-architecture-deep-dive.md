# EdgeRunner Architecture: What We Found When We Read the Whole Damn Codebase

## The TL;DR upfront

EdgeRunner is a Metal-accelerated LLM inference engine for Apple Silicon. It runs GGUF models (Llama, Qwen, Mistral, Gemma, and more) with on-the-fly GPU weight dequantization, mega-kernels that collapse 5 GPU dispatches into 1, and a KV cache that handles overflow via circular buffer rotation.

Decode throughput: ~160 tok/s on Qwen3 0.6B Q8_0. Memory footprint: 310MB peak RSS.

This post is the distillation of a thorough architectural audit. Not a tutorial. Not a benchmark. What we found that was actually interesting.

---

## The quantization system is a three-tier strategy, not a feature list

Most documentation lists supported quantization formats. EdgeRunner has 10 of them (Q4_0, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, plus F16/F32). That's not the interesting part.

The interesting part is the three-tier dequantization strategy:

**Tier 1 — Raw Q8_0 zero-copy.** The raw Q8_0 buffer is passed directly to the fused GEMV kernel. No float32 materialization. The quantized bytes go straight into the matmul, with the scale living in a 2-byte header at the start of each 32-element block.

**Tier 2 — GPU dequantization kernels.** For Q4_0, Q5_0, Q5_1, and the K-quant family (Q2_K through Q6_K), dedicated Metal kernels dequantize on-GPU during the GEMV operation. The dequantization math happens inline with the matmul — not a separate pass.

**Tier 3 — CPU fallback.** Embedding lookups and tied head computations use CPU SIMD loops that decode directly into destination buffers.

EdgeRunner never dequantizes everything upfront. It dequantizes only what's needed, where it's needed, in the kernel that needs it.

---

## The mega-kernels: where dispatch overhead goes to die

EdgeRunner's progressive fusion strategy collapses multiple GPU dispatches into one. It doesn't just fuse adjacent operations; it fuses progressively more operations as kernel complexity justifies the compile time investment.

### `fused_qk_norm_rope_gqa`

This kernel fuses Q/K RMSNorm + RoPE + GQA attention into a single dispatch with zero threadgroup barriers. It caught my attention because of how it handles thread synchronization.

**How it works:**

- Threadgrid: `(32, numHeads+numKVHeads)` — 32 threads per head
- Each thread handles 4 elements: positions `[dimIdx, dimIdx+32, dimIdx+64, dimIdx+96]` of a 128-dim head
- 32 threads × 4 elements = 128-dim complete coverage, all in registers
- SIMD reduction (`simd_sum`) across 32 lanes gives the full dot product with no barrier

**The critical detail:** The Q threads (32 threads per query head) also run the GQA. The KV threads (32 per KV head) compute their RMSNorm + RoPE and write directly to the KV cache, then exit. No barrier between phases — the KV threads simply `return` when done.

```metal
if (!isQ) {
    kCache[cacheBase + ...] = half(q_a0/q_a1/q_b0/q_b1)
    return;  // K threads done — no barrier needed
}
// Q threads continue to GQA phase
```

The online softmax in GQA also uses `simd_broadcast_first` to broadcast corrections across lanes without memory:

```metal
if (dimIdx == 0) {
    nextRunMax = max(runMax, score)
    correction = exp(runMax - nextRunMax)
}
runMax = simd_broadcast_first(nextRunMax);
```

### `dequant_q8_0_fused_ffn_block`

The FFN mega-kernel uses 1024 threads per threadgroup — 32 SIMD groups × 32 threads. One dispatch replaces:

1. Wo GEMV + residual add
2. RMSNorm (attention output residual)
3. Gate GEMV + Up GEMV + SwiGLU
4. Down GEMV + residual add

**Phase structure:**

```
Phase 1: 1024 threads, each = 1 output row
  afterAttn[tid] = Wo[tid,:] · attnOut + residual[tid]
  threadgroup_barrier

Phase 2: Cooperative RMSNorm across 1024 elements
  Each SG (32 threads): simd_sum of squares → partial_sums[sgIdx]
  SG0: reduce 32 partials → rmsScale
  Broadcast via threadgroup memory

Phase 3: Gate + Up + SwiGLU GEMVs (1024 threads, 3 rows each)
Phase 4: Down GEMV + residual add
```

The cross-SIMDgroup RMSNorm works like this: 32 simdgroups each compute `simd_sum` → 32 partial sums in threadgroup memory. Then SIMD group 0 reduces those 32 values via another `simd_sum`. No serial reduction. No separate normalization kernel. Clever stuff.

---

## The KV cache: circular buffers with overflow handling

The KV cache is a circular buffer per layer, with `writePos` tracking the current position and `totalWritten` tracking absolute token count.

```swift
struct LayerState: Sendable {
    var writePos = 0     // Current write position (circular)
    var totalWritten = 0 // Total tokens ever written
}
```

When `totalWritten > maxSeqLen`, the cache enters "rotated" mode. Retrieval handles this by reading in two chunks: `[writePos...maxSeqLen]` then `[0...writePos]`. This correctly reconstructs the causal view for attention without re-computing anything.

GPU kernels write K/V directly to the cache at computed offsets:

```swift
let cacheWriteOffF16 = (t + startPosition) * kvDim * halfStride
enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
```

No separate cache management kernel. The write position is known at dispatch time.

---

## Three inference modes, auto-detected

EdgeRunner has three GPU paths, chosen automatically per call based on token sequence analysis:

**Decode mode:** `tokenIDs.count == previousTokenIDs.count + 1` and common prefix covers everything. Single new token embedded. Full KV cache used. The fast path.

**Prefix reuse mode:** Sequence extends a cached prefix. Only suffix tokens are embedded, but RoPE positions are offset by `commonPrefixLen` and GQA attends over the full KV cache. This is what makes multi-turn conversations fast.

**Full prefill mode:** No useful prefix match. KV cache reset. Entire sequence processed. Also runs 5 dummy decode passes first to warm the GPU JIT pipeline, then re-runs the actual prefill.

The mode detection is in `forwardLogitsBuffer`:

```swift
let commonPrefixLen = countMatchingPrefix(previousTokenIDs, tokenIDs)
let isDecodeMode = commonPrefixLen == previousTokenIDs.count
    && tokenIDs.count == commonPrefixLen + 1
let isPrefixReuseMode = commonPrefixLen > 0
    && commonPrefixLen == previousTokenIDs.count
    && tokenIDs.count > commonPrefixLen + 1
```

No configuration. No hints. The model just gets faster on subsequent turns.

---

## Zero-allocation inference via scratch buffers

EdgeRunner pre-allocates 19 persistent GPU buffers at model init, sized for `maxSeqLen`. These cover every intermediate tensor:

| Buffer | Purpose |
|--------|---------|
| `normed` | RMSNorm output |
| `afterAttn` | Post-attention residual |
| `outputA`, `outputB` | Layer output ping-pong |
| `allQ`, `allK`, `allV` | Full Q/K/V tensors (prefill) |
| `ropeQ`, `ropeK` | RoPE output |
| `gateOut`, `upOut` | SwiGLU intermediates |
| `logits` | Final vocab output |

Ping-pong buffers (`outputA`/`outputB`) alternate every layer for residual addition without an extra copy. The residual add is just reading from the buffer written by the previous layer.

This eliminates ~17 MTLBuffer allocations per forward pass. Forward passes never call `device.makeBuffer()`.

---

## The sampling pipeline: composable and explicit

EdgeRunner's sampling is built on a `SamplingPipeline` that composes transforms and a selector:

```
Logits → RepetitionPenalty → Temperature → TopK → TopP → Greedy/Stochastic
```

Each transform is a value type conforming to `LogitsTransform`. The pipeline is constructed from `SamplingConfiguration`:

```swift
public func toPipeline() -> SamplingPipeline {
    if temperature <= 0 {
        return SamplingPipeline(
            transforms: [],
            selector: GreedySampler()
        )
    } else {
        return SamplingPipeline(
            transforms: [
                TemperatureSampler(temperature: temperature),
                TopKSampler(k: topK),
                TopPSampler(p: topP)
            ],
            selector: StochasticSampler(randomSource: &rng)
        )
    }
}
```

The repetition penalty is applied before temperature. This is the right order — you penalize before sharpening the distribution.

---

## Hardware optimization details

**`powr` vs `pow`:** RoPE frequency computation uses `powr(params.theta, exponent)` instead of `pow`. On Apple Silicon, `powr` uses hardware reciprocal units. The difference is measurable in the RoPE kernel — not huge, but free.

**Half-precision strategy:** KV cache is Float16 (50% memory savings). Q8_0 scale is Float16 (2 bytes sufficient). Inner GEMV dot accumulates in Float16 (2× ALU throughput on Apple 2:1 f16:f32 ratio). Outer accumulator stays Float32 (prevents drift over many blocks). Attention scores stay Float32 (numerical stability in exp).

**Command batching:** One encoder per forward pass. All dispatches within the encoder are implicitly ordered. Creating a new encoder per dispatch costs ~0.01ms each — for a 40-layer model, that's 4ms just in encoder overhead.

---

## What's in the full document

The complete architecture reference at `docs/arch/EdgeRunner_Complete_Reference.md` covers:

- All 10 quantization formats with byte-level layout diagrams
- Every Metal shader with threadgroup dimensions and math
- The complete inference pipeline from tokenization to logits
- Memory management architecture (BufferCache, BarrierTracker, ResidencyManager)
- The three-tier dequantization strategy
- KV cache mechanics including overflow rotation
- The composable sampling pipeline
- Full public API reference with usage examples

---

## Try it

```swift
import EdgeRunner

let model = try await ModelLoader.load(from: modelURL)
let session = GenerationSession(model: model, maxTokens: 1024)

for try await text in session.stream(prompt: "Write a story") {
    print(text, terminator: "")
}
```

Full docs: `docs/arch/EdgeRunner_Complete_Reference.md`
Repo: https://github.com/YourHandle/EdgeRunner
