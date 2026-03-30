# Thread — EdgeRunner Architecture Deep Dive — 2026-03-25

## 1/ (Hook)

We just fully documented EdgeRunner — a Metal-accelerated LLM inference engine for Apple Silicon.

Not a blog post. Not an overview. ~2,500 lines of architecture analysis covering every quantization format, every kernel, every inference mode.

Here's what we found that was actually interesting 🧵

---

## 2/

The quantization system isn't the interesting part. The interesting part is how you use it.

EdgeRunner has a three-tier dequantization strategy:

→ Raw Q8_0: zero-copy pass-through to GEMV kernel
→ Other quants: GPU dequant kernel per format
→ Embeddings: CPU fallback with SIMD loops

Most runtimes dequantize everything upfront. EdgeRunner only dequantizes what's needed, where it's needed.

📎 Image: `../assets/diagrams/2026-03-25-dequant-tiers.svg`

---

## 3/

The most technically interesting kernel is `fused_qk_norm_rope_gqa`.

It computes per-head Q/K RMSNorm + RoPE + GQA attention for a 128-dim head.

How it works:
- 32 threads per head (one full SIMDgroup)
- Each thread handles 4 elements: [i, i+32, i+64, i+96]
- SIMD sum across 32 threads covers the full 128-dim head
- GQA dot products also use SIMD reductions
- Zero threadgroup_barrier calls

---

## 4/

The contrast with naive attention is stark.

Naive: RMSNorm kernel → Q GEMV → K GEMV → RoPE Q → RoPE K → GQA kernel
Each kernel = separate GPU dispatch, separate synchronization.

`fused_qk_norm_rope_gqa`: All in one dispatch. One synchronization event.

The performance difference isn't algorithmic — it's the reduction of dispatch overhead and barrier latency.

---

## 5/

The FFN mega-kernel is even more aggressive.

`dequant_q8_0_fused_ffn_block` uses 1024 threads per threadgroup (32 SIMD groups).

What it does in one dispatch:
1. Wo GEMV + residual add
2. Cooperative RMSNorm across 1024 elements (cross-SIMDgroup barrier)
3. Gate + Up + SwiGLU GEMVs
4. Down GEMV + residual add

That's 5 operations. 1 dispatch. 0 context switches.

---

## 6/

The KV cache design is worth understanding.

Per-layer circular buffers. `writePos` tracks current position. `totalWritten` tracks absolute count.

When `totalWritten > maxSeqLen`, the cache enters "rotated" mode — retrieval reads in two chunks: `[writePos...maxSeqLen]` then `[0...writePos]`.

This correctly reconstructs the causal view for attention without re-computing anything.

---

## 7/

EdgeRunner has three inference modes, auto-detected per call:

**Decode** (common): single new token, KV cache valid. Only the new token is embedded.

**Prefix Reuse** (multi-turn): sequence extends cached prefix. Only suffix tokens are embedded. RoPE positions offset. Full KV cache used.

**Full Prefill**: no useful prefix. KV cache reset. Entire sequence processed.

The framework chooses. You don't configure it.

---

## 8/

One detail that surprised me: scratch buffer pre-allocation.

EdgeRunner pre-allocates 19 persistent GPU buffers at model init, sized for `maxSeqLen`.

These buffers cover every intermediate tensor: normed outputs, attention projections, FFN intermediates, logits.

Forward passes never allocate GPU memory. ~17 MTLBuffer allocations eliminated per inference step.

This is zero-allocation not as a policy but as a data structure design.

---

## 9/

Memory management: a layered approach.

- `BufferCache`: LRU cache, 50% VRAM max, size bucketed
- `BarrierTracker`: tracks buffer writes, inserts MTLBarrier only when needed
- `CommandBatcher`: batches dispatches into single encoder, threshold varies by chip (50 for apple9, 40 for apple8, 30 for others)
- `ResidencyManager`: MTLResidencySet hints on Metal 4

None of this is novel. All of it is necessary.

---

## 10/ (Closer)

The full architecture doc is at `docs/arch/EdgeRunner_Complete_Reference.md`.

2,500 lines covering:
→ 10 quantization formats with byte-level layouts
→ All Metal shaders with threadgroup dimensions
→ The three-tier dequantization strategy
→ KV cache circular buffer mechanics
→ Sampling pipeline composition
→ Public API surface

If you're building Metal compute kernels, this might save you some research.

🔗 https://github.com/YourHandle/EdgeRunner

The TL;DR: the interesting optimizations in EdgeRunner aren't algorithmic breakthroughs. They're dispatch consolidation, memory layout awareness, and hardware-specific SIMD usage.
