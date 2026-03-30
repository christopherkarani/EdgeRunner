# What I found reading the whole EdgeRunner codebase

We just finished a thorough architectural documentation of EdgeRunner — a Metal LLM inference engine for Apple Silicon.

The results are interesting even if you don't care about Metal kernels.

**The headline numbers:** ~160 tok/s decode throughput on Qwen3 0.6B Q8_0. 310MB memory footprint. Three inference modes, auto-detected per call.

But the architecture is where it gets worth reading.

**The most interesting design decision:** Progressive kernel fusion.

Most LLM runtimes have a "fuse what you can" approach. EdgeRunner has a progressive strategy where the fusion depth depends on the hardware capability. The most aggressive kernel — `fused_qk_norm_rope_gqa` — computes Q/K RMSNorm + RoPE + GQA attention in one dispatch with zero threadgroup barriers. Not fused carefully. Fused completely.

How it works: 32 threads per head, each handling 4 elements of a 128-dim vector. SIMD sum across lanes gives the full dot product without a barrier. The KV threads write directly to the KV cache and exit. The Q threads continue to GQA. No synchronization event between the phases.

**The insight that transfers:** Scratch buffer pre-allocation.

EdgeRunner pre-allocates 19 persistent GPU buffers at model init. Every intermediate tensor has a fixed slot. Forward passes never allocate GPU memory.

This isn't about reducing allocation overhead (though that helps). It's about making the inference loop a fixed memory access pattern. No dynamic allocation means no allocator behavior to model. The GPU knows exactly what memory it'll touch before the first kernel fires.

**One detail worth knowing:** `powr` vs `pow` in RoPE.

RoPE frequency computation uses `powr(base, exp)` instead of `pow(base, exp)`. On Apple Silicon, `powr` uses hardware reciprocal units. Same mathematical result, faster execution.

**What this means for on-device ML:**

The constraints on Apple Silicon are real. Unified memory means you can't hide latency behind copies. The Neural Engine has its own memory domain. Metal lets you access all of it, but you have to think about memory bandwidth across the whole system, not just GPU compute.

EdgeRunner's approach is: know the access pattern, pre-allocate for it, fuse to reduce synchronization, use the right precision at every layer.

Full architecture doc: `docs/arch/EdgeRunner_Complete_Reference.md`

What's your on-device inference setup looking like? Curious what numbers others are seeing on M3 Max vs M4.
