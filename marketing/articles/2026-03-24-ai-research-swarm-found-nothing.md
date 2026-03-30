# I Built a 6-Agent AI Research Swarm to Optimize My Code. It Found Nothing.

## What happens when the scientific method meets stubborn performance ceilings

I spent three days running an autonomous research swarm on EdgeRunner, my Swift/Metal LLM inference engine. Six specialist agents, rigorous methodology, falsifiable hypotheses.

The result? 228 → 230.8 tok/s.

Not a typo. Two failed experiments, one marginal gain, and a hard lesson about where optimization actually happens.

---

## The Setup

EdgeRunner runs quantized LLMs on Apple Silicon. Think llama.cpp but Swift-native, Metal-first, and designed for iOS/macOS apps that need on-device inference.

The baseline: 228 tok/s decode throughput on an M3 Max (Qwen 0.6B Q8_0).

The target: Beat MLX's 277 tok/s — the gold standard for Apple Silicon inference.

The gap: 49 tok/s. About 17%.

Instead of randomly trying optimizations, I built a proper research infrastructure.

---

## The Swarm

Six agents, each with a specific role:

**Researcher** — Deep dives on whitepapers, llama.cpp source, MLX implementations. Finds techniques, assesses applicability to Metal/Swift.

**Designer** — Forms falsifiable hypotheses with quantified predictions. "If we change X, then metric Y will improve by Z."

**Implementer** — Surgical code changes. Preserves correctness, minimal diffs.

**Benchmarker** — Statistical rigor. 10+ runs, p-values, confidence intervals.

**Analyst** — Keep/rollback decisions. Is this a breakthrough (>3%, p<0.05) or not?

**Logger** — Complete experiment database. Every hypothesis, result, and lesson learned.

The scientific method, automated.

---

## Experiment 33: SIMD-Optimized Softmax

**Hypothesis:** Replace threadgroup-based softmax reduction with `simd_sum` and `simd_max` for 10-15% speedup.

**Mechanism:** The current GQA attention kernel uses threadgroup barriers for cross-simdgroup reduction. Metal's `simd_sum` operates in hardware, theoretically faster.

**Predicted:** 228 → 250 tok/s

**Implementation:** Modified the mega-kernel in RoPE.metal. Each thread maintains running max/sum, uses `simd_max` and `simd_sum` to combine across the simdgroup.

**Result:** Complete correctness failure.

The tokens went from `[1, 1479, 35, 5371, 1]` to `[1, 101828, 122053]`. The SIMD-optimized renormalization had a subtle bug in how per-thread accumulators combined across the simdgroup.

Online softmax is numerically unstable. The "obvious" optimization broke it.

**Status:** ROLLED BACK

**Lesson:** SIMD optimizations require extreme care with numerical stability. Profile correctness before performance.

---

## Experiment 34: Metal Indirect Command Buffers

**Hypothesis:** Pre-record the 142-dispatch decode sequence into a Metal Indirect Command Buffer (ICB), eliminating per-decode CPU encoding overhead.

**Mechanism:** ICBs allow recording a command sequence once, then replaying it with only buffer offset updates. Theoretically eliminates ~0.5ms of encoding overhead per decode.

**Predicted:** 228 → 260 tok/s (+14%)

**Implementation:** Attempted to create `DecodeICBState.swift` with ICB recording during warmup and replay during decode.

**Result:** COMPILATION FAILED

```swift
// This works on iOS:
icbDesc.commandTypes = .concurrentCompute  // ✅

// On macOS:
// ❌ Type 'MTLIndirectCommandType' has no member 'concurrentCompute'
// ❌ 'indirectComputeCommand(at:)' is unavailable in macOS
```

Metal ICBs on macOS only support render commands, not compute. They're designed for draw-call-heavy rendering pipelines, not compute-heavy LLM inference.

**Status:** ABANDONED

**Lesson:** Read the SDK headers *before* designing the experiment. ICBs are render-focused; compute is iOS-only.

---

## Experiment 35: Tile-Based GEMV

**Hypothesis:** Restructure Q8_0 GEMV to use cooperative tile-based access patterns, increasing effective DRAM bandwidth from 207 GB/s to 250+ GB/s.

**Mechanism:** Current kernel has strided memory access (each thread loads elements 32 apart), causing DRAM row buffer thrashing. Tile-based approach: 32 threads cooperatively load a contiguous tile into threadgroup memory, then access from fast SRAM.

**Predicted:** 228 → 275 tok/s (+20%)

**Implementation:** New kernel `dequant_q8_0_gemv_tiled` with cooperative tile loading:

```metal
threadgroup float tile[TILE_SIZE];
// Cooperatively load tile
for (uint i = tid.x; i < TILE_SIZE; i += 32) {
    tile[i] = x[i];  // Coalesced loads
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Process from tile
```

**Result:** 228 → 230.8 tok/s (+1.2%)

The existing Q8_0 GEMV kernel already had reasonable memory access patterns. Threadgroup synchronization overhead offset most of the gains.

**Status:** KEPT (code is cleaner, but no breakthrough)

**Lesson:** Profile first. The existing implementation was closer to optimal than expected.

---

## The Real Bottleneck

Here's what profiling revealed:

| Component | Time | % of Total |
|-----------|------|------------|
| GPU execution | 3.4ms | 77% |
| Swift async overhead | 0.23ms | 5% |
| Inter-dispatch idle | ~0.5ms | 11% |
| TLB/page overhead | ~0.3ms | 7% |

The GPU time is essentially fixed. At 604MB of weight data per decode and ~400 GB/s theoretical DRAM bandwidth, the floor is ~1.5ms for memory alone. Add compute, and you get 3.4ms.

The remaining ~1ms is architectural overhead:
- Swift's async/await per token
- 142 dispatches with inter-dispatch idle time
- TLB misses from 196 distinct weight buffers

More kernel fusion won't help. We've gone from 15 dispatches/layer down to 5. The mega-kernel already fuses Q/K norm, RoPE, and GQA into one dispatch.

The path to 280 tok/s requires:
1. Protocol changes to eliminate async overhead (not micro-optimizations)
2. Metal ICBs (but macOS doesn't support compute ICBs)
3. Fundamental memory layout changes (consolidate weight buffers)

---

## The Scientific Method Works

The autoresearch swarm "failed" to hit the 280 tok/s target. But it succeeded at exactly what it was designed to do:

- Falsified two hypotheses quickly (3 days vs. weeks of blind optimization)
- Documented exactly why they failed
- Found the real bottleneck (architectural, not kernel-level)
- Prevented shipping broken optimizations

The scientific method doesn't guarantee breakthroughs. It guarantees you'll know when you don't have one.

That's worth more than fake metrics.

---

## Current State

**EdgeRunner performance:**
- Decode: 230 tok/s (M3 Max, Qwen 0.6B Q8_0)
- 26% faster than llama.cpp reference (183 tok/s)
- 17% behind MLX (277 tok/s)
- Memory: 269MB peak RSS (18.5x improvement from earlier)

**What worked:**
- 6-agent swarm with clear role separation
- Falsifiable hypotheses with quantified predictions
- Rigorous correctness checks (token hash matching)
- Complete experiment logging

**What didn't:**
- SIMD softmax — numerical stability issues
- Metal ICBs — macOS API limitations
- Tile GEMV — existing code already near-optimal

---

## The Honest Conclusion

Sometimes the ceiling is closer than you think.

We've done the obvious optimizations: fused kernels, Q8_0 direct read, single-simdgroup dispatches, mega-kernels. The remaining gap to MLX isn't in kernel efficiency — it's in Swift/Metal interop overhead and memory architecture.

That's not a quick fix. That's a rewrite.

And that's fine. The experiment swarm told us exactly where we stand.

---

**Repo:** github.com/chriskarani/EdgeRunner

**Experiment log:** benchmarks/experiment_log.md

**Honest about limitations:** Always.
