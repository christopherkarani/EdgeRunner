# Standalone Tweets — 2026-03-24

## Tweet 1 — Hot take

3 experiments. 2 failures. 1.2% improvement.

This is what optimization actually looks like. Not the 10x threads — the failed SIMD attempts, the API limitations you discover too late, the hypotheses that don't survive contact with reality.

The scientific method doesn't guarantee breakthroughs. It guarantees you'll know when you don't have one.

📎 Image: experiment summary

---

## Tweet 2 — Metric bomb

228 → 230.8 tok/s

After 3 days with a 6-agent autoresearch swarm. After studying llama.cpp and MLX source. After three carefully designed experiments with falsifiable hypotheses.

The gap to my 280 tok/s target? Still 47 tok/s.

Sometimes the ceiling is closer than you think.

🔗 Repo in bio

---

## Tweet 3 — Code flex

TIL Metal Indirect Command Buffers don't support compute on macOS.

```swift
// This works on iOS:
icbDesc.commandTypes = .concurrentCompute

// On macOS? Doesn't exist.
// ICBs are render-only. Compute needs direct encoding.
```

Read the headers *before* you design the experiment. Not after.

📎 Image: code snippet

---

## Tweet 4 — Insight

The most expensive optimization I tried: SIMD-optimized softmax.

Replaced threadgroup barriers with `simd_sum`. Cleaner code. Better theoretical throughput.

Result: Completely wrong tokens. Numerical stability in online softmax is subtle — the "obvious" SIMD optimization broke renormalization.

Rolled back. Lesson learned.

---

## Tweet 5 — Question hook

What if you could run 6 specialist AI agents in parallel to optimize your code?

Researcher. Designer. Implementer. Benchmarker. Analyst. Logger.

Each with a specific role. Each following the scientific method. Each experiment logged, measured, rolled back if it fails.

Just tried it. Here's what happened 🧵

---

## Tweet 6 — Architecture insight

Profiling revealed the truth: my LLM inference isn't kernel-bound. It's overhead-bound.

- GPU time: 3.4ms per decode
- Swift async overhead: 0.23ms
- The rest: inter-dispatch idle, TLB misses

More kernel fusion won't help. Need protocol changes to eliminate async hops.

---

## Tweet 7 — TIL

Tile-based GEMV with cooperative threadgroup loading:

Expected: 20% bandwidth improvement (207 → 250 GB/s)
Actual: 1.2%

The existing Q8_0 kernel already had decent memory access patterns. Threadgroup synchronization ate the gains.

Profile first. Optimize second. Or third.

---

## Tweet 8 — API showcase

Scientific method in code:

```swift
// Hypothesis: ICBs eliminate encoding overhead
// Predicted: 228 → 260 tok/s (+14%)
// Failure mode: <5% improvement
// Actual: COMPILATION FAILED (macOS limitation)
// Verdict: HYPOTHESIS ABANDONED
```

Not every experiment succeeds. The process is the point.

📎 Image: code snippet

---
