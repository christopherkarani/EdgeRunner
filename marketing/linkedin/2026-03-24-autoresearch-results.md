# I ran a 6-agent AI research swarm for 3 days. Here's what actually happened.

Not a breakthrough story. Not a 10x improvement thread.

Two failed experiments, one 1.2% gain, and some hard-won lessons about optimization work.

**The setup**

I've been building EdgeRunner — a Swift/Metal inference engine for running LLMs on Apple Silicon. The baseline was 228 tok/s decode on an M3 Max. My target was 280 tok/s to beat MLX's 277 tok/s.

Instead of guessing, I built an autonomous research swarm:

• Researcher — Deep dives on whitepapers and prior art
• Designer — Forms falsifiable hypotheses
• Implementer — Surgical code changes
• Benchmarker — Statistical rigor (10+ runs, p-values)
• Analyst — Keep/rollback decisions
• Logger — Complete experiment database

**The results**

Experiment 33: SIMD-optimized softmax
Expected: 228 → 250 tok/s (+10%)
Actual: Complete correctness failure. Wrong tokens.
Lesson: Numerical stability in parallel reduction is subtle.

Experiment 34: Metal Indirect Command Buffers
Expected: Eliminate CPU encoding overhead
Actual: macOS doesn't support compute ICBs. Abandoned.
Lesson: Read the headers before designing experiments.

Experiment 35: Tile-based GEMV
Expected: 228 → 275 tok/s (+20%)
Actual: 230.8 tok/s (+1.2%)
Lesson: Profile first. The existing kernel was already decent.

**The real insight**

Profiling revealed the truth: the bottleneck isn't kernel efficiency. It's architectural overhead.

GPU time is fixed at ~3.4ms (bandwidth limited). The remaining ~1ms is Swift async overhead, inter-dispatch idle, and TLB misses.

More kernel fusion won't help. The path to 280 tok/s requires protocol changes to eliminate async hops — not micro-optimizations.

**Why this matters**

Most optimization posts show the wins. The 10x improvements. But real engineering is mostly this — failed experiments, wrong assumptions, and incremental gains.

The scientific method doesn't guarantee breakthroughs. It guarantees you'll know when you don't have one. That you'll understand where the ceiling actually is.

Current state: 230 tok/s, 26% faster than llama.cpp, honest about limitations.

The code is open. The experiment log is detailed. The "failures" are documented.

Because that's how you actually improve.

---

EdgeRunner: github.com/chriskarani/EdgeRunner

#metal #swift #llm #performance #iosdevelopment
