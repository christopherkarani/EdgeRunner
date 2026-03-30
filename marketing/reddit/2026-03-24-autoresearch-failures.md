# When Your "10% Speedup" Experiment Actually Regresses Performance

I spent the last 3 days running an autonomous research swarm on my Metal LLM inference engine. 6 agents, 3 experiments, rigorous scientific methodology.

The result? Two failures, one 1.2% improvement, and a hard lesson about optimization.

**What we tried:**

**Experiment 33: SIMD-optimized softmax**
Hypothesis: Replace threadgroup barriers with `simd_sum`/`simd_max` for 10-15% speedup.
Reality: Correctness regression. Wrong tokens. Rolled back.
Lesson: Numerical stability in parallel reduction is subtle. The "obvious" optimization broke online softmax renormalization.

**Experiment 34: Metal Indirect Command Buffers**
Hypothesis: Pre-record 142 dispatches to eliminate CPU encoding overhead. 14% expected gain.
Reality: `MTLIndirectCommandType.compute` doesn't exist on macOS. Only available on iOS/tvOS.
Lesson: Read the headers before designing the experiment. ICBs are render-focused, not compute-focused.

**Experiment 35: Tile-based GEMV**
Hypothesis: Cooperative tile loading into threadgroup memory for 20% bandwidth improvement.
Reality: +1.2%. Existing kernel already had reasonable access patterns.
Lesson: Profile first. The bottleneck wasn't where I thought.

**The numbers:**
- Baseline: 228 tok/s
- After 3 days of work: 230.8 tok/s
- Gap to target (beat MLX at 277 tok/s): Still 47 tok/s

**Why I'm posting this:**

Most optimization posts show the wins. The "we achieved 10x speedup" threads. But real engineering is mostly this — failed experiments, wrong assumptions, and incremental gains.

The autoresearch swarm worked exactly as designed. Scientific method: hypothesis → experiment → measure → decide. Two hypotheses falsified, one partially validated. That's science working correctly, even when the result is "no breakthrough."

**Current state of the project:**
- EdgeRunner: Swift/Metal LLM inference
- 0.6B models at 230 tok/s on M3 Max
- 26% faster than llama.cpp reference
- Still 17% behind MLX

The remaining gap is likely architectural — Swift async overhead, not kernel efficiency. That needs protocol changes, not micro-optimizations.

**Repo:** github.com/chriskarani/EdgeRunner

Happy to answer questions about Metal compute, quantization kernels, or why your "obvious" optimization might not work.

---

*Edit: To save the "did you try..." comments — yes, we profiled. Yes, we checked bandwidth utilization. Yes, we tried simdgroup_matrix. Yes, we tried NR=4. They all either failed or regressed. There's a detailed experiment log in benchmarks/experiment_log.md if you want the full autopsy.*
