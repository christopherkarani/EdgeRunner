# EdgeRunner Autoresearch Experiment Log

**Model:** Qwen 3 0.6B Q8_0 (610 MB)
**Device:** Apple M3 Max, 28 GB unified memory
**Metric:** Autoregressive decode tokens/sec (greedy, 4 tokens)

---

### Experiment 0: Baseline
- **Hypothesis:** Establish initial performance measurement
- **Change:** First working Qwen 3 inference — naive per-token GEMV loop, CPU LM head, no KV cache reuse
- **Files modified:** LlamaLanguageModel.swift (initial implementation)
- **Result:** 0.0000 → 0.0578 tok/s
- **Status:** KEPT
- **Commit:** cb4d887
