# Autoresearch: Beat MLX on Qwen3-0.6B Q8_0 Decode

## Current Status
- MLX (Python): **277.8 tok/s** median decode (128 tokens)
- llama.cpp: **200.3 tok/s** median decode
- **EdgeRunner: 234.8 tok/s** median decode (128 tokens)
- Gap to MLX: **43 tok/s (15.5%)**
- **EdgeRunner beats llama.cpp by 17%**

## Completed Experiments
- [x] **Exp 21: Single-Simdgroup GQA** -- 207.5 -> 234.8 tok/s (+13.2%) KEPT
- [x] **Exp 22: f16acc GEMV Kernels** -- NaN, ROLLED BACK
- [x] **Exp 23: GQA Loop Unrolling** -- Correctness failure, ROLLED BACK
- [x] **Exp 24: Fast Math** -- No improvement, ROLLED BACK
- [x] **Exp 25: Reusable Logits Array** -- Slower (COW issues), ROLLED BACK
- [x] **GPU Profiling** -- Identified exact bottleneck split

## Bottleneck Analysis (from GPU profiling)
At 234.8 tok/s = 4.26ms/token average:
- **Weight GEMV**: 3.07ms (72%) -- 207 GB/s effective, 635MB data
- **GQA attention**: 0.65ms (15%) -- grows 9.8us per KV position
- **Dispatch overhead**: 0.31ms (7%) -- 142 dispatches x 2.2us
- **CPU/async overhead**: 0.23ms (5%) -- array copy + continuation

## Next Optimizations (priority order)
- [ ] **Flash-Decode GQA** -- Parallelize KV scan into chunks with separate threadgroups. Each chunk scans a portion of KV cache, then reduce partial results. Expected: -0.3 to -0.5ms at avg kvLen=64.
- [ ] **Reduce dispatch count** -- Merge norm+LM head, or use ICBs. Expected: -0.1 to -0.2ms.
- [ ] **GEMV bandwidth improvement** -- Target 230+ GB/s (currently 207). Investigate memory prefetch, cache-friendly access patterns. Expected: -0.2 to -0.4ms.

## Autoresearch Infrastructure
- `autoresearch/run_loop.sh` -- Automated build + correctness + benchmark script
- `benchmarks/experiment_log.md` -- Full experiment history (Exp 0-25)
- `benchmarks/framework_comparison.json` -- MLX vs llama.cpp vs EdgeRunner data

## Experiment Log
See benchmarks/experiment_log.md
