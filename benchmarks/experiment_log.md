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

### Experiment 1: Weight Buffer Caching + GPU LM Head
- **Hypothesis:** Recreating MTLBuffers from [Float] for every GEMV call wastes ~1.68GB of allocation+copy per forward pass. Moving the 151K-row tied-embedding LM head from CPU to GPU should also be a large win.
- **Change:** Added MetalBufferCacheActor to cache weight MTLBuffers; added GEMVKernel.executeWithWeightBuffer() that takes pre-allocated MTLBuffer; replaced CPU computeTiedLMHead with single GPU GEMV dispatch.
- **Files modified:** GEMVKernel.swift, LlamaLanguageModel.swift
- **Result:** 0.0578 → 3.5743 tok/s (+6084.6%, 61.8x)
- **Status:** KEPT
- **Commit:** 8ea7935

### Experiment 2a: KV Cache + Single-Token Decode (ROLLED BACK)
- **Hypothesis:** Cache K/V per layer and only process 1 new token per decode step to avoid recomputing the full sequence.
- **Change:** Added DecoderStateActor, decodeTransformerLayer with KV cache append/retrieve, padded Q for GQA compatibility.
- **Files modified:** LlamaLanguageModel.swift
- **Result:** 3.57 → 5.56 tok/s (+55%) BUT correctness FAILED: [1,1479,21456,56682,4102] ≠ expected
- **Root cause:** GQA kernel expects [H,S,D] layout but code builds [S,H,D]. Zero-padded Q is read from wrong memory region. The layout mismatch is baked into the expected tokens (scrambled attention produces deterministic but "incorrect" results that the benchmark treats as ground truth).
- **Status:** ROLLED BACK

### Experiment 2b: Fix GQA Layout Mismatch (ROLLED BACK)
- **Hypothesis:** Adding transposes before/after GQA to fix [S,H,D]→[H,S,D] layout would enable KV cache.
- **Change:** Added transposeSHDtoHSD/transposeHSDtoSHD helpers, applied before/after GQA.
- **Result:** Tokens changed to [1,1479,1479,1479,1479] — degenerate repetition. The model's expected behavior depends on the scrambled GQA layout.
- **Status:** ROLLED BACK

### Experiment 2c: Batch Command Buffers for Independent GEMVs
- **Hypothesis:** Q/K/V projections (3 GEMVs) and gate/up projections (2 GEMVs) are independent and can share a single MTLCommandBuffer, reducing GPU synchronization overhead by ~84 sync points per forward pass.
- **Change:** Added GEMVKernel.executeBatchedWithWeightBuffers() that encodes N GEMV dispatches into 1 command buffer. Applied to Q/K/V and gate/up batches in transformerLayer.
- **Files modified:** GEMVKernel.swift, LlamaLanguageModel.swift
- **Result:** 3.5743 → 5.0873 tok/s (+42.3%)
- **Status:** KEPT
- **Commit:** dbc99a8

### Experiment 3: Fused Command Buffers + RMSNorm Buffer Caching + Batched RoPE
- **Hypothesis:** Encoding dependent operations (RMSNorm → GEMV, RoPE Q+K) into shared command buffers reduces GPU sync overhead. Caching RMSNorm weights as MTLBuffers eliminates 56 small buffer allocations per pass.
- **Change:** Added `encode()` methods to RMSNormKernel and GEMVKernel for command buffer reuse. Fused attention RMSNorm + Q/K/V into single command buffer (4 dispatches, 1 sync). Fused FFN RMSNorm + gate/up (3 dispatches, 1 sync). Batched RoPE Q+K into single command buffer. Cached RMSNorm weights as MTLBuffers.
- **Files modified:** GEMVKernel.swift, RMSNormKernel.swift, RoPEKernel.swift, LlamaLanguageModel.swift
- **Result:** 5.09 → 5.52 tok/s (+8.4%)
- **Status:** KEPT
- **Commit:** 5fb15e3

### Experiment 4: Fully-Fused GPU Pipeline (Single Command Buffer)
- **Hypothesis:** Encoding ALL 28 transformer layers into a single MTLCommandBuffer eliminates ~224 GPU sync points (reduced to 1). Keeping data as MTLBuffers between operations eliminates [Float]↔MTLBuffer round-trips.
- **Change:** Added encode() to GQAKernel, RoPEKernel, ActivationKernels. Added GPU elementwise_add for residual connections. Rewrote forward pass as `fusedForwardPass` — one command buffer for the entire transformer stack (420+ dispatches, 1 sync).
- **Files modified:** GEMVKernel.swift, GQAKernel.swift, RoPEKernel.swift, ActivationKernels.swift, RMSNormKernel.swift, LlamaLanguageModel.swift
- **Result:** 5.52 → 16.08 tok/s (+191.3%)
- **Status:** KEPT
- **Commit:** d3160c8

### Experiment 5: Single Encoder + Pre-loaded Weights + Pre-allocated Buffers
- **Hypothesis:** Reducing encoder creation overhead (422 → 1 encoder), eliminating async actor hops (252 → 0 per pass), and pre-allocating scratch buffers (17 allocs → 0 per pass) should reduce CPU overhead.
- **Change:** Single MTLComputeCommandEncoder for entire pass. Pre-loaded LayerWeightBuffers array. Pre-allocated ScratchBuffers reused across all calls. Fast embedding memcpy.
- **Files modified:** All kernel files (expose pipelines) + LlamaLanguageModel.swift
- **Result:** 16.08 → 17.0 tok/s (+5.7%) — modest, overhead was smaller than expected
- **Status:** KEPT
- **Commit:** bfe96c4

### Experiment 6: Fused Q8_0 Dequant+GEMV Kernel
- **Hypothesis:** Reading quantized Q8_0 weights directly in the GEMV kernel (1.06 bytes/element) instead of pre-dequantized float32 (4 bytes/element) gives ~3.8x memory bandwidth reduction.
- **Change:** Created `dequant_q8_0_gemv` Metal kernel using simd_sum for warp-level reduction. Applied to all 7 GEMV projections per layer + LM head. Cached raw Q8_0 buffers via bytesNoCopy.
- **Files modified:** Dequant_Q8_0.metal, LlamaLanguageModel.swift
- **Result:** 17.0 → 24.0 tok/s (+41%) debug, 52.7 → 62.8 tok/s release
- **Status:** KEPT
- **Commit:** c8d7cab

### Experiment 7: Concatenated Q/K/V Single-Dispatch (ROLLED BACK)
- **Hypothesis:** Merging Q+K+V weight matrices into one buffer and using a single GEMV dispatch (M=4096) instead of 3 separate dispatches would reduce dispatch overhead.
- **Change:** Concatenated raw Q8_0 weight buffers, single-dispatch QKV for seqLen=1.
- **Result:** Correctness failed (output layout mismatch for seqLen>1). After fixing, performance was WORSE (22.1 vs 24.1) due to less efficient large dispatch.
- **Status:** ROLLED BACK

---

## Performance Summary

| Experiment | tok/s (debug) | tok/s (release) | vs Baseline |
|---|---|---|---|
| 0. Baseline | 0.058 | — | — |
| 1. Weight buffer caching + GPU LM head | 3.57 | — | 61.6x |
| 2. Batched command buffers | 5.09 | — | 87.8x |
| 3. Fused RMSNorm+QKV cmd buffers | 5.52 | — | 95.2x |
| 4. Fully fused GPU pipeline (1 cmd buf) | 16.1 | 28.6 | 277x |
| 5. Single encoder + pre-loaded weights | 17.0 | — | 293x |
| 6. Fused Q8_0 dequant+GEMV + pre-alloc | 24.0 | 62.8 | 414x |

**Final: 0.058 → 24.7 tok/s (426x improvement)**

### Experiment 8: Q/K Per-Head Norm + NeoX RoPE + [S,H,D] GQA (CORRECTNESS FIX)
- **Root cause found**: Qwen3 has per-head RMSNorm on Q/K (attn_q_norm/attn_k_norm) that was completely missing. Without it, self-attention overwhelms cross-attention, causing degenerate output with correct layout. The scrambled [H,S,D] layout accidentally worked by mixing heads.
- **Change**: Load Q/K norm weights from GGUF, apply per-head RMSNorm after Q/K projection. Fix GQA to [S,H,D]. Add NeoX split-halves RoPE kernel.
- **Result**: Non-degenerate output [1, 1479, 35, 5371, 1] — correct attention pipeline
- **Status**: KEPT
- **Commit**: daca271

### Experiment 9: KV Cache for Single-Token Decode
- **Change**: Extended GQA kernel with kvSeqLen/qOffset for asymmetric Q/KV. Direct GPU buffer writes to per-layer cache. DecoderStateStore for decode detection.
- **Result**: KV cache working — decode processes 1 token instead of full sequence
- **Status**: KEPT
- **Commit**: 7e54202

## Current Performance

| Mode | tok/s | ms/token |
|------|-------|----------|
| Debug | 24.2 | 41 |
| Release | 63 | 16 |
| llama.cpp (reference) | 180 | 5.6 |

## Remaining Path to 300 tok/s

1. **GEMV kernel efficiency**: Current Q8_0 GEMV uses 32 threads per row with scalar ops. llama.cpp uses simdgroup_matrix_multiply for ~3x throughput.
2. **Float16 intermediates**: Halves bandwidth for KV cache and scratch buffers.
3. **Reduce dispatch count**: Kernel fusion (RMSNorm+GEMV, SwiGLU+down).
4. **Flash attention**: Fused QKV attention instead of separate dispatches.

### Experiment 10: Kernel Fusion — Eliminate Tiny Dispatch Overhead
- **Root cause found via profiling**: GPU dispatch encoding is only 1.2μs (not 25μs). The 15.5ms GPU time was from Q8_0 GEMV achieving only 40 GB/s on small matrices (10% bandwidth utilization). 196 tiny non-GEMV dispatches (RMSNorm, Q/K norm, RoPE, conversions) each cost ~30μs of GPU launch latency.
- **Phase 1**: Fused RMSNorm into QKV GEMV and Gate+Up+SwiGLU kernels. Cooperative RMSNorm: all threads compute sum of squares, simd_sum reduces, then norm applied inline during x-value loading. Saves 2 dispatches × 28 layers = 56.
- **Result**: 64 → 107 tok/s (+67%)
- **Commit**: 7cd8cb4

- **Phase 2**: Fused Q/K per-head norm + RoPE Q + RoPE K→f16 into single kernel. Thread grid (halfDim, numHeads+numKVHeads). Cross-simdgroup norm reduction via threadgroup memory. Saves 3 dispatches × 28 layers = 84.
- **Result**: 107 → 120 tok/s (+12%)
- **Commit**: 76ed83a

### Experiment 11: 32-Thread Single-Simdgroup GEMV Architecture  
- **Change**: Rewrote all 5 Q8_0 GEMV kernels to use 32 threads (1 simdgroup) per threadgroup, each thread processing 1 full block (32 elements). Eliminated cross-simdgroup reduction overhead. Each kernel has independent LOCAL_NR=2.
- **Result**: No measurable change (GEMV is bandwidth-bound, not compute-bound for our matrix sizes)
- **Commit**: 257d48f

## Current Performance

| Mode | tok/s | ms/token | Dispatches/token |
|------|-------|----------|-----------------|
| Release | 120 | 8.3 | 170 |
| llama.cpp ref | 183 | 5.5 | — |

## Current Per-Layer Decode Dispatches (6 total)

1. Fused RMSNorm + QKV GEMV (LARGE — reads wq+wk+wv Q8_0)
2. Fused Q/K norm + RoPE Q + RoPE K→f16 (medium — per-head norm + rotation)
3. GQA f16kv (tiny for small kvLen)
4. Fused Wo GEMV + residual add (LARGE — reads wo Q8_0)
5. Fused RMSNorm + Gate+Up+SwiGLU GEMV (LARGE — reads gate+up Q8_0)
6. Fused Down GEMV + residual add (LARGE — reads down Q8_0)

## Remaining Path to 300 tok/s

At 120 tok/s = 8.3ms/token with 170 dispatches:
- GEMV weight bandwidth: ~5ms (627MB at ~125 GB/s effective)
- GQA + dispatch overhead: ~2ms (170 × ~6μs + GQA compute)  
- Other: ~1.3ms

Reaching 300 tok/s = 3.3ms/token requires:
1. Higher GEMV bandwidth utilization (125 → 200+ GB/s)
2. Further dispatch count reduction
3. Flash attention for GQA

### Experiment 12: Fuse Prefill Path to Match Decode
- **Root cause**: Profiling showed prefill step (seqLen=1) takes 14.7ms vs decode at 5.2ms. The prefill path still used separate RMSNorm, Q/K norm, RoPE dispatches.
- **Change**: Applied all 5 kernel fusions from decode path to prefill when seqLen==1: fused RMSNorm+QKV, fused Q/K norm+RoPE, fused RMSNorm+Gate+Up+SwiGLU, fused Wo+add, fused Down+add.
- **GPU time breakdown (before)**: prefill 14.7ms + 3× decode 5.2-6.1ms = 32ms/4 tokens
- **GPU time breakdown (after)**: all paths ~5ms = 20ms/4 tokens  
- **Result**: 120 → 148 tok/s median (165 peak)
- **Commit**: (below)

## Updated Performance

| Mode | tok/s | ms/token |
|------|-------|----------|
| Release (median) | 148 | 6.8 |
| Release (peak) | 166 | 6.0 |
| Decode-only | ~192 | ~5.2 |
| llama.cpp ref | 183 | 5.5 |
