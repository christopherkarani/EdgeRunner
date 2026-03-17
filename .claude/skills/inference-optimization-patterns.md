---
name: inference-optimization-patterns
description: Metal LLM inference optimization patterns extracted from llama.cpp and MLX. Use when optimizing LlamaLanguageModel.swift for maximum tokens/sec on Apple Silicon.
---

# LLM Inference Optimization Patterns

Reference patterns from llama.cpp (~100 tok/s) and MLX (~80 tok/s) for reaching
competitive performance on Apple Silicon. EdgeRunner baseline: 0.058 tok/s.

## Pattern 1: Single-Token Decode Path (Critical — 100x improvement)

**The Problem:** EdgeRunner recomputes the FULL sequence every decode step.
For a 10-token context, that's 10× the work of processing just the new token.

**llama.cpp pattern:**
```
// llama.cpp distinguishes prefill (n_tokens > 1) from decode (n_tokens == 1)
if (n_queued_tokens == 1) {
    // DECODE PATH: only process the new token
    // Use KV cache for all previous K/V — no recomputation
    t_eval_us += elapsed;
} else {
    // PREFILL PATH: process all tokens in batch
    t_p_eval_us += elapsed;
}
```

**How to implement in EdgeRunner:**
```swift
// In LlamaLanguageModel:
private var cachedPosition: Int = 0

func logits(for tokenIDs: [Int]) async throws -> [Float] {
    if tokenIDs.count == cachedPosition + 1 {
        // DECODE: only process the last token
        return try await decodeSingleToken(
            tokenID: tokenIDs.last!,
            position: cachedPosition
        )
    } else {
        // PREFILL: process all tokens, populate KV cache
        cachedPosition = 0
        kvCache.reset()
        return try await prefillTokens(tokenIDs)
    }
}

func decodeSingleToken(tokenID: Int, position: Int) async throws -> [Float] {
    var hidden = embeddingLookup(tokenIDs: [tokenID]) // 1 token, not seqLen

    for layer in 0..<config.layerCount {
        // RMSNorm on 1 token (rows=1)
        // Q/K/V projection: 1 GEMV each (not seqLen GEMVs)
        // RoPE on 1 token
        // Append new K/V to cache
        // Attention: new Q against ALL cached K/V
        // FFN: 1 token through gate/up/down
        hidden = try await decodeSingleTokenLayer(hidden, layer, position)
    }

    cachedPosition += 1
    // LM head on 1 token
}
```

**Expected speedup:** For context length N, decode goes from O(N) to O(1) per token.
At seqLen=5 (our benchmark), that's ~5x. At seqLen=100, it's ~100x.

## Pattern 2: Batched Projections via GEMM (10-50x improvement)

**The Problem:** EdgeRunner loops `for t in 0..<seqLen` and dispatches separate GEMVs.
Each GEMV launches its own Metal command buffer — massive dispatch overhead.

**llama.cpp pattern:**
```c
// llama.cpp uses a single ggml_mul_mat for all tokens at once:
// Q = W_q @ input   (input is [n_tokens, dim], not per-token)
// K = W_k @ input
// V = W_v @ input
// All done as GEMM, not per-token GEMV
struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
```

**How to implement in EdgeRunner:**
For prefill (n_tokens > 1), replace per-token GEMV loop with GEMM:
```swift
// BEFORE (current — dispatches seqLen × command buffers):
for t in 0..<seqLen {
    let tokenHidden = Array(normed[t * dim..<(t + 1) * dim])
    let q = try await gemvKernel.execute(a: wq, x: tokenHidden, M: qDim, K: dim, ...)
    ...
}

// AFTER (one command buffer for all tokens):
// For GEMM: C[M,N] = A[M,K] × B[K,N]
// Where A = weight [qDim, dim], B = input [dim, seqLen], C = output [qDim, seqLen]
// But our GEMVKernel doesn't support batched input.
// Solution: use the GEMMKernel directly, or reshape input as a matrix.
```

For decode (n_tokens == 1), GEMV is correct and optimal — one vector per projection.

## Pattern 3: Computation Graph + Single Command Buffer (2-5x improvement)

**The Problem:** Each kernel call (GEMV, RMSNorm, RoPE, etc.) creates its own
MTLCommandBuffer, commits it, and waits. That's ~200 command buffers per forward pass.

**llama.cpp pattern:**
```c
// llama.cpp builds a computation graph FIRST, then executes it all at once:
// 1. Build graph (no GPU work yet)
ggml_build_forward_expand(gf, cur);

// 2. Execute entire graph in one shot
ggml_backend_sched_graph_compute(sched, gf);
// This encodes ALL operations into a minimal number of command buffers
```

**MLX pattern:**
```python
# MLX uses lazy evaluation — operations are recorded, not executed
# Only when .eval() or array access happens does the graph run
q = x @ self.wq  # recorded
k = x @ self.wk  # recorded
# ... more ops ...
mx.eval(output)   # executes entire graph
```

**How to implement in EdgeRunner:**
The `ComputeGraph` + `CommandBatcher` already exist in EdgeRunner! The problem is
`LlamaLanguageModel` doesn't use them — it calls kernels one at a time.

Option A: Use `MetalBackend.shared` (actor) which batches commands internally.
Option B: Build all ops into a single command buffer manually:
```swift
let commandBuffer = commandQueue.makeCommandBuffer()!
// Encode ALL layer operations into this one buffer
for layer in 0..<layerCount {
    // RMSNorm → encoder.dispatch(...)
    // Q/K/V projections → encoder.dispatch(...)
    // RoPE → encoder.dispatch(...)
    // Attention → encoder.dispatch(...)
    // FFN → encoder.dispatch(...)
}
commandBuffer.commit()
await commandBuffer.completed()
// ONE round-trip to GPU instead of 200
```

## Pattern 4: Quantized Matmul Without Dequantization (2-3x improvement)

**The Problem:** EdgeRunner dequantizes Q8_0 weights to Float32 arrays, then does
GEMV with float32. This wastes memory bandwidth (read 34 bytes, expand to 128 bytes).

**llama.cpp pattern:**
```c
// llama.cpp's Metal kernels operate DIRECTLY on quantized data:
// kernel_mul_mv_q8_0_f32 reads Q8_0 blocks and multiplies in-place
// No intermediate Float32 buffer needed
// Memory bandwidth: read 34 bytes → produce 1 float (scale × dot product)
```

**MLX pattern:**
```metal
// MLX fused quantized matmul kernel:
// Reads quantized weights, dequantizes in registers, accumulates in threadgroup
template <typename T, int group_size, int bits>
kernel void qmv(
    device const uint32_t* w,  // quantized weights (NOT dequantized)
    device const T* x,          // input vector
    device T* y,                // output
    ...
) {
    // Dequantize in registers, accumulate partial sums in simdgroup
    // Uses simd_sum for fast reduction
    // Never materializes full float32 weight matrix
}
```

**EdgeRunner already has this!** `DequantQ4_0Kernel.fusedDequantGEMV()`:
```swift
dequantQ4_0.fusedDequantGEMV(
    quantisedRows: rawBytes,  // Q4_0 data, NOT dequantized
    x: hidden,                // input vector
    rows: M, cols: K,
    commandQueue: queue
) -> [Float]   // output vector
```

Use this instead of `dequant → cache → GEMV` for every weight projection.
For Q8_0, you'd need to add a similar `fusedDequantGEMV` kernel.

## Pattern 5: Metal Simdgroup Optimizations (1.5-2x improvement)

**llama.cpp Metal kernel patterns:**
```metal
// simdgroup matrix multiply (AMX on Apple Silicon)
simdgroup_matrix<float, 8, 8> sgA, sgB, sgC;
simdgroup_load(sgA, ...);
simdgroup_load(sgB, ...);
simdgroup_multiply_accumulate(sgC, sgA, sgB, sgC);

// simd_sum for fast reductions (softmax, RMSNorm)
float sum = simd_sum(thread_value);

// threadgroup memory for shared data
threadgroup float shared[BLOCK_SIZE];
```

**MLX patterns:**
```metal
// MLX uses simd shuffle for warp-level reductions
T val = simd_shuffle_down(val, offset);

// Tiled loading with async copies
threadgroup_barrier(mem_flags::mem_threadgroup);

// Block quantized processing — 32 elements at a time
// matches Q4_0/Q8_0 block size exactly
for (int k = 0; k < K; k += 32) {
    // Process one quantized block per iteration
}
```

## Pattern 6: KV Cache Memory Layout (Important for attention)

**llama.cpp KV cache:**
```c
// Contiguous K/V buffers per layer
// K shape: [n_kv, n_embd_k_gqa]  (all KV heads concatenated)
// V shape: [n_kv, n_embd_v_gqa]  (transposed for efficient attention)
// On decode: only append 1 row to K and V
// Attention: Q[1, head_dim] × K[n_kv, head_dim]^T → scores[1, n_kv]
```

**EdgeRunner already has KVCache** with ring buffer. The gap is USING it:
```swift
// In decodeSingleToken:
// 1. Project new K, V for this token
let newK = try await gemvKernel.execute(a: wk, x: hidden, M: kvDim, K: dim, ...)
let newV = try await gemvKernel.execute(a: wv, x: hidden, M: kvDim, K: dim, ...)

// 2. Append to cache
try kvCache.append(layer: layerIndex, keys: newK, values: newV)

// 3. Retrieve all cached K, V
let (allK, allV) = try kvCache.retrieve(layer: layerIndex, asType: Float.self)

// 4. Attention: new Q against all cached K/V
// Q is [1, numHeads * headDim], K is [cacheLen, numKVHeads * headDim]
```

## Pattern 7: Tied Embedding LM Head on GPU (2-5x for large vocab)

**The Problem:** EdgeRunner's `computeTiedLMHead()` runs on CPU — 151K dot products.

**llama.cpp pattern:**
```c
// LM head is just another ggml_mul_mat — runs on GPU like everything else:
cur = build_lora_mm(model.output, cur);  // [vocab_size, dim] × [dim, 1] → [vocab_size, 1]
```

**Fix for EdgeRunner:**
```swift
// Instead of CPU dot product loop, use GEMV:
// But embedding is Q8_0 — need fusedDequantGEMV for Q8_0
// OR dequantize embedding once at load time (593MB but fast thereafter)

// Best approach: add fusedDequantGEMV for Q8_0 (like the Q4_0 version)
let logits = try await dequantQ8_0.fusedDequantGEMV(
    quantisedRows: embeddingRawBytes,
    x: lastTokenHidden,
    rows: vocabSize,  // 151936
    cols: dim,        // 1024
    commandQueue: commandQueue
)
```

## Optimization Priority Order

| # | Optimization | Expected Speedup | Complexity |
|---|-------------|-----------------|------------|
| 1 | Single-token decode path + KV cache | 5-100x | Medium |
| 2 | Batched projections (GEMM for prefill) | 10-50x | Medium |
| 3 | Single command buffer per forward | 2-5x | Low |
| 4 | Fused dequant+GEMV for Q8_0 | 2-3x | Medium |
| 5 | GPU LM head (fused Q8_0 GEMV) | 2-5x | Low-Medium |
| 6 | Simdgroup optimizations in kernels | 1.5-2x | High |

Combined theoretical maximum: ~0.058 × 5 × 10 × 3 × 2 × 3 = **52 tok/s**
(Conservative estimate accounting for Amdahl's law: **15-30 tok/s** realistic target)

## Quick Reference: EdgeRunner Metal Kernel APIs

```swift
// GEMV: y[M] = A[M,K] × x[K]
gemvKernel.execute(a:x:M:K:commandQueue:) -> [Float]

// Fused Q4_0 dequant+GEMV (no float32 intermediate)
dequantQ4_0.fusedDequantGEMV(quantisedRows:x:rows:cols:commandQueue:) -> [Float]

// GQA attention
gqaKernel.execute(q:k:v:seqLen:headDim:numHeads:numKVHeads:causal:commandQueue:) -> [Float]

// KV Cache
kvCache.append(layer:keys:values:)
kvCache.retrieve(layer:asType:) -> ([T], [T])
kvCache.reset()

// MetalBackend (actor, batches commands)
MetalBackend.shared.synchronize()
```
