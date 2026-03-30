# EdgeRunner — Complete Technical Reference

**Author:** EdgeRunner Research Swarm
**Date:** 2026-03-25
**Scope:** Complete architectural documentation of the EdgeRunner Metal-accelerated LLM inference engine.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Package Architecture](#2-package-architecture)
3. [Core Data Types](#3-core-data-types)
4. [Quantization System](#4-quantization-system)
5. [Metal Shaders & Compute Kernels](#5-metal-shaders--compute-kernels)
6. [Transformer Architecture](#6-transformer-architecture)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Memory & Compute Management](#8-memory--compute-management)
9. [KV Cache](#9-kv-cache)
10. [Sampling & Generation](#10-sampling--generation)
11. [Public API Reference](#11-public-api-reference)
12. [Unique Design Innovations](#12-unique-design-innovations)

---

## 1. Executive Summary

EdgeRunner is a Swift-native Metal-accelerated inference engine for LLM models (Llama, Qwen, Mistral, Gemma, Phi3 families) running on Apple Silicon. It implements autoregressive generation with on-the-fly GPU weight dequantization, highly fused compute kernels, KV caching, and prefix reuse.

**Core architectural pipeline:**
```
tokens → embedding → [RMSNorm → RoPE + GQA + KVCache → SwiGLU FFN] × N → RMSNorm → LM head → logits
```

**Key capabilities:**
- 10 quantization formats (Q4_0, Q5_0, Q5_1, Q8_0 + K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- 3 inference modes auto-detected per call: Full Prefill, Decode, Prefix Reuse
- Mega-kernels reducing 5 dispatches per layer to 1
- Zero-allocation inference via pre-allocated scratch buffers
- Metal 4 (macOS 26+) argument table dispatch and GPU residency management
- Streaming generation via `AsyncThrowingStream`
- Tool calling support via `EdgeRunnerTool` protocol

---

## 2. Package Architecture

```
EdgeRunner (main product, @_exported re-exports all modules)
├── EdgeRunnerCore       # Tensor ops, sampling pipeline, tokenizers, chat templates
├── EdgeRunnerIO         # GGUF parsing, model loading, quantization, weight maps
├── EdgeRunnerMetal      # Metal compute backend, all GPU kernels
├── EdgeRunnerSharedTypes # C interop headers (params structs)
└── ANEInteropIO         # Apple Neural Engine integration
```

**Platforms:** iOS 26+, macOS 26+

Each submodule is a separate Swift package target. The main `EdgeRunner` module uses `@_exported import` to re-export all public types, so users typically only need:

```swift
import EdgeRunner
```

---

## 3. Core Data Types

### 3.1 `LlamaConfig` — GGUF Metadata Container

Parsed from GGUF file metadata (keys like `llama.embedding_length`, `llama.block_count`, etc.):

```swift
public struct LlamaConfig: Sendable, Equatable {
    public let embeddingDim: Int       // e.g., 4096
    public let layerCount: Int         // number of transformer layers
    public let headCount: Int          // query heads (Q)
    public let kvHeadCount: Int        // KV heads (typically fewer for GQA)
    public let vocabSize: Int
    public let intermediateDim: Int     // FFN intermediate (~4× embeddingDim)
    public let ropeFreqBase: Double    // RoPE base (e.g., 10000 or 500000)
    public let rmsNormEpsilon: Double
    public let explicitHeadDim: Int?   // Qwen 3: key_length != embedding_dim/head_count

    public var headDim: Int { explicitHeadDim ?? (embeddingDim / headCount) }
    public var gqaRatio: Int { headCount / kvHeadCount }
}
```

### 3.2 `TensorDataType` — All Supported Quantization Types

```swift
public enum TensorDataType: UInt32, Sendable, Equatable {
    case float32 = 0
    case float16 = 1
    case q4_0 = 2        // 4-bit, 18 bytes/block (32 weights)
    case q4_1 = 3        // 4-bit with bias, 20 bytes/block
    case q5_0 = 6        // 5-bit, 22 bytes/block
    case q5_1 = 7        // 5-bit with bias, 24 bytes/block
    case q8_0 = 8        // 8-bit, 34 bytes/block
    case q8_1 = 9        // NOT SUPPORTED
    case q2_K = 10       // 2-bit K-quant, 84 bytes/256 weights
    case q3_K = 11       // 3-bit K-quant, 110 bytes/256 weights
    case q4_K = 12       // 4-bit K-quant, 144 bytes/256 weights
    case q5_K = 13       // 5-bit K-quant, 176 bytes/256 weights
    case q6_K = 14       // 6-bit K-quant, 210 bytes/256 weights
    case q8_K = 15       // incomplete support
    case i8 = 16, i16 = 17, i32 = 18, i64 = 19, f64 = 20, bfloat16 = 30
}
```

### 3.3 `TensorStorage` — GPU Weight Container

```swift
public struct TensorStorage: @unchecked Sendable {
    public let buffer: MTLBuffer          // raw GPU buffer
    public let byteOffset: Int
    public let dataType: TensorDataType
    public let shape: [Int]
    public let name: String
}
```

### 3.4 `LlamaModel` — Loadable Model Container

```swift
public struct LlamaModel: LoadableModel, Sendable {
    public let config: LlamaConfig
    public let layers: [LlamaBlock]
    public private(set) var loadedWeights: [String: TensorStorage] = [:]
}
```

### 3.5 Key Internal Structs

| Struct | File | Purpose |
|--------|------|---------|
| `ScratchBuffers` | `LlamaLanguageModel.swift` | 19 pre-allocated GPU buffers for zero-allocation inference |
| `PreloadedWeightsStore` | `LlamaLanguageModel.swift` | Caches all layer weights + final norm + LM head on first prefill |
| `DecoderStateStore` | `LlamaLanguageModel.swift` | Tracks previous token IDs for KV cache hit detection |
| `Metal4State` | `LlamaLanguageModel.swift` | Metal 4 (macOS 26+) GPU state: arg table, residency set, command allocator |
| `LayerWeightBuffers` | `LlamaLanguageModel.swift` | Per-layer: wq/wk/wv/wo/gate/up/down + raw Q8_0 buffers |
| `KVCache` | `KVCache.swift` | Per-layer circular K/V buffers with position tracking |

---

## 4. Quantization System

### 4.1 Format Overview

EdgeRunner supports **10 quantization formats** in two families:

**Plain formats (32-element blocks):**

| Format | Block Size | Bits/Wt | Scale | Zero-Point |
|--------|-----------|---------|-------|------------|
| Q8_0 | 34 bytes | 8 | f16 | none |
| Q4_0 | 18 bytes | 4 | f16 | -8 offset |
| Q5_0 | 24 bytes | 5 | f16 | none |
| Q5_1 | 24 bytes | 5 | f16 (d,m) | m offset |

**K-quants (256-element superblocks, mixed block structure):**

| Format | SuperBlock Bytes | Bits/Wt | Notes |
|--------|----------------|---------|-------|
| Q2_K | 84 | ~2.06 | 16 sub-block scales |
| Q3_K | 110 | ~3.06 | |
| Q4_K_M | 144 | ~4.52 | "medium", GGUF default |
| Q5_K | 176 | ~5.53 | |
| Q6_K | 210 | ~6.56 | Near Q8 quality |

### 4.2 K-Quant Superblock Layout

All K-quants use a two-level structure:
- **Superblock** (256 elements): contains scales and metadata
- **Blocks** within superblock: contain quantized values

**Q4_K_M (144 bytes / 256 weights):**
```
[0..1]    d (f16 master scale)
[2..3]    dmin (f16)
[4..11]   8 scale bytes (6 bits each, high bits from [12..15])
[12..15]  high bits for scales
[16..143] 128 bytes nibble-packed weights (256 nibbles)
-- 8 sub-blocks × 32 weights each --
```

**Q6_K (210 bytes / 256 weights):**
```
[0..127]  ql nibble-packed (lower 4 bits)
[128..191] qh (upper 2 bits, 4 weights per byte)
[192..207] scales (signed int8, 16 sub-blocks × 16 weights)
[208..209] d (f16)
```

**Q2_K (84 bytes / 256 weights):**
```
[0..15]   scale/metadata (sc | m<<4 per sub-block)
[16..79]  qs (2 bits per weight, 4 per byte)
[80..81]  d (f16)
[82..83]  dmin (f16)
```

### 4.3 Three-Tier Dequantization Strategy

**Tier 1 — Raw Q8_0 Zero-Copy (fastest):**
The raw Q8_0 buffer is passed directly to fused GPU kernels. No float32 materialization.
```swift
bytesNoCopy: storage.buffer.contents() + storage.byteOffset
```

**Tier 2 — GPU Dequantization Kernel:**
Dedicated Metal kernels (`DequantQ4_0Kernel`, `DequantQ6KKernel`, etc.) dequantize on-GPU to float32/float16 during GEMV.

**Tier 3 — CPU Fallback:**
For embedding lookups and tied head computations, CPU SIMD loops decode directly into destination buffers.

### 4.4 Fused Dequant + GEMV Kernels

Rather than separate dequantization then GEMV, fused kernels dequantize directly during the matrix multiply:

- `dequant_q8_0_gemv` — Q8_0 weight × input vector
- `dequant_q8_0_fused_qkv` — RMSNorm + Q + K + V projections (3→1 dispatch)
- `dequant_q8_0_fused_gate_up_silu` — RMSNorm + Gate + Up + SwiGLU (4→1)
- `dequant_q8_0_fused_ffn_block` — Wo + RMSNorm + Gate/Up/SwiGLU + Down (5→1 dispatch)
- `dequant_q8_0_gemv_tiled` — Tile-based to prevent DRAM row-buffer thrashing

---

## 5. Metal Shaders & Compute Kernels

### 5.1 Threadgroup Layouts Reference

| Kernel | Threads/TG | SIMDGroups/TG | Rows/TG | Purpose |
|--------|-----------|--------------|---------|---------|
| `dequant_q8_0` | 1 | — | — | Plain Q8_0 dequant |
| `dequant_q8_0_gemv` | 32 | 1 | 2 | Q8_0 GEMV |
| `dequant_q8_0_gemv_tiled` | 32 | 1 | 2 | Tiled Q8_0 GEMV |
| `gemv_f32/f16` | 256 | 8 | 1 | Full-precision GEMV |
| `rmsnorm_parallel_f32` | 256 | 8 | 1 | Decode RMSNorm |
| `softmax_f32` | 256 | 8 | 1 | Full softmax |
| `fused_qk_norm_rope_gqa` | 32 | 1 | — | Mega-kernel (no barriers) |
| `dequant_q8_0_fused_ffn_block` | 1024 | 32 | — | Full layer mega-kernel |

### 5.2 Q8_0 GEMV Kernel (Base)

**Threadgroup:** 32 threads, 2 rows per TG, strided block iteration.

```metal
struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};
// buffer(0): uchar* quantizedW (Q8_0)
// buffer(1): float* x (input)
// buffer(2): float* y (output)
```

**Execution per threadgroup (row0, row0+1):**
```
Phase 1: Register-cache x[32] via strided loads
Phase 2: For each block ib (strided 0,32,64...):
    scale = f16(block[0..1])
    qs = block[2..33]       // 32 int8 values
    sumq = simd_sum(qs[i] * xl[i])  // dot product
    sumf[row] += sumq * scale
Phase 3: simd_sum reduction (32→1)
Phase 4: Lane 0 writes final output
```

**Key optimizations:**
- `xl[32]` lives in registers — not reloaded from memory per block
- `LOCAL_NR = 2` amortizes x loading across two GEMV rows
- SIMD reduction = no threadgroup_barrier

### 5.3 Mega-Kernel: `dequant_q8_0_fused_ffn_block`

The crown jewel: **one dispatch** replaces 5 GPU dispatches per layer.

**Threadgroup:** 1024 threads (32 simdgroups × 32 threads).

```
Phase 1: Wo GEMV + residual add (threads 0..1023, each = 1 output row)
Phase 2: Cross-SIMDgroup barrier + Cooperative RMSNorm over 1024 elements
Phase 3: Gate + Up + SwiGLU GEMVs (threads each compute 3 rows)
Phase 4: Down GEMV + residual add
```

**Cross-SIMDgroup RMSNorm detail:**
- 32 simdgroups × 32 threads = 1024 threads
- Each SG computes `simd_sum` → 32 partial sums in `threadgroup float partial_sums[32]`
- SG0 reduces 32 partials via `simd_sum` → broadcast via threadgroup memory

### 5.4 Mega-Kernel: `fused_qk_norm_rope_gqa`

**No threadgroup barriers at all** — pure SIMD reductions. Replaces fused Q/K Norm + RoPE + GQA.

**Threadgrid:** `(32, numHeads+numKVHeads)` — 32 threads per head (one full SIMDgroup).

**Phase 1: Per-head RMSNorm + RoPE (4 elements per thread):**
Each thread processes positions `[dimIdx, dimIdx+32, dimIdx+64, dimIdx+96]` of a 128-dim head:
```metal
// 32 threads × 4 elements = 128-dim complete coverage, NO barrier
sq = raw_a0^2 + raw_a1^2 + raw_b0^2 + raw_b1^2;
sumSq = simd_sum(sq);  // pure SIMD reduction

rs = rsqrt(sumSq / headDim + eps);
x_a0 = raw_a0 * rs * nw[dimIdx];  // norm applied to all 4 positions
// RoPE applied...
```

**Phase 2: GQA attention (Q threads only):**
```metal
// 32 threads cooperatively compute full 128-dim dot per KV position
// simd_sum gives full dot with ZERO barriers
for kv in 0..kvSeqLen:
    partial = q_a0*dk_a0 + q_a1*dk_a1 + q_b0*dk_b0 + q_b1*dk_b1
    score = simd_sum(partial) * attnScale  // NO barrier!

    // Online softmax: lane 0 computes correction, broadcast via simd_broadcast_first
    runMax = simd_broadcast_first(nextRunMax);
```

### 5.5 Fusion Architecture Summary

| Dispatches Replaced | Kernel | What It Fuses |
|--------------------|--------|--------------|
| 3 | `dequant_q8_0_fused_qkv` | RMSNorm + Q + K + V GEMVs |
| 4 | `dequant_q8_0_fused_gate_up_silu` | RMSNorm + Gate + Up + SwiGLU |
| 5 | `dequant_q8_0_fused_ffn_block` | Wo + add + RMSNorm + Gate/Up/SwiGLU + Down + add |
| 4 | `fused_qk_norm_rope_neox` | Q RMSNorm + K RMSNorm + RoPE Q + RoPE K→f16 |
| 2 | `fused_qk_norm_rope_gqa` | fused_qk_norm_rope_neox + GQA attention |

### 5.6 Hardware Optimizations

**`powr` vs `pow` in RoPE:**
```metal
float frequency = 1.0f / powr(params.theta, exponent);  // hardware reciprocal
```
`powr` uses hardware-optimized reciprocal units on Apple Silicon. Standard `pow` does not.

**SIMD intrinsics:**
| Intrinsic | Operation | Use |
|-----------|-----------|-----|
| `simd_sum(x)` | 32-lane sum | GEMV accumulation, RMSNorm |
| `simd_max(x)` | 32-lane max | Softmax max, attention scores |
| `simd_broadcast_first(x)` | Copy lane 0 → all | Online softmax broadcast |
| `as_type<T>(x)` | Bit reinterpret | f16↔f32 scale reinterpret |
| `rsqrt(x)` | Fast 1/sqrt | RMSNorm, normalization |

**Half-precision strategy:**
| Where | Precision | Rationale |
|-------|-----------|-----------|
| KV Cache | half | 50% memory, acceptable precision |
| Q8_0 scale | half | 2-byte scale, sufficient range |
| Inner GEMV dot (f16acc) | half | 2× ALU throughput (Apple 2:1 ratio) |
| Outer GEMV accumulator | float | No drift over many blocks |
| Attention scores | float | Numerical stability in exp() |
| Softmax probs | float | Numerical stability |

**Tiled GEMV for DRAM row-buffer prevention:**
Base GEMV's strided `x[]` access across 32 threads causes DRAM row buffer thrashing. The tiled variant cooperatively loads a contiguous 1024-element tile into threadgroup SRAM (4KB), then processes from fast memory.

---

## 6. Transformer Architecture

### 6.1 Per-Layer Computation

```
input hidden [seqLen, dim]
  → attention_norm [seqLen, dim] (RMSNorm)
  → Wq, Wk, Wv projections → Q [seqLen, qDim], K [seqLen, kvDim], V [seqLen, kvDim]
  → Q/K per-head RMSNorm (optional, Qwen3)
  → RoPE on Q and K
  → K/V write to KV cache
  → GQA (grouped query attention) → attn_out [seqLen, qDim]
  → residual: hidden + Wo @ attn_out → after_attn [seqLen, dim]
  → ffn_norm [seqLen, dim] (RMSNorm)
  → SwiGLU FFN: gate @ ffn_norm → gate_out; up @ ffn_norm → up_out; SiLU(gate_out) * up_out
  → down @ activated → down_out
  → residual: after_attn + down_out → layer_output [seqLen, dim]
```

### 6.2 GQA (Grouped Query Attention)

GQA reduces KV head count vs query head count. All query heads attend to all KV heads.

```swift
ERGQAParams(
    seqLen: UInt32(seqLen),
    headDim: UInt32(headDim),
    numHeads: UInt32(numHeads),
    numKVHeads: UInt32(numKVHeads),
    groupSize: UInt32(groupSize),  // numHeads / numKVHeads
    scale: 1.0 / sqrt(Float(headDim)),
    causal: 1,
    kvBlockSize: UInt32(gqaBlockSize),
    qBlockSize: UInt32(gqaBlockSize),
    kvSeqLen: UInt32(totalKVSeqLen),
    qOffset: UInt32(startPosition)
)
```

`qOffset` supports prefix reuse: queries at position `qOffset + q_idx` attend only to KV positions `0..q_offset+q_idx`.

### 6.3 RMSNorm

Root Mean Square Layer Normalization: `output = input * RMS(input)^{-1} * weight`

```c
typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} ERRMSNormParams;
```

Fused into projection kernels for decode (seqLen=1): RMSNorm weight applied inline without a separate dispatch.

### 6.4 RoPE (Rotary Position Embedding)

Encodes absolute position with rotation matrices applied to Q and K:

```c
typedef struct {
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t startPos;       // Position offset (for prefill continuation)
    float theta;             // Base frequency (e.g., 10000.0 or 500000.0)
    float scalingFactor;     // Scaling factor for extended context
} ERRoPEParams;
```

**Standard Llama RoPE:** `(2d, 2d+1)` pairs.
**NeoX RoPE (Qwen3):** `(d, d+halfDim)` pairs.

### 6.5 SwiGLU FFN

SwiGLU = SiLU(Gate) × Up (Gated Linear Unit with Swish activation):
```
output = down_proj * (silu(gate_proj * x) * up_proj * x)
where silu(x) = x * sigmoid(x)
```

Fused into a single kernel for seqLen=1 decode via `FusedGateUpSiluParams`.

---

## 7. Inference Pipeline

### 7.1 Three Auto-Detected Modes

The `forwardLogitsBuffer` method detects which GPU path to use:

**Decode Mode** (single new token, KV cache valid):
```
commonPrefixLen == previousTokenIDs.count && tokenIDs.count == commonPrefixLen + 1
```
Processes only the single new token using KV cache.

**Prefix Reuse Mode** (multiple new tokens extending cached prefix):
```
commonPrefixLen > 0 && commonPrefixLen == previousTokenIDs.count && tokenIDs.count > commonPrefixLen + 1
```
Prefills suffix tokens with correct RoPE positions; GQA attends over full KV cache.

**Full Prefill Mode** (no useful prefix match):
```
commonPrefixLen == 0 || commonPrefixLen < previousTokenIDs.count
```
Resets KV cache and recomputes entire sequence. Also runs 5 warmup dummy decodes to JIT warm GPU pipeline.

### 7.2 Full Prefill Path

```
Input tokens [seqLen]
  → fillEmbeddings → hiddenBuf [seqLen, dim]
  → kvCache.reset()
  → for each layer:
      → fusedQKVPipeline (Q8) or RMSNorm + GEMVs (fallback)
      → fusedNormRoPEGQAPipeline (mega) or separate RoPE + GQA
      → gemvAddPipeline (Q8) or separate Wo + residual add
      → fusedGateUpSiluPipeline (Q8) or separate RMSNorm + gate + up + SwiGLU
      → gemvAddPipeline (Q8) or separate down + residual add
  → fusedFinalNormGemvPipeline or separate RMSNorm + GEMV
  → logits [vocabSize]
  → run 5 dummy decodes for warmup, then re-run actual prefill
```

### 7.3 Decode Path

```
Single new token [1]
  → memcpy embedding into decodeHidden [dim]
  → runDecodePass → fusedDecodePass / fusedDecodePassOpt / fusedDecodePassMetal4
  → logits [vocabSize]
```

### 7.4 Three Decode Dispatch Variants

1. **Base Metal 3** (`fusedDecodePass`): Standard approach with per-layer dispatches
2. **Optimized Metal 3** (`fusedDecodePassOpt`): Pre-allocated params buffer with 7×256-byte slots, reducing `setBytes` overhead
3. **Metal 4** (`fusedDecodePassMetal4`): Uses argument table dispatch (`setArgumentTable` once), `setAddress` for per-dispatch buffer updates, execution-only barriers, residency set

### 7.5 LlamaLanguageModel Key Fields

```swift
public struct LlamaLanguageModel: LogitsModel, @unchecked Sendable {
    // Config & weights
    private let config: LlamaConfig
    private let weights: [String: TensorStorage]
    private let tokenizer: (any Tokenizer)?

    // Metal infrastructure
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Fused compute pipelines
    private let fusedQ8GemvPipeline: MTLComputePipelineState
    private let fusedQ8GemvTiledPipeline: MTLComputePipelineState
    private let fusedQKVPipeline: MTLComputePipelineState
    private let fusedGateUpSiluPipeline: MTLComputePipelineState
    private let fusedNormRoPEGQAPipeline: MTLComputePipelineState
    private let fusedFinalNormGemvPipeline: MTLComputePipelineState

    // One dequant kernel per quantization type
    private let dequantQ4_0: DequantQ4_0Kernel
    private let dequantQ8_0: DequantQ8_0Kernel
    private let dequantQ4KM: DequantQ4KMKernel
    private let dequantQ5K: DequantQ5KKernel
    private let dequantQ6K: DequantQ6KKernel
    private let dequantQ3K: DequantQ3KKernel
    private let dequantQ2K: DequantQ2KKernel
    private let dequantQ5_0: DequantQ5_0Kernel
    private let dequantQ5_1: DequantQ5_1Kernel

    // KV cache
    private let kvCache: KVCache
    private let layerKCaches: [MTLBuffer]
    private let layerVCaches: [MTLBuffer]

    // Preloaded weights
    private let preloadedWeights: PreloadedWeightsStore

    // Zero-allocation scratch buffers
    private let scratch: ScratchBuffers

    // Metal 4 state (macOS 26+)
    private let metal4State: Metal4State?

    // Optimized Metal 3 params buffer
    private let decodeParamsBuffer: MTLBuffer?
}
```

---

## 8. Memory & Compute Management

### 8.1 BufferCache (LRU Cache)

**File:** `Sources/EdgeRunnerMetal/BufferCache.swift`

Thread-safe (via `Mutex`) LRU cache for Metal buffer reuse.

- **Storage:** `storageModeShared | hazardTrackingModeUntracked` (CPU-GPU unified memory)
- **Size bucketing:** Returns buffer in `[size, size*2]`
- **Max cache:** 50% of `device.recommendedMaxWorkingSetSize` (min 64MB)
- **Eviction:** LRU within size buckets when `totalBytes + newLength > maxBytes`

### 8.2 ScratchBuffers (19 Persistent Pre-Allocated)

Pre-allocated once at model init, sized for `maxSeqLen`. These eliminate ~17 MTLBuffer allocations per forward pass:

| Buffer | Size |
|--------|------|
| `normed` | `maxSeqLen × dim × 4B` |
| `afterAttn` | `maxSeqLen × dim × 4B` |
| `ffnNormed` | `maxSeqLen × dim × 4B` |
| `outputA`, `outputB` | `maxSeqLen × dim × 4B` (ping-pong) |
| `allQ`, `allK`, `allV` | `maxSeqLen × qDim/kvDim × 4B` |
| `ropeQ`, `ropeK` | `maxSeqLen × qDim/kvDim × 4B` |
| `attnOut` | `maxSeqLen × qDim × 4B` |
| `proj` | `maxSeqLen × dim × 4B` |
| `gateOut`, `upOut` | `maxSeqLen × interDim × 4B` |
| `activ` | `maxSeqLen × interDim × 4B` |
| `downOut` | `maxSeqLen × dim × 4B` |
| `finalOut` | `maxSeqLen × dim × 4B` |
| `logits` | `vocabSize × 4B` |
| `decodeHidden` | `dim × 4B` |

Ping-pong buffers (`outputA`/`outputB`) alternate every layer for residual addition without extra copy.

### 8.3 CommandBatcher

**File:** `Sources/EdgeRunnerMetal/CommandBatcher.swift`

Manages a single `MTLCommandBuffer` + `MTLComputeCommandEncoder` pair, flushing when threshold reached:

```swift
final class CommandBatcher {
    private let commandQueue: MTLCommandQueue
    private var currentCommandBuffer: MTLCommandBuffer?
    private var currentEncoder: MTLComputeCommandEncoder?
    private var currentOpCount: Int = 0
    private let maxOpsPerBuffer: Int  // apple9=50, apple8=40, other=30
}
```

**Why one encoder?** Metal guarantees sequential execution + implicit barriers between dispatches within the same encoder. Creating 422 encoders (for a 40-layer model) costs ~4.2ms in encoder overhead.

### 8.4 BarrierTracker

**File:** `Sources/EdgeRunnerMetal/BarrierTracker.swift`

Tracks which buffers have been written since last `reset()`, inserts `MTLBarrier` before reads to prevent RAW hazards:

```swift
final class BarrierTracker {
    private var writtenBuffers: Set<ObjectIdentifier> = []
    func needsBarrier(forReading buffer: MTLBuffer) -> Bool
    func recordWrite(_ buffer: MTLBuffer)
    func insertBarrierIfNeeded(forReading buffer: MTLBuffer, encoder: MTLComputeCommandEncoder)
    func reset()  // Called after flushAndWait
}
```

Barriers use `memoryBarrier(scope: .buffers)`. Reset only after `flushAndWait`, not after every `flush`.

### 8.5 ResidencyManager

**File:** `Sources/EdgeRunnerMetal/ResidencyManager.swift`

Uses `MTLResidencySet` (Metal 4 feature, graceful fallback on older GPUs) to hint GPU which buffers should stay resident:

```swift
let descriptor = MTLResidencySetDescriptor()
descriptor.initialCapacity = 256
let set = try device.makeResidencySet(descriptor: descriptor)
set.requestResidency()
commandQueue.addResidencySet(set)
```

### 8.6 KernelRegistry

**File:** `Sources/EdgeRunnerMetal/KernelRegistry.swift`

Cached `MTLComputePipelineState` lookup with Mutex protection:

```swift
package final class KernelRegistry {
    private let library: MetalLibraryHandle
    private let cache: Mutex<PipelineCache>
    private let device: MTLDevice
}

package func pipeline(for name: String) throws -> MTLComputePipelineState {
    // Check cache first (Mutex-protected)
    // Fall back to library.makeFunction(name:)
    // Compile with MTLComputePipelineDescriptor(supportIndirectCommandBuffers: true)
}
```

**Library loading priority:**
1. Pre-compiled `.metallib` from Xcode bundle
2. Runtime compilation from `.metal` files with `MTLCompileOptions`

### 8.7 Memory Allocation Summary

| Category | When Allocated | Recycling |
|----------|--------------|-----------|
| KV cache | Model init | Never (persistent) |
| Scratch buffers | Model init | Never (persistent) |
| Weight buffers | Model init or first prefill | Never (persistent) |
| Short-lived temps | Per-call | Returned to `BufferCache` |
| Kernel temps | Per-call | Not recycled |

### 8.8 Thread Safety

| Component | Mechanism |
|----------|-----------|
| `MetalBackend` | Swift `actor` |
| `BufferCache` | `Mutex<CacheState>` |
| `KVCache` | `Mutex<[LayerState]>` |
| `KernelRegistry` | `Mutex<PipelineCache>` |
| `BarrierTracker` | Not thread-safe (caller ensures single-threaded access) |

---

## 9. KV Cache

### 9.1 Structure

```swift
public final class KVCache: Sendable {
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision  // .float32, .float16, .float8

    private let keyBuffers: [MetalBufferHandle]   // [numLayers]
    private let valueBuffers: [MetalBufferHandle]
    private let layerStates: Mutex<[LayerState]>
}

struct LayerState: Sendable {
    var writePos = 0    // Current write position (circular)
    var totalWritten = 0  // Total tokens ever written
}
```

### 9.2 Circular Buffer Design

`writePos` cycles from 0 to maxSeqLen-1. `totalWritten` tracks absolute token count. When `totalWritten > maxSeqLen`, oldest tokens are overwritten. Retrieval handles this via two-chunk reading: `[writePos...maxSeqLen]` then `[0...writePos]`.

### 9.3 Precision

KV cache initialized as Float16 (`.float16`). `ERKVPrecision` enum values: `float32=0`, `float16=1`, `float8=2`.

### 9.4 GPU Direct Write

GPU kernels write K/V directly to cache buffers:
```swift
let cacheWriteOffF16 = (t + startPosition) * kvDim * halfStride
enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
```

---

## 10. Sampling & Generation

### 10.1 SamplingPipeline Architecture

Composable transforms + selector:

```swift
public struct SamplingPipeline: Sendable {
    private let transforms: [any LogitsTransform]
    private let selector: any TokenSelector
    private let repetitionPenalty: RepetitionPenalty?
}
```

**Flow:**
```
Logits → RepetitionPenalty → TemperatureSampler → TopKSampler → TopPSampler → TokenSelector → Token ID
```

### 10.2 LogitsTransform Implementations

| Transform | Behavior |
|-----------|----------|
| `TemperatureSampler` | Divides logits by temperature |
| `TopKSampler` | Sets all logits below top-K to `-infinity` |
| `TopPSampler` | Sets tokens outside cumulative nucleus to `-infinity` |
| `RepetitionPenalty` | Penalizes tokens based on frequency |

### 10.3 TokenSelector Implementations

| Selector | Behavior |
|----------|----------|
| `GreedySampler` | Returns index of maximum logit |
| `StochasticSampler<RNG>` | Samples from softmax distribution |

### 10.4 RepetitionPenalty Algorithm

```
for each token in previousTokens:
    if logits[token] > 0: logits[token] /= penalty
    else:                 logits[token] *= penalty
    if frequencyPenalty > 0:
        logits[token] -= frequencyPenalty * count(token)
```

### 10.5 GenerationSession

```swift
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    public let maxTokens: Int

    public init(
        model: Model,
        sampling: SamplingConfiguration = SamplingConfiguration(),
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    )

    public func stream(prompt: String) -> AsyncThrowingStream<String, Error>
    public func generate(prompt: String) async throws -> String
}
```

### 10.6 Conversation (Multi-turn Chat)

```swift
public struct Conversation: Sendable {
    public private(set) var messages: [ChatMessage]

    public init(systemPrompt: String? = nil)
    public mutating func addUser(_ content: String)
    public mutating func addAssistant(_ content: String)
    public mutating func addSystem(_ content: String)
    public mutating func reset(keepSystem: Bool = true)
}
```

---

## 11. Public API Reference

### 11.1 Core Protocols

**`EdgeRunnerLanguageModel`** — Central protocol for all models:
```swift
public protocol EdgeRunnerLanguageModel: Sendable {
    static var modelIdentifier: String { get }
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self
    func tokenize(_ text: String) -> [Int]
    func detokenize(_ ids: [Int]) -> String
    var eosTokenID: Int { get }
    var bosTokenID: Int? { get }
    var vocabularySize: Int { get }
    func applyChatTemplate(messages: [ChatMessage], addGenerationPrompt: Bool) -> String?
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}
```

**`LogitsModel`** — Metal-accelerated models with raw logits access:
```swift
public protocol LogitsModel: EdgeRunnerLanguageModel {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}
```

**`EdgeRunnerTool`** — Tool/function calling:
```swift
public protocol EdgeRunnerTool: Sendable {
    static var name: String { get }
    static var description: String { get }
    static var parameters: [ToolParameter] { get }
    func invoke(arguments: [String: Any]) async throws -> String
}
```

### 11.2 Configuration Types

**`ModelConfiguration`:**
```swift
public struct ModelConfiguration: Sendable {
    public var maxTokens: Int                    // default: 2048
    public var contextWindowSize: Int            // default: 4096
    public var useMemoryMapping: Bool            // default: true
    public var tokenizerURL: URL?                 // optional external tokenizer
}
```

**`SamplingConfiguration`:**
```swift
public struct SamplingConfiguration: Sendable {
    public var temperature: Float       // default: 1.0
    public var topK: Int              // default: 40
    public var topP: Float           // default: 0.9
    public var repetitionPenalty: Float // default: 1.0
    public var seed: UInt64?          // optional, for reproducibility
}
```

### 11.3 Error Types

| Error | Cause |
|-------|-------|
| `GenerationError.modelLoadFailed` | GPU unavailable, buffer allocation failed |
| `GenerationError.decodingFailed` | NaN/Inf in logits during greedy decode |
| `GenerationError.contextWindowExceeded` | Requested > maximum context |
| `ModelLoadError.unsupportedFormat` | Unknown model format |
| `ModelLoadError.unknownArchitecture` | Unknown model architecture |
| `WeightLoaderError.unsupportedDataType` | Quantization type not supported |
| `WeightLoaderError.checksumMismatch` | GGUF checksum validation failed |

### 11.4 BackendRegistry

Registry for loading models by format:

```swift
public final class BackendRegistry: Sendable {
    public func register<T: EdgeRunnerLanguageModel>(_ type: T.Type, for format: String)
    public func backend(for format: String) -> (any EdgeRunnerLanguageModel.Type)?
    public func load(from url: URL, format: String, configuration: ModelConfiguration) async throws -> any EdgeRunnerLanguageModel
}
```

### 11.5 ModelLoader

High-level loading with automatic architecture detection:

```swift
public enum ModelLoader: Sendable {
    public static func load(from url: URL, configuration: ModelConfiguration = ModelConfiguration()) async throws -> any EdgeRunnerLanguageModel
}
```

**Supported architectures:** `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`, `mistral`, `starcoder`, `starcoder2`, `internlm2`, `yi`, `deepseek`, `deepseek2`, `command-r`, `falcon`

### 11.6 Quick Usage Examples

**Basic generation:**
```swift
let model = try await ModelLoader.load(from: modelURL)
let tokens = model.tokenize("Hello, world!")
for _ in 0..<50 {
    let next = try await model.nextToken(for: tokens, sampling: SamplingConfiguration())
    tokens.append(next)
}
let text = model.detokenize(tokens)
```

**Streaming:**
```swift
let stream = model.stream("Tell me a story")
for try await text in stream {
    print(text, terminator: "")
}
```

**GenerationSession:**
```swift
let session = GenerationSession(model: model, sampling: SamplingConfiguration(temperature: 0.7), maxTokens: 1024)
let stream = session.stream(prompt: "Write a haiku")
for try await text in stream { print(text, terminator: "") }
let result = try await session.generate(prompt: "Write a haiku")
```

**Tool calling:**
```swift
struct CalculatorTool: EdgeRunnerTool {
    static let name = "calculator"
    static let description = "Perform calculations"
    static let parameters: [ToolParameter] = [...]
    func invoke(arguments: [String: Any]) async throws -> String { ... }
}
let executor = ToolExecutor(tools: [CalculatorTool()])
let results = try await executor.executeAll(parsedToolCalls)
```

---

## 12. Unique Design Innovations

### 12.1 Progressive Fusion Strategy

Every multi-step operation that can be merged into one GPU dispatch IS merged. The mega-kernel pattern (`fused_qk_norm_rope_gqa` + `dequant_q8_0_fused_ffn_block`) is the most impactful single optimization — reducing per-layer dispatches from ~20 to 2.

### 12.2 Zero-Copy Q8_0 Path

The raw Q8_0 buffer is passed directly to matmul kernels without float32 materialization, achieving ~3.8× bandwidth reduction vs naive separate passes.

### 12.3 Three-Tier Dequantization

GPU dynamically selects the optimal dequantization path:
1. Raw Q8_0 zero-copy (fastest)
2. GPU dequant kernel (flexible, handles all other formats)
3. CPU fallback (embedding lookups)

### 12.4 GPU Pipeline Warmup

On first prefill, 5 dummy decode passes warm the GPU JIT compilation pipeline before re-running the actual prefill. This prevents the first-real-inference latency spike.

### 12.5 No Threadgroup-Barrier Mega-Kernel

The `fused_qk_norm_rope_gqa` kernel achieves pure SIMD reductions across the full 128-dim attention head with zero `threadgroup_barrier` calls — eliminating the most expensive synchronization primitive in GPU compute.

### 12.6 Circular KV Cache

Handles sequences longer than `maxSeqLen` via wraparound, with `writePos` and `totalWritten` tracking logical position. GPU writes K/V directly at computed offsets; no separate cache management kernel needed.

### 12.7 Tiled GEMV for Memory Coalescing

The `dequant_q8_0_gemv_tiled` variant solves DRAM row-buffer thrashing from strided x[] access by cooperatively loading a 1024-element contiguous tile into threadgroup SRAM (4KB), processing from fast memory.

### 12.8 Metal 4 Argument Table Dispatch

On macOS 26+, `setArgumentTable` is called ONCE and Metal snapshots buffer bindings at dispatch time. Only changed buffer addresses are updated via `setAddress`, and execution-only barriers (`visibilityOptions: []`) avoid cache flush on unified memory.

### 12.9 Pre-allocated Scratch Buffers

All 19 scratch buffers are allocated once at model init and reused across all forward passes. Forward passes never allocate GPU memory, eliminating allocator overhead entirely.

### 12.10 Multi-Turn Prefix Reuse

The system auto-detects when a new sequence extends a cached prefix, computing only the suffix while attending over the full KV cache — enabling efficient multi-turn conversations with automatic KV cache reuse.

---

## Appendix: File Map

| File | Purpose |
|------|---------|
| `EdgeRunner/EdgeRunnerLanguageModel.swift` | Core model protocol |
| `EdgeRunner/Models/LlamaLanguageModel.swift` | Main Metal implementation |
| `EdgeRunner/Streaming/GenerationSession.swift` | Streaming generation |
| `EdgeRunner/Streaming/TokenStream.swift` | Token stream types |
| `EdgeRunner/SamplingConfiguration.swift` | Sampling config |
| `EdgeRunner/Conversation.swift` | Chat history |
| `EdgeRunnerCore/Sampling/SamplingPipeline.swift` | Composable sampling |
| `EdgeRunnerCore/Sampling/GreedySampler.swift` | Argmax |
| `EdgeRunnerCore/Sampling/StochasticSampler.swift` | Random sampling |
| `EdgeRunnerCore/Sampling/TemperatureSampler.swift` | Temperature |
| `EdgeRunnerCore/Sampling/TopKSampler.swift` | Top-K |
| `EdgeRunnerCore/Sampling/TopPSampler.swift` | Nucleus |
| `EdgeRunnerCore/Sampling/RepetitionPenalty.swift` | Repetition penalty |
| `EdgeRunnerMetal/KVCache.swift` | KV cache |
| `EdgeRunnerMetal/BufferCache.swift` | LRU buffer cache |
| `EdgeRunnerMetal/CommandBatcher.swift` | Command batching |
| `EdgeRunnerMetal/BarrierTracker.swift` | RAW hazard tracking |
| `EdgeRunnerMetal/ResidencyManager.swift` | GPU residency |
| `EdgeRunnerMetal/KernelRegistry.swift` | Pipeline cache |
| `EdgeRunnerMetal/MetalBackend.swift` | GPU actor |
| `EdgeRunnerIO/LlamaConfig.swift` | GGUF config parsing |
| `EdgeRunnerIO/LlamaModel.swift` | Model container |
| `EdgeRunnerIO/WeightMap.swift` | Weight storage types |
| `EdgeRunnerSharedTypes/include/*.h` | C params structs |
| `EdgeRunnerMetal/Shaders/*.metal` | All GPU shaders |
