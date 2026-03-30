# EdgeRunner Framework: Deep Technical Documentation

**Compiled:** March 25, 2026
**Status:** Deep Research covering Framework Architecture, Memory Management, Quantization, and Performance Engineering

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Decomposition](#3-module-decomposition)
4. [GGUF Model Loading](#4-gguf-model-loading)
5. [Quantization System](#5-quantization-system)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Metal Compute Kernels](#7-metal-compute-kernels)
8. [Memory Management](#8-memory-management)
9. [Streaming API and Public Interface](#9-streaming-api-and-public-interface)
10. [Benchmark Infrastructure](#10-benchmark-infrastructure)
11. [Performance Engineering Patterns](#11-performance-engineering-patterns)
12. [Unique Innovations](#12-unique-innovations)

---

## 1. Executive Summary

EdgeRunner is a high-performance, on-device LLM inference engine written entirely in Swift, targeting Apple's Metal GPU API for compute acceleration. It loads GGUF-format quantized models and performs autoregressive text generation entirely on-device with no network dependency.

**Key differentiating characteristics:**

- **Zero-copy memory-mapped weight loading** from GGUF files via mmap
- **On-GPU dequantization**: quantized weights are never materialized to float32. Instead, dequantization happens inside Metal compute kernels during matmul operations
- **Fused mega-kernels** that merge 4 to 11 operations per GPU dispatch, including RMSNorm+QKV, Q/K norm+RoPE+GQA, Gate+Up+SwiGLU, and FFN blocks
- **Triple decode mode detection**: automatic switching between full prefill, prefix reuse, and incremental decode
- **Metal 4 argument table dispatch** (macOS 26+): eliminates per-dispatch CPU overhead via snapshot binding
- **9 quantization types** with specialized block layouts, including Q2_K through Q6_K, Q4_0, Q5_0, Q5_1, and Q8_0
- **Pre-loaded weight store**: eliminates async actor hops during the forward pass

---

## 2. Architecture Overview

### 2.1 Module Hierarchy

```
EdgeRunner (umbrella)
├── EdgeRunnerCore       - Tokenizer, model protocols, sampling pipelines
├── EdgeRunnerIO          - GGUF loading, memory-mapping, model config
├── EdgeRunnerMetal      - Compute kernels, GPU buffer management, KV cache
└── EdgeRunnerSharedTypes - C-interop structs for GPU params
```

### 2.2 Core Data Flow

```
GGUF File (mmap)
    -> GGUFLoader -> TensorStorage (memory-mapped MTLBuffer)
    -> LlamaLanguageModel.init
        -> KernelRegistry (compiles Metal shaders at runtime)
        -> PreloadedWeightsStore (pre-loads weights to GPU)
        -> KVCache (allocates GPU memory for K/V)
        -> ScratchBuffers (pre-allocates 19 scratch buffers)
    -> GenerationSession.stream
        -> tokenize(prompt) -> [Int] token IDs
        -> fusedPrefillPass (full layer-by-layer GPU encoding) OR
        -> fusedDecodePassMetal4 / fusedDecodePassOpt (single-token decode)
        -> SamplingConfiguration -> next token
```

### 2.3 Transformer Architecture

The LlamaLanguageModel implements the standard Llama architecture with SwiGLU activation and Grouped-Query Attention:

```
token embeddings -> [RMSNorm -> QKV projection -> RoPE -> GQA -> RMSNorm -> SwiGLU FFN] x N_layers
    -> RMSNorm -> LM head -> logits -> sampling -> next token
```

**Supported models:** Llama 2, Llama 3, Qwen, Mistral, and any GGUF model using the standard Llama architecture with SwiGLU + GQA + RoPE.

---

## 3. Module Decomposition

### 3.1 EdgeRunnerCore (`Sources/EdgeRunnerCore/`)

**Tokenizer** (`TokenizerProtocol.swift`):

- `Tokenizer` protocol with `encode`, `decode`, `applyChatTemplate`
- Supports both BPE and SentencePiece tokenizers loaded from GGUF metadata
- `TokenizerFactory.create` dispatches based on metadata

**SamplingPipeline** (`SamplingConfiguration.swift`):

- Composable pipeline: `LogitsTransform` x N -> `TokenSelector`
- Temperature, Top-K, Top-P, repetition penalty
- Greedy sampler for temperature 0 or less
- Seeded random source for deterministic generation

### 3.2 EdgeRunnerIO (`Sources/EdgeRunnerIO/`)

**GGUFLoader.swift** implements `EdgeRunnerWeightLoader`:

- `prepare(url)`: reads GGUF header, metadata kv pairs, tensor infos
- `load(from)`: creates `TensorStorage` with memory-mapped `MTLBuffer` for each weight tensor
- All tensor data remains memory-mapped (zero copy): no redundant allocation
- `tensorByteCount()` computes byte size for all 13 GGUF quantization types

**MemoryMappedFile.swift** provides mmap-based file access:

- `MAP_PRIVATE` read-only mapping
- `makeMetalBufferRegion()` handles page-aligned offsets for GPU buffer creation
- Falls back to copy if `bytesNoCopy` buffer creation fails
- Proper cleanup via `munmap` in `deinit`

**ModelConfig** is parsed from GGUF metadata k/v pairs. This includes architecture name, layer count, head count, embedding dim, intermediate dim, RoPE theta, and more.

### 3.3 EdgeRunnerMetal (`Sources/EdgeRunnerMetal/`)

**MetalBackend.swift** provides actor-encapsulated GPU management:

- `KernelRegistry`: pipeline state cache with mutex protection
- `BufferCache`: LRU cache at 50% of recommended working set size
- `ResidencyManager`: `MTLResidencySet` for page residency hints
- `CommandBatcher`: batches 30 to 50 ops per command buffer based on GPU family
- `BarrierTracker`: memory hazard tracking

**KVCache.swift** implements a ring-buffer KV cache:

- Per-layer K and V buffers in float16 (default) or float32
- `writePos` cycles modulo `maxSeqLen` (circular buffer)
- `totalWritten` tracks absolute token count for wrap detection
- `metalBuffers(layer)` exposes raw `MTLBuffer` handles for GPU kernel direct writes
- `advanceWritePosition(layer, count)`: called after GPU writes to update positions

**GEMVKernel.swift** handles general matrix-vector multiply:

- Row-parallel: 256 threads per threadgroup cooperatively reduce across K
- `simd_sum` for warp-level reduction, threadgroup memory for cross-warp reduction
- `executeBatchedWithWeightBuffers`: N independent GEMVs in one command buffer, using just 1 sync round-trip

**KernelRegistry.swift** handles Metal shader compilation:

- Tries pre-compiled `.metallib` first for Xcode builds
- Falls back to runtime compilation of all `.metal` files concatenated
- Pipeline states cached under mutex
- Supports `MTLFunctionConstantValues` for compile-time specialization

**CommandBatcher.swift** handles command buffer batching:

- `apple9` family (M4 Max/Ultra, A17 Pro): 50 ops max
- `apple8` family (M3, A16): 40 ops max
- Older devices: 30 ops max
- `dispatchType: .concurrent` encoder for parallel kernel execution

### 3.4 EdgeRunnerSharedTypes (`Sources/EdgeRunnerSharedTypes/`)

C-interop header with GPU parameter structs:

- `ERGEMVParams`: M, K, lda
- `ERDequantGEMVParams`: rows, cols, blocksPerRow
- `ERDequantParams`: blockCount, outputOffset
- `ERRMSNormParams`: rows, cols, eps
- `ERRoPEParams`: seqLen, numHeads, headDim, startPos, theta, scalingFactor
- `ERGQAParams`: seqLen, headDim, numHeads, numKVHeads, groupSize, scale, causal, kvBlockSize, qBlockSize, kvSeqLen, qOffset
- `ERKVCacheParams`: maxSeqLen, currentLen, writePos, numKVHeads, headDim, precision
- `ERElementwiseParams`: elementCount
- `ERActivationParams`: count

---

## 4. GGUF Model Loading

### 4.1 Memory-Mapped Loading

The GGUF file is opened with `mmap(PROT_READ, MAP_PRIVATE)`. This means:

- The OS handles page faults for on-demand loading
- No redundant RAM allocation: weights stay in the file, paged in as needed
- Fast startup: no full file read required
- Safe: `MAP_PRIVATE` means modifications do not affect the file

### 4.2 TensorStorage

Each weight tensor is wrapped in a `TensorStorage` struct:

```swift
struct TensorStorage {
    let buffer: MTLBuffer        // memory-mapped GPU buffer
    let byteOffset: Int         // offset into the buffer
    let dataType: TensorDataType // .float32, .float16, .q8_0, etc.
    let shape: [Int]             // dimensions
    let name: String
    let owner: MemoryMappedFile // keeps mmap alive
}
```

### 4.3 Dequantization-on-Demand

For Q8_0 weights, `makeRawQ8BufferIfAvailable()` returns the memory-mapped buffer directly: no float32 materialization. The GPU kernel reads the Q8_0 blocks and dequantizes in-register during the matmul.

For other quantization types (Q2_K through Q6_K, Q4_0, Q5_0, Q5_1), dequantization to float32 happens via CPU fallback kernels before GPU upload.

### 4.4 GGUF Format Details

The GGUF format (from llama.cpp) stores:

- **Header**: magic, version, tensor count, metadata kv count
- **Metadata k/v pairs**: model hyperparameters (config, tokenizer vocab, etc.)
- **Tensor infos**: name, type enum, dimensions, offset in data section
- **Data section**: raw weight bytes, 32-byte aligned

Byte sizes per quantization type (for N elements):

| Type  | Block Size | Bytes per Block | Notes |
|-------|-----------|----------------|-------|
| f32   | 1         | 4N             | Float32 |
| f16   | 1         | 2N             | Float16 |
| q4_0  | 32        | 18 x N/32      | 4-bit, scale only |
| q4_K  | 256       | 144 x N/256    | K-quant, 4-bit with scales+min per 256 |
| q5_0  | 32        | 22 x N/32      | 5-bit, scale only |
| q5_1  | 32        | 24 x N/32      | 5-bit, scale+zero |
| q5_K  | 256       | 176 x N/256    | K-quant, 5-bit with qh+qs nibble packing |
| q6_K  | 256       | 210 x N/256    | 6-bit: ql nibble + qh 2-bit + int8 scales |
| q8_0  | 32        | 34 x N/32      | 8-bit, scale as float16 |
| q2_K  | 256       | 84 x N/256     | 2-bit with scales+min per 256 |
| q3_K  | 256       | 110 x N/256    | 3-bit with hmask for high bit |

---

## 5. Quantization System

### 5.1 Quantization Types

EdgeRunner supports 9 quantization types across two families:

**Legacy block types** (32 elements per block):

- **Q4_0**: 4-bit, scale stored as float16, no zero point. 18 bytes/block.
- **Q5_0**: 5-bit, scale only. 22 bytes/block.
- **Q5_1**: 5-bit, scale + zero point. 24 bytes/block.

**K-quant family** (256 elements per super-block, variable sub-block structure):

- **Q2_K**: 2-bit quantization. 84 bytes/256 elements (1.64 bits/element). Lowest precision K-quant.
- **Q3_K**: 3-bit quantization. 110 bytes/256 elements (2.44 bits/element).
- **Q4_K_M**: 4-bit with separate scale+min per super-block. 144 bytes/256 elements (2.56 bits/element). Popular balanced choice.
- **Q5_K**: 5-bit with nibble-packed high bits. 176 bytes/256 elements (3.44 bits/element).
- **Q6_K**: 6-bit: ql (4-bit) + qh (2-bit) packed separately. 210 bytes/256 elements (4.06 bits/element).

**Q8_0**: 8-bit per element. 34 bytes/32 elements. Highest precision, 4 bits/element. This is the only type that avoids float32 materialization entirely via raw buffer passing.

### 5.2 On-GPU Dequantization

The key innovation here is that Q8_0 dequantization happens inside Metal compute kernels. The weight matrix never exists as float32 in GPU memory. Instead:

1. The raw Q8_0 bytes are passed directly as an `MTLBuffer`
2. Inside the kernel, each thread reads its 32-element block
3. `scale = Float(Float16(bitPattern: scaleBits))` reconstructs the float scale
4. `value = scale * Float(qval)` dequantizes each qval
5. The float value is immediately used in the matmul accumulation

This eliminates the 4x memory bandwidth that float32 would require versus Q8_0's 1x (plus scale overhead).

### 5.3 Q8_0 Block Layout

```
Byte layout per 32-element block (34 bytes):
[0-1]   scale: Float16 (little-endian)
[2-33]  32 x Int8 quantized values (offset 0)
```

### 5.4 K-Quant Super-Block Layout

The K-quant family uses a two-level structure. A super-block (256 elements) contains metadata like scales, mins, and bit packing info. Sub-blocks (32 or 64 elements) hold the actual quantized values.

This design allows different precision tradeoffs within the same weight tensor.

---

## 6. Inference Pipeline

### 6.1 Triple Decode Mode Detection

The `DecoderStateStore` tracks `previousTokenIDs`. When `nextToken(for:)` is called, the system detects one of three modes:

1. **Full prefill** (cold start): `previousTokenIDs` is empty, so a full prefill runs
2. **Prefix reuse** (cache hit): new tokens are `previousTokenIDs` plus exactly 1 new token, and cached logits exist for `previousTokenIDs`. The system reuses cached attention keys/values for the prefix and only computes attention for the new token
3. **Incremental decode** (normal decode): new tokens are `previousTokenIDs` plus 1 new token. The KV cache is already populated, so only Q is computed for the new position and the system attends to the full KV

Detection logic in `forwardLogitsBuffer`:

```swift
let isPrefill = previousTokenIDs.isEmpty
let isPrefixReuse = !isPrefill
    && tokenIDs.count == previousTokenIDs.count + 1
    && tokenIDs.dropLast() == previousTokenIDs
    && decoderState.cachedLogits != nil
let isDecode = !isPrefill && !isPrefixReuse
```

### 6.2 Prefill Pass (Full Context Encoding)

`fusedPrefillPass` processes the entire prompt in a single forward pass:

1. **Embedding lookup**: `fillEmbeddings(tokenIDs)` reads embedding rows directly from memory-mapped storage (handles float32, float16, q8_0, q4_0)
2. **Layer loop** (N layers):
   - `RMSNorm -> QKV projection` (fused Q8_0 GEMV)
   - `RoPE` (standard or NeoX)
   - `GQA attention` (blocked KV cache access)
   - `RMSNorm -> SwiGLU FFN` (fused Gate+Up+SiLU)
3. **Final RMSNorm -> LM head** (fused final norm + GEMV)
4. Returns `logits[vocabSize]`

All 19 scratch buffers are used, and all operations are fused GPU kernels.

### 6.3 Decode Pass

**Metal 4 Path** (`fusedDecodePassMetal4`, macOS 26+):

- `setArgumentTable` is called once at the start
- Only changed buffer addresses are updated via `setAddress` per dispatch
- Pre-allocated 256-byte-aligned params buffer slots: zero `setBytes` copies
- Single `MTL4ComputeCommandEncoder` for the entire forward pass
- Execution-only barriers (`MTL4VisibilityOptionNone`): no cache flushes on unified memory
- Residency set pre-populated with all buffers before first decode

**Optimized Metal 3 Path** (`fusedDecodePassOpt`):

- Pre-allocated params buffer (7 x 256 bytes)
- Constant params written once via `memcpy`, only varying params (`startPos`, `kvSeqLen`) updated per call
- Eliminates `setBytes` overhead for all dispatches after first init

**Base Metal 3 Path**: Full per-dispatch setup (fallback for bisection and debugging).

### 6.4 Decode Warmup

On first decode call, 5 dummy decode passes are run to warm up the GPU pipeline:

```swift
for _ in 0..<5 {
    try await runDecodePass(hiddenBuf: decodeHiddenBuf, currentPos: 0, ...)
}
```

This ensures the GPU compiler has optimized the pipelines before measuring performance.

### 6.5 Fused Mega-Kernels

The most aggressive fusion in the decode path merges operations that would otherwise require 2 to 4 separate GPU dispatches.

**`dequant_q8_0_fused_qkv`**: RMSNorm + Wqx + Wkx + Wvx in one dispatch. It uses cooperative RMSNorm across heads, three independent Q8_0 GEMV operations, and outputs allQ, allK (half precision), and allV (written directly to KV cache).

**`fused_qk_norm_rope_gqa`** (6-in-1 mega-kernel): Q RMSNorm + Q RoPE + K RMSNorm + K RoPE + K write to cache + GQA attention in one dispatch. It uses 32 threads per head (single simdgroup), no threadgroup barriers (only simdgroup-level `simd_sum`), and threadgroup memory only for the attention output tile. Requires Q/K norm to be present (Qwen3 architecture).

**`dequant_q8_0_fused_gate_up_silu`** (4-in-1): RMSNorm + gate_projx + up_projx + SwiGLU in one dispatch. It performs a single GEMV with two Q8_0 weight matrices (gate + up), applies `silu` element-wise on the gate output, and produces the activated intermediate tensor.

**`dequant_q8_0_fused_ffn_block`** (11-in-1 mega-kernel for prefill): Wo GEMV + residual add + RMSNorm + Gate GEMV + Up GEMV + SwiGLU + Down GEMV + residual add in 6 phases. This runs the full FFN block in a single dispatch, with each phase separated by a threadgroup barrier, achieving massive bandwidth reduction versus 8 separate dispatches.

**`dequant_q8_0_gemv_add`**: GEMV + residual add fused. Used for the Wo projection and Down projection, saving one dispatch per occurrence.

### 6.6 SwiGLU Activation

SwiGLU (Swish-Gated Linear Unit) is the activation function used in Llama and Mistral models:

```
SwiGLU(x) = gate(x) x silu(up(x))
         = (W_gate x x) x (silu(W_up x x))
```

In `Activations.metal`:

```metal
inline float silu(float value) {
    return value / (1.0f + exp(-value));
}
kernel void swiglu_f32(...) {
    output[gid] = silu(gate[gid]) * up[gid];
}
```

The fused kernel avoids materializing the gate and up projections separately.

---

## 7. Metal Compute Kernels

### 7.1 Shader File Map

| Shader File           | Kernels                                                                                                                                                                                                                                           |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Dequant_Q8_0.metal`  | `dequant_q8_0`, `dequant_q8_0_gemv`, `dequant_q8_0_gemv_f16out`, `dequant_q8_0_gemv_add`, `dequant_q8_0_gemv_tiled`, `dequant_q8_0_fused_qkv`, `dequant_q8_0_fused_gate_up_silu`, `dequant_q8_0_fused_ffn_block`, `dequant_q8_0_fused_final_norm_gemv` |
| `Dequant_Q4_0.metal`  | `dequant_q4_0`, `dequant_q4_0_gemv`                                                                                                                                                                                                               |
| `Dequant_Q4_K_M.metal`| `dequant_q4_k_m`, `dequant_q4_k_m_gemv`                                                                                                                                                                                                           |
| `Dequant_Q2_K.metal`  | `dequant_q2_k`, `dequant_q2_k_gemv`                                                                                                                                                                                                               |
| `Dequant_Q3_K.metal`  | `dequant_q3_k`, `dequant_q3_k_gemv`                                                                                                                                                                                                               |
| `Dequant_Q5_K.metal`  | `dequant_q5_k`, `dequant_q5_k_gemv`                                                                                                                                                                                                               |
| `Dequant_Q6_K.metal`  | `dequant_q6_k`, `dequant_q6_k_gemv`                                                                                                                                                                                                               |
| `Dequant_Q5_0.metal`  | `dequant_q5_0`, `dequant_q5_0_gemv`                                                                                                                                                                                                               |
| `Dequant_Q5_1.metal`  | `dequant_q5_1`, `dequant_q5_1_gemv`                                                                                                                                                                                                               |
| `GEMV.metal`          | `gemv_f32`, `gemv_f16`                                                                                                                                                                                                                           |
| `GEMM.metal`          | `gemm_f32`, `gemm_f16`                                                                                                                                                                                                                           |
| `GQA.metal`          | `gqa_attention_f32`, `gqa_attention_f16kv`                                                                                                                                                                                                       |
| `RoPE.metal`          | `rope_f32`, `rope_neox_f32`, `rope_neox_f32_to_f16`, `fused_qk_norm_rope_neox`, `fused_qk_norm_rope_gqa`                                                                                                                               |
| `RMSNorm.metal`       | `rmsnorm_f32`, `rmsnorm_parallel_f32`                                                                                                                                                                                                             |
| `Softmax.metal`       | `softmax_f32`                                                                                                                                                                                                                                   |
| `Activations.metal`   | `sigmoid_f32`, `gelu_f32`, `swiglu_f32`                                                                                                                                                                                                         |
| `Elementwise.metal`   | `elementwise_add_float`, `convert_f32_to_f16`                                                                                                                                                                                                   |
| `LayerNorm.metal`     | `layernorm_f32`                                                                                                                                                                                                                                  |
| `FusedPatterns.metal` | `fused_add_activate_float`, `fused_mul_activate_float`, `fused_activate_float`                                                                                                                                                                   |
| `FlashAttention.metal` | Flash attention variants                                                                                                                                                                                                                         |
| `Reduction.metal`      | Reduction utilities                                                                                                                                                                                                                               |
| `Transpose.metal`      | Tensor transpose                                                                                                                                                                                                                                |
| `StitchableOps.metal`  | Dynamic dispatch via function constants                                                                                                                                                                                                           |

### 7.2 GEMV Kernel (GEMV.metal)

Row-parallel matrix-vector multiply. Each threadgroup (256 threads) handles one output row:

1. Each thread accumulates a partial dot product across K elements (strided access)
2. `simd_sum` reduces within a warp (32 threads)
3. `threadgroup_barrier` + `shared_sums[32]` for cross-warp reduction
4. First warp finalizes with another `simd_sum`

```
Threadgroup: 256 threads (8 warps x 32 threads/warp)
Grid: M threadgroups (one per output row)
```

For the float16 variant, accumulation happens in float32 for numerical stability, with the final result converted back to half.

### 7.3 GQA Kernel (GQA.metal)

Blocked KV attention with threadgroup tiles (16 x 128):

- `gqa_attention_f32`: K/V stored as float32 in KV cache
- `gqa_attention_f16kv`: K/V stored as half in KV cache, which halves KV bandwidth. Q and O remain float32 (fresh from GEMV). K/V are converted to float in threadgroup memory.

The block-wise attention algorithm works like this:

1. Each threadgroup loads a tile of K and V (16 rows x 128 headDim)
2. Computes attention scores for Q rows in the group
3. Uses online softmax (max/sum trick) for numerical stability
4. Accumulates weighted values per Q row
5. Divides by sum at the end for the final softmax

The threadgroup tile avoids loading the entire K/V head for each block, which is critical for long contexts.

### 7.4 RoPE (RoPE.metal)

**Standard RoPE** (`rope_f32`):

- Processes half-dimensions at a time (pairs of elements)
- Uses `cos` and `sin` precomputed or computed on-the-fly
- In-place rotation: `x_out[i] = x[i] * cos - x[i+half] * sin`

**NeoX RoPE** (`rope_neox_f32`):

- Different rotation pattern (used by Qwen and Falcon)
- Applied per-head rather than per-dimension-pair

**Fused variants**:

- `fused_qk_norm_rope_neox`: Q norm + Q RoPE + K norm + K RoPE in one kernel
- `fused_qk_norm_rope_gqa`: The 6-in-1 mega-kernel combining norm and rope for both Q/K, plus KV write and GQA

### 7.5 RMSNorm (RMSNorm.metal)

**Row-parallel** (`rmsnorm_f32`): Each thread handles one element. It computes `sqrt(rms + eps)` where `rms = sum(x^2) / N`, using single-threaded reduction to find the max, then parallel normalization.

**Parallel** (`rmsnorm_parallel_f32`): 8 simdgroups x 32 threads cooperatively process columns. It uses simdgroup-level `simd_max` for max reduction, a threadgroup barrier plus a second pass for normalization. This variant is used in prefill for better utilization.

### 7.6 StitchableOps (StitchableOps.metal)

Uses `MTLFunctionConstantValues` for compile-time kernel specialization. The `activation_type` constant supports values 0=none, 1=relu, 2=sigmoid, 3=gelu, 4=silu. The compiler eliminates dead branches, producing zero-overhead dispatch.

---

## 8. Memory Management

### 8.1 Scratch Buffers

Pre-allocated `MTLBuffer` pool (19 buffers) reused across all forward passes:

```swift
struct ScratchBuffers {
    let normed: MTLBuffer       // RMSNorm input
    let afterAttn: MTLBuffer    // after attention residual
    let ffnNormed: MTLBuffer    // FFN RMSNorm input
    let outputA, outputB: MTLBuffer  // ping-pong layer outputs
    let allQ, allK, allV: MTLBuffer   // Q/K/V projections
    let ropeQ, ropeK: MTLBuffer     // RoPE outputs
    let attnOut: MTLBuffer     // attention output
    let proj: MTLBuffer        // projection temp
    let gateOut, upOut: MTLBuffer  // FFN gate/up temps
    let activ: MTLBuffer       // SwiGLU activation output
    let downOut: MTLBuffer     // down projection temp
    let finalOut: MTLBuffer   // final RMSNorm output
    let logits: MTLBuffer      // LM head output (vocabSize)
    let decodeHidden: MTLBuffer // pre-allocated embedding buffer
}
```

Total allocation: roughly `(maxSeqLen * dim * 6 + maxSeqLen * qDim * 4 + vocabSize) * 4 bytes`. For a Qwen 0.6B model (dim=896, vocabSize=151,936, maxSeqLen=4096), this comes to about 100MB of scratch space.

### 8.2 PreloadedWeightsStore

`PreloadedWeightsStore` is write-once, read-many. It uses `NSLock` plus `OSAllocatedUnfairLock` for safe concurrent initialization:

```swift
final class PreloadedWeightsStore: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var layers: [LayerWeightBuffers] = []
    private(set) var finalNorm: MTLBuffer?
    private(set) var lmHead: MTLBuffer?
    private(set) var lmHeadRaw: MTLBuffer?  // Q8_0 raw buffer
    private(set) var isLoaded = false

    func load(...) { // called exactly once
        lock.lock(); defer { lock.unlock() }
        self.layers = layers; self.isLoaded = true
    }
}
```

For Q8_0 layers, only the raw quantized buffer is stored (no float32 materialization). Non-Q8_0 layers get a float32 fallback buffer.

### 8.3 LayerWeightBuffers

```swift
struct LayerWeightBuffers {
    let attnNorm: MTLBuffer     // float32 RMSNorm weight
    let wq, wk, wv, wo: MTLBuffer!           // float32 fallback
    let qNorm, kNorm: MTLBuffer? // per-head Q/K norm (Qwen3)
    let ffnNorm: MTLBuffer     // FFN RMSNorm
    let gate, up, down: MTLBuffer!           // FFN weights

    // Raw Q8_0 quantized buffers (nil if not Q8_0)
    let wqRaw, wkRaw, wvRaw, woRaw: MTLBuffer?
    let gateRaw, upRaw, downRaw: MTLBuffer?
}
```

### 8.4 BufferCache (MetalBackend)

LRU cache at 50% of GPU recommended working set:

- Reuse policy: buffer length must be in the range `[size, size * 2]`
- Thread-safe via `Mutex<CacheState>`
- Used for temporary buffers during kernel execution

### 8.5 ResidencyManager

`MTLResidencySet` provides page residency hints. It is pre-populated with all model weight buffers before the first inference, telling the GPU which pages to keep resident in VRAM and reducing page fault overhead during inference.

---

## 9. Streaming API and Public Interface

### 9.1 Core Protocol

```swift
public protocol EdgeRunnerLanguageModel: Sendable {
    static var modelIdentifier: String { get }
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self

    func tokenize(_ text: String) -> [Int]
    func detokenize(_ ids: [Int]) -> String
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}
```

### 9.2 LogitsModel Sub-Protocol

For raw logits access:

```swift
public protocol LogitsModel: EdgeRunnerLanguageModel {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}
```

### 9.3 GenerationSession

Generic streaming session manager wrapping any `EdgeRunnerLanguageModel`:

```swift
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    public func stream(prompt: String) -> AsyncThrowingStream<String, Error>
    public func generate(prompt: String) async throws -> String
}
```

The flow works like this:

1. Tokenizes the prompt (adds BOS if needed)
2. Runs prefill (full context encoding)
3. Loops: calls `nextToken`, detokenizes, yields text, appends to tokenIDs
4. Continues until EOS or `maxTokens`

### 9.4 ModelConfiguration

```swift
public struct ModelConfiguration: Sendable {
    public var maxTokens: Int = 2048
    public var contextWindowSize: Int = 4096
    public var useMemoryMapping: Bool = true
    public var tokenizerURL: URL?
    // internal: LlamaDecodeOverrides for debugging
}
```

### 9.5 SamplingConfiguration

```swift
public struct SamplingConfiguration: Sendable {
    public var temperature: Float = 1.0
    public var topK: Int = 40
    public var topP: Float = 0.9
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64?
}
```

Pipeline building works like this:

- Temperature 0 or less produces greedy (argmax) sampling
- Temperature, top-K, and top-P are applied via `TemperatureSampler`, `TopKSampler`, and `TopPSampler`
- Repetition penalty is applied before sampling

---

## 10. Benchmark Infrastructure

### 10.1 Benchmark Files

| File                          | Purpose                                |
|--------------------------------|----------------------------------------|
| `BenchmarkHelpers.swift`      | GPU timing helpers, command buffer profiling |
| `BenchmarkReportGenerator.swift` | JSON report generation for benchmarking |
| `LFMBenchmark.swift`           | Long-form generation benchmark          |
| `LongStoryGenerationTest.swift` | Long story generation tests           |
| `KVCacheBenchmarks.swift`      | KVCache performance benchmarks         |
| `BufferCacheTests.swift`       | Buffer cache hit rate tests            |

### 10.2 Benchmark Approach

The benchmark system uses Metal's `CMTime` for GPU-side timing:

1. Create a `MTLCommandBuffer` with completion handler
2. Encode benchmark kernel(s)
3. `commit()` with timestamp before and `completed` timestamp after
4. GPU time equals `completionTime - startTime`

Decode throughput is measured in tokens per second:

- Time from first token decode to last token
- Warmup runs (5 dummy passes) are excluded
- Prefill time can be measured separately

### 10.3 Decode Debug Options

Environment variables for benchmarking:

- `EDGERUNNER_DECODE_FORCE_BASE`: force base decode path
- `EDGERUNNER_DECODE_DISABLE_MEGA_GQA`: disable mega-kernel
- `EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD`: disable fused final norm
- `EDGERUNNER_DECODE_DISABLE_KV_BARRIER`: disable KV cache barrier
- `EDGERUNNER_DECODE_PREFER_METAL4`: prefer Metal 4 path

The mega-kernel auto-disables when `headCount + kvHeadCount > 24`.

---

## 11. Performance Engineering Patterns

### 11.1 Zero-Copy Weight Access

The architecture eliminates copies at every stage:

1. GGUF file -> mmap (OS handles page faults, no copy into userspace)
2. Memory-mapped file -> `MTLBuffer` via `bytesNoCopy` (no copy into GPU driver)
3. Q8_0 weights -> GPU kernel reads directly (no float32 materialization)
4. KV cache -> K/V written directly by GPU kernel (no CPU round-trip)

### 11.2 Async Preloading

`PreloadedWeightsStore.load()` is called during the first `logits()` call. Weights are uploaded to the GPU in the first forward pass, then cached. Subsequent passes have zero weight-loading overhead.

### 11.3 Scratch Buffer Ping-Pong

Layer outputs alternate between `outputA` and `outputB`:

```swift
let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB
```

This allows the previous layer's output to be used as residual while the new output is being computed.

### 11.4 SimdGroup-Only Reductions

The mega-kernel GQA uses `simd_sum` without cross-threadgroup barriers. It uses 32 threads (1 simdgroup) per head, with `threadgroup_barrier` only for the attention output tile write. The `simdgroup_index_in_threadgroup` is used for cross-warp reduction. This is the lowest-overhead synchronization primitive on Apple Silicon.

### 11.5 Threadgroup Memory for Coalesced Access

The tiled GEMV kernel (`dequant_q8_0_gemv_tiled`) uses threadgroup memory:

1. Each threadgroup loads a tile of the weight matrix
2. Cooperatively computes GEMV for a subset of rows
3. Avoids uncoalesced global memory access patterns
4. Particularly effective for wide matrices with large K dimension

### 11.6 Function Constants for Zero-Cost Dispatch

The `FusedPatterns.metal` shader uses `[[function_constant(0)]]` for activation type. The compiler performs dead code elimination at compile time, producing separate specialized `MTLComputePipelineState` instances cached per activation type. Runtime dispatch is a hash table lookup (O(1)).

---

## 12. Unique Innovations

### 12.1 Triple Decode Mode

No other open-source Metal LLM engine automatically detects and switches between:

1. **Full prefill**: cold start, entire context encoded
2. **Prefix reuse**: repeated prompt prefix, KV cache reused
3. **Incremental decode**: single new token, only Q computed

The `DecoderStateStore` plus `cachedLogits` tracking enables this without any user configuration.

### 12.2 Argument Table Dispatch (Metal 4)

On macOS 26 and later, `MTL4ArgumentTable` allows binding buffer addresses once. Metal snapshots at each dispatch, with only `setAddress` for changed buffers (not the full `setBuffer`). Execution-only barriers mean no coherence transactions on unified memory, and pre-allocated 256-byte-aligned params slots eliminate `setBytes` copying.

This is the lowest-CPU-overhead path for autoregressive decode on Apple Silicon.

### 12.3 11-in-1 Mega-Kernel

The `dequant_q8_0_fused_ffn_block` kernel in prefill merges 11 operations in a single GPU dispatch, eliminating 10 synchronization and dispatch overhead events per layer.

### 12.4 K-Quant Family with Super-Block Structure

The K-quant family (Q2_K through Q6_K) uses a two-level super-block structure that allows fine-grained precision control. Different bit depths apply to different model components, scales are stored as float16 for reduced memory, and separate sub-block quantization handles different sensitivity regions.

### 12.5 F16 KV Cache

The `gqa_attention_f16kv` kernel stores K/V in float16 rather than float32, which halves KV cache memory bandwidth. Q remains float32 for matmul precision. K/V are converted to float in threadgroup memory during attention computation.

### 12.6 Residency Set Pre-Population

Before the first inference, all weight buffers, scratch buffers, and KV cache buffers are added to an `MTLResidencySet`. This hints to the GPU driver to page all relevant weights into VRAM before the latency-sensitive first inference begins.

---

## Appendix: Key File Locations

| Component          | Path                                                                   |
|--------------------|------------------------------------------------------------------------|
| Main model         | `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`                  |
| GGUF loading       | `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift`                          |
| Memory mapping     | `Sources/EdgeRunnerIO/GGUF/MemoryMappedFile.swift`                   |
| Metal backend      | `Sources/EdgeRunnerMetal/MetalBackend.swift`                           |
| KV cache          | `Sources/EdgeRunnerMetal/KVCache.swift`                                |
| GEMV kernel       | `Sources/EdgeRunnerMetal/GEMVKernel.swift`                           |
| Q8_0 shader       | `Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_0.metal`                  |
| GQA shader        | `Sources/EdgeRunnerMetal/Shaders/GQA.metal`                            |
| RoPE shader       | `Sources/EdgeRunnerMetal/Shaders/RoPE.metal`                           |
| Fused patterns     | `Sources/EdgeRunnerMetal/Shaders/FusedPatterns.metal`                  |
| GPU params        | `Sources/EdgeRunnerSharedTypes/include/DequantParams.h`              |
| Streaming          | `Sources/EdgeRunner/Streaming/GenerationSession.swift`              |
| Sampling          | `Sources/EdgeRunner/SamplingConfiguration.swift`                      |
| Tokenizer         | `Sources/EdgeRunnerCore/Tokenizer/TokenizerProtocol.swift`             |
| Model config      | `Sources/EdgeRunner/ModelConfiguration.swift`                        |
