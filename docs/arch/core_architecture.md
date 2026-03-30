# EdgeRunner Core Architecture

## Overview

EdgeRunner is a Metal-accelerated LLM inference engine for Apple Silicon. It implements the Llama transformer architecture (Llama 2, Llama 3, Qwen, Mistral) with support for GGUF model loading, on-the-fly GPU weight dequantization, and highly fused compute kernels.

**Architecture pipeline:**
```
tokens → embedding → [RMSNorm → RoPE + GQA + KVCache → RMSNorm → SwiGLU FFN] × N → RMSNorm → LM head → logits
```

---

## 1. Core Model Types

### 1.1 LlamaConfig

Defined in `Sources/EdgeRunnerIO/LlamaConfig.swift`. Holds the static configuration parsed from GGUF metadata.

```swift
public struct LlamaConfig: Sendable, Equatable {
    public let embeddingDim: Int           // Model dimension (e.g., 4096)
    public let layerCount: Int             // Number of transformer layers
    public let headCount: Int              // Number of query heads (Q)
    public let kvHeadCount: Int            // Number of KV heads (usually fewer than Q for GQA)
    public let vocabSize: Int              // Vocabulary size
    public let intermediateDim: Int        // FFN intermediate dimension (usually ~4× embeddingDim)
    public let ropeFreqBase: Double        // RoPE base frequency (e.g., 10000 or 500000)
    public let rmsNormEpsilon: Double      // RMSNorm epsilon (e.g., 1e-5)
    public let explicitHeadDim: Int?       // Optional explicit head dim (Qwen 3: key_length != embedding_dim/head_count)

    public var headDim: Int {
        explicitHeadDim ?? (embeddingDim / headCount)
    }

    public var gqaRatio: Int {
        headCount / kvHeadCount
    }
}
```

Parsed from GGUF metadata keys like `llama.embedding_length`, `llama.block_count`, `llama.attention.head_count`, etc.

### 1.2 LlamaModel

Defined in `Sources/EdgeRunnerIO/LlamaModel.swift`. A loadable model container that holds config and weight references.

```swift
public struct LlamaModel: LoadableModel, Sendable {
    public let config: LlamaConfig
    public let layers: [LlamaBlock]
    public private(set) var loadedWeights: [String: TensorStorage] = [:]
}
```

Weight name mapping: GGUF tensor names (e.g., `blk.0.attn_q.weight`) are mapped to canonical names (e.g., `layers.0.attention.wq.weight`) via `LlamaWeightNameMapper`.

### 1.3 TensorStorage & TensorDataType

Defined in `Sources/EdgeRunnerIO/WeightMap.swift`.

```swift
public enum TensorDataType: UInt32, Sendable, Equatable {
    case float32 = 0
    case float16 = 1
    case q4_0 = 2        // 4-bit, 32 elements/block + scale (18 bytes/block)
    case q4_1 = 3        // 4-bit with bias, 32 elements/block + scale + bias (20 bytes/block)
    case q5_0 = 6        // 5-bit, 32 elements/block + scale (22 bytes/block)
    case q5_1 = 7        // 5-bit with bias, 32 elements/block + scale + bias (24 bytes/block)
    case q8_0 = 8        // 8-bit, 32 elements/block + scale (34 bytes/block)
    case q8_1 = 9        // 8-bit with bias (not supported)
    case q2_K = 10       // 2-bit K-quant, 256 elements/superblock (84 bytes/superblock)
    case q3_K = 11       // 3-bit K-quant, 256 elements/superblock (110 bytes/superblock)
    case q4_K = 12       // 4-bit K-quant, 256 elements/superblock (144 bytes/superblock)
    case q5_K = 13       // 5-bit K-quant, 256 elements/superblock (176 bytes/superblock)
    case q6_K = 14       // 6-bit K-quant, 256 elements/superblock (210 bytes/superblock)
    case q8_K = 15       // 8-bit K-quant (not fully supported)
    case i8 = 16, i16 = 17, i32 = 18, i64 = 19, f64 = 20, bfloat16 = 30
}

public struct TensorStorage: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let byteOffset: Int
    public let dataType: TensorDataType
    public let shape: [Int]
    public let name: String
}
```

**Supported quantization types** (validated at load time):
- `Q4_0`, `Q4_K_M` (Q4_K variant), `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `Q2_K`, `Q3_K`
- Plus full-precision: `F16`, `F32`

---

## 2. LlamaLanguageModel - Main Inference Engine

Defined in `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`. Conforms to `LogitsModel`.

### 2.1 Instance Fields

```swift
public struct LlamaLanguageModel: LogitsModel, @unchecked Sendable {
    // Config & weights
    private let config: LlamaConfig
    private let weights: [String: TensorStorage]
    private let tokenizer: (any Tokenizer)?

    // Metal infrastructure
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Metal kernels
    private let rmsNormKernel: RMSNormKernel
    private let ropeKernel: RoPEKernel
    private let gqaKernel: GQAKernel
    private let activationKernels: ActivationKernels
    private let gemvKernel: GEMVKernel

    // Fused compute pipelines (key optimization)
    private let fusedQ8GemvPipeline: MTLComputePipelineState        // Q8_0 dequant + GEMV
    private let fusedQ8GemvTiledPipeline: MTLComputePipelineState   // Tile-based variant
    private let fusedQKVPipeline: MTLComputePipelineState          // Fused QKV (RMSNorm + 3× GEMV)
    private let fusedGateUpSiluPipeline: MTLComputePipelineState    // Fused gate+up+SwiGLU
    private let fusedNormRoPEGQAPipeline: MTLComputePipelineState   // Mega-kernel: Q/K norm + RoPE + GQA
    private let fusedFinalNormGemvPipeline: MTLComputePipelineState // Final RMSNorm + LM head

    // Dequantization kernels (one per quantization type)
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
    private let layerKCaches: [MTLBuffer]  // Per-layer direct MTLBuffer access
    private let layerVCaches: [MTLBuffer]

    // Preloaded weights (write-once, then read-only)
    private let preloadedWeights: PreloadedWeightsStore

    // Pre-allocated scratch buffers (zero allocation per call)
    private let scratch: ScratchBuffers

    // Metal 4 state (macOS 26+)
    private let metal4State: Metal4State?

    // Optimized Metal 3 params buffer
    private let decodeParamsBuffer: MTLBuffer?
}
```

### 2.2 Model Loading

```swift
public static func load(
    from url: URL,
    configuration: ModelConfiguration
) async throws -> LlamaLanguageModel
```

Flow:
1. `GGUFLoader` reads the GGUF file via memory-mapped I/O
2. `LlamaConfig` is parsed from metadata
3. Weights are validated for quantization type compatibility
4. `LlamaLanguageModel` is initialized with all Metal kernels and buffers

### 2.3 Inference Entry Points

```swift
// Primary: compute logits for next token
public func logits(for tokenIDs: [Int]) async throws -> [Float]

// Greedy argmax without materializing full Swift array
func greedyToken(for tokenIDs: [Int]) async throws -> (token: Int, hasNonFinite: Bool)

// Generate next token with sampling
public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
```

### 2.4 Inference Modes

The `forwardLogitsBuffer` method detects three modes:

**Decode Mode** (single new token, KV cache valid):
```
commonPrefixLen == previousTokenIDs.count && tokenIDs.count == commonPrefixLen + 1
```
Processes only the single new token using KV cache. Most efficient path.

**Prefix Reuse Mode** (multiple new tokens extending cached prefix):
```
commonPrefixLen > 0 && commonPrefixLen == previousTokenIDs.count && tokenIDs.count > commonPrefixLen + 1
```
Prefills suffix tokens with correct RoPE positions; GQA attends over full KV cache.

**Full Prefill Mode** (no useful prefix match):
```
commonPrefixLen == 0 || commonPrefixLen < previousTokenIDs.count
```
Resets KV cache and recomputes entire sequence. Also runs 5 warmup dummy decodes to JIT warm GPU pipeline, then re-runs actual prefill.

---

## 3. Quantization & Weight Dequantization

### 3.1 Quantization Format Summary

| Type | Block Size | Bytes/Block | Notes |
|------|-----------|-------------|-------|
| Q4_0 | 32 elements | 18 bytes | 4-bit, scale as float16 |
| Q4_1 | 32 elements | 20 bytes | 4-bit with float16 bias |
| Q5_0 | 32 elements | 22 bytes | 5-bit, scale as float16 |
| Q5_1 | 32 elements | 24 bytes | 5-bit with float16 bias |
| Q8_0 | 32 elements | 34 bytes | 8-bit, scale as float16 |
| Q2_K | 256 superblock | 84 bytes/superblock | 2-bit quantized scales + quants |
| Q3_K | 256 superblock | 110 bytes/superblock | 3-bit |
| Q4_K | 256 superblock | 144 bytes/superblock | 4-bit |
| Q5_K | 256 superblock | 176 bytes/superblock | 5-bit |
| Q6_K | 256 superblock | 210 bytes/superblock | 6-bit |

### 3.2 Q8_0 Layout (32 elements/block)

Each block: `[2 bytes scale (float16)][32 × 1 byte quantized values]`

```swift
// In makeRawQ8BufferIfAvailable:
let blockCount = storage.elementCount / 32
let byteCount = blockCount * 34  // 2 bytes scale + 32 bytes data
```

### 3.3 K-Quant Layout (Q4_K, Q6_K, etc.)

K-quant types use a two-level structure:
- **Superblock** (256 elements): contains scales
- **Blocks** within superblock: contain quantized values

Example Q6_K: 210 bytes per 256-element superblock = scales (4 bits + 6 bits + 2 bits) + 128 bytes quantized (6 bits × 256 / 8).

### 3.4 Weight Dequantization Strategy

EdgeRunner uses **three tiers** of dequantization:

**Tier 1 - Raw Q8_0 (most efficient)**: For Q8_0 weights, the raw quantized buffer is passed directly to fused GPU kernels. No float32 materialization.

```swift
// makeRawQ8BufferIfAvailable returns a zero-copy view of the Q8_0 buffer:
bytesNoCopy: storage.buffer.contents() + storage.byteOffset
```

**Tier 2 - GPU Dequantization Kernel**: For non-Q8_0 quantized weights, dedicated Metal kernels (`DequantQ4_0Kernel`, `DequantQ6KKernel`, etc.) dequantize on-GPU to float32/float16. The dequantized result is used for the current forward pass.

```swift
// Example: dequantize Q6_K
let superBlockCount = elementCount / 256
let byteCount = superBlockCount * 210
return try await dequantQ6K.dequantise(blockData: blockData, ...)
```

**Tier 3 - CPU Fallback**: For embedding lookups and tied head computations, CPU dequantization decodes directly into destination buffers using SIMD-optimized loops (see `fillEmbeddings`).

---

## 4. KV Cache

Defined in `Sources/EdgeRunnerMetal/KVCache.swift`.

### 4.1 KVCache Structure

```swift
public final class KVCache: Sendable {
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision  // .float32, .float16, .float8

    private let keyBuffers: [MetalBufferHandle]   // [numLayers], each maxSeqLen × numKVHeads × headDim
    private let valueBuffers: [MetalBufferHandle]
    private let layerStates: Mutex<[LayerState]>  // writePos, totalWritten per layer
}

public enum Precision: Sendable {
    case float32, float16, float8
}
```

### 4.2 Circular Buffer Design

The KV cache uses a **circular buffer** with `writePos` tracking the current position and `totalWritten` tracking total tokens seen. This handles wraparound when `totalWritten > maxSeqLen`.

```swift
private struct LayerState: Sendable {
    var writePos = 0    // Current write position in circular buffer
    var totalWritten = 0  // Total tokens written (can exceed maxSeqLen)
}
```

### 4.3 Direct GPU Access

The most performance-critical path: GPU kernels write K/V directly to cache buffers.

```swift
// From fusedPrefillPass and fusedDecodePass:
// K is written to layerKCache at position (tokenOffset + startPosition) * kvDim * halfStride
let cacheWriteOffF16 = (t + startPosition) * kvDim * halfStride
enc.setBuffer(layerVCache, offset: cacheWriteOffF16, index: 6)
```

### 4.4 KVCacheParams (C header)

```c
typedef struct {
    uint32_t maxSeqLen;
    uint32_t currentLen;
    uint32_t writePos;
    uint32_t numKVHeads;
    uint32_t headDim;
    uint32_t precision;  // 0=float32, 1=float16, 2=float8
} ERKVCacheParams;
```

---

## 5. Transformer Layer Data Flow

### 5.1 Per-Layer Computation

Each transformer layer computes:

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

### 5.2 GQA (Grouped Query Attention)

GQA reduces KV head count vs query head count. All query heads attend to all KV heads.

```swift
// GQA params
ERGQAParams(
    seqLen: UInt32(seqLen),
    headDim: UInt32(headDim),
    numHeads: UInt32(numHeads),
    numKVHeads: UInt32(numKVHeads),
    groupSize: UInt32(groupSize),  // numHeads / numKVHeads
    scale: 1.0 / sqrt(Float(headDim)),
    causal: 1,  // causal masking enabled
    kvBlockSize: UInt32(gqaBlockSize),
    qBlockSize: UInt32(gqaBlockSize),
    kvSeqLen: UInt32(totalKVSeqLen),
    qOffset: UInt32(startPosition)
)
```

The `qOffset` parameter supports prefix reuse: queries at position `qOffset + q_idx` attend only to KV positions `0..q_offset+q_idx`.

---

## 6. RMSNorm

Root Mean Square Layer Normalization: `output = input * RMS(input)^{-1} * weight + bias`

```swift
// ERRMSNormParams (C header)
typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} ERRMSNormParams;
```

Fused into projection kernels for decode (seqLen=1): the RMSNorm weight is applied inline without a separate kernel dispatch.

---

## 7. RoPE (Rotary Position Embedding)

RoPE encodes absolute position with rotation matrices applied to Query and Key vectors.

```swift
// ERRoPEParams (C header)
typedef struct {
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t startPos;       // Position offset (for prefill continuation)
    float theta;             // Base frequency (e.g., 10000.0 or 500000.0)
    float scalingFactor;     // Scaling factor for extended context
} ERRoPEParams;
```

Two variants:
- **Standard Llama RoPE**: Applied after Q/K projections
- **NeoX RoPE** (for Qwen3 with Q/K norm): Applied with per-head Q/K normalization in the mega-kernel

The half-dim rotation formula:
```swift
// For head_dim = 128, half_dim = 64
// cos/sin computed for positions startPos..startPos+seqLen-1
// RoPE(x)[2i]   = x[2i]   * cos - x[2i+1] * sin
// RoPE(x)[2i+1] = x[2i+1] * cos + x[2i]   * sin
```

---

## 8. SwiGLU FFN

SwiGLU = SiLU(Gate) * Up (Gated Linear Unit with Swish activation)

```swift
// FFN: output = down_proj * (silu(gate_proj * x) * up_proj * x)
// Where silu(x) = x * sigmoid(x)
```

Fused into a single kernel for seqLen=1 decode: `FusedGateUpSiluParams` drives a fused kernel that applies RMSNorm + gate + up + SiLU in one dispatch.

---

## 9. Compute Pipeline Dispatch

### 9.1 Fused Prefill Pass

Encodes ALL transformer layers + final norm + LM head into a **single command buffer** with one compute encoder.

Dispatch per layer (standard path, seqLen > 1 or non-Q8):
1. RMSNorm (attention)
2. Q, K, V projections (seqLen × 3 dispatches)
3. F32→F16 V conversion
4. Q/K per-head RMSNorm (Qwen3, per-token)
5. RoPE Q + RoPE K
6. K F32→F16 conversion + KV cache write
7. GQA
8. Wo projection
9. Residual add
10. RMSNorm (FFN)
11. Gate + Up projections (seqLen × 2)
12. SwiGLU activation
13. Down projection
14. Residual add

**Fused path (seqLen=1 + Q8_0)**: Merges steps 1+2, 4+5+6+7, 7+8, 10+11, 13+14 — dramatically reducing kernel dispatches.

### 9.2 Fused Decode Pass

For single-token decode with KV cache. Three variants:

1. **Base Metal 3** (`fusedDecodePass`): Standard approach with per-layer dispatches
2. **Optimized Metal 3** (`fusedDecodePassOpt`): Pre-allocated params buffer with 7 × 256-byte slots, reducing `setBytes` overhead
3. **Metal 4** (`fusedDecodePassMetal4`): Uses argument table dispatch (`setArgumentTable` called once), `setAddress` for per-dispatch buffer updates, execution-only barriers, residency set for explicit GPU memory management

### 9.3 Mega-Kernel (Qwen3)

The `fusedNormRoPEGQAPipeline` fuses Q/K per-head RMSNorm + RoPE + GQA into a single dispatch:

```swift
// 32 threads per head (single SIMD group, zero barriers)
// grid: [32, numHeads + numKVHeads]
enc.dispatchThreads(
    MTLSize(width: 32, height: totalHeads, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
)
```

---

## 10. Scratch Buffers

Pre-allocated reusable GPU buffers eliminate ~17 MTLBuffer allocations per forward pass.

```swift
private struct ScratchBuffers: @unchecked Sendable {
    let normed: MTLBuffer      // RMSNorm output
    let afterAttn: MTLBuffer   // After attention residual add
    let ffnNormed: MTLBuffer   // FFN RMSNorm output
    let outputA: MTLBuffer     // Layer output (even layers)
    let outputB: MTLBuffer     // Layer output (odd layers)
    let allQ: MTLBuffer        // Q after projection
    let allK: MTLBuffer        // K after projection
    let allV: MTLBuffer        // V (float32 scratch)
    let ropeQ: MTLBuffer       // RoPE Q temp
    let ropeK: MTLBuffer       // RoPE K temp
    let attnOut: MTLBuffer     // GQA output
    let proj: MTLBuffer        // Projection temp
    let gateOut: MTLBuffer     // FFN gate output
    let upOut: MTLBuffer       // FFN up output
    let activ: MTLBuffer       // SwiGLU activation
    let downOut: MTLBuffer     // FFN down output
    let finalOut: MTLBuffer   // Final norm output
    let logits: MTLBuffer     // LM head output
    let decodeHidden: MTLBuffer // Pre-allocated embedding buffer (dim × float)
}
```

---

## 11. Preloaded Weights Store

```swift
private struct LayerWeightBuffers {
    let attnNorm: MTLBuffer    // Attention RMSNorm weight
    let wq, wk, wv, wo: MTLBuffer!   // Q/K/V/O projection (float32 fallback)
    let qNorm, kNorm: MTLBuffer?     // Per-head Q/K RMSNorm (Qwen3)
    let ffnNorm: MTLBuffer    // FFN RMSNorm weight
    let gate, up, down: MTLBuffer!   // FFN weights (float32 fallback)

    // Raw Q8_0 quantized buffers (nil if not Q8_0)
    let wqRaw, wkRaw, wvRaw, woRaw: MTLBuffer?
    let gateRaw, upRaw, downRaw: MTLBuffer?
}
```

Weights are loaded once on the first forward pass and cached in `PreloadedWeightsStore`. Raw Q8_0 buffers are preferred to avoid float32 materialization.

---

## 12. Decoder State Store

Tracks state for efficient KV cache utilization:

```swift
private final class DecoderStateStore: @unchecked Sendable {
    private var _previousTokenIDs: [Int] = []
    private var _cachedLogits: [Float]?       // Cached logits for last input
    private var _cachedLogitsInput: [Int]?     // The input that produced cached logits
    private var _decodeWarmedUp: Bool = false  // Whether GPU pipeline is warmed up
}
```

Key optimization: if current `tokenIDs` exactly matches `cachedLogitsInput`, the cached logits are returned without GPU computation.

---

## 13. Metal4State

Metal 4 (macOS 26+) introduces GPU-managed memory residency and argument table dispatch:

```swift
@available(macOS 26.0, iOS 26.0, *)
private final class Metal4State: @unchecked Sendable {
    let commandQueue: any MTL4CommandQueue
    let commandBuffer: any MTL4CommandBuffer
    let allocator: any MTL4CommandAllocator
    let argumentTable: any MTL4ArgumentTable  // 11 max buffer slots
    let residencySet: any MTLResidencySet     // Explicit GPU memory management
    let paramsBuffer: MTLBuffer                // 7 × 256-byte slots
}
```

Key optimizations:
- `setArgumentTable` called ONCE; Metal snapshots at dispatch time
- Only changed buffer addresses updated via `setAddress`
- Execution-only barriers (`visibilityOptions: []`) — no cache flush on unified memory
- Residency set pre-populated with all buffers for the decode workload

---

## 14. Data Flow Summary

### Full Prefill Path:
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

### Decode Path:
```
Single new token [1]
  → memcpy embedding into decodeHidden [dim]
  → runDecodePass → fusedDecodePass / fusedDecodePassOpt / fusedDecodePassMetal4
  → logits [vocabSize]
```

---

## 15. Key Architectural Insights

1. **Fused Kernels as Primary Optimization**: Every multi-step operation that can be merged into one GPU dispatch is fused. The "mega-kernel" pattern (Q/K norm + RoPE + GQA in one dispatch) is the most impactful single optimization.

2. **Zero-Copy Q8_0 Path**: The raw Q8_0 buffer is passed directly to matmul kernels without float32 materialization, achieving ~3.8× bandwidth reduction vs naively dequantizing first.

3. **Three Inference Modes**: The system auto-detects decode (1 new token), prefix-reuse (extends cached prefix), and full-prefill (no useful cache), choosing the appropriate GPU path.

4. **GPU Memory Pre-allocation**: All scratch buffers, KV cache buffers, and weight buffers are allocated once at model load time. Forward passes never allocate GPU memory, eliminating allocator overhead.

5. **Circular KV Cache**: Handles sequences longer than `maxSeqLen` via wraparound, with `writePos` and `totalWritten` tracking the logical position.

6. **Qwen3 Per-Head Q/K Norm**: Models like Qwen3 apply per-head RMSNorm to Q and K separately (not shared attention norm), enabling the mega-kernel fusion path.
