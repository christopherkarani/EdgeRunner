# EdgeRunner Memory & Compute Management

## Overview

EdgeRunner uses a layered memory management architecture built on Metal, combining LRU buffer caching, GPU residency tracking, per-layer KV cache management, and pre-allocated scratch buffers. Compute dispatch is managed through a `CommandBatcher` that consolidates operations into single command buffers to minimize GPU overhead.

---

## 1. Buffer Allocation & Caching

### 1.1 BufferCache (LRU Cache)

**File:** `Sources/EdgeRunnerMetal/BufferCache.swift`

The `BufferCache` is a thread-safe (via `Mutex`) LRU cache for Metal buffer reuse. It eliminates repeated `device.makeBuffer()` calls by recycling buffers of similar size.

**Key characteristics:**

- **Storage format:** `storageModeShared | hazardTrackingModeUntracked`
  - `storageModeShared` — CPU-GPU unified memory, no explicit GPU->CPU copies needed
  - `hazardTrackingModeUntracked` — Disables Metal's hazard tracking for performance; correctness is maintained via explicit command buffer ordering

- **Size bucketing:** Returns a buffer whose length is in `[size, size*2]`. If no cached buffer fits, allocates fresh from the device.

- **Max cache size:** 50% of `device.recommendedMaxWorkingSetSize` (capped at 64MB minimum)
  ```swift
  let maxCacheBytes = Int(Double(device.recommendedMaxWorkingSetSize) * 0.5)
  self.bufferCache = BufferCache(device: device, maxBytes: max(maxCacheBytes, 64 * 1024 * 1024))
  ```

- **Eviction:** LRU within size buckets. Eviction runs whenever `totalBytes + newLength > maxBytes`.

- **Wrapper:** `MetalBufferHandle` wraps `MTLBuffer` with `@unchecked Sendable`, keeping Metal objects inside package-internal APIs without leaking across actor boundaries.

### 1.2 ScratchBuffers (Persistent Pre-allocated)

**File:** `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` (private struct, line ~2775)

Pre-allocated once at model init, sized for `maxSeqLen`. These are the backbone of zero-allocation inference:

| Buffer | Purpose | Size |
|--------|---------|------|
| `normed` | RMSNorm output | `maxSeqLen × dim × 4B` |
| `afterAttn` | Post-attention residual | `maxSeqLen × dim × 4B` |
| `ffnNormed` | FFN RMSNorm input | `maxSeqLen × dim × 4B` |
| `outputA`, `outputB` | Layer output ping-pong | `maxSeqLen × dim × 4B` |
| `allQ` | Full Q tensor (prefill) | `maxSeqLen × qDim × 4B` |
| `allK`, `allV` | Full K/V tensors (prefill) | `maxSeqLen × kvDim × 4B` |
| `ropeQ`, `ropeK` | RoPE output | `maxSeqLen × qDim/kvDim × 4B` |
| `attnOut` | Attention projection output | `maxSeqLen × qDim × 4B` |
| `proj` | Attention output projection | `maxSeqLen × dim × 4B` |
| `gateOut`, `upOut` | SwiGLU gate/up | `maxSeqLen × interDim × 4B` |
| `activ` | SiLU activation output | `maxSeqLen × interDim × 4B` |
| `downOut` | FFN down projection | `maxSeqLen × dim × 4B` |
| `finalOut` | Final RMSNorm output | `maxSeqLen × dim × 4B` |
| `logits` | Vocab-sized logits | `vocabSize × 4B` |
| `decodeHidden` | Single-token embedding | `dim × 4B` |

Ping-pong buffers (`outputA`/`outputB`) alternate every layer (even→A, odd→B) for residual addition without an extra copy.

### 1.3 Weight Buffer Pre-loading

**File:** `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` (~line 814)

On first `fusedPrefillPass` call, ALL layer weight buffers and the final norm + LM head weights are loaded from the `weights` dictionary into pre-loaded `MTLBuffer`s. This eliminates ~254 actor hops per subsequent forward pass. The loaded weights are stored in `PreloadedWeightsStore`:

```swift
struct PreloadedWeightsStore {
    var layers: [LayerWeightBuffers]  // wq, wk, wv, wo, gate, up, down, attnNorm, ffnNorm, qNorm, kNorm
    var finalNorm: MTLBuffer
    var lmHead: MTLBuffer?
    var lmHeadRaw: MTLBuffer?  // Q8_0 raw for fused path
    var lmHeadCols: Int
}
```

Quantized weights (Q8_0, Q4_K_M, etc.) are kept in "raw" `MTLBuffer`s (`wqRaw`, `wkRaw`, etc.) for direct fused dequant+GEMV kernels, avoiding a float32 intermediate.

---

## 2. GPU Residency Management

### 2.1 ResidencyManager

**File:** `Sources/EdgeRunnerMetal/ResidencyManager.swift`

Uses `MTLResidencySet` (Metal 4 feature, falls back gracefully on older GPUs):

```swift
let descriptor = MTLResidencySetDescriptor()
descriptor.initialCapacity = 256
let set = try device.makeResidencySet(descriptor: descriptor)
set.requestResidency()
commandQueue.addResidencySet(set)
```

**Responsibilities:**
- `addBuffer(_:)` — Adds a buffer to the residency set and commits
- `addHeap(_:)` — Adds a heap allocation to the set and commits

The residency set hints the GPU which buffers should be kept resident in GPU memory, reducing page fault overhead during dispatch. On Metal 4 (macOS 26+), this is used for the full decode pass with `m4CmdBuf.useResidencySet(state.residencySet)`.

---

## 3. KV Cache

**File:** `Sources/EdgeRunnerMetal/KVCache.swift`

### 3.1 Structure

Per-layer circular buffers for key and value tensors:

```swift
public final class KVCache: Sendable {
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision  // .float32, .float16, .float8

    private let keyBuffers: [MetalBufferHandle]   // [numLayers]
    private let valueBuffers: [MetalBufferHandle] // [numLayers]
    private let layerStates: Mutex<[LayerState]>  // Per-layer write position tracking
}
```

Buffer layout per layer:
```
bufferLength = maxSeqLen × numKVHeads × headDim × bytesPerElement
```

### 3.2 Circular Write Tracking

`LayerState` tracks position atomically via `Mutex`:

```swift
private struct LayerState: Sendable {
    var writePos = 0   // Current write position (circular)
    var totalWritten = 0  // Total tokens ever written (for overflow detection)
}
```

On GPU write, `advanceWritePosition(layer:count:)` is called to advance positions. On prefill reset, `setPosition(_:)` sets all layers to the same position.

### 3.3 Precision Support

| Precision | `ERKVPrecision` | Bytes/Element | Type |
|-----------|----------------|---------------|------|
| Float32 | `ERKVPrecisionFloat32` | 4 | `Float` |
| Float16 | `ERKVPrecisionFloat16` | 2 | `Float16` |
| Float8 | `ERKVPrecisionFloat8` | 1 | `UInt8` |

The KV cache is initialized as Float16 (`.float16`) in `LlamaLanguageModel`.

### 3.4 Rotation Handling

When `totalWritten > maxSeqLen`, the cache enters "rotated" mode. Retrieval (`retrieve(...)`) handles this by reading in two chunks: `[writePos...maxSeqLen]` then `[0...writePos]`. This correctly reconstructs the causal view for attention.

---

## 4. Command Buffer Management

### 4.1 CommandBatcher

**File:** `Sources/EdgeRunnerMetal/CommandBatcher.swift`

Manages a single "current" `MTLCommandBuffer` + `MTLComputeCommandEncoder` pair, flushing when a threshold is reached:

```swift
final class CommandBatcher {
    private let commandQueue: MTLCommandQueue
    private var currentCommandBuffer: MTLCommandBuffer?
    private var currentEncoder: MTLComputeCommandEncoder?
    private var currentOpCount: Int = 0

    private let maxOpsPerBuffer: Int  // apple9=50, apple8=40, other=30
}
```

**`encoder()`** — Returns the current encoder, creating a new command buffer if `currentOpCount >= maxOpsPerBuffer`.

**`flush()`** — Ends encoding, commits the command buffer, resets state. The GPU will execute asynchronously.

**`flushAndWait()`** — Same as flush but calls `waitUntilCompleted()` before returning. Used for `MetalBackend.synchronize()`.

### 4.2 MetalBackend Dispatch

**File:** `Sources/EdgeRunnerMetal/MetalBackend.swift` (actor)

`MetalBackend` is a `public actor` that serializes all GPU operations. Its `dispatch()` method:

```swift
private func dispatch(
    pipeline: MTLComputePipelineState,
    buffers: [(MTLBuffer, Int)],  // buffer + binding index
    threadgroups: MTLSize,
    threadsPerThreadgroup: MTLSize
) {
    let (_, encoder) = commandBatcher.encoder()

    // Insert barrier if any input buffer was previously written
    for (buffer, _) in buffers {
        barrierTracker.insertBarrierIfNeeded(forReading: buffer, encoder: encoder)
    }

    encoder.setComputePipelineState(pipeline)
    for (buffer, index) in buffers {
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

    // Mark last buffer as written (for subsequent barriers)
    if let (outBuffer, _) = buffers.last {
        barrierTracker.recordWrite(outBuffer)
    }
}
```

### 4.3 BarrierTracker

**File:** `Sources/EdgeRunnerMetal/BarrierTracker.swift`

Tracks which buffers have been written since the last `reset()`, and inserts `MTLBarrier` before reads to prevent RAW hazards:

```swift
final class BarrierTracker {
    private var writtenBuffers: Set<ObjectIdentifier> = []

    func needsBarrier(forReading buffer: MTLBuffer) -> Bool
    func recordWrite(_ buffer: MTLBuffer)
    func insertBarrierIfNeeded(forReading buffer: MTLBuffer, encoder: MTLComputeCommandEncoder)
    func reset()  // Called after flushAndWait (full GPU sync)
}
```

Barriers use `memoryBarrier(scope: .buffers)`. Reset is only called after `flushAndWait`, not after every `flush`, meaning barriers persist across batched dispatches within the same command buffer.

---

## 5. Prefill vs Decode: Command Buffer Split

### 5.1 Prefill — Single Command Buffer

**File:** `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` (`fusedPrefillPass`, ~line 778)

Full prefill (and full prefill after prefix reuse) encodes ALL transformer layers + final norm + LM head into **ONE command buffer with ONE encoder**:

```swift
guard let cmdBuf = commandQueue.makeCommandBuffer() else { ... }
guard let enc = cmdBuf.makeComputeCommandEncoder() else { ... }

// Layer loop encodes ~20-40 dispatches per layer into the SAME encoder
for layerIndex in 0..<config.layerCount {
    // RMSNorm
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(currentHidden, offset: 0, index: 0)
    enc.setBuffer(lw.attnNorm, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.dispatchThreads(...)

    // QKV GEMV (per token in sequence)
    for t in 0..<seqLen {
        enc.setComputePipelineState(fusedQ8PSO)
        enc.setBuffer(lw.wqRaw!, offset: 0, index: 0)
        // ...
        enc.dispatchThreadgroups(...)
    }

    // RoPE, GQA, projection, SwiGLU FFN...
}

// Final RMSNorm + LM head dispatch

enc.endEncoding()
cmdBuf.commit()
await cmdBuf.completed()
```

**Why one encoder:** Metal guarantees sequential execution + implicit barriers between dispatches within the same encoder. Creating 422 encoders (for a 40-layer model) would cost ~4.2ms in encoder overhead alone.

**Decode warmup:** After first prefill, 5 dummy decode passes run to warm the GPU JIT pipeline for decode-specific kernel variants, then KV cache is zeroed and prefill is re-run.

### 5.2 Decode — Three Paths

**File:** `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` (`runDecodePass`, ~line 693)

Single-token decode has three dispatch paths:

**Path 1: Fused Metal 4 (macOS 26+, preferred)**
- `fusedDecodePassMetal4()` — Uses `MTL4ComputeCommandEncoder` with argument table dispatch
- `setArgumentTable()` called ONCE, then only changed buffer addresses updated via `setAddress()`
- Pre-allocated 7-slot params ring buffer (256-byte aligned slots) for zero-copy param passing
- Execution-only barriers via `enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])`
- No cache flushes on unified memory

**Path 2: Fused Metal 3 Optimized**
- `fusedDecodePassOpt()` — Uses pre-allocated `decodeParamsBuffer` (7 × 256 bytes)
- Writes params once into ring buffer, then dispatches with `setBytes` from that buffer

**Path 3: Base (fallback)**
- `fusedDecodePass()` — Uses standard `GEMVKernel`, `RoPEKernel`, `GQAKernel` encoding
- Each kernel creates its own temporary buffers and dispatches independently

### 5.3 Prefix Reuse Mode

When the new sequence is a strict extension of the cached sequence, only the suffix tokens are processed. `commonPrefixLen` is computed by comparing tokens, and `kvCache.setPosition()` updates the write position without clearing existing KV data.

---

## 6. Kernel Registry

**File:** `Sources/EdgeRunnerMetal/KernelRegistry.swift`

Cached `MTLComputePipelineState` lookup:

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
2. Runtime compilation: concatenates all `.metal` files from bundle, compiles with `MTLCompileOptions`

Pipeline descriptor always sets `supportIndirectCommandBuffers = true`.

---

## 7. Quantized Weight Memory

### 7.1 Supported Quantization Types

| Type | Kernel | Format |
|------|--------|--------|
| Q4_0 | `DequantQ4_0Kernel` | 4-bit, 1 block size |
| Q4_K_M | `DequantQ4KMKernel` | 4-bit, mixed block size |
| Q5_0 | `DequantQ5_0Kernel` | 5-bit, 1 block size |
| Q5_1 | `DequantQ5_1Kernel` | 5-bit, 1 block size |
| Q5_K | `DequantQ5KKernel` | 5-bit, mixed block |
| Q6_K | `DequantQ6KKernel` | 6-bit, mixed block |
| Q3_K | `DequantQ3KKernel` | 3-bit, mixed block |
| Q2_K | `DequantQ2KKernel` | 2-bit, mixed block |
| Q8_0 | `DequantQ8_0Kernel` | 8-bit, 1 block size |

### 7.2 Fused Dequant + GEMV

Rather than separate dequantization then GEMV, fused kernels (`dequant_q8_0_gemv`, `dequant_q8_0_fused_qkv`, `dequant_q8_0_gemv_tiled`) dequantize directly during the matrix multiply, achieving 3.8x bandwidth reduction for Q8_0 vs naive separate passes.

---

## 8. Memory Model Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     MetalBackend (actor)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ BufferCache  │  │CommandBatcher│  │ BarrierTracker │    │
│  │  (LRU, 50%   │  │  (≤50 ops/   │  │ (write tracking│    │
│  │  VRAM)       │  │  cmdbuf)     │  │  for barriers) │    │
│  └──────────────┘  └──────────────┘  └────────────────┘    │
│  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │ResidencySet  │  │      KernelRegistry (pipeline    │  │
│  │(MTLResidency │  │       cache, Mutex-protected)    │  │
│  │  Set)        │  │                                  │  │
│  └──────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌────────────────────┐
│  KVCache      │   │  ScratchBuffers │   │  PreloadedWeights  │
│  (per-layer   │   │  (19 persistent │   │  (layer weights,   │
│   K/V rings)  │   │   maxSeqLen-    │   │   Q8_0 raw bufs)   │
│               │   │   sized)        │   │                    │
│  Float16 only │   │                 │   │                    │
└───────────────┘   └─────────────────┘   └────────────────────┘
```

### Allocation Strategy

| Category | Allocation | Recycling |
|----------|-----------|-----------|
| KV cache | At model init, never freed | Never recycled (persistent) |
| Scratch buffers | At model init, never freed | Never recycled (persistent) |
| Weight buffers | At model init or first prefill | Never recycled (persistent) |
| Short-lived temps | `BufferCache.acquire()` | Returned to `BufferCache.recycle()` |
| Standalone kernel temps | `device.makeBuffer()` per call | Not recycled |

### Thread Safety

- `MetalBackend` — `actor` (Swift actor isolation)
- `BufferCache` — `Mutex<CacheState>`
- `KVCache` — `Mutex<[LayerState]>`
- `KernelRegistry` — `Mutex<PipelineCache>`
- `BarrierTracker` — **Not thread-safe** — caller must ensure single-threaded access per batch (guaranteed by actor)

---

## 9. Compute Dispatch Flow

### Full Prefill
```
CPU: embed tokens → memcpy to hiddenBuf
      ↓
GPU: [encoder] RMSNorm → QKV GEMV × seqLen → RoPE → GQA → Proj → FFN (SwiGLU)
      × numLayers (one encoder, sequential dispatches)
      ↓
GPU: [encoder] Final RMSNorm → LM head GEMV → logits
      ↓
CPU: await completion → argmax → next token
```

### Single-Token Decode
```
CPU: embed token → memcpy to decodeHidden
      ↓
GPU: [encoder] FusedQKV → MegaKernel(Q/K Norm + RoPE + GQA) → ProjAdd → FFN → DownAdd
      × numLayers (Metal 4: arg table, 1 setArgumentTable + N setAddress)
      ↓
GPU: [encoder] Final RMSNorm → LM head → logits
      ↓
CPU: await completion → argmax → next token
      ↓
GPU: (async) KV cache write continues from previous command buffer
```
