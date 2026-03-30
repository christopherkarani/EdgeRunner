# EdgeRunner Metal Shaders & Compute Kernels

**Author:** Metal Shader Research Agent
**Date:** 2026-03-25
**Sources:** All `.metal` files in `Sources/EdgeRunnerMetal/Shaders/`, all headers in `Sources/EdgeRunnerSharedTypes/include/`

---

## Table of Contents

1. [Quantization Format Reference](#1-quantization-format-reference)
2. [Dequantization Kernels (Plain)](#2-dequantization-kernels-plain)
3. [Q8_0 GEMV Family (Fused Dequant + MatVec)](#3-q8_0-gemv-family-fused-dequant--matvec)
4. [GEMM & GEMV Kernels](#4-gemm--gemv-kernels)
5. [Attention Kernels](#5-attention-kernels)
6. [Normalization Kernels](#6-normalization-kernels)
7. [RoPE Kernels](#7-rope-kernels)
8. [Elementwise, Activation & Reduction](#8-elementwise-activation--reduction)
9. [Mega-Kernels & Fused Patterns](#9-mega-kernels--fused-patterns)
10. [Hardware Optimizations Summary](#10-hardware-optimizations-summary)

---

## 1. Quantization Format Reference

EdgeRunner supports **10 quantization formats** across three families:

### 1.1 Plain Formats (no super-block structure)

| Format | Block Size | Weights/Block | Bits/Weight | Scale | Zero-point |
|--------|-----------|---------------|-------------|-------|------------|
| Q8_0 | 34 bytes | 32 | 8 | f16 | none |
| Q4_0 | 18 bytes | 32 | 4 | f16 | -8 offset |
| Q5_0 | 24 bytes | 32 | 5 | f16 | none |
| Q5_1 | 24 bytes | 32 | 5 | f16 (d,m) | m offset |

### 1.2 K-Quants Family (super-block structure, 256 weights/superblock)

| Format | SuperBlock Bytes | Bits/Weight | Scales | Notes |
|--------|-----------------|-------------|--------|-------|
| Q2_K | 84 | ~2.06 | 16 sub-block (sc,m) | |
| Q3_K | 110 | ~3.06 | 16 sub-block (signed) | |
| Q4_K_M | 144 | ~4.52 | 8 sub-block + min | "medium" |
| Q5_K | 176 | ~5.53 | 8 sub-block + min | |
| Q6_K | 210 | ~6.56 | 16 sub-block (int8) | |

### 1.3 K-Quants Memory Layout

All K-quants formats use a **super-block** of 256 quantized weights, with sub-block scales packed into the first ~100 bytes. The exact layout varies per format:

```
Q4_K_M (144 bytes superblock):
  [0..1]    d (f16 master scale)
  [2..3]    dmin (f16)
  [4..11]   8 scale bytes (6 bits each, high bits from [12..15])
  [12..15]  high bits for scales
  [16..143] 128 bytes nibble-packed weights (256 nibbles)
  -- 8 sub-blocks × 32 weights each --
  -- each sub-block: 1 scale byte + 1 min byte + 32 nibbles (16 bytes) --

Q5_K (176 bytes superblock):
  [0..1]    d (f16)
  [2..3]    dmin (f16)
  [4..15]   scale/min packing (same scheme as Q4_K_M)
  [16..47]  qh bit-field (5th bit of each weight)
  [48..175] ql nibble-packed (lower 4 bits)
  -- Each weight: q5 = (ql & 0x0F) | (qh_bit << 4) --

Q6_K (210 bytes superblock):
  [0..127]  ql nibble-packed (lower 4 bits of each weight)
  [128..191] qh (upper 2 bits, 4 weights per byte)
  [192..207] scales (signed int8, 16 sub-blocks × 16 weights each)
  [208..209] d (f16)
  -- q6 = (ql & 0x0F) | ((qh >> (i%4)*2) & 0x03) << 4 --

Q3_K (110 bytes superblock):
  [0..31]   hmask (high bit of each weight)
  [32..95]  qs nibble-packed (lower 2 bits)
  [96..107] scale nibbles (6 bits each, 16 sub-blocks)
  [104..109] upper bits for scales
  [108..109] d (f16)
  -- q3 = (qs & 0x03) | (hmask_bit << 2) --

Q2_K (84 bytes superblock):
  [0..15]   scale/metadata (sc | m<<4 per sub-block)
  [16..79]  qs (2 bits per weight, 4 per byte)
  [80..81]  d (f16)
  [82..83]  dmin (f16)
  -- q2 = (qs >> (i%4)*2) & 0x03 --
```

---

## 2. Dequantization Kernels (Plain)

All plain dequant kernels use:
- `[[thread_position_in_grid]]` — 1D grid, one thread per block
- `[[buffer(0)]]` — device uchar* input (quantized)
- `[[buffer(1)]]` — device float* output
- `[[buffer(2)]]` — constant params struct

### 2.1 `dequant_q8_0` — Q8_0 Dequant

**Block layout:** 34 bytes = 2 bytes scale (f16) + 32 bytes int8

```metal
struct ERDequantQ8_0Params {
    uint blockCount;
    uint outputOffset;
};
constant uint q8_0BlockBytes = 34;
constant uint q8_0WeightsPerBlock = 32;
```

**Threadgroup:** None (1 thread per block, 1D grid)

**Math per thread:**
```
scale = f16_to_float(*(ushort*)block)          // reinterpret scale bytes as f16
for i in 0..31:
    output[outputBase + i] = scale * int8(block[2 + i])
```

**Key details:**
- Uses `as_type<half>()` reinterpret cast for scale (f16 → f32)
- Quantized values are **signed int8** (range -128..127), multiplied directly by scale
- No zero-point offset (pure scaling)
- Grid size = `blockCount`, each thread fully independent

### 2.2 `dequant_q4_0` — Q4_0 Dequant

**Block layout:** 18 bytes = 2 bytes scale (f16) + 16 bytes nibble-packed

```metal
constant uint Q4_0_BLOCK_BYTES = 18;
constant uint Q4_0_BLOCK_WEIGHTS = 32;
```

**Math:**
```
scale = f16_to_float(*reinterpret_cast<half*>(block))
for i in 0..15:
    packed = block[2 + i]
    low  = (packed & 0x0F) - 8   // signed 4-bit: -8..7
    high = (packed >> 4) - 8
    output[outBase + i]     = scale * float(low)
    output[outBase + i+16]  = scale * float(high)
```

**Key details:**
- Zero-point offset of -8 baked into the nibble value
- Two nibbles unpacked per byte, processed in one loop iteration
- Grid size = `params.blockCount`, processes 32 weights per thread

### 2.3 `dequant_q5_0` — Q5_0 Dequant

**Block layout:** 24 bytes = 2 bytes d (f16) + 2 bytes m (f16) + 8 bytes qs + 4 bytes qh

```metal
constant uint Q5_1_BLOCK_BYTES = 24;
constant uint Q5_1_WEIGHTS_PER_BLOCK = 32;
```

**Math:**
```
d = f16(block[0..1])
m = f16(block[2..3])
for i in 0..31:
    qs_nibble = block[8 + i/2]     // nibble from qs
    qh_bit    = block[4 + i/8] >> (i%8) & 1  // bit from qh
    q5 = (qs_nibble & 0x0F) | (qh_bit << 4)  // 5-bit value
    output[outBase + i] = d * float(q5) + m
```

### 2.4 `dequant_q5_1` — Q5_1 Dequant

Same as Q5_0 but uses `as_type<half>()` reinterpret cast:
```
d = as_type<half>(*(ushort*)block)
m = as_type<half>(*(ushort*)(block+2))
output[i] = d * q5 + m   // additive zero-point
```

---

## 3. Q8_0 GEMV Family (Fused Dequant + MatVec)

The Q8_0 GEMV family is the **most sophisticated kernel family**, with multiple variants covering every placement in the transformer architecture. Each kernel fuses dequantization with the matrix-vector product, eliminating a separate materialization pass.

### 3.1 Base Variant: `dequant_q8_0_gemv`

**Interface:**
```metal
struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};
// buffer(0): uchar* quantizedW  (Q8_0 weights)
// buffer(1): float* x           (input vector)
// buffer(2): float* y           (output vector)
// buffer(3): constant params
```

**Threadgroup layout:**
- `[[threadgroup_position_in_grid]]` → `uint tgIndex` (row group index)
- `[[thread_index_in_simdgroup]]` → `ushort tiisg` (0..31)
- **Threads per threadgroup: 32** (1 full SIMD group)
- **Rows per threadgroup: 2** (`LOCAL_NR = 2`)
- Grid: `(rows + 1) / 2` threadgroups

**Execution model:**
```
row0 = tgIndex * LOCAL_NR          // e.g., threadgroup 0 → rows 0,1

for each block ib in [tiisg, nb, 32]:  // strided: 0,32,64,...
    // x cache: each thread loads 1 element, but 32 threads cover 32 contiguous elements
    xb = x + ib * 32
    xl[i] = xb[i]  for i in 0..31    // register cache

    for row in 0..1:
        block = ax[row] + ib * q8_0BlockBytes
        scale = f16(block[0..1])
        qs = block[2..33]            // 32 int8 values
        sumq = sum_i(qs[i] * xl[i])  // dot product
        sumf[row] += sumq * scale

// SIMD reduction (32 → 1 value per row)
sumf[row] = simd_sum(sumf[row])

// lane 0 writes final output
if (tiisg == 0):
    y[row0 + row] = sumf[row]
```

**Key optimizations:**
1. **Register caching of x**: `xl[32]` lives in registers, not memory. Each thread block iteration reuses this cache for both rows.
2. **Two rows per threadgroup**: `LOCAL_NR = 2` amortizes x loading across two GEMV rows
3. **Single simd_sum reduction**: No threadgroup_barrier for reduction — SIMD group does it in hardware
4. **Safe row clamping**: `uint r = row0 + row; ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes` — bounds-check to avoid out-of-bounds reads
5. **Strided block iteration**: `for (short ib = tiisg; ib < nb; ib += 32)` — each thread iterates over non-overlapping blocks

### 3.2 F16-Accumulation Variant: `dequant_q8_0_gemv_f16acc`

**Difference from base:** Inner dot product (`sumq`) uses `half` accumulation instead of `float`.

```metal
half sumq = 0.h;
for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
sumf[row] += float(sumq) * scale;
```

**Why it works:**
- Apple Silicon has 2:1 f16:f32 ALU ratio (each f16 op is half-rate)
- `half` dot product accumulator halves register pressure and doubles throughput
- Outer `sumf[row]` stays float32 to prevent accumulation drift over many blocks

### 3.3 F16-Output Variant: `dequant_q8_0_gemv_f16out`

Writes `half` directly to KV cache, eliminating the separate f32→f16 conversion dispatch:
```metal
y[row0 + row] = half(sumf[row]);  // writes buffer(2) as device half*
```

### 3.4 Residual-Add Variant: `dequant_q8_0_gemv_add`

Fuses the residual addition into the GEMV output:
```metal
y[row0 + row] = sumf[row] + residual[row0 + row];
```
Used for output/down projections that need residual connections.

### 3.5 Fused QKV Kernel: `dequant_q8_0_fused_qkv`

Single dispatch replaces: RMSNorm + Q KV GEMV (3 dispatches) → 1 dispatch.

```metal
struct ERFusedQKVParams {
    uint qRows;       // numHeads * headDim
    uint kvRows;      // numKVHeads * headDim
    uint cols;        // dim
    uint blocksPerRow;
    float rmsEps;
};
```

**Execution:**
1. Cooperative RMSNorm: each thread sums squares for its blocks, `simd_sum` across 32 lanes
2. `rmsScale = rsqrt(sumSq / cols + eps)`
3. Main loop: normed_x = x * rmsScale * normWeight (inline)
4. GEMV for Q, K, V simultaneously (Q and K → float, V → half for KV cache)
5. Output routing by row index range

**Threadgroup layout:** Same as base GEMV (32 threads, 2 rows per TG)

### 3.6 Fused Gate+Up+SwiGLU: `dequant_q8_0_fused_gate_up_silu`

Single dispatch replaces: RMSNorm + Gate GEMV + Up GEMV + SwiGLU (4 dispatches) → 1 dispatch.

```metal
silu_fn(x) = x / (1.0f + exp(-x))  // SiLU / Swish

// Computes: activated = silu(gate) * up
// where gate = dequant(Wg) * normed, up = dequant(Wu) * normed
```

**Two-pass structure:**
- Phase 1: Cooperative RMSNorm (same as fused QKV)
- Phase 2: Gate GEMV + Up GEMV in same loop, then SiLU fusion
  - `sumGate[row] = simd_sum(...)`
  - `sumUp[row] = simd_sum(...)`
  - `activated[row0 + row] = silu_fn(sumGate[r]) * sumUp[r]`

### 3.7 Tiled GEMV: `dequant_q8_0_gemv_tiled`

Addresses DRAM row-buffer thrashing from strided x[] access.

**Problem with base GEMV:**
- Each thread accesses `x[ib * 32 + i]` with stride 32 across threads
- 32 separate memory streams → DRAM row buffer thrashing
- Theoretical bandwidth: 207 GB/s

**Solution: 2D tile-based access**
- Threadgroup memory: `threadgroup float tile[1024]` (4KB, fits in SRAM)
- All 32 threads cooperatively load a **contiguous 1024-element tile** of x[]
- Then process Q8_0 blocks within the tile from fast SRAM

```metal
threadgroup float tile[TILE_SIZE];  // 1024

// Phase 1: cooperative load into threadgroup memory
for (uint i = tiisg; i < tileLen; i += 32):
    tile[i] = x[tileOffset + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 2: process blocks within tile (fast SRAM access)
for (ib in tileStartBlock..tileEndBlock):
    xl[i] = tile[blockStartInTile + i];  // fast!
    // ... GEMV ...
threadgroup_barrier(mem_flags::mem_threadgroup);
```

**Expected improvement:** 207 GB/s → 250+ GB/s (20% bandwidth increase)

### 3.8 Mega-Kernel: `dequant_q8_0_fused_ffn_block`

The crown jewel: **one dispatch** replaces 5 GPU dispatches per layer:
1. Wo GEMV + residual add
2. RMSNorm
3. Gate GEMV + Up GEMV + SwiGLU
4. Down GEMV + residual add

**Threadgroup layout:** 1024 threads per threadgroup (32 simdgroups × 32 threads)

```metal
// Phase 1: Wo GEMV + residual add (threads 0..1023 each compute 1 row)
{
    // Each thread = 1 output row
    rowPtr = woRaw + tid * woNb * q8_0BlockBytes
    acc = 0.0f
    for ib in 0..woNb:
        // half f16 accumulation
        sumq = sum_j(qs[j] * xb[j])  // half
        acc += float(sumq) * scale
    afterAttn[tid] = acc + residual[tid]
}
threadgroup_barrier(mem_flags::mem_device)

// Phase 2: Cooperative RMSNorm over 1024 elements
// simd_sum per SG → threadgroup partial_sums[32] → simd_sum → broadcast
myVal = afterAttn[tid]
mySq = myVal * myVal
sgSum = simd_sum(mySq)
if (laneIdx == 0) partial_sums[sgIdx] = sgSum
threadgroup_barrier(mem_flags::mem_threadgroup)
// First SG reduces 32 partial sums → totalSq → rmsScale
totalSq = simd_sum(partial_sums[laneIdx])  // within SG0
afterAttn[tid] = normed  // overwrite with normed values
threadgroup_barrier(mem_flags::mem_device)

// Phase 3: Gate + Up + SwiGLU (threads each compute 3 rows: interDim/dim = 3)
for r in 0..2:
    gate = GEMV(Wgate, normed)
    up   = GEMV(Wup, normed)
    activBuf[row] = silu(gate) * up
threadgroup_barrier(mem_flags::mem_device)

// Phase 4: Down GEMV + residual add
{
    // myVal still holds pre-norm afterAttn = correct FFN residual
    layerOutput[tid] = downGEMV + myVal
}
```

**Cross-simdgroup RMSNorm detail:**
- 32 simdgroups × 32 threads = 1024 threads
- Each SG computes `simd_sum` → 32 partial sums in `threadgroup float partial_sums[32]`
- SG0 then reduces these 32 values via `simd_sum`
- Result broadcast via threadgroup memory

---

## 4. GEMM & GEMV Kernels

### 4.1 `gemm_f32` / `gemm_f16`

Naive element-wise GEMM. Grid: `uint2 gid [[thread_position_in_grid]]`.

```metal
// gemm_f32
uint row = gid.y;  uint col = gid.x;
float sum = 0.0;
for (uint k = 0; k < params.K; k++):
    sum += A[row*lda + k] * B[k*ldb + col];
C[row*ldc + col] = sum;

// gemm_f16: same but half sum, A/B/C as half*
```

**Threadgroup:** None — 1 thread per output element. Naive implementation with no tiling or shared memory.

### 4.2 `gemv_f32` / `gemv_f16`

Threadgroup: 256 threads per row (8 SIMD groups × 32 threads).

```metal
uint row = group_id;
float partial = 0.0f;
for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW):
    partial += a_row[j] * x[j];

partial = simd_sum(partial);  // warp-level reduction

// Cross-warp reduction via threadgroup memory
threadgroup float shared_sums[32];
if (simd_lane == 0) shared_sums[simd_group] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_group == 0) {
    float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
    val = simd_sum(val);
    if (simd_lane == 0) y[row] = val;
}
```

**Reduction tree:**
- 256 threads → 8 × `simd_sum(4 elements each)` → 8 partials in shared memory
- Threadgroup barrier
- SIMD group 0 reduces 8 partials → final scalar
- Lane 0 writes result

### 4.3 `dequant_q4_0_gemv`

Same architecture as `dequant_q8_0_gemv` but for Q4_0 weights.

```metal
// 32 threads per threadgroup, 2 rows per TG, strided block iteration
for (uint blockIndex = localID; blockIndex < params.blocksPerRow; blockIndex += 32):
    // Dequant Q4_0 block: 2 bytes scale + nibble-packed weights
    scale = f16(*reinterpret_cast<half*>(block))
    for i in 0..15:
        packed = block[2+i]
        low  = float(int(packed & 0x0F) - 8)
        high = float(int(packed >> 4) - 8)
        partial += low  * x[colBase + i]
        partial += high * x[colBase + i + 16]

partial = simd_sum(partial);
if (simdLane == 0) y[row] = partial;
```

---

## 5. Attention Kernels

### 5.1 `flash_attention_f32`

Implements **block-wise online softmax** (Flash Attention algorithm).

**Threadgroup layout:**
- `[[threadgroup_position_in_grid]]` → `group_id` (Q row group)
- `[[thread_position_in_threadgroup]]` → `local_id` (thread within group)
- Threadgroup size: `br` (Q block size, typically 16)
- Grid: `(seqLen + br - 1) / br` Q blocks

**Threadgroup memory:**
```metal
threadgroup float kTile[16 * 128];    // blockSize × headDim
threadgroup float vTile[16 * 128];
threadgroup float outputScratch[16 * 128];
```

**Algorithm (per Q row):**
```
runningMax = -∞, runningSum = 0.0
init outputScratch to 0 (each thread zeros its headDim elements)

for each kvBlock:
    // Load K,V tile cooperatively (all threads participate)
    kTile[local_id * headDim + dim] = K[(kvStart+local_id)*headDim + dim]
    vTile[local_id * headDim + dim] = V[(kvStart+local_id)*headDim + dim]
    threadgroup_barrier

    // Compute block-wise attention scores
    for kvIndex in 0..kvCount:
        if causal && (kvStart+kvIndex > qRow): scores[kvIndex] = -∞
        else: dot = sum_dim(Q[qRow] * kTile[kvIndex])
               scores[kvIndex] = dot * scale

    blockMax = max(scores[:])

    // Online softmax correction
    nextMax = max(runningMax, blockMax)
    correction = exp(runningMax - nextMax)

    // Compute exp(scores - nextMax) and sum
    for kvIndex: probs[kvIndex] = exp(scores[kvIndex] - nextMax)
    blockSum = sum(probs[:])

    runningSum = runningSum * correction + blockSum

    // Update output accumulator
    for dim:
        value = outputScratch[local_id*headDim + dim] * correction
        for kvIndex: value += probs[kvIndex] * vTile[kvIndex*headDim + dim]
        outputScratch[...] = value

    runningMax = nextMax
    threadgroup_barrier

// Final normalization
invSum = 1.0f / runningSum
O[qRow*headDim + dim] = outputScratch[local_id*headDim + dim] * invSum
```

**Key optimizations:**
1. **Online softmax**: Avoids materializing the full exp matrix; `runningMax` and `runningSum` maintained incrementally
2. **Block-wise processing**: K,V loaded into fast threadgroup memory (SRAM)
3. **Cooperative K,V loading**: All threads participate regardless of Q validity, avoiding divergence at barrier
4. **Fused exp+sum**: computed together before weight accumulation

### 5.2 `gqa_attention_f32` / `gqa_attention_f16kv`

Grouped Query Attention with block-wise online softmax. Same overall structure as Flash Attention but with GQA grouping.

**Threadgroup layout:** `uint2 group_id [[threadgroup_position_in_grid]]`
- `group_id.x` = Q block index
- `group_id.y` = head index
- Threadgroup size: `blockSize × 1` (second dimension is 1D)

```metal
uint qRow = qBlockIndex * blockSize + local_id.x;
uint kvHeadIndex = headIndex / params.groupSize;  // GQA grouping
```

**f16kv variant:** K,V stored as `half` in KV cache, converted to float inside the kernel:
```metal
kTile[local_id.x * headDim + dim] = float(K[kBase + dim]);
vTile[local_id.x * headDim + dim] = float(V[kBase + dim]);
```
This halves KV cache bandwidth at the cost of f16→f32 conversion inside the attention kernel.

---

## 6. Normalization Kernels

### 6.1 `rmsnorm_f32` — Naive RMSNorm

1 thread per row, serial reduction over cols.

```metal
// RMSNorm formula: output[i] = input[i] * scale * weight[i]
// scale = rsqrt(mean(input^2) + eps)
//       = rsqrt(sum(input^2) / cols + eps)

meanSq = sum_i(input[offset + i]^2) / cols
scale = rsqrt(meanSq + eps)
for i: output[offset + i] = input[offset + i] * scale * weight[i]
```

**Naive**: O(cols) serial reduction. For large cols (e.g., 4096), this wastes parallelism when rows=1.

### 6.2 `rmsnorm_parallel_f32` — Parallel RMSNorm for Decode

Optimized for the single-row decode path (rows=1). Uses 256 threads (8 SIMDgroups) cooperatively.

```metal
// Threadgroup: (rows, 1) grid. 8 simdgroups × 32 threads = 256 threads per row.
uint row = tgid;
uint tid = sgitg * 32 + tiisg;   // 0..255 within row
const uint stride = 256;

// Phase 1: Parallel sum-of-squares
localSumSq = 0.0f;
for (col = tid; col < cols; col += stride):
    v = input[offset + col];
    localSumSq += v * v;

localSumSq = simd_sum(localSumSq);  // within simdgroup

// Cross-SG reduction via threadgroup memory
threadgroup float tg_partial[8];
if (tiisg == 0) tg_partial[sgitg] = localSumSq;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute scale once
if (sgitg == 0 && tiisg == 0):
    total = sum_i(tg_partial[i])
    tg_partial[0] = rsqrt(total / float(cols) + eps)
threadgroup_barrier(mem_flags::mem_threadgroup);
scale = tg_partial[0];

// Phase 2: Parallel scale+weight multiply
for (col = tid; col < cols; col += stride):
    output[offset + col] = input[offset + col] * scale * weight[col];
```

### 6.3 `layernorm_f32`

Standard LayerNorm with mean+variance:

```metal
mean = sum_i(input[offset+i]) / cols
variance = sum_i((input[offset+i] - mean)^2) / cols
invStd = rsqrt(variance + eps)
for i: output[i] = (input[i] - mean) * invStd * gamma[i] + beta[i]
```

---

## 7. RoPE Kernels

### 7.1 `rope_f32` — Standard RoPE (Llama-style)

Threadgrid: `uint3 tid [[thread_position_in_grid]]` with dims `(halfDim, numHeads, seqLen)`.

```metal
uint dimPair = tid.x;  // 0..31 for headDim=64
uint head = tid.y;
uint seq = tid.z;

// RoPE formula: rotate pairs (2d, 2d+1) by angle based on position and frequency
exponent = float(2*dimPair) / float(headDim)
frequency = 1.0f / powr(theta, exponent)      // hardware-optimized reciprocal
angle = float(seq + startPos) * (frequency / scalingFactor)
cosValue = cos(angle)
sinValue = sin(angle)

baseIndex = (seq*numHeads*headDim) + (head*headDim) + (2*dimPair)
x0 = input[baseIndex]
x1 = input[baseIndex + 1]
output[baseIndex]     = x0*cosValue - x1*sinValue
output[baseIndex + 1] = x0*sinValue + x1*cosValue
```

**Key detail:** Uses `powr(base, exponent)` instead of `pow(base, exponent)`. `powr` is hardware-optimized for `1/pow(x,y)` with reciprocal, yielding higher throughput on Apple Silicon GPUs.

### 7.2 `rope_neox_f32` — NeoX-style RoPE (Qwen, StableLM)

Same threadgrid as standard RoPE but pairs `(d, d+halfDim)` instead of `(2d, 2d+1)`:

```metal
headBase = (seq*numHeads*headDim) + (head*headDim)
x0 = input[headBase + dimPair]
x1 = input[headBase + dimPair + halfDim]
output[headBase + dimPair]           = x0*cos - x1*sin
output[headBase + dimPair + halfDim] = x0*sin + x1*cos
```

### 7.3 `fused_qk_norm_rope_neox` — Fused Q/K Norm + RoPE

Single dispatch replaces 4: Q norm + K norm + RoPE Q + RoPE K→f16.

Threadgrid: `uint2 tid [[thread_position_in_grid]]` → `(halfDim, numHeads+numKVHeads)`

**Per-head RMSNorm:** 32 threads per head (dimPair 0..31). Two simdgroups per head when halfDim=64.

```metal
bool isQ = headIdx < numHeads;
raw0 = src[hb + dimPair];
raw1 = src[hb + dimPair + halfDim];

// Per-head RMSNorm across 128 dims (2 simdgroups per head)
pairSq = raw0^2 + raw1^2;
sumSq = simd_sum(pairSq);  // within simdgroup
// cross-SG reduction
if (dimPair % 32 == 0) tgSq[headIdx * 2 + sgIdx] = sumSq;
threadgroup_barrier
sumSq = tgSq[headIdx*2] + tgSq[headIdx*2+1];

rs = rsqrt(sumSq / headDim + eps);
x0 = raw0 * rs * nw[dimPair];
x1 = raw1 * rs * nw[dimPair + halfDim];

// RoPE
freq = 1.0f / pow(theta, exp);   // Note: uses pow here (non-critical path)
angle = startPos * (freq / scaling);
c = cos(angle), s = sin(angle);
o0 = x0*c - x1*s;  o1 = x0*s + x1*c;

if (isQ) outQ[hb+...] = o0/o1;
else     outK[hb+...] = half(o0/o1);  // K→f16 for KV cache
```

### 7.4 `fused_qk_norm_rope_gqa` — Ultra-Fused Norm+RoPE+GQA

**No threadgroup barriers at all** — pure SIMD reductions. Replaces 2 dispatches.

Threadgrid: `(32, numHeads+numKVHeads)`. **32 threads per head (one full SIMDgroup)**.

Each thread processes **4 elements**: positions `[dimIdx, dimIdx+32, dimIdx+64, dimIdx+96]` of a 128-dim head vector.

**Phase 1: Per-head RMSNorm + RoPE** (4 elements per thread)
```metal
// NeoX pairs: (i, i+halfDim) and (i+32, i+32+halfDim)
raw_a0 = src[hb + dimIdx];          // position i
raw_a1 = src[hb + dimIdx+halfDim];   // position i+64
raw_b0 = src[hb + dimIdx+32];        // position i+32
raw_b1 = src[hb + dimIdx+32+halfDim]; // position i+96

// sumSq: 32 threads × 4 elements each = 128-dim complete coverage
sq = raw_a0^2 + raw_a1^2 + raw_b0^2 + raw_b1^2;
sumSq = simd_sum(sq);  // 32 threads cover 128 dims — NO barrier!

rs = rsqrt(sumSq / headDim + eps);
x_a0 = raw_a0 * rs * nw[dimIdx];  // norm applied to all 4 positions
x_a1 = raw_a1 * rs * nw[dimIdx+halfDim];
x_b0 = raw_b0 * rs * nw[dimIdx+32];
x_b1 = raw_b1 * rs * nw[dimIdx+32+halfDim];

// RoPE applied to pair_a (frequency from dimIdx) and pair_b (freq from dimIdx+32)
q_a0 = x_a0*ca - x_a1*sa;  q_a1 = x_a0*sa + x_a1*ca;
q_b0 = x_b0*cb - x_b1*sb;  q_b1 = x_b0*sb + x_b1*cb;

// K heads: write to cache and exit
if (!isQ):
    kCache[cacheBase + ...] = half(q_a0/q_a1/q_b0/q_b1)
    return;  // K threads done — no barrier needed!
```

**Phase 2: GQA attention** (Q threads only, 32 threads)
```metal
// 32 threads cooperatively compute full 128-dim dot product per kv position
// Each thread handles 4 elements: simd_sum gives full dot with ZERO barriers
for kv in 0..kvSeqLen:
    dk = kCache[kv*stride + kvHead*headDim + dimIdx ... dimIdx+32+halfDim]
    partial = q_a0*dk_a0 + q_a1*dk_a1 + q_b0*dk_b0 + q_b1*dk_b1
    score = simd_sum(partial) * attnScale  // full 128-dim dot, NO barrier!

    // Online softmax: lane 0 computes correction, broadcast via simd_broadcast_first
    if (dimIdx == 0):
        nextRunMax = max(runMax, score)
        correction = exp(runMax - nextRunMax)
        prob = exp(score - nextRunMax)
        nextRunSum = runSum * correction + prob
    runMax = simd_broadcast_first(nextRunMax)
    runSum = simd_broadcast_first(nextRunSum)
    // accumulate V
    acc_a0 += prob * vCache[...]; ...
```

---

## 8. Elementwise, Activation & Reduction

### 8.1 Elementwise Kernels

Each kernel: 1 thread per element, 1D grid. Straightforward.

```metal
// add, sub, mul, div — float and half variants
out[tid] = a[tid] op b[tid];

// f32↔f16 conversion
out[tid] = half(input[tid]);   // or float(input[tid])
```

### 8.2 Activation Kernels

```metal
// sigmoid: 1/(1+exp(-x))
output[gid] = 1.0f / (1.0f + exp(-value));

// GELU (exact): 0.5 * x * (1 + tanh(0.79788... * (x + 0.044715 * x^3)))
const float coefficient = 0.7978845608028654f;
output = value * 0.5f * (1.0f + tanh(coefficient * (value + 0.044715f * value^3)));

// SwiGLU: silu(gate) * up
output[gid] = silu(gate[gid]) * up[gid];
// silu(x) = x / (1 + exp(-x))
```

### 8.3 Fused Add/Activate: `Elementwise.metal`

Uses `function_constant(0)` for compile-time activation selection:
```metal
constant int activation_type [[function_constant(0)]];
// Compiler eliminates dead branches → zero runtime overhead
```

Supports: none (0), relu (1), sigmoid (2), gelu (3), silu (4).

### 8.4 Reduction Kernels

```metal
// Sum/Mean: serial reduction per output element
reduce_sum_float:  sum_i(input[base+i])
reduce_mean_float: sum_i(input[base+i]) / reductionSize

// Max: serial max per output element
reduce_max_float: max_i(input[base+i])
```

Grid: `outerSize` threads. Each thread processes one output element.

### 8.5 Softmax: `softmax_f32`

Full SIMD + threadgroup reduction tree:

```metal
// Threadgroup: 256 threads (8 simdgroups × 32)
// Grid: rows (one threadgroup per row)

// Step 1: Find row max
threadMax = max(rowIn[local_id, local_id+256, ...])  // strided
threadMax = simd_max(threadMax)
if (simd_lane == 0) shared[simd_group] = threadMax
threadgroup_barrier

// Cross-SG reduction (SG0 reduces 8 partials)
if (simd_group == 0):
    value = simd_lane < 8 ? shared[simd_lane] : -INFINITY
    value = simd_max(value)
    shared[0] = value
threadgroup_barrier
rowMax = shared[0]

// Step 2: Compute exp and sum
threadSum = sum_i(exp(rowIn[i] - rowMax))
threadSum = simd_sum(threadSum)
if (simd_lane == 0) shared[simd_group] = threadSum
threadgroup_barrier

if (simd_group == 0):
    value = simd_lane < 8 ? shared[simd_lane] : 0.0f
    value = simd_sum(value)
    shared[0] = value
threadgroup_barrier
rowSum = shared[0]
invSum = 1.0f / rowSum

// Step 3: Normalize
rowOut[i] = exp(rowIn[i] - rowMax) * invSum
```

### 8.6 StitchableOps: `[[stitchable]]` Functions

Metal 2.3+ runtime function stitching. These are linkable function symbols:

```metal
[[stitchable]] float op_relu_float(float x)    { return max(x, 0.0f); }
[[stitchable]] float op_silu_float(float x)    { return x / (1.0f + exp(-x)); }
[[stitchable]] float op_gelu_float(float x)   { return 0.5f * x * (1.0f + tanh(...)); }
[[stitchable]] float op_add_float(float a, float b)  { return a + b; }
[[stitchable]] float op_mul_float(float a, float b)  { return a * b; }
// ... (11 total stitchable ops)
```

These compose into pipelines via `MTLFunctionStitching` at runtime without recompiling shader source.

---

## 9. Mega-Kernels & Fused Patterns

### 9.1 Fusion Architecture Overview

EdgeRunner's Metal shaders follow a **progressive fusion strategy**:

| Dispatches Replaced | Kernel | Saves |
|--------------------|--------|-------|
| 3 | `dequant_q8_0_fused_qkv` | RMSNorm + Q GEMV + K GEMV + V GEMV |
| 4 | `dequant_q8_0_fused_gate_up_silu` | RMSNorm + Gate GEMV + Up GEMV + SwiGLU |
| 5 | `dequant_q8_0_fused_ffn_block` | Wo GEMV + add + RMSNorm + Gate/Up/SwiGLU + Down GEMV + add |
| 4 | `fused_qk_norm_rope_neox` | Q RMSNorm + K RMSNorm + RoPE Q + RoPE K→f16 |
| 2 | `fused_qk_norm_rope_gqa` | fused_qk_norm_rope + GQA attention |

### 9.2 FFN Mega-Kernel: Full Layer Breakdown

The `dequant_q8_0_fused_ffn_block` kernel is the most complex:

```
Phase 1 (1024 threads): Wo GEMV + residual
  Thread tid computes: afterAttn[tid] = Wo[tid,:] · attnOut + residual[tid]

Phase 2 (1024 threads): Cross-simdgroup barrier + Cooperative RMSNorm
  Each SG (32 threads): simd_sum of squares → partial_sums[sgIdx]
  SG0: reduce 32 partials → rmsScale
  Broadcast via threadgroup memory

Phase 3 (1024 threads): Write normed, barrier, Gate+Up+SwiGLU GEMVs
  Each thread: 3 rows of Gate/Up GEMV, then silu(gate)*up
  Wait: normed was distributed (1 element per thread)
  Solution: write to afterAttn[tid], barrier, read back as xb

Phase 4 (1024 threads): Barrier, Down GEMV + residual add
  afterAttn[tid] (pre-norm) + Down[tid,:] · activBuf → layerOutput[tid]
```

---

## 10. Hardware Optimizations Summary

### 10.1 `powr` vs `pow`

In `rope_f32`:
```metal
float frequency = 1.0f / powr(params.theta, exponent);  // GOOD: hardware reciprocal
float frequency = 1.0f / pow(params.theta, exponent);   // BAD: generic pow
```

`powr(base, exp)` computes `base^exp` with `exp` treated as a reciprocal internally, enabling hardware-optimized reciprocal units. On Apple Silicon, `powr` is significantly faster than `pow` for the `1/pow(x, y)` pattern used in RoPE frequency computation.

Note: `fused_qk_norm_rope_neox` uses `pow` for RoPE (non-critical path), while `rope_f32`/`rope_neox_f32` use `powr`.

### 10.2 SIMD Intrinsics Used

| Intrinsic | Operation | Use Case |
|-----------|-----------|----------|
| `simd_sum(x)` | 32-lane sum reduction | GEMV accumulation, RMSNorm |
| `simd_max(x)` | 32-lane max reduction | Softmax max, attention scores |
| `simd_broadcast_first(x)` | Copy lane 0 to all lanes | Online softmax broadcast |
| `as_type<T>(x)` | Bit reinterpretation | f16↔f32 scale reinterpret |
| `rsqrt(x)` | Fast 1/sqrt | RMSNorm scale, normalization |

### 10.3 Memory Coalescing Patterns

| Kernel | Pattern | Why |
|--------|---------|-----|
| Plain dequant | Contiguous block read | 1 thread reads 1 block, no striding |
| Base GEMV | x[] strided, W[] contiguous | x striding is unavoidable for GEMV; W is row-major |
| Tiled GEMV | Cooperative contiguous x tile → SRAM | Solves strided x[] DRAM thrashing |
| Flash Attention | K,V loaded as contiguous tiles | Threadgroup memory tiles of [block×headDim] |
| RoPE | 3D grid, contiguous per-thread | Each thread accesses 2 consecutive floats |

### 10.4 Threadgroup Memory Usage

| Kernel | Threadgroup Memory | Purpose |
|--------|-------------------|---------|
| Flash Attention | `kTile[16×128] + vTile[16×128] + output[16×128]` | K,V tile + output scratch |
| GQA Attention | Same as Flash Attention | Same |
| Tiled GEMV | `tile[1024]` | x[] input tile (4KB) |
| `rmsnorm_parallel_f32` | `tg_partial[8]` | Cross-SG reduction |
| `softmax_f32` | `shared[32]` | Cross-SG max/sum reduction |
| `gemv_f32` | `shared_sums[32]` | Cross-warp sum reduction |
| Fused FFN Block | `partial_sums[32]` | Cross-simdgroup RMSNorm |
| `fused_qk_norm_rope_neox` | `tgSq[48]` | Cross-SG RMSNorm (2 SG/head) |

### 10.5 Half-Precision Strategy

EdgeRunner uses half-precision strategically:

| Where | Precision | Rationale |
|-------|-----------|-----------|
| KV Cache | half | 50% memory, acceptable precision loss |
| Q8_0 scale | half | 2-byte scale, sufficient range |
| Inner GEMV dot (f16acc variants) | half | 2× ALU throughput, Apple Silicon has 2:1 ratio |
| Outer GEMV accumulator | float | Accumulate over 32-100+ blocks without drift |
| Attention scores | float | Numerical stability in exp() |
| Softmax probs | float | Numerical stability |
| Output projections | float (or half for cache) | Final accuracy |

### 10.6 Quantization Format Bit-Precision Reference

| Format | Bits/Weight | Block Overhead | Effective Bits | Use Case |
|--------|-------------|----------------|----------------|----------|
| Q8_0 | 8 | 0.5 bits (scale) | ~7.5 | Baseline, highest quality |
| Q4_0 | 4 | 0.5 bits (scale) | ~3.5 | Good quality, 2× compression |
| Q5_0 | 5 | 1 bit (d,m) | ~4 | Better than Q4, still compact |
| Q5_1 | 5 | 1 bit (d,m) | ~4 | Similar to Q5_0, different zero-point |
| Q2_K | ~2.06 | ~0.53 bits | ~1.5 | Very aggressive, testing |
| Q3_K | ~3.06 | ~0.69 bits | ~2.4 | Medium-low quality |
| Q4_K_M | ~4.52 | ~0.56 bits | ~3.96 | Good balance (GGUF default) |
| Q5_K | ~5.53 | ~0.69 bits | ~4.84 | Better than Q4_K |
| Q6_K | ~6.56 | ~0.44 bits | ~6.1 | Near Q8 quality |

**Block overhead calculation:** `(scale bytes) / weights_per_block × 8` bits per weight.

---

## Appendix: Header Files Reference

### DequantParams.h
```c
ERDequantParams         { blockCount, outputOffset }           // Q8_0, Q4_0, Q5_0, Q5_1
ERDequantGEMVParams     { rows, cols, blocksPerRow }             // plain GEMV dequant
ERDequantQ4KParams      { superBlockCount, outputOffset }        // K-quants variants
ERDequantQ8GEMVTiledParams { rows, cols, blocksPerRow, tileSize, tilesPerRow }
```

### AttentionParams.h
```c
ERFlashAttentionParams  { seqLen, headDim, scale, causal, kvBlockSize, qBlockSize }
ERGQAParams             { seqLen, headDim, numHeads, numKVHeads, groupSize,
                          scale, causal, kvBlockSize, qBlockSize, kvSeqLen, qOffset }
```

### RoPEParams.h
```c
ERRoPEParams            { seqLen, numHeads, headDim, startPos, theta, scalingFactor }
```

### GEMMParams.h / GEMVParams.h
```c
ERGEMMParams            { M, N, K, lda, ldb, ldc }
ERGEMVParams            { M, K, lda }
```
