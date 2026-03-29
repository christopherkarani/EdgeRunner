#include <metal_stdlib>
using namespace metal;

struct ERDequantQ8_0Params {
    uint blockCount;
    uint outputOffset;
};

constant uint q8_0BlockBytes = 34;
constant uint q8_0WeightsPerBlock = 32;

kernel void dequant_q8_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ8_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;
    device const uchar* block = input + (tid * q8_0BlockBytes);
    float scale = float(as_type<half>(*(device const ushort*)block));
    uint outputBase = params.outputOffset + (tid * q8_0WeightsPerBlock);
    for (uint index = 0; index < q8_0WeightsPerBlock; index++) {
        output[outputBase + index] = scale * float(as_type<char>(block[2 + index]));
    }
}

// === High-Performance Fused Q8_0 GEMV ===
// Architecture: 32 threads (1 simdgroup) per threadgroup, 2 rows per TG.
// Each thread processes 1 full Q8_0 block (32 elements) per iteration.
// Single simd_sum reduction — no cross-simdgroup overhead.
//
// y[row] = sum_k dequant(W_q8[row, k]) * x[k]

struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

kernel void dequant_q8_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    // Pointers to weight rows
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    // Main loop: each thread handles 1 full block (32 elements) per iteration
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        // Cache x in registers (reused across LOCAL_NR rows)
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);

            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    // Single simd_sum — no cross-SG reduction needed
    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// === Fused RMSNorm + Q+K+V projection ===
// Single dispatch replaces: RMSNorm + QKV GEMV (saves 1 dispatch per layer).
// Computes: normed = RMSNorm(x, normWeight, eps), then Q/K/V = dequant(W) * normed
// RMSNorm is computed cooperatively: each threadgroup calculates the normalization
// factor from the input, then uses it while processing weight blocks.

struct ERFusedQKVParams {
    uint qRows;          // Q output rows (numHeads * headDim)
    uint kvRows;         // K/V output rows (numKVHeads * headDim)
    uint cols;           // input columns (dim)
    uint blocksPerRow;   // Q8_0 blocks per row
    uint tokenCount;     // number of input tokens in the batch
    float rmsEps;        // RMSNorm epsilon
};

kernel void dequant_q8_0_fused_qkv(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],     // V writes f16 directly to cache
    device const float* normWeight [[buffer(8)]],  // RMSNorm weight [cols]
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    // Total rows = qRows + kvRows + kvRows. Each threadgroup handles LOCAL_NR consecutive rows
    // across the concatenated Q/K/V output space.
    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device half* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    // Determine which weight matrix each row belongs to
    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    // === Cooperative RMSNorm: compute normalization factor ===
    // Each thread sums squares for its assigned blocks, then simd_sum reduces.
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    // === Main GEMV loop with inline RMSNorm ===
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        // Apply RMSNorm inline: normed_x = x * rmsScale * normWeight
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[r] += sumq * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// === Fused RMSNorm + Gate+Up+SwiGLU ===
// Single dispatch replaces: FFN RMSNorm + gate GEMV + up GEMV + SwiGLU (saves 1 dispatch per layer).
// Computes: normed = RMSNorm(x, normWeight, eps)
//           activated[row] = silu(gate_proj[row]) * up_proj[row]
// where gate_proj = dequant(Wg) * normed, up_proj = dequant(Wu) * normed

inline float silu_fn(float x) { return x / (1.0f + exp(-x)); }

struct ERFusedGateUpSiluParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
    uint tokenCount;
    float rmsEps;
};

kernel void dequant_q8_0_fused_final_norm_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const float* normWeight [[buffer(3)]],
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);

            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

kernel void dequant_q8_0_fused_gate_up_silu(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(5)]],  // RMSNorm weight [cols]
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= params.rows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenActivated = activated + tokenIndex * params.rows;

    // === Cooperative RMSNorm ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumGate[LOCAL_NR] = { 0.f };
    float sumUp[LOCAL_NR] = { 0.f };

    device const uchar* axGate[LOCAL_NR];
    device const uchar* axUp[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint row = row0 + r;
        uint safeRow = row < params.rows ? row : row0;
        axGate[r] = wGate + safeRow * nb * q8_0BlockBytes;
        axUp[r] = wUp + safeRow * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        // Apply RMSNorm inline
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;

            // Gate
            device const uchar* blockG = axGate[r] + ib * q8_0BlockBytes;
            float scaleG = float(as_type<half>(*(device const ushort*)blockG));
            device const char* qsG = (device const char*)(blockG + 2);
            float sqG = 0.f;
            for (short i = 0; i < 32; i++) sqG += float(qsG[i]) * xl[i];
            sumGate[r] += sqG * scaleG;

            // Up
            device const uchar* blockU = axUp[r] + ib * q8_0BlockBytes;
            float scaleU = float(as_type<half>(*(device const ushort*)blockU));
            device const char* qsU = (device const char*)(blockU + 2);
            float sqU = 0.f;
            for (short i = 0; i < 32; i++) sqU += float(qsU[i]) * xl[i];
            sumUp[r] += sqU * scaleU;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumGate[r] = simd_sum(sumGate[r]);
        sumUp[r] = simd_sum(sumUp[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;
            tokenActivated[row0 + r] = silu_fn(sumGate[r]) * sumUp[r];
        }
    }
}

// === Float16 output variant — writes half directly to KV cache ===
// Eliminates separate f32->f16 conversion dispatch.
kernel void dequant_q8_0_gemv_f16out(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = half(sumf[row]);
        }
    }
}

// === GEMV + Residual Add fused — y[i] = sum_k dequant(W[i,k])*x[k] + residual[i] ===
// Eliminates separate elementwise_add dispatch after output/down projections.
kernel void dequant_q8_0_gemv_add(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ERDequantQ8GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row] + residual[row0 + row];  // fused add!
        }
    }
}

// =============================================================================
// === Tile-Based GEMV with Coalesced Memory Access ============================
// =============================================================================
// This kernel restructures Q8_0 GEMV to use 2D tile-based access patterns.
// Problem: Current kernel has each thread loading x[] directly from DRAM with
// strided access (each thread loads elements 32 apart -> 32 separate memory
// streams per simdgroup, causing DRAM row buffer thrashing).
// Solution: All 32 threads cooperatively load a contiguous tile of x[] (1024
// elements) into threadgroup memory, then access from fast SRAM with coalesced
// patterns.
//
// Expected improvement: 207 GB/s -> 250+ GB/s (20% bandwidth increase)

kernel void dequant_q8_0_gemv_tiled(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    constexpr uint TILE_SIZE = 1024;  // 4KB, fits comfortably in threadgroup memory

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const uint tilesPerRow = (params.cols + TILE_SIZE - 1) / TILE_SIZE;

    float sumf[LOCAL_NR] = { 0.f };

    // Pointers to weight rows
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    // Threadgroup memory for the tile - shared across all 32 threads
    threadgroup float tile[TILE_SIZE];

    // Process the row in tiles
    for (uint tileIdx = 0; tileIdx < tilesPerRow; tileIdx++) {
        const uint tileOffset = tileIdx * TILE_SIZE;
        const uint remainingCols = params.cols - tileOffset;
        const uint tileLen = min(TILE_SIZE, remainingCols);

        // === Phase 1: Cooperatively load tile into threadgroup memory ===
        // All 32 threads participate in loading the tile contiguously
        // This creates coalesced memory access patterns
        for (uint i = tiisg; i < tileLen; i += 32) {
            tile[i] = x[tileOffset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Phase 2: Process Q8_0 blocks within this tile ===
        // Each thread processes blocks that fall within the current tile
        // A Q8_0 block covers 32 elements, so we process blocks [tileStartBlock, tileEndBlock)
        const uint tileStartBlock = (tileOffset) / 32;
        const uint tileEndBlock = min((tileOffset + tileLen + 31) / 32, (uint)nb);

        for (short ib = tileStartBlock + tiisg; ib < tileEndBlock; ib += 32) {
            // Calculate position within tile for this block
            const uint blockStartInTile = (ib * 32) - tileOffset;

            // Load x values for this block from tile (fast SRAM access)
            float xl[32];
            for (short i = 0; i < 32; i++) {
                uint tilePos = blockStartInTile + i;
                if (tilePos < TILE_SIZE) {
                    xl[i] = tile[tilePos];
                } else {
                    // Fallback to device memory for edge cases (shouldn't happen with proper tile sizing)
                    xl[i] = x[ib * 32 + i];
                }
            }

            // Process this block for all rows
            for (short row = 0; row < LOCAL_NR; row++) {
                if (row0 + row >= params.rows) break;

                device const uchar* block = ax[row] + ib * q8_0BlockBytes;
                float scale = float(as_type<half>(*(device const ushort*)block));
                device const char* qs = (device const char*)(block + 2);

                float sumq = 0.f;
                for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
                sumf[row] += sumq * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Single simd_sum reduction
    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// =============================================================================
// === f16 Accumulation Variant of Tiled GEMV ==================================
// =============================================================================
kernel void dequant_q8_0_gemv_tiled_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    constexpr uint TILE_SIZE = 1024;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const uint tilesPerRow = (params.cols + TILE_SIZE - 1) / TILE_SIZE;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    threadgroup float tile[TILE_SIZE];

    for (uint tileIdx = 0; tileIdx < tilesPerRow; tileIdx++) {
        const uint tileOffset = tileIdx * TILE_SIZE;
        const uint remainingCols = params.cols - tileOffset;
        const uint tileLen = min(TILE_SIZE, remainingCols);

        // Cooperatively load tile
        for (uint i = tiisg; i < tileLen; i += 32) {
            tile[i] = x[tileOffset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tileStartBlock = (tileOffset) / 32;
        const uint tileEndBlock = min((tileOffset + tileLen + 31) / 32, (uint)nb);

        for (short ib = tileStartBlock + tiisg; ib < tileEndBlock; ib += 32) {
            const uint blockStartInTile = (ib * 32) - tileOffset;

            half xl[32];
            for (short i = 0; i < 32; i++) {
                uint tilePos = blockStartInTile + i;
                if (tilePos < TILE_SIZE) {
                    xl[i] = half(tile[tilePos]);
                } else {
                    xl[i] = half(x[ib * 32 + i]);
                }
            }

            for (short row = 0; row < LOCAL_NR; row++) {
                if (row0 + row >= params.rows) break;

                device const uchar* block = ax[row] + ib * q8_0BlockBytes;
                float scale = float(as_type<half>(*(device const ushort*)block));
                device const char* qs = (device const char*)(block + 2);

                half sumq = 0.h;
                for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
                sumf[row] += float(sumq) * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}
// These kernels use half-precision for the inner dot product (xl[] cache and
// per-block sumq), halving register pressure and doubling ALU throughput on
// Apple Silicon. The outer cross-block accumulator (sumf[]) stays float32 to
// prevent drift over hundreds of blocks.

// --- 1. dequant_q8_0_gemv_f16acc ---
kernel void dequant_q8_0_gemv_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// TurboQuant decode variant: same fused RMSNorm + Q/K/V projection, but V stays in f32
// so it can be packed into TurboQuant immediately after RoPE(K) without falling back to
// separate RMSNorm + Q/K/V GEMV dispatches.
kernel void dequant_q8_0_fused_qkv_turbo(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device float* outV [[buffer(6)]],
    device const float* normWeight [[buffer(8)]],
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device float* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[r] += sumq * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumf[r] = simd_sum(sumf[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = total;
            }
        }
    }
}

// --- 2. dequant_q8_0_fused_qkv_f16acc ---
kernel void dequant_q8_0_fused_qkv_f16acc(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],
    device const float* normWeight [[buffer(8)]],
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device half* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    // === Cooperative RMSNorm: stays in f32 (runs once, not on hot path) ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    // === Main GEMV loop: f16 inner accumulation ===
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = half(xb[i] * rmsScale * normWeight[ib * 32 + i]);
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[r] += float(sumq) * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// --- 3. dequant_q8_0_fused_gate_up_silu_f16acc ---
kernel void dequant_q8_0_fused_gate_up_silu_f16acc(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(5)]],
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= params.rows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenActivated = activated + tokenIndex * params.rows;

    // === Cooperative RMSNorm: stays in f32 ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumGate[LOCAL_NR] = { 0.f };
    float sumUp[LOCAL_NR] = { 0.f };

    device const uchar* axGate[LOCAL_NR];
    device const uchar* axUp[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint row = row0 + r;
        uint safeRow = row < params.rows ? row : row0;
        axGate[r] = wGate + safeRow * nb * q8_0BlockBytes;
        axUp[r] = wUp + safeRow * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = half(xb[i] * rmsScale * normWeight[ib * 32 + i]);
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;

            // Gate
            device const uchar* blockG = axGate[r] + ib * q8_0BlockBytes;
            float scaleG = float(as_type<half>(*(device const ushort*)blockG));
            device const char* qsG = (device const char*)(blockG + 2);
            half sqG = 0.h;
            for (short i = 0; i < 32; i++) sqG += half(qsG[i]) * xl[i];
            sumGate[r] += float(sqG) * scaleG;

            // Up
            device const uchar* blockU = axUp[r] + ib * q8_0BlockBytes;
            float scaleU = float(as_type<half>(*(device const ushort*)blockU));
            device const char* qsU = (device const char*)(blockU + 2);
            half sqU = 0.h;
            for (short i = 0; i < 32; i++) sqU += half(qsU[i]) * xl[i];
            sumUp[r] += float(sqU) * scaleU;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumGate[r] = simd_sum(sumGate[r]);
        sumUp[r] = simd_sum(sumUp[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;
            tokenActivated[row0 + r] = silu_fn(sumGate[r]) * sumUp[r];
        }
    }
}

// --- 4. dequant_q8_0_gemv_f16out_f16acc ---
kernel void dequant_q8_0_gemv_f16out_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = half(sumf[row]);
        }
    }
}

// =============================================================================
// === Fused FFN Block (Mega-Kernel) ==========================================
// =============================================================================
// Merges 3 GPU dispatches into 1 per transformer layer:
//   Phase 1: Wo GEMV + residual add     (replaces Dispatch 3)
//   Phase 2: RMSNorm                     (was implicit in Dispatch 4)
//   Phase 3: Gate + Up + SwiGLU GEMV     (replaces Dispatch 4)
//   Phase 4: Down GEMV + residual add    (replaces Dispatch 5)
//
// Architecture: 1 threadgroup x 1024 threads (32 simdgroups).
// Dispatched once per layer (28 times total).
// Uses threadgroup_barrier instead of pipeline drains between phases.

struct ERFusedFFNBlockParams {
    uint dim;               // 1024 (Wo output rows, Down output rows)
    uint qDim;              // 2048 (Wo input cols = attn output dim)
    uint interDim;          // 3072 (Gate/Up output rows, Down input cols)
    uint woBlocksPerRow;    // qDim/32 = 64
    uint ffnBlocksPerRow;   // dim/32 = 32
    uint downBlocksPerRow;  // interDim/32 = 96
    float rmsEps;           // RMSNorm epsilon
};

kernel void dequant_q8_0_fused_ffn_block(
    device const uchar*  woRaw       [[buffer(0)]],   // Wo weights Q8_0 [dim x qDim]
    device const float*  attnOut     [[buffer(1)]],   // attention output [qDim]
    device const float*  residual    [[buffer(2)]],   // currentHidden (residual for Wo add) [dim]
    device       float*  afterAttn   [[buffer(3)]],   // Wo output dest (also FFN residual) [dim]
    device const uchar*  gateRaw     [[buffer(4)]],   // gate weights Q8_0 [interDim x dim]
    device const uchar*  upRaw       [[buffer(5)]],   // up weights Q8_0 [interDim x dim]
    device const float*  normWeight  [[buffer(6)]],   // FFN RMSNorm weight [dim]
    device       float*  activBuf    [[buffer(7)]],   // intermediate activated [interDim]
    device const uchar*  downRaw     [[buffer(8)]],   // down weights Q8_0 [dim x interDim]
    device       float*  layerOutput [[buffer(9)]],   // final output [dim]
    constant ERFusedFFNBlockParams& params [[buffer(10)]],
    uint  tid    [[thread_index_in_threadgroup]],
    ushort sgIdx [[simdgroup_index_in_threadgroup]],
    ushort laneIdx [[thread_index_in_simdgroup]]
) {
    // Shared memory for cross-simdgroup RMSNorm reduction (32 simdgroups)
    threadgroup float partial_sums[32];

    const uint dim      = params.dim;        // 1024
    const uint qDim     = params.qDim;       // 2048
    const uint interDim = params.interDim;   // 3072

    // =========================================================================
    // Phase 1: Wo GEMV + residual add
    //   afterAttn[i] = dot(Wo[i,:], attnOut[:]) + residual[i]
    //   Each thread computes 1 output row (tid < dim=1024, all threads active)
    // =========================================================================
    {
        const uint woNb = params.woBlocksPerRow;  // qDim/32 = 64
        device const uchar* rowPtr = woRaw + tid * woNb * q8_0BlockBytes;

        float acc = 0.0f;
        for (uint ib = 0; ib < woNb; ib++) {
            device const uchar* block = rowPtr + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            device const float* xb = attnOut + ib * q8_0WeightsPerBlock;

            half sumq = 0.h;
            for (ushort j = 0; j < 32; j++) {
                sumq += half(qs[j]) * half(xb[j]);
            }
            acc += float(sumq) * scale;
        }

        afterAttn[tid] = acc + residual[tid];
    }

    // =========================================================================
    // Phase 2: Barrier — all 1024 Wo outputs must be visible
    // =========================================================================
    threadgroup_barrier(mem_flags::mem_device);

    // =========================================================================
    // Phase 3: Cooperative RMSNorm over afterAttn[dim=1024]
    //   normed[i] = afterAttn[i] * scale * normWeight[i]
    //   where scale = rsqrt(mean(afterAttn^2) + eps)
    //
    //   Each thread reads afterAttn[tid], squares it.
    //   Reduce within simdgroup via simd_sum, then cross-SG via threadgroup mem.
    // =========================================================================
    float myVal = afterAttn[tid];
    float mySq  = myVal * myVal;

    // Intra-simdgroup reduction
    float sgSum = simd_sum(mySq);

    // Cross-simdgroup reduction: lane 0 of each SG writes to shared mem
    if (laneIdx == 0) {
        partial_sums[sgIdx] = sgSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the 32 partial sums
    float totalSq = 0.0f;
    if (sgIdx == 0) {
        totalSq = (laneIdx < 32) ? partial_sums[laneIdx] : 0.0f;
        totalSq = simd_sum(totalSq);
    }

    // Broadcast the total to all threads via shared memory
    if (tid == 0) {
        partial_sums[0] = totalSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    totalSq = partial_sums[0];

    float rmsScale = rsqrt(totalSq / float(dim) + params.rmsEps);
    float normed = myVal * rmsScale * normWeight[tid];

    // =========================================================================
    // Phase 4: Gate + Up + SwiGLU GEMV
    //   interDim=3072 output rows, 1024 threads => 3 rows per thread
    //   For each assigned row r:
    //     gate = dot(Wgate[r,:], normed_vec[:])
    //     up   = dot(Wup[r,:],   normed_vec[:])
    //     activBuf[r] = silu(gate) * up
    //
    //   Problem: normed_vec is distributed (each thread holds 1 element).
    //   Solution: store normed to device memory, barrier, read back.
    // =========================================================================

    // Write normed value to afterAttn (reuse as temporary — we still have myVal
    // for Phase 6 residual, and afterAttn is the FFN residual anyway)
    afterAttn[tid] = normed;
    threadgroup_barrier(mem_flags::mem_device);

    // Each thread computes 3 rows of gate+up
    const uint rowsPerThread = interDim / dim;  // 3072/1024 = 3
    const uint ffnNb = params.ffnBlocksPerRow;  // dim/32 = 32

    for (uint r = 0; r < rowsPerThread; r++) {
        uint row = tid * rowsPerThread + r;
        if (row >= interDim) break;

        device const uchar* gateRow = gateRaw + row * ffnNb * q8_0BlockBytes;
        device const uchar* upRow   = upRaw   + row * ffnNb * q8_0BlockBytes;

        float accGate = 0.0f;
        float accUp   = 0.0f;

        for (uint ib = 0; ib < ffnNb; ib++) {
            device const float* nb_x = afterAttn + ib * q8_0WeightsPerBlock;

            // Gate block
            device const uchar* gBlock = gateRow + ib * q8_0BlockBytes;
            float gScale = float(as_type<half>(*(device const ushort*)gBlock));
            device const char* gQs = (device const char*)(gBlock + 2);

            // Up block
            device const uchar* uBlock = upRow + ib * q8_0BlockBytes;
            float uScale = float(as_type<half>(*(device const ushort*)uBlock));
            device const char* uQs = (device const char*)(uBlock + 2);

            half sqG = 0.h;
            half sqU = 0.h;
            for (ushort j = 0; j < 32; j++) {
                half xv = half(nb_x[j]);
                sqG += half(gQs[j]) * xv;
                sqU += half(uQs[j]) * xv;
            }
            accGate += float(sqG) * gScale;
            accUp   += float(sqU) * uScale;
        }

        activBuf[row] = silu_fn(accGate) * accUp;
    }

    // =========================================================================
    // Phase 5: Barrier — all interDim=3072 activated values must be visible
    // =========================================================================
    threadgroup_barrier(mem_flags::mem_device);

    // =========================================================================
    // Phase 6: Down GEMV + residual add
    //   layerOutput[i] = dot(Wdown[i,:], activBuf[:]) + afterAttn_before_norm[i]
    //   Each thread computes 1 output row (tid < dim=1024)
    //   Residual is the Wo output + old residual, which we stored in afterAttn
    //   before we overwrote it with normed values.
    //
    //   Wait — we overwrote afterAttn with normed in Phase 4.
    //   We need the pre-norm value for the residual.
    //   Solution: use (acc + residual[tid]) which we computed in Phase 1.
    //   We saved myVal = afterAttn[tid] before norming, and
    //   afterAttn[tid] was (acc + residual[tid]) from Phase 1.
    //   So myVal IS the correct residual for Phase 6.
    // =========================================================================
    {
        const uint downNb = params.downBlocksPerRow;  // interDim/32 = 96
        device const uchar* rowPtr = downRaw + tid * downNb * q8_0BlockBytes;

        float acc = 0.0f;
        for (uint ib = 0; ib < downNb; ib++) {
            device const uchar* block = rowPtr + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            device const float* xb = activBuf + ib * q8_0WeightsPerBlock;

            half sumq = 0.h;
            for (ushort j = 0; j < 32; j++) {
                sumq += half(qs[j]) * half(xb[j]);
            }
            acc += float(sumq) * scale;
        }

        // myVal holds the Phase 1 output (Wo GEMV + residual) = correct FFN residual
        layerOutput[tid] = acc + myVal;
    }
}

// --- 5. dequant_q8_0_gemv_add_f16acc ---
kernel void dequant_q8_0_gemv_add_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ERDequantQ8GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row] + residual[row0 + row];
        }
    }
}
