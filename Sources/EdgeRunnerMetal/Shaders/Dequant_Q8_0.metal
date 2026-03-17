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

// === Fused Q+K+V projection — 3 outputs from 1 input in a single dispatch ===
// Reads input x once, multiplies by 3 different weight matrices, writes 3 outputs.
// Eliminates 2 dispatches per layer (3->1) and shares x-value cache across all 3 projections.

struct ERFusedQKVParams {
    uint qRows;          // Q output rows (numHeads * headDim)
    uint kvRows;         // K/V output rows (numKVHeads * headDim)
    uint cols;           // input columns (dim)
    uint blocksPerRow;   // Q8_0 blocks per row
};

kernel void dequant_q8_0_fused_qkv(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],     // V writes f16 directly to cache
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    // Total rows = qRows + kvRows + kvRows. Each threadgroup handles LOCAL_NR consecutive rows
    // across the concatenated Q/K/V output space.
    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= totalRows) return;

    const short nb = params.blocksPerRow;

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

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

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
                outQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                outK[globalRow - params.qRows] = total;
            } else {
                outV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// === Fused Gate+Up+SwiGLU — 2 GEMVs + activation in 1 dispatch ===
// Computes: activated[row] = silu(gate_proj[row]) * up_proj[row]
// where gate_proj = dequant(Wg) * x, up_proj = dequant(Wu) * x
// Eliminates 2 dispatches per layer (gate + up + swiglu -> 1).

inline float silu_fn(float x) { return x / (1.0f + exp(-x)); }

kernel void dequant_q8_0_fused_gate_up_silu(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    constant ERDequantQ8GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

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
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

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
            activated[row0 + r] = silu_fn(sumGate[r]) * sumUp[r];
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
