#include <metal_stdlib>
using namespace metal;

struct ERDequantQ1_0_g128Params {
    uint blockCount;
    uint outputOffset;
    uint scaleByteOffset;
    uint bitDataOffset;
    uint bitOrderMSBFirst;
    uint oneBitIsNegative;
};

struct ERDequantQ1_0_g128GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint q1_0_g128BlockBytes = 18;
constant uint q1_0_g128WeightsPerBlock = 128;

// ─── Shared v2 helper ────────────────────────────────────────────────
// Computes the bit-selected sum B for one 4-byte (32-weight) sub-block.
// Returns: scale × (2×B - xSum), the dot product contribution.
inline float q1_subblock_dot(
    thread const float* xl, // 32 cached x (or normed-x) values in thread address space
    float xSum,             // precomputed sum of xl[0..31]
    device const uchar* qs, // 4 bytes of Q1 bit data for this sub-block
    float scale             // Q1 block scale (shared across 4 sub-blocks)
) {
    float B = 0.f;
    for (short bi = 0; bi < 4; bi++) {
        uchar bits = qs[bi];
        const short base = bi * 8;
        // Vectorized: extract bit masks → float4, dot with x values
        float4 x0 = float4(xl[base+0], xl[base+1], xl[base+2], xl[base+3]);
        float4 x1 = float4(xl[base+4], xl[base+5], xl[base+6], xl[base+7]);
        float4 m0 = float4(float((bits>>0)&1), float((bits>>1)&1),
                           float((bits>>2)&1), float((bits>>3)&1));
        float4 m1 = float4(float((bits>>4)&1), float((bits>>5)&1),
                           float((bits>>6)&1), float((bits>>7)&1));
        B += dot(m0, x0) + dot(m1, x1);
    }
    return scale * (2.f * B - xSum);
}

// ─── Basic dequantization kernel ─────────────────────────────────────

kernel void dequant_q1_0_g128(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ1_0_g128Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;
    device const uchar* block = input + (tid * q1_0_g128BlockBytes);
    uint scaleOffset = min(params.scaleByteOffset, q1_0_g128BlockBytes - 2);
    float scale = float(as_type<half>(*(device const ushort*)(block + scaleOffset)));
    uint outputBase = params.outputOffset + (tid * q1_0_g128WeightsPerBlock);
    for (uint byteIndex = 0; byteIndex < 16; byteIndex++) {
        uint bitOffset = min(params.bitDataOffset + byteIndex, q1_0_g128BlockBytes - 1);
        uchar bits = block[bitOffset];
        uint baseOutput = outputBase + (byteIndex * 8);
        for (uint bitIndex = 0; bitIndex < 8; bitIndex++) {
            uint shift = params.bitOrderMSBFirst != 0 ? (7u - bitIndex) : bitIndex;
            uchar bit = (bits >> shift) & 1;
            bool positive = params.oneBitIsNegative != 0 ? (bit == 0) : (bit != 0);
            output[baseOutput + bitIndex] = positive ? scale : -scale;
        }
    }
}

// ─── v1 GEMV (legacy, kept for EDGERUNNER_Q1_USE_V2_KERNEL=0 fallback) ──

kernel void dequant_q1_0_g128_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;
    const uint nb = params.blocksPerRow;
    float sumf[LOCAL_NR] = { 0.f };
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }
    for (uint ib = tiisg; ib < nb; ib += 32) {
        uint xBaseIdx = ib * q1_0_g128WeightsPerBlock;
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block + 0)));
            device const uchar* qs = block + 2;
            float sumq = 0.f;
            for (uint byteIdx = 0; byteIdx < 16; byteIdx++) {
                uchar bits = qs[byteIdx];
                uint elemBase = xBaseIdx + byteIdx * 8;
                float4 xv0 = float4(x[elemBase], x[elemBase+1], x[elemBase+2], x[elemBase+3]);
                float4 xv1 = float4(x[elemBase+4], x[elemBase+5], x[elemBase+6], x[elemBase+7]);
                float4 w0 = float4(
                    ((bits >> 0) & 1) ? scale : -scale,
                    ((bits >> 1) & 1) ? scale : -scale,
                    ((bits >> 2) & 1) ? scale : -scale,
                    ((bits >> 3) & 1) ? scale : -scale
                );
                float4 w1 = float4(
                    ((bits >> 4) & 1) ? scale : -scale,
                    ((bits >> 5) & 1) ? scale : -scale,
                    ((bits >> 6) & 1) ? scale : -scale,
                    ((bits >> 7) & 1) ? scale : -scale
                );
                sumq += dot(w0, xv0) + dot(w1, xv1);
            }
            sumf[row] += sumq;
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

// ─── v2 standalone GEMV (sign-flip + sub-block granularity) ──────────

kernel void dequant_q1_0_g128_gemv_v2(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }

    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        device const float* xp = x + xBase;
        for (short i = 0; i < 32; i++) { xl[i] = xp[i]; xSum += xl[i]; }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[row] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);
    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// ─── v2 fused QKV: RMSNorm + Q + K + V in one dispatch ──────────────

kernel void dequant_q1_0_g128_fused_qkv_v2(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(7)]],
    device const float* normWeight [[buffer(8)]],
    constant float& rmsEps [[buffer(9)]],
    constant uint& qRows [[buffer(10)]],
    constant uint& kvRows [[buffer(11)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint totalRows = qRows + kvRows + kvRows;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= totalRows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    const uint cols = params.cols;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < qRows) { localRow = globalRow; weights = wq; }
        else if (globalRow < qRows + kvRows) { localRow = globalRow - qRows; weights = wk; }
        else { localRow = globalRow - qRows - kvRows; weights = wv; }
        ax[r] = weights + localRow * nb * q1_0_g128BlockBytes;
    }

    // Cooperative RMSNorm
    float sumSq = 0.0f;
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint xBase = (isb / 4) * q1_0_g128WeightsPerBlock + (isb % 4) * 32;
        for (short i = 0; i < 32; i++) { float v = x[xBase + i]; sumSq += v * v; }
    }
    sumSq = simd_sum(sumSq);
    const float rmsScale = rsqrt(sumSq / float(cols) + rmsEps);

    // GEMV with inline RMSNorm
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        for (short i = 0; i < 32; i++) {
            xl[i] = x[xBase + i] * rmsScale * normWeight[xBase + i];
            xSum += xl[i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[r] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);
    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;
            float total = sumf[r];
            if (globalRow < qRows) { outQ[globalRow] = total; }
            else if (globalRow < qRows + kvRows) { outK[globalRow - qRows] = total; }
            else { outV[globalRow - qRows - kvRows] = half(total); }
        }
    }
}

// ─── v2 fused GEMV + residual add ────────────────────────────────────

kernel void dequant_q1_0_g128_gemv_add_v2(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }

    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        device const float* xp = x + xBase;
        for (short i = 0; i < 32; i++) { xl[i] = xp[i]; xSum += xl[i]; }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[row] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
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

// ─── v2 fused RMSNorm + Gate + Up (SwiGLU applied by caller) ────────

kernel void dequant_q1_0_g128_fused_gate_up_v2(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(4)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(5)]],
    constant float& rmsEps [[buffer(6)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint rows = params.rows;
    const uint totalRows = rows * 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= totalRows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    const uint cols = params.cols;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wGate; continue; }
        if (globalRow < rows) { ax[r] = wGate + globalRow * nb * q1_0_g128BlockBytes; }
        else { ax[r] = wUp + (globalRow - rows) * nb * q1_0_g128BlockBytes; }
    }

    // Cooperative RMSNorm
    float sumSq = 0.0f;
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint xBase = (isb / 4) * q1_0_g128WeightsPerBlock + (isb % 4) * 32;
        for (short i = 0; i < 32; i++) { float v = x[xBase + i]; sumSq += v * v; }
    }
    sumSq = simd_sum(sumSq);
    const float rmsScale = rsqrt(sumSq / float(cols) + rmsEps);

    // GEMV with inline RMSNorm
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        for (short i = 0; i < 32; i++) {
            xl[i] = x[xBase + i] * rmsScale * normWeight[xBase + i];
            xSum += xl[i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[r] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);
    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;
            activated[globalRow] = sumf[r];
        }
    }
}

// ─── Fused Q1 final norm + LM head (v1 pattern, 256 threads/TG) ─────

struct Q1FusedLMHeadParams {
    uint2 dims;  // x=vocabSize, y=dim
    float rmsEps;
};

kernel void dequant_q1_0_g128_fused_final_norm_gemv(
    device const uchar* lmHeadW [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    device const float* rmsNormW [[buffer(3)]],
    constant Q1FusedLMHeadParams& params [[buffer(4)]],
    uint ti [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]
) {
    constexpr uint THREADS = 256;
    const uint vocabSize = params.dims.x;
    const uint dim = params.dims.y;
    const float rmsEps = params.rmsEps;
    const uint nb = dim / q1_0_g128WeightsPerBlock;
    const uint row = tg * THREADS + ti;
    if (row >= vocabSize) return;

    threadgroup float xNorm[2048];
    const uint floatsPerThread = (dim + THREADS - 1) / THREADS;
    const uint loadStart = ti * floatsPerThread;
    const uint loadEnd = min(loadStart + floatsPerThread, dim);

    float localSumSq = 0.f;
    for (uint i = loadStart; i < loadEnd; i++) {
        localSumSq += hidden[i] * hidden[i];
    }

    threadgroup float sharedSum[256];
    sharedSum[ti] = localSumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (ti < s) { sharedSum[ti] += sharedSum[ti + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float invRms = rsqrt(sharedSum[0] * (1.0f / float(dim)) + rmsEps);
    for (uint i = loadStart; i < loadEnd; i++) {
        xNorm[i] = hidden[i] * invRms * rmsNormW[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.f;
    device const uchar* wRow = lmHeadW + row * nb * q1_0_g128BlockBytes;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar* block = wRow + ib * q1_0_g128BlockBytes;
        float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const uchar* qs = block + 2;
        const uint xBase = ib * q1_0_g128WeightsPerBlock;
        for (uint bi = 0; bi < 16; bi++) {
            uchar bits = qs[bi];
            const uint eb = xBase + bi * 8;
            float4 xv0 = float4(xNorm[eb],   xNorm[eb+1], xNorm[eb+2], xNorm[eb+3]);
            float4 xv1 = float4(xNorm[eb+4], xNorm[eb+5], xNorm[eb+6], xNorm[eb+7]);
            float4 s0 = float4(
                (((bits >> 0) & 1) * 2 - 1) * scale,
                (((bits >> 1) & 1) * 2 - 1) * scale,
                (((bits >> 2) & 1) * 2 - 1) * scale,
                (((bits >> 3) & 1) * 2 - 1) * scale
            );
            float4 s1 = float4(
                (((bits >> 4) & 1) * 2 - 1) * scale,
                (((bits >> 5) & 1) * 2 - 1) * scale,
                (((bits >> 6) & 1) * 2 - 1) * scale,
                (((bits >> 7) & 1) * 2 - 1) * scale
            );
            sum += dot(s0, xv0) + dot(s1, xv1);
        }
    }
    logits[row] = sum;
}
