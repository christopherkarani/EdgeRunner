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

/// Fused Q1 QKV GEMV: processes Q, K, V matrices in one dispatch.
/// Reads x once, computes all three projections, writes three separate outputs.
/// Eliminates 2 redundant x reads and 2 dispatch overheads per layer.
///
/// Layout: Q weights [qRows×cols], K weights [kvRows×cols], V weights [kvRows×cols]
/// Outputs: Q out [qRows], K out [kvRows], V out [kvRows]
///
/// Dispatch: 1 threadgroup of 256 threads, each processes totalRows/256 rows
kernel void dequant_q1_0_g128_fused_qkv(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* qOut [[buffer(4)]],
    device float* kOut [[buffer(5)]],
    device float* vOut [[buffer(6)]],
    constant uint3& dims [[buffer(7)]],  // x=qRows, y=kvRows, z=cols
    uint ti [[thread_position_in_threadgroup]]
) {
    constexpr uint THREADS = 256;
    const uint qRows = dims.x;
    const uint kvRows = dims.y;
    const uint cols = dims.z;
    const uint nb = cols / q1_0_g128WeightsPerBlock;
    const uint totalRows = qRows + kvRows + kvRows;

    // Phase 1: Cache x in threadgroup memory (cooperative load)
    threadgroup float xShared[8192];
    const uint floatsPerThread = (cols + THREADS - 1) / THREADS;
    const uint loadStart = ti * floatsPerThread;
    const uint loadEnd = min(loadStart + floatsPerThread, cols);
    for (uint i = loadStart; i < loadEnd; i++) {
        xShared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Each thread processes its assigned rows across Q, K, V
    const uint rowsPerThread = (totalRows + THREADS - 1) / THREADS;
    const uint rowStart = ti * rowsPerThread;
    const uint rowEnd = min(rowStart + rowsPerThread, totalRows);

    for (uint row = rowStart; row < rowEnd; row++) {
        float sum = 0.f;
        device const uchar* wRow;
        if (row < qRows) {
            wRow = wq + row * nb * q1_0_g128BlockBytes;
        } else if (row < qRows + kvRows) {
            wRow = wk + (row - qRows) * nb * q1_0_g128BlockBytes;
        } else {
            wRow = wv + (row - qRows - kvRows) * nb * q1_0_g128BlockBytes;
        }

        for (uint ib = 0; ib < nb; ib++) {
            device const uchar* block = wRow + ib * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            device const uchar* qs = block + 2;
            const uint xBase = ib * q1_0_g128WeightsPerBlock;

            for (uint bi = 0; bi < 16; bi++) {
                uchar bits = qs[bi];
                const uint eb = xBase + bi * 8;
                float4 xv0 = float4(xShared[eb],   xShared[eb+1], xShared[eb+2], xShared[eb+3]);
                float4 xv1 = float4(xShared[eb+4], xShared[eb+5], xShared[eb+6], xShared[eb+7]);
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

        // Write to correct output buffer
        if (row < qRows) {
            qOut[row] = sum;
        } else if (row < qRows + kvRows) {
            kOut[row - qRows] = sum;
        } else {
            vOut[row - qRows - kvRows] = sum;
        }
    }
}

/// Fused Q1 Gate+Up GEMV: processes Gate and Up matrices in one dispatch.
/// Same pattern as fused QKV but for FFN gate/up projections.
kernel void dequant_q1_0_g128_fused_gate_up(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* gateOut [[buffer(3)]],
    device float* upOut [[buffer(4)]],
    constant uint2& dims [[buffer(5)]],  // x=rows, y=cols
    uint ti [[thread_position_in_threadgroup]]
) {
    constexpr uint THREADS = 256;
    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint nb = cols / q1_0_g128WeightsPerBlock;
    const uint totalRows = rows * 2;  // Gate + Up

    // Phase 1: Cache x
    threadgroup float xShared[8192];
    const uint floatsPerThread = (cols + THREADS - 1) / THREADS;
    const uint loadStart = ti * floatsPerThread;
    const uint loadEnd = min(loadStart + floatsPerThread, cols);
    for (uint i = loadStart; i < loadEnd; i++) {
        xShared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Each thread processes Gate + Up rows
    const uint rowsPerThread = (totalRows + THREADS - 1) / THREADS;
    const uint rowStart = ti * rowsPerThread;
    const uint rowEnd = min(rowStart + rowsPerThread, totalRows);

    for (uint row = rowStart; row < rowEnd; row++) {
        float sum = 0.f;
        device const uchar* wRow;
        if (row < rows) {
            wRow = wGate + row * nb * q1_0_g128BlockBytes;
        } else {
            wRow = wUp + (row - rows) * nb * q1_0_g128BlockBytes;
        }

        for (uint ib = 0; ib < nb; ib++) {
            device const uchar* block = wRow + ib * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            device const uchar* qs = block + 2;
            const uint xBase = ib * q1_0_g128WeightsPerBlock;

            for (uint bi = 0; bi < 16; bi++) {
                uchar bits = qs[bi];
                const uint eb = xBase + bi * 8;
                float4 xv0 = float4(xShared[eb],   xShared[eb+1], xShared[eb+2], xShared[eb+3]);
                float4 xv1 = float4(xShared[eb+4], xShared[eb+5], xShared[eb+6], xShared[eb+7]);
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

        if (row < rows) {
            gateOut[row] = sum;
        } else {
            upOut[row - rows] = sum;
        }
    }
}
