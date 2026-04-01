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

constant uint q1_0_g128BlockBytes = 18;  // 2 bytes scale + 16 bytes bits
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

    // 16 bytes = 128 bits, each bit is one weight.
    // GGUF variants may place the fp16 scale at byte 0 or byte 16.
    for (uint byteIndex = 0; byteIndex < 16; byteIndex++) {
        uint bitOffset = min(params.bitDataOffset + byteIndex, q1_0_g128BlockBytes - 1);
        uchar bits = block[bitOffset];
        uint baseOutput = outputBase + (byteIndex * 8);

        for (uint bitIndex = 0; bitIndex < 8; bitIndex++) {
            // Extract bit: 0 → -scale, 1 → +scale
            uint shift = params.bitOrderMSBFirst != 0 ? (7u - bitIndex) : bitIndex;
            uchar bit = (bits >> shift) & 1;
            bool positive = params.oneBitIsNegative != 0 ? (bit == 0) : (bit != 0);
            output[baseOutput + bitIndex] = positive ? scale : -scale;
        }
    }
}

/// Fused Q1_0_g128 GEMV: y = W @ x
/// Each SIMD group handles 2 rows; each thread processes 1 block (128 elements) per iteration.
/// Reads x directly from device memory to avoid 128-float register spill.
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
