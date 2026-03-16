#include <metal_stdlib>
using namespace metal;

struct ERDequantQ4_0Params {
    uint blockCount;
    uint outputOffset;
};

struct ERDequantQ4_0GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint Q4_0_BLOCK_BYTES = 18;
constant uint Q4_0_BLOCK_WEIGHTS = 32;

kernel void dequant_q4_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ4_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + tid * Q4_0_BLOCK_BYTES;
    device const half* scalePtr = reinterpret_cast<device const half*>(block);
    float scale = float(scalePtr[0]);
    uint outBase = params.outputOffset + tid * Q4_0_BLOCK_WEIGHTS;

    for (uint i = 0; i < 16; ++i) {
        uchar packed = block[2 + i];
        int low = int(packed & 0x0F) - 8;
        int high = int(packed >> 4) - 8;
        output[outBase + i] = scale * float(low);
        output[outBase + i + 16] = scale * float(high);
    }
}

kernel void dequant_q4_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ4_0GEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint localID [[thread_position_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    uint rowOffset = row * params.blocksPerRow;

    for (uint blockIndex = localID; blockIndex < params.blocksPerRow; blockIndex += 32) {
        device const uchar* block = quantisedW + (rowOffset + blockIndex) * Q4_0_BLOCK_BYTES;
        device const half* scalePtr = reinterpret_cast<device const half*>(block);
        float scale = float(scalePtr[0]);
        uint colBase = blockIndex * Q4_0_BLOCK_WEIGHTS;

        for (uint i = 0; i < 16; ++i) {
            uchar packed = block[2 + i];
            float low = scale * float(int(packed & 0x0F) - 8);
            float high = scale * float(int(packed >> 4) - 8);
            partial += low * x[colBase + i];
            partial += high * x[colBase + i + 16];
        }
    }

    partial = simd_sum(partial);
    if (simdLane == 0) {
        y[row] = partial;
    }
}
