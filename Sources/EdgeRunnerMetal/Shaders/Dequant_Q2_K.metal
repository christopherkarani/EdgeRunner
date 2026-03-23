#include <metal_stdlib>
using namespace metal;

struct ERDequantQ2KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q2_K_BLOCK_BYTES = 84;
constant uint Q2_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q2_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ2KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q2_K_BLOCK_BYTES;

    // d at offset 80, dmin at offset 82 (float16)
    device const half* dPtr = reinterpret_cast<device const half*>(block + 80);
    float d = float(dPtr[0]);
    float dmin = float(dPtr[1]);

    uint outBase = params.outputOffset + tid * Q2_K_WEIGHTS_PER_BLOCK;
    for (uint idx = 0; idx < Q2_K_WEIGHTS_PER_BLOCK; ++idx) {
        uint sub = idx / 16;

        // scales packed at offset 0..15 (sc | m<<4)
        uchar scaleByte = block[sub];
        float sc = float(scaleByte & 0x0F);
        float m = float(scaleByte >> 4);

        // 2-bit quant from qs at offset 16..79
        uchar qsByte = block[16 + idx / 4];
        float q2 = float((qsByte >> ((idx % 4) * 2)) & 0x03);

        output[outBase + idx] = d * sc * q2 - dmin * m;
    }
}
