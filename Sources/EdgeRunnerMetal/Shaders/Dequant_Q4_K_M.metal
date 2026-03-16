#include <metal_stdlib>
using namespace metal;

struct ERDequantQ4KMParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q4_K_M_BLOCK_BYTES = 144;
constant uint Q4_K_M_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q4_k_m(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ4KMParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q4_K_M_BLOCK_BYTES;
    device const half* masterScales = reinterpret_cast<device const half*>(block);
    float d = float(masterScales[0]);
    float dmin = float(masterScales[1]);

    float scales[8];
    float mins[8];
    for (uint subBlock = 0; subBlock < 4; ++subBlock) {
        uchar scaleByte = block[4 + subBlock];
        uchar minByte = block[8 + subBlock];
        uchar highBits = block[12 + subBlock];

        scales[subBlock] = d * float(scaleByte & 0x3F);
        scales[subBlock + 4] = d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
        mins[subBlock] = dmin * float(minByte & 0x3F);
        mins[subBlock + 4] = dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
    }

    uint outBase = params.outputOffset + tid * Q4_K_M_WEIGHTS_PER_BLOCK;
    for (uint subBlock = 0; subBlock < 8; ++subBlock) {
        float scale = scales[subBlock];
        float minValue = mins[subBlock];

        for (uint index = 0; index < 32; ++index) {
            uint byteIndex = 16 + (subBlock * 16 + index / 2);
            uchar packed = block[byteIndex];
            uchar nibble = index % 2 == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            output[outBase + (subBlock * 32) + index] = (scale * float(nibble)) - minValue;
        }
    }
}
