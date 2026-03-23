#include <metal_stdlib>
using namespace metal;

struct ERDequantQ5KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q5_K_BLOCK_BYTES = 176;
constant uint Q5_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q5_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ5KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q5_K_BLOCK_BYTES;
    device const half* masterScales = reinterpret_cast<device const half*>(block);
    float d = float(masterScales[0]);
    float dmin = float(masterScales[1]);

    // Unpack scales and mins from 12 bytes at offset 4 (identical to Q4_K_M)
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

    uint outBase = params.outputOffset + tid * Q5_K_WEIGHTS_PER_BLOCK;

    for (uint subBlock = 0; subBlock < 8; ++subBlock) {
        float scale = scales[subBlock];
        float minValue = mins[subBlock];

        for (uint index = 0; index < 32; ++index) {
            uint globalIdx = subBlock * 32 + index;

            // Lower 4 bits from qs (offset 48, 128 bytes nibble-packed)
            uint qsByteIndex = 48 + globalIdx / 2;
            uchar qsByte = block[qsByteIndex];
            uchar lower4 = (globalIdx % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

            // 5th bit from qh (offset 16, 32 bytes bit-packed)
            uchar qhByte = block[16 + globalIdx / 8];
            uchar bit5 = (qhByte >> (globalIdx % 8)) & 1;

            uint q5 = uint(lower4) | (uint(bit5) << 4);
            output[outBase + globalIdx] = scale * float(q5) - minValue;
        }
    }
}
