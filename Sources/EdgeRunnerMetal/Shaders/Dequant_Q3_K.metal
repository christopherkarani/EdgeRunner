#include <metal_stdlib>
using namespace metal;

struct ERDequantQ3KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q3_K_BLOCK_BYTES = 110;
constant uint Q3_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q3_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ3KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q3_K_BLOCK_BYTES;

    // Master scale d (float16) at offset 108
    device const half* dPtr = reinterpret_cast<device const half*>(block + 108);
    float d = float(dPtr[0]);

    // Unpack 16 sub-block scales from 12 bytes starting at offset 96
    float scales[16];
    for (uint i = 0; i < 16; ++i) {
        uint byteIdx = i / 2;
        uchar lower4 = (block[96 + byteIdx] >> ((i % 2) * 4)) & 0x0F;

        uint upperByteIdx = 104 + i / 4;
        uchar upper2 = (block[upperByteIdx] >> ((i % 4) * 2)) & 0x03;

        int raw6 = int(lower4) | (int(upper2) << 4); // 0..63
        int signedScale = raw6 - 32;                 // -32..31
        scales[i] = d * float(signedScale);
    }

    uint outBase = params.outputOffset + tid * Q3_K_WEIGHTS_PER_BLOCK;
    for (uint idx = 0; idx < Q3_K_WEIGHTS_PER_BLOCK; ++idx) {
        uint sub = idx / 16;

        // Lower 2 bits from qs at offset 32
        uchar qsByte = block[32 + idx / 4];
        uchar lower2 = (qsByte >> ((idx % 4) * 2)) & 0x03;

        // High bit from hmask at offset 0
        uchar hmaskByte = block[idx / 8];
        uchar highBit = (hmaskByte >> (idx % 8)) & 0x01;

        uint q3 = uint(lower2) | (uint(highBit) << 2); // 0..7
        output[outBase + idx] = scales[sub] * float(int(q3) - 4);
    }
}
