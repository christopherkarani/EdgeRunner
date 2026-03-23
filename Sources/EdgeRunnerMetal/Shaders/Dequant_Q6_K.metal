#include <metal_stdlib>
using namespace metal;

struct ERDequantQ6KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q6_K_BLOCK_BYTES = 210;
constant uint Q6_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q6_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ6KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q6_K_BLOCK_BYTES;

    // d at offset 208 (float16)
    device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
    float d = float(dPtr[0]);

    uint outBase = params.outputOffset + tid * Q6_K_WEIGHTS_PER_BLOCK;

    for (uint i = 0; i < 256; ++i) {
        uint sub = i / 16;

        // Lower 4 bits from ql (offset 0, 128 bytes nibble-packed)
        uchar qlByte = block[i / 2];
        uchar lower4 = (i % 2 == 0) ? (qlByte & 0x0F) : ((qlByte >> 4) & 0x0F);

        // Upper 2 bits from qh (offset 128, 64 bytes, 4 per byte)
        uchar qhByte = block[128 + i / 4];
        uchar upper2 = (qhByte >> ((i % 4) * 2)) & 0x03;

        int q6 = int(lower4) | (int(upper2) << 4);

        // scales is signed int8 at offset 192
        char scale = as_type<char>(block[192 + sub]);

        output[outBase + i] = d * float(scale) * float(q6 - 32);
    }
}
