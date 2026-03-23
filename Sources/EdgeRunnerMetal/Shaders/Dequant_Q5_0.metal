#include <metal_stdlib>
using namespace metal;

struct ERDequantQ5_0Params {
    uint blockCount;
    uint outputOffset;
};

constant uint Q5_0_BLOCK_BYTES = 22;
constant uint Q5_0_WEIGHTS_PER_BLOCK = 32;

kernel void dequant_q5_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ5_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + tid * Q5_0_BLOCK_BYTES;
    float d = float(as_type<half>(*reinterpret_cast<device const ushort*>(block)));

    uint outBase = params.outputOffset + tid * Q5_0_WEIGHTS_PER_BLOCK;
    for (uint i = 0; i < Q5_0_WEIGHTS_PER_BLOCK; ++i) {
        // Lower 4 bits from qs at offset 6
        uchar qsByte = block[6 + i / 2];
        uchar lower4 = (i % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

        // 5th bit from qh at offset 2
        uchar qhByte = block[2 + i / 8];
        uchar bit5 = (qhByte >> (i % 8)) & 0x01;

        uint q5 = uint(lower4) | (uint(bit5) << 4);
        output[outBase + i] = d * float(int(q5) - 16);
    }
}
