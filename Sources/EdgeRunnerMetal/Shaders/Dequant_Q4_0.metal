#include <metal_stdlib>
using namespace metal;

struct ERDequantParams {
    uint blockCount;
    uint outputOffset;
};

struct ERDequantGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint q4_0BlockBytes = 18;
constant uint q4_0WeightsPerBlock = 32;

kernel void dequant_q4_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + (tid * q4_0BlockBytes);
    const device ushort* scaleBits = reinterpret_cast<const device ushort*>(block);
    half scale = as_type<half>(*scaleBits);

    uint outputBase = params.outputOffset + (tid * q4_0WeightsPerBlock);
    for (uint index = 0; index < 16; index++) {
        uchar packed = block[2 + index];
        int low = int(packed & 0x0F) - 8;
        int high = int(packed >> 4) - 8;
        output[outputBase + index] = float(scale) * float(low);
        output[outputBase + index + 16] = float(scale) * float(high);
    }
}

kernel void dequant_q4_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    uint rowBlockOffset = row * params.blocksPerRow;

    for (uint blockIndex = lane; blockIndex < params.blocksPerRow; blockIndex += 32) {
        device const uchar* block = quantisedW + ((rowBlockOffset + blockIndex) * q4_0BlockBytes);
        const device ushort* scaleBits = reinterpret_cast<const device ushort*>(block);
        float scale = float(as_type<half>(*scaleBits));
        uint columnBase = blockIndex * q4_0WeightsPerBlock;

        for (uint index = 0; index < 16; index++) {
            uchar packed = block[2 + index];
            float low = scale * float(int(packed & 0x0F) - 8);
            float high = scale * float(int(packed >> 4) - 8);
            partial += low * x[columnBase + index];
            partial += high * x[columnBase + index + 16];
        }
    }

    partial = simd_sum(partial);
    if (lane == 0) {
        y[row] = partial;
    }
}
