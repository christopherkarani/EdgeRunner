#include <metal_stdlib>
using namespace metal;

struct ERDequantQ8_0Params {
    uint blockCount;
    uint outputOffset;
};

constant uint q8_0BlockBytes = 34;
constant uint q8_0WeightsPerBlock = 32;

kernel void dequant_q8_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ8_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + (tid * q8_0BlockBytes);
    const device ushort* scaleBits = reinterpret_cast<const device ushort*>(block);
    float scale = float(as_type<half>(*scaleBits));

    uint outputBase = params.outputOffset + (tid * q8_0WeightsPerBlock);
    for (uint index = 0; index < q8_0WeightsPerBlock; index++) {
        char quantised = as_type<char>(block[2 + index]);
        output[outputBase + index] = scale * float(quantised);
    }
}

// === Fused Dequant+GEMV for Q8_0 ===
// Reads quantized weights directly, dequantizes in GPU registers, and multiplies.
// Reduces memory bandwidth by ~3.8x vs pre-dequantized float32 GEMV.
// y[row] = sum_k dequant(W_q8[row, k]) * x[k]

struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

kernel void dequant_q8_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint localID [[thread_position_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    if (row >= params.rows) return;

    float partial = 0.0f;
    uint rowOffset = row * params.blocksPerRow;

    for (uint blockIndex = localID; blockIndex < params.blocksPerRow; blockIndex += 32) {
        device const uchar* block = quantisedW + (rowOffset + blockIndex) * q8_0BlockBytes;
        device const ushort* scaleBits = reinterpret_cast<device const ushort*>(block);
        float scale = float(as_type<half>(*scaleBits));
        uint colBase = blockIndex * q8_0WeightsPerBlock;

        for (uint j = 0; j < q8_0WeightsPerBlock; j++) {
            char quantised = as_type<char>(block[2 + j]);
            partial += scale * float(quantised) * x[colBase + j];
        }
    }

    partial = simd_sum(partial);
    if (simdLane == 0) {
        y[row] = partial;
    }
}
