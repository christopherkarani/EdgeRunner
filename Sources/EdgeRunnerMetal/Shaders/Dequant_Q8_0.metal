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

// === Optimized Fused Dequant+GEMV for Q8_0 ===
// Inspired by llama.cpp's ggml Metal kernel optimizations:
//   1. 2 rows per threadgroup — doubles output per dispatch, Y-value reuse across rows
//   2. Y-value register caching — load x[k] once, multiply by 2 different weight rows
//   3. Loop unrolling for instruction pipelining
//   4. 32 threads (1 simdgroup) with simd_sum reduction
//
// y[row] = sum_k dequant(W_q8[row, k]) * x[k]

struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

// Process 2 rows per threadgroup — halves dispatch count and reuses x[] loads
kernel void dequant_q8_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    uint localID [[thread_position_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup processes 2 rows
    uint row0 = tgIndex * 2;
    uint row1 = row0 + 1;
    bool hasRow0 = row0 < params.rows;
    bool hasRow1 = row1 < params.rows;
    if (!hasRow0) return;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    uint rowOffset0 = row0 * params.blocksPerRow;
    uint rowOffset1 = row1 * params.blocksPerRow;

    for (uint blockIndex = localID; blockIndex < params.blocksPerRow; blockIndex += 32) {
        uint colBase = blockIndex * q8_0WeightsPerBlock;

        // Load x values once into registers (reused across both rows)
        float yl[32];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < q8_0WeightsPerBlock; j++) {
            yl[j] = x[colBase + j];
        }

        // Row 0
        {
            device const uchar* block = quantisedW + (rowOffset0 + blockIndex) * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            float sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < q8_0WeightsPerBlock; j++) {
                sum += float(as_type<char>(block[2 + j])) * yl[j];
            }
            partial0 += scale * sum;
        }

        // Row 1 (if valid)
        if (hasRow1) {
            device const uchar* block = quantisedW + (rowOffset1 + blockIndex) * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            float sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < q8_0WeightsPerBlock; j++) {
                sum += float(as_type<char>(block[2 + j])) * yl[j];
            }
            partial1 += scale * sum;
        }
    }

    // Reduce across simdgroup
    partial0 = simd_sum(partial0);
    if (hasRow1) partial1 = simd_sum(partial1);

    if (simdLane == 0) {
        y[row0] = partial0;
        if (hasRow1) y[row1] = partial1;
    }
}
