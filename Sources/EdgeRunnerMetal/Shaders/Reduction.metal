#include <metal_stdlib>
using namespace metal;

struct ERReductionParams {
    uint elementCount;
    uint reductionSize;
    uint outerSize;
};

kernel void reduce_sum_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum;
}

kernel void reduce_mean_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum / float(params.reductionSize);
}

kernel void reduce_max_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    uint base = tid * params.reductionSize;
    float maxVal = input[base];
    for (uint i = 1; i < params.reductionSize; i++) {
        maxVal = max(maxVal, input[base + i]);
    }
    output[tid] = maxVal;
}
