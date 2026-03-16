#include <metal_stdlib>
using namespace metal;

struct ERSoftmaxParams {
    uint rows;
    uint cols;
};

constant uint SOFTMAX_THREADS = 256;

kernel void softmax_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERSoftmaxParams &params [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) {
        return;
    }

    device const float *rowIn = input + row * params.cols;
    device float *rowOut = output + row * params.cols;

    float threadMax = -INFINITY;
    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        threadMax = max(threadMax, rowIn[col]);
    }

    threadMax = simd_max(threadMax);

    threadgroup float shared[32];
    if (simd_lane == 0) {
        shared[simd_group] = threadMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint simdGroupCount = (SOFTMAX_THREADS + 31) / 32;
        float value = simd_lane < simdGroupCount ? shared[simd_lane] : -INFINITY;
        value = simd_max(value);
        shared[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowMax = shared[0];

    float threadSum = 0.0f;
    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        float expValue = exp(rowIn[col] - rowMax);
        rowOut[col] = expValue;
        threadSum += expValue;
    }

    threadSum = simd_sum(threadSum);
    if (simd_lane == 0) {
        shared[simd_group] = threadSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint simdGroupCount = (SOFTMAX_THREADS + 31) / 32;
        float value = simd_lane < simdGroupCount ? shared[simd_lane] : 0.0f;
        value = simd_sum(value);
        shared[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowSum = shared[0];
    float invSum = 1.0f / rowSum;

    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        rowOut[col] *= invSum;
    }
}
