#include <metal_stdlib>
using namespace metal;

struct ERRMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERRMSNormParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.rows) {
        return;
    }

    uint offset = row * params.cols;
    float meanSq = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        float value = input[offset + col];
        meanSq += value * value;
    }
    meanSq /= float(params.cols);
    float scale = rsqrt(meanSq + params.eps);

    for (uint col = 0; col < params.cols; col++) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}

/// Parallel RMSNorm for single-row decode (rows=1).
/// Uses 256 threads (8 simdgroups) to cooperatively process cols elements.
/// Much faster than rmsnorm_f32 which dispatches 1 thread for rows=1.
kernel void rmsnorm_parallel_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERRMSNormParams &params [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    if (row >= params.rows) return;

    const uint cols = params.cols;
    const uint offset = row * cols;
    const uint tid = sgitg * 32 + tiisg;
    const uint stride = 256;  // 8 simdgroups × 32 threads

    // Phase 1: Parallel sum-of-squares reduction
    float localSumSq = 0.0f;
    for (uint col = tid; col < cols; col += stride) {
        float v = input[offset + col];
        localSumSq += v * v;
    }

    // Intra-simdgroup reduction
    localSumSq = simd_sum(localSumSq);

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float tg_partial[8];
    if (tiisg == 0) {
        tg_partial[sgitg] = localSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread computes final scale
    if (sgitg == 0 && tiisg == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; i++) total += tg_partial[i];
        tg_partial[0] = rsqrt(total / float(cols) + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = tg_partial[0];

    // Phase 2: Parallel scale + weight multiply
    for (uint col = tid; col < cols; col += stride) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}
