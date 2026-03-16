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
