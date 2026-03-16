#include <metal_stdlib>
using namespace metal;

struct ERLayerNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void layernorm_f32(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERLayerNormParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.rows) {
        return;
    }

    uint offset = row * params.cols;
    float mean = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        mean += input[offset + col];
    }
    mean /= float(params.cols);

    float variance = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        float delta = input[offset + col] - mean;
        variance += delta * delta;
    }
    variance /= float(params.cols);
    float invStd = rsqrt(variance + params.eps);

    for (uint col = 0; col < params.cols; col++) {
        output[offset + col] = (input[offset + col] - mean) * invStd * gamma[col] + beta[col];
    }
}
