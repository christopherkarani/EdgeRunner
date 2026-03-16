#include <metal_stdlib>
using namespace metal;

struct ERActivationParams {
    uint count;
};

kernel void sigmoid_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERActivationParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    float value = input[gid];
    output[gid] = 1.0f / (1.0f + exp(-value));
}

kernel void gelu_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERActivationParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    float value = input[gid];
    const float coefficient = 0.7978845608028654f;
    output[gid] = value * 0.5f * (1.0f + tanh(coefficient * (value + 0.044715f * value * value * value)));
}

inline float silu(float value) {
    return value / (1.0f + exp(-value));
}

kernel void swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERActivationParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    output[gid] = silu(gate[gid]) * up[gid];
}
