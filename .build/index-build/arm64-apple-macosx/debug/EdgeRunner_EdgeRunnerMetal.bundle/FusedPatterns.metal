#include <metal_stdlib>
using namespace metal;

// function_constant(0): selects which activation to apply after the binary op.
// Values: 0 = none, 1 = relu, 2 = sigmoid, 3 = gelu, 4 = silu.
// The Metal compiler eliminates dead branches at pipeline-creation time,
// producing a specialised kernel with zero runtime overhead.
constant int activation_type [[function_constant(0)]];

inline float apply_activation(float x) {
    if (activation_type == 1) {
        return max(x, 0.0f);
    } else if (activation_type == 2) {
        return 1.0f / (1.0f + exp(-x));
    } else if (activation_type == 3) {
        const float kSqrt2OverPi = 0.7978845608f;
        float cube = x * x * x;
        return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + 0.044715f * cube)));
    } else if (activation_type == 4) {
        return x / (1.0f + exp(-x));
    }
    return x; // activation_type == 0: identity
}

// Fused add + activation kernel (float32).
kernel void fused_add_activate_float(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    out[tid] = apply_activation(a[tid] + b[tid]);
}

// Fused multiply + activation kernel (float32).
kernel void fused_mul_activate_float(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    out[tid] = apply_activation(a[tid] * b[tid]);
}

// Fused unary activation kernel (float32).
kernel void fused_activate_float(
    device const float* input    [[buffer(0)]],
    device       float* output   [[buffer(1)]],
    constant     uint&  elementCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    output[tid] = apply_activation(input[tid]);
}
