#include <metal_stdlib>
using namespace metal;

// [[stitchable]] marks these functions as linkable building blocks for
// runtime function stitching (Metal 2.3+ / macOS 12+ / iOS 15+).
// FusionEngine (Task 10) uses them to compose fused pipelines at runtime
// without recompiling shader source.

[[stitchable]] float op_relu_float(float x) { return max(x, 0.0f); }

[[stitchable]] float op_sigmoid_float(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[[stitchable]] float op_gelu_float(float x) {
    const float kSqrt2OverPi = 0.7978845608f;
    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + 0.044715f * cube);
    return 0.5f * x * (1.0f + tanh(inner));
}

[[stitchable]] float op_silu_float(float x) {
    return x / (1.0f + exp(-x));
}

[[stitchable]] float op_neg_float(float x) { return -x; }

[[stitchable]] float op_abs_float(float x) { return abs(x); }

[[stitchable]] float op_sqrt_float(float x) { return sqrt(x); }

[[stitchable]] float op_exp_float(float x) { return exp(x); }

[[stitchable]] float op_log_float(float x) { return log(x); }

[[stitchable]] float op_tanh_float(float x) { return tanh(x); }

[[stitchable]] float op_add_float(float a, float b) { return a + b; }

[[stitchable]] float op_sub_float(float a, float b) { return a - b; }

[[stitchable]] float op_mul_float(float a, float b) { return a * b; }

[[stitchable]] float op_div_float(float a, float b) { return a / b; }
