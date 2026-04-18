#include <metal_stdlib>
using namespace metal;

struct GeGLUParams {
    uint count;
};

// Fused GeGLU with PyTorch tanh-approx GELU for Gemma 4 (E4B).
// y[i] = gelu_tanh(gate[i]) * up[i]
// gelu_tanh(x) = x * 0.5 * (1 + tanh(c * (x + 0.044715 * x^3))), c = sqrt(2/pi)
kernel void gelu_tanh_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float *out        [[buffer(2)]],
    constant GeGLUParams &p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) {
        return;
    }
    float g = gate[gid];
    const float c = 0.7978845608028654f;
    float inner = c * (g + 0.044715f * g * g * g);
    float gelu = g * 0.5f * (1.0f + tanh(inner));
    out[gid] = gelu * up[gid];
}
