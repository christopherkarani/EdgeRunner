#include <metal_stdlib>
using namespace metal;

kernel void logit_softcap_f32(
    device float *logits   [[buffer(0)]],
    constant float &cap    [[buffer(1)]],
    constant uint &count   [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = logits[gid];
    logits[gid] = tanh(x / cap) * cap;
}
