#include <metal_stdlib>
using namespace metal;

struct ERSlidingMaskParams {
    uint seqLen;
    uint window;
};

/// Generates an additive sliding-window causal attention mask as Float.
///
/// For query position `q` attending to key position `k`:
///   - k > q              → -INFINITY (causal constraint; cannot attend to future)
///   - q >= window &&
///     (q - k) >= window  → -INFINITY (outside sliding window)
///   - otherwise          → 0.0       (attention allowed)
///
/// The `q >= p.window` guard is REQUIRED to prevent unsigned underflow when
/// q < window. Without it, `q - k` wraps around and erroneously masks valid
/// positions at the start of the sequence.
///
/// Global (full causal) attention is expressed as `window >= seqLen`:
/// the sliding-window check is impossible (q - k < window always holds),
/// so only the causal constraint applies.
///
/// Output layout: row-major `[seqLen, seqLen]` with index `q * seqLen + k`.
kernel void sliding_causal_mask_f32(
    device float *mask [[buffer(0)]],
    constant ERSlidingMaskParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q = gid.y;
    uint k = gid.x;
    if (q >= params.seqLen || k >= params.seqLen) {
        return;
    }
    bool outOfWindow =
        (k > q) ||
        (q >= params.window && (q - k) >= params.window);
    mask[q * params.seqLen + k] = outOfWindow ? -INFINITY : 0.0f;
}
