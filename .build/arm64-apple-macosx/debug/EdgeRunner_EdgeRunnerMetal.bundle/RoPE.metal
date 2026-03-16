#include <metal_stdlib>
using namespace metal;

struct ERRoPEParams {
    uint seqLen;
    uint numHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
};

kernel void rope_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERRoPEParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint head = tid.y;
    uint seq = tid.z;
    uint halfDim = params.headDim / 2;

    if (dimPair >= halfDim || head >= params.numHeads || seq >= params.seqLen) {
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / pow(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    uint baseIndex = (seq * params.numHeads * params.headDim) + (head * params.headDim) + (2 * dimPair);
    float x0 = input[baseIndex];
    float x1 = input[baseIndex + 1];
    output[baseIndex] = x0 * cosValue - x1 * sinValue;
    output[baseIndex + 1] = x0 * sinValue + x1 * cosValue;
}
