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

/// NeoX-style RoPE: pairs (d, d+halfDim) instead of (2d, 2d+1).
/// Used by Qwen, GPT-NeoX, StableLM, and other models with split-halves layout.
kernel void rope_neox_f32(
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

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);
    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = x0 * cosValue - x1 * sinValue;
    output[headBase + dimPair + halfDim] = x0 * sinValue + x1 * cosValue;
}

// NeoX RoPE with float16 output — writes directly to KV cache, eliminating conversion dispatch.
kernel void rope_neox_f32_to_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant ERRoPEParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint head = tid.y;
    uint seq = tid.z;
    uint halfDim = params.headDim / 2;

    if (dimPair >= halfDim || head >= params.numHeads || seq >= params.seqLen) return;

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / pow(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);
    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = half(x0 * cosValue - x1 * sinValue);
    output[headBase + dimPair + halfDim] = half(x0 * sinValue + x1 * cosValue);
}
