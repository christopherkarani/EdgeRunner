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

/// Fused Q/K per-head norm + NeoX RoPE in a SINGLE dispatch.
/// Replaces 4 dispatches per layer (Q norm + K norm + RoPE Q + RoPE K→f16) with 1.
/// Thread grid: (halfDim, numHeads+numKVHeads). Threads 0..<numHeads do Q, rest do K.
struct ERFusedNormRoPEParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    float rmsEps;
};

kernel void fused_qk_norm_rope_neox(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half  *outK [[buffer(5)]],
    constant ERFusedNormRoPEParams &p [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    device const float* src = isQ ? Q : K;
    device const float* nw = isQ ? qNormW : kNormW;
    uint hb = head * p.headDim;

    float raw0 = src[hb + dimPair];
    float raw1 = src[hb + dimPair + halfDim];

    // Per-head RMSNorm: sum of squares across headDim elements
    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    // halfDim=64 means 2 simdgroups per head. Need cross-SG reduction.
    threadgroup float tgSq[48]; // max totalHeads=24, 2 SG each
    uint sgIdx = dimPair / 32;
    if (dimPair % 32 == 0) tgSq[headIdx * 2 + sgIdx] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[headIdx * 2] + tgSq[headIdx * 2 + 1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    // RoPE
    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle), s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        outQ[hb + dimPair] = o0;
        outQ[hb + dimPair + halfDim] = o1;
    } else {
        outK[hb + dimPair] = half(o0);
        outK[hb + dimPair + halfDim] = half(o1);
    }
}

/// NeoX RoPE with f16 output — eliminates separate f32→f16 conversion dispatch.
/// Used for K before writing to float16 KV cache.
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
