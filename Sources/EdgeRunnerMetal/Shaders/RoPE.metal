#include <metal_stdlib>
using namespace metal;

struct ERRoPEParams {
    uint seqLen;
    uint numHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    // Fraction of head_dim to rotate (pRoPE). 1.0 = rotate all channels (standard RoPE).
    // <1.0 = only rotate channels with 2*pair < headDim * partialRotaryFactor; rest pass through.
    float partialRotaryFactor;
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

    uint baseIndex = (seq * params.numHeads * params.headDim) + (head * params.headDim) + (2 * dimPair);

    // pRoPE pass-through: channels beyond the partial boundary are copied verbatim.
    // rotatedPairs = floor(halfDim * partialRotaryFactor). For partial=1.0 this == halfDim,
    // so the guard never fires (full rotation, identical to standard RoPE).
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[baseIndex] = input[baseIndex];
        output[baseIndex + 1] = input[baseIndex + 1];
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

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

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);

    // pRoPE pass-through (NeoX layout pairs (d, d+halfDim)). For partial=1.0, rotatedPairs == halfDim
    // and this guard never fires — standard NeoX RoPE behavior preserved.
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[headBase + dimPair]           = input[headBase + dimPair];
        output[headBase + dimPair + halfDim] = input[headBase + dimPair + halfDim];
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

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

struct ERFusedNormRoPEPrefillParams {
    uint seqLen;
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

kernel void fused_qk_norm_rope_neox_prefill_f16in(
    device const half *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half *outK [[buffer(5)]],
    constant ERFusedNormRoPEPrefillParams &p [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgTid [[thread_position_in_threadgroup]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint seq = tid.z;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads || seq >= p.seqLen) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    uint inputHeadCount = isQ ? p.numHeads : p.numKVHeads;
    uint base = (seq * inputHeadCount + head) * p.headDim;
    device const half *src = isQ ? Q : K;
    device const float *nw = isQ ? qNormW : kNormW;

    float raw0 = float(src[base + dimPair]);
    float raw1 = float(src[base + dimPair + halfDim]);

    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    threadgroup float tgSq[2];
    uint sgIdx = tgTid.x / 32;
    if ((tgTid.x % 32) == 0) {
        tgSq[sgIdx] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[0] + tgSq[1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(seq + p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle);
    float s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        uint outBase = (seq * p.numHeads + head) * p.headDim;
        outQ[outBase + dimPair] = o0;
        outQ[outBase + dimPair + halfDim] = o1;
    } else {
        uint outBase = (seq * p.numKVHeads + head) * p.headDim;
        outK[outBase + dimPair] = half(o0);
        outK[outBase + dimPair + halfDim] = half(o1);
    }
}

kernel void fused_qk_norm_rope_neox_prefill_f16in_kpacked(
    device const half *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half *outK [[buffer(5)]],
    constant ERFusedNormRoPEPrefillParams &p [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgTid [[thread_position_in_threadgroup]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint seq = tid.z;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads || seq >= p.seqLen) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    uint inputHeadCount = isQ ? p.numHeads : p.numKVHeads;
    uint base = (seq * inputHeadCount + head) * p.headDim;
    device const half *src = isQ ? Q : K;
    device const float *nw = isQ ? qNormW : kNormW;

    float raw0 = float(src[base + dimPair]);
    float raw1 = float(src[base + dimPair + halfDim]);

    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    threadgroup float tgSq[2];
    uint sgIdx = tgTid.x / 32;
    if ((tgTid.x % 32) == 0) {
        tgSq[sgIdx] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[0] + tgSq[1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(seq + p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle);
    float s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        uint outBase = (seq * p.numHeads + head) * p.headDim;
        outQ[outBase + dimPair] = o0;
        outQ[outBase + dimPair + halfDim] = o1;
    } else {
        uint lane = dimPair % 32;
        uint outBase = (seq * p.numKVHeads + head) * p.headDim + lane * 4;
        if (dimPair < 32) {
            outK[outBase + 0] = half(o0);
            outK[outBase + 2] = half(o1);
        } else {
            outK[outBase + 1] = half(o0);
            outK[outBase + 3] = half(o1);
        }
    }
}

/// Fused Q/K norm + RoPE + GQA in a SINGLE dispatch.
/// Replaces norm+RoPE + GQA = 2 dispatches per layer with 1.
/// Phase 1: Q/K heads compute norm + RoPE
/// Phase 2: Q heads cooperatively compute attention against KV cache
///
/// ARCHITECTURE: 32 threads (1 simdgroup) per head. Each thread processes
/// 4 elements: positions [i, i+32, i+64, i+96] of the 128-dim head vector.
/// This eliminates ALL threadgroup_barriers — pure simd_sum reductions only.
/// Thread grid: (32, numHeads+numKVHeads)
struct ERFusedNormRoPEGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    float rmsEps;
    uint kvSeqLen;       // total K/V cache positions
    float attnScale;     // 1/sqrt(headDim)
};

kernel void fused_qk_norm_rope_gqa(
    device const float *Q [[buffer(0)]],          // raw Q from GEMV [numHeads*headDim]
    device const float *K [[buffer(1)]],          // raw K from GEMV [numKVHeads*headDim]
    device const float *qNormW [[buffer(2)]],     // Q norm weight [headDim]
    device const float *kNormW [[buffer(3)]],     // K norm weight [headDim]
    device float *outAttn [[buffer(4)]],          // attention output [numHeads*headDim]
    device half  *kCache [[buffer(5)]],           // K cache [kvSeqLen, numKVHeads, headDim]
    device half  *vCache [[buffer(6)]],           // V cache [kvSeqLen, numKVHeads, headDim]
    constant ERFusedNormRoPEGQAParams &p [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint dimIdx = tid.x;       // 0..31 (32 threads per head)
    uint headIdx = tid.y;
    uint halfDim = p.headDim / 2;  // 64
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimIdx >= 32 || headIdx >= totalHeads) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    device const float* src = isQ ? Q : K;
    device const float* nw = isQ ? qNormW : kNormW;
    uint hb = head * p.headDim;

    // === Phase 1: Per-head RMSNorm + RoPE ===
    // Each thread handles 4 elements: [dimIdx, dimIdx+32, dimIdx+64, dimIdx+96]
    // NeoX RoPE pairs: (dimIdx, dimIdx+halfDim) and (dimIdx+32, dimIdx+32+halfDim)
    float raw_a0 = src[hb + dimIdx];                    // position i
    float raw_a1 = src[hb + dimIdx + halfDim];           // position i+64 (NeoX pair)
    float raw_b0 = src[hb + dimIdx + 32];                // position i+32
    float raw_b1 = src[hb + dimIdx + 32 + halfDim];      // position i+96

    // RMSNorm: sum of squares of all 4 values, simd_sum gives full head total
    float sq = raw_a0 * raw_a0 + raw_a1 * raw_a1 + raw_b0 * raw_b0 + raw_b1 * raw_b1;
    float sumSq = simd_sum(sq);  // 32 threads cover all 128 dims — NO barrier needed

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);

    // Apply norm weights
    float x_a0 = raw_a0 * rs * nw[dimIdx];
    float x_a1 = raw_a1 * rs * nw[dimIdx + halfDim];
    float x_b0 = raw_b0 * rs * nw[dimIdx + 32];
    float x_b1 = raw_b1 * rs * nw[dimIdx + 32 + halfDim];

    // RoPE for pair_a (frequency based on dimIdx)
    float freq_a = 1.0f / pow(p.theta, float(2 * dimIdx) / float(p.headDim));
    float angle_a = float(p.startPos) * (freq_a / p.scalingFactor);
    float ca = cos(angle_a), sa = sin(angle_a);
    float q_a0 = x_a0 * ca - x_a1 * sa;
    float q_a1 = x_a0 * sa + x_a1 * ca;

    // RoPE for pair_b (frequency based on dimIdx+32)
    float freq_b = 1.0f / pow(p.theta, float(2 * (dimIdx + 32)) / float(p.headDim));
    float angle_b = float(p.startPos) * (freq_b / p.scalingFactor);
    float cb = cos(angle_b), sb = sin(angle_b);
    float q_b0 = x_b0 * cb - x_b1 * sb;
    float q_b1 = x_b0 * sb + x_b1 * cb;

    // K heads: write 4 values to cache and exit
    if (!isQ) {
        uint cacheBase = p.startPos * p.numKVHeads * p.headDim + hb;
        kCache[cacheBase + dimIdx]              = half(q_a0);
        kCache[cacheBase + dimIdx + halfDim]    = half(q_a1);
        kCache[cacheBase + dimIdx + 32]         = half(q_b0);
        kCache[cacheBase + dimIdx + 32 + halfDim] = half(q_b1);
        return;  // K threads done — only Q threads continue to GQA
    }

    // === Phase 2: GQA — Q threads compute attention inline ===
    // 32 threads per Q head, each handling 4 dims. simd_sum gives full 128-dim
    // dot product with ZERO barriers.
    uint kvHead = headIdx / (p.numHeads / p.numKVHeads);  // GQA grouping
    uint kvStride = p.numKVHeads * p.headDim;

    // The newest K slice is written by separate K threadgroups in this same dispatch.
    // Metal provides no cross-threadgroup barrier, so Q threadgroups must not read that
    // just-written position back out of global cache. Instead, each Q head recomputes the
    // current-token K vector for its shared kvHead locally and uses that value only for the
    // current position; K threadgroups still write the canonical cache slice for future steps.
    uint kvBaseCurrent = kvHead * p.headDim;
    float raw_k_a0 = K[kvBaseCurrent + dimIdx];
    float raw_k_a1 = K[kvBaseCurrent + dimIdx + halfDim];
    float raw_k_b0 = K[kvBaseCurrent + dimIdx + 32];
    float raw_k_b1 = K[kvBaseCurrent + dimIdx + 32 + halfDim];
    float kSq = raw_k_a0 * raw_k_a0 + raw_k_a1 * raw_k_a1 + raw_k_b0 * raw_k_b0 + raw_k_b1 * raw_k_b1;
    float kSumSq = simd_sum(kSq);
    float kRs = rsqrt(kSumSq / float(p.headDim) + p.rmsEps);
    float k_a0 = raw_k_a0 * kRs * kNormW[dimIdx];
    float k_a1 = raw_k_a1 * kRs * kNormW[dimIdx + halfDim];
    float k_b0 = raw_k_b0 * kRs * kNormW[dimIdx + 32];
    float k_b1 = raw_k_b1 * kRs * kNormW[dimIdx + 32 + halfDim];
    float current_k_a0 = k_a0 * ca - k_a1 * sa;
    float current_k_a1 = k_a0 * sa + k_a1 * ca;
    float current_k_b0 = k_b0 * cb - k_b1 * sb;
    float current_k_b1 = k_b0 * sb + k_b1 * cb;

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_b0 = 0.0f, acc_b1 = 0.0f;

    for (uint kv = 0; kv < p.kvSeqLen; kv++) {
        uint kvBase = kv * kvStride + kvHead * p.headDim;

        // Dot product: Q[head] . K[kv, kvHead] — 4 elements per thread
        float dk_a0;
        float dk_a1;
        float dk_b0;
        float dk_b1;
        if (kv == p.startPos) {
            dk_a0 = current_k_a0;
            dk_a1 = current_k_a1;
            dk_b0 = current_k_b0;
            dk_b1 = current_k_b1;
        } else {
            dk_a0 = float(kCache[kvBase + dimIdx]);
            dk_a1 = float(kCache[kvBase + dimIdx + halfDim]);
            dk_b0 = float(kCache[kvBase + dimIdx + 32]);
            dk_b1 = float(kCache[kvBase + dimIdx + 32 + halfDim]);
        }

        float partial = q_a0 * dk_a0 + q_a1 * dk_a1 + q_b0 * dk_b0 + q_b1 * dk_b1;
        float score = simd_sum(partial) * p.attnScale;  // full dot product, NO barrier!

        // Online softmax. `score` is identical across the full simdgroup after `simd_sum`,
        // so compute the scalar update once and broadcast it instead of repeating the
        // same transcendental work on all 32 lanes.
        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (dimIdx == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        // Weighted V accumulation
        acc_a0 = acc_a0 * correction + prob * float(vCache[kvBase + dimIdx]);
        acc_a1 = acc_a1 * correction + prob * float(vCache[kvBase + dimIdx + halfDim]);
        acc_b0 = acc_b0 * correction + prob * float(vCache[kvBase + dimIdx + 32]);
        acc_b1 = acc_b1 * correction + prob * float(vCache[kvBase + dimIdx + 32 + halfDim]);
    }

    // Normalize and write 4 output values
    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    outAttn[hb + dimIdx]              = acc_a0 * invSum;
    outAttn[hb + dimIdx + halfDim]    = acc_a1 * invSum;
    outAttn[hb + dimIdx + 32]         = acc_b0 * invSum;
    outAttn[hb + dimIdx + 32 + halfDim] = acc_b1 * invSum;
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

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);

    // pRoPE pass-through (NeoX f32->f16 variant). For partial=1.0, rotatedPairs == halfDim.
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[headBase + dimPair]           = half(input[headBase + dimPair]);
        output[headBase + dimPair + halfDim] = half(input[headBase + dimPair + halfDim]);
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = half(x0 * cosValue - x1 * sinValue);
    output[headBase + dimPair + halfDim] = half(x0 * sinValue + x1 * cosValue);
}
