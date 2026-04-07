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
    float frequency = 1.0f / powr(params.theta, exponent);
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
    float frequency = 1.0f / powr(params.theta, exponent);
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
    uint2 tid [[thread_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgIdx [[simdgroup_index_in_threadgroup]]
) {
    // 64 threads per TG (2 simdgroups). Each SG has 32 threads covering all 128 head dims.
    // K heads: SG 1 exits immediately (only SG 0 does K work).
    // Q heads: both SGs run Phase 1 identically, then split KV positions in Phase 2.
    uint dimIdx = tiisg;           // 0..31 (same dimension mapping in both SGs)
    uint headIdx = tid.y;
    uint halfDim = p.headDim / 2;  // 64
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (headIdx >= totalHeads) return;

    bool isQ = headIdx < p.numHeads;

    // K heads only need 1 simdgroup — SG 1 exits immediately
    if (!isQ && sgIdx == 1) return;

    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    device const float* src = isQ ? Q : K;
    device const float* nw = isQ ? qNormW : kNormW;
    uint hb = head * p.headDim;

    // === Phase 1: Per-head RMSNorm + RoPE ===
    // Both SGs compute identically (redundant for Q, but avoids a barrier)
    float raw_a0 = src[hb + dimIdx];
    float raw_a1 = src[hb + dimIdx + halfDim];
    float raw_b0 = src[hb + dimIdx + 32];
    float raw_b1 = src[hb + dimIdx + 32 + halfDim];

    float sq = raw_a0 * raw_a0 + raw_a1 * raw_a1 + raw_b0 * raw_b0 + raw_b1 * raw_b1;
    float sumSq = simd_sum(sq);

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);

    float x_a0 = raw_a0 * rs * nw[dimIdx];
    float x_a1 = raw_a1 * rs * nw[dimIdx + halfDim];
    float x_b0 = raw_b0 * rs * nw[dimIdx + 32];
    float x_b1 = raw_b1 * rs * nw[dimIdx + 32 + halfDim];

    float freq_a = 1.0f / pow(p.theta, float(2 * dimIdx) / float(p.headDim));
    float angle_a = float(p.startPos) * (freq_a / p.scalingFactor);
    float ca = cos(angle_a), sa = sin(angle_a);
    float q_a0 = x_a0 * ca - x_a1 * sa;
    float q_a1 = x_a0 * sa + x_a1 * ca;

    float freq_b = 1.0f / pow(p.theta, float(2 * (dimIdx + 32)) / float(p.headDim));
    float angle_b = float(p.startPos) * (freq_b / p.scalingFactor);
    float cb = cos(angle_b), sb = sin(angle_b);
    float q_b0 = x_b0 * cb - x_b1 * sb;
    float q_b1 = x_b0 * sb + x_b1 * cb;

    // K heads: write 4 values to cache and exit (SG 0 only, SG 1 already returned)
    if (!isQ) {
        uint cacheBase = p.startPos * p.numKVHeads * p.headDim + hb;
        kCache[cacheBase + dimIdx]              = half(q_a0);
        kCache[cacheBase + dimIdx + halfDim]    = half(q_a1);
        kCache[cacheBase + dimIdx + 32]         = half(q_b0);
        kCache[cacheBase + dimIdx + 32 + halfDim] = half(q_b1);
        return;
    }

    // === Phase 2: GQA — 2 simdgroups per Q head, position-split ===
    // SG 0 processes even KV positions, SG 1 processes odd positions.
    // This halves the serial dependency chain, reducing GQA latency.
    uint kvHead = headIdx / (p.numHeads / p.numKVHeads);
    uint kvStride = p.numKVHeads * p.headDim;

    // Re-derive current-position K locally (avoids cross-TG race with K writers)
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

    // Position-split loop: SG 0 handles even positions, SG 1 handles odd
    for (uint kvPair = 0; kvPair < (p.kvSeqLen + 1) / 2; kvPair++) {
        uint kv = kvPair * 2 + sgIdx;
        if (kv >= p.kvSeqLen) break;

        uint kvBase = kv * kvStride + kvHead * p.headDim;

        float dk_a0, dk_a1, dk_b0, dk_b1;
        if (kv == p.startPos) {
            dk_a0 = current_k_a0; dk_a1 = current_k_a1;
            dk_b0 = current_k_b0; dk_b1 = current_k_b1;
        } else {
            dk_a0 = float(kCache[kvBase + dimIdx]);
            dk_a1 = float(kCache[kvBase + dimIdx + halfDim]);
            dk_b0 = float(kCache[kvBase + dimIdx + 32]);
            dk_b1 = float(kCache[kvBase + dimIdx + 32 + halfDim]);
        }

        float partial = q_a0 * dk_a0 + q_a1 * dk_a1 + q_b0 * dk_b0 + q_b1 * dk_b1;
        float score = simd_sum(partial) * p.attnScale;

        float oldMax = runMax;
        runMax = max(runMax, score);
        float correction = exp(oldMax - runMax);
        float prob = exp(score - runMax);
        runSum = runSum * correction + prob;

        acc_a0 = acc_a0 * correction + prob * float(vCache[kvBase + dimIdx]);
        acc_a1 = acc_a1 * correction + prob * float(vCache[kvBase + dimIdx + halfDim]);
        acc_b0 = acc_b0 * correction + prob * float(vCache[kvBase + dimIdx + 32]);
        acc_b1 = acc_b1 * correction + prob * float(vCache[kvBase + dimIdx + 32 + halfDim]);
    }

    // === Merge: combine SG 0 and SG 1 accumulators via threadgroup memory ===
    threadgroup float tg_acc[128];  // SG 1's per-dim accumulator values
    threadgroup float tg_max1;
    threadgroup float tg_sum1;

    if (sgIdx == 1) {
        tg_acc[dimIdx]              = acc_a0;
        tg_acc[dimIdx + halfDim]    = acc_a1;
        tg_acc[dimIdx + 32]         = acc_b0;
        tg_acc[dimIdx + 32 + halfDim] = acc_b1;
        if (dimIdx == 0) {
            tg_max1 = runMax;
            tg_sum1 = runSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgIdx == 0) {
        float max1 = tg_max1;
        float sum1 = tg_sum1;

        // Online softmax merge of two independent accumulators
        float maxFinal = max(runMax, max1);
        float c0 = exp(runMax - maxFinal);
        float c1 = exp(max1 - maxFinal);
        float sumFinal = runSum * c0 + sum1 * c1;
        float invSum = sumFinal > 0.0f ? 1.0f / sumFinal : 0.0f;

        float other_a0 = tg_acc[dimIdx];
        float other_a1 = tg_acc[dimIdx + halfDim];
        float other_b0 = tg_acc[dimIdx + 32];
        float other_b1 = tg_acc[dimIdx + 32 + halfDim];

        outAttn[hb + dimIdx]              = (acc_a0 * c0 + other_a0 * c1) * invSum;
        outAttn[hb + dimIdx + halfDim]    = (acc_a1 * c0 + other_a1 * c1) * invSum;
        outAttn[hb + dimIdx + 32]         = (acc_b0 * c0 + other_b0 * c1) * invSum;
        outAttn[hb + dimIdx + 32 + halfDim] = (acc_b1 * c0 + other_b1 * c1) * invSum;
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
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);
    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = half(x0 * cosValue - x1 * sinValue);
    output[headBase + dimPair + halfDim] = half(x0 * sinValue + x1 * cosValue);
}
