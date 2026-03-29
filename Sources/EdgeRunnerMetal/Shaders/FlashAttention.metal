#include <metal_stdlib>
using namespace metal;

struct ERFlashAttentionParams {
    uint seqLen;
    uint headDim;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
};

struct ERFlashGQAParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;
    uint qOffset;
};

kernel void flash_attention_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashAttentionParams &params [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    const uint br = params.qBlockSize;
    const uint bc = params.kvBlockSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;

    uint qRow = group_id * br + local_id;
    if (qRow >= seqLen) {
        return;
    }

    threadgroup float kTile[16 * 128];
    threadgroup float vTile[16 * 128];
    threadgroup float outputScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    for (uint dim = 0; dim < headDim; dim++) {
        outputScratch[local_id * headDim + dim] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (seqLen + bc - 1) / bc;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * bc;
        uint kvEnd = min(kvStart + bc, seqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id < kvCount) {
            for (uint dim = 0; dim < headDim; dim++) {
                kTile[local_id * headDim + dim] = K[(kvStart + local_id) * headDim + dim];
                vTile[local_id * headDim + dim] = V[(kvStart + local_id) * headDim + dim];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float blockMax = -INFINITY;
        float scores[16];
        for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
            if (params.causal != 0 && kvStart + kvIndex > qRow) {
                scores[kvIndex] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (uint dim = 0; dim < headDim; dim++) {
                dot += Q[qRow * headDim + dim] * kTile[kvIndex * headDim + dim];
            }
            scores[kvIndex] = dot * params.scale;
            blockMax = max(blockMax, scores[kvIndex]);
        }

        float nextMax = max(runningMax, blockMax);
        float correction = exp(runningMax - nextMax);

        float blockSum = 0.0f;
        float probs[16];
        for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
            if (scores[kvIndex] == -INFINITY) {
                probs[kvIndex] = 0.0f;
            } else {
                probs[kvIndex] = exp(scores[kvIndex] - nextMax);
            }
            blockSum += probs[kvIndex];
        }

        runningSum = runningSum * correction + blockSum;

        for (uint dim = 0; dim < headDim; dim++) {
            float value = outputScratch[local_id * headDim + dim] * correction;
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                value += probs[kvIndex] * vTile[kvIndex * headDim + dim];
            }
            outputScratch[local_id * headDim + dim] = value;
        }

        runningMax = nextMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    for (uint dim = 0; dim < headDim; dim++) {
        O[qRow * headDim + dim] = outputScratch[local_id * headDim + dim] * invSum;
    }
}

kernel void flash_attention_gqa_simd_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint kvStride = params.numKVHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint kvBase = kv * kvStride + kvHeadIndex * params.headDim;
        float partial =
            q0 * K[kvBase + lane] +
            q1 * K[kvBase + lane + 32] +
            q2 * K[kvBase + lane + 64] +
            q3 * K[kvBase + lane + 96];
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
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

        acc0 = acc0 * correction + prob * V[kvBase + lane];
        acc1 = acc1 * correction + prob * V[kvBase + lane + 32];
        acc2 = acc2 * correction + prob * V[kvBase + lane + 64];
        acc3 = acc3 * correction + prob * V[kvBase + lane + 96];
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}

kernel void flash_attention_gqa_simd_qf32_kvf16(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint kvStride = params.numKVHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint kvBase = kv * kvStride + kvHeadIndex * params.headDim;
        float partial =
            q0 * float(K[kvBase + lane]) +
            q1 * float(K[kvBase + lane + 32]) +
            q2 * float(K[kvBase + lane + 64]) +
            q3 * float(K[kvBase + lane + 96]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
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

        acc0 = acc0 * correction + prob * float(V[kvBase + lane]);
        acc1 = acc1 * correction + prob * float(V[kvBase + lane + 32]);
        acc2 = acc2 * correction + prob * float(V[kvBase + lane + 64]);
        acc3 = acc3 * correction + prob * float(V[kvBase + lane + 96]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}
