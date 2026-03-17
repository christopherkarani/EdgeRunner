#include <metal_stdlib>
using namespace metal;

struct ERGQAParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
};

kernel void gqa_attention_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;
    const uint blockSize = params.qBlockSize;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    if (qRow >= seqLen) {
        return;
    }

    device const float *qHead = Q + headIndex * seqLen * headDim;
    device const float *kHead = K + kvHeadIndex * seqLen * headDim;
    device const float *vHead = V + kvHeadIndex * seqLen * headDim;
    device float *oHead = O + headIndex * seqLen * headDim;

    threadgroup float kTile[16 * 128];
    threadgroup float vTile[16 * 128];
    threadgroup float outputScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    for (uint dim = 0; dim < headDim; dim++) {
        outputScratch[local_id.x * headDim + dim] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (seqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, seqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id.x < kvCount) {
            for (uint dim = 0; dim < headDim; dim++) {
                kTile[local_id.x * headDim + dim] = kHead[(kvStart + local_id.x) * headDim + dim];
                vTile[local_id.x * headDim + dim] = vHead[(kvStart + local_id.x) * headDim + dim];
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
                dot += qHead[qRow * headDim + dim] * kTile[kvIndex * headDim + dim];
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
            float value = outputScratch[local_id.x * headDim + dim] * correction;
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                value += probs[kvIndex] * vTile[kvIndex * headDim + dim];
            }
            outputScratch[local_id.x * headDim + dim] = value;
        }

        runningMax = nextMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    for (uint dim = 0; dim < headDim; dim++) {
        oHead[qRow * headDim + dim] = outputScratch[local_id.x * headDim + dim] * invSum;
    }
}
