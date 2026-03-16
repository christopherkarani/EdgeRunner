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
