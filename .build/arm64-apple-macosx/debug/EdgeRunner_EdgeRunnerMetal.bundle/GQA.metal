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
    uint kvSeqLen;    // K/V sequence length (0 = same as seqLen)
    uint qOffset;     // offset for Q positions in causal mask (0 = default)
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
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;  // causal mask: Q position = qRow + qOffset
    const uint blockSize = params.qBlockSize;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    // Track whether this thread has a valid Q position.
    // Inactive Q threads still participate in KV tile loading and barriers.
    bool activeQ = (qRow < seqLen);

    // [S, H, D] layout strides
    const uint qStride = numHeads * headDim;       // stride between sequence positions for Q/O
    const uint kvStride = numKVHeads * headDim;     // stride between sequence positions for K/V

    threadgroup float kTile[16 * 128];
    threadgroup float vTile[16 * 128];
    threadgroup float outputScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim = 0; dim < headDim; dim++) {
            outputScratch[local_id.x * headDim + dim] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        // ALL threads participate in KV tile loading (not just active Q threads)
        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint kBase = kvPos * kvStride + kvHeadIndex * headDim;
            for (uint dim = 0; dim < headDim; dim++) {
                kTile[local_id.x * headDim + dim] = K[kBase + dim];
                vTile[local_id.x * headDim + dim] = V[kBase + dim];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                for (uint dim = 0; dim < headDim; dim++) {
                    dot += Q[qBase + dim] * kTile[kvIndex * headDim + dim];
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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        for (uint dim = 0; dim < headDim; dim++) {
            O[oBase + dim] = outputScratch[local_id.x * headDim + dim] * invSum;
        }
    }
}

// === Float16 KV variant ===
// K/V stored as half in KV cache — halves attention memory bandwidth.
// Q and O remain float32 (fresh from GEMV). K/V converted to float in threadgroup memory.
kernel void gqa_attention_f16kv(
    device const float *Q [[buffer(0)]],
    device const half  *K [[buffer(1)]],
    device const half  *V [[buffer(2)]],
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
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    const uint qStride = numHeads * headDim;
    const uint kvStride = numKVHeads * headDim;

    threadgroup float kTile[16 * 128];
    threadgroup float vTile[16 * 128];
    threadgroup float outputScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim = 0; dim < headDim; dim++)
            outputScratch[local_id.x * headDim + dim] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint kBase = kvPos * kvStride + kvHeadIndex * headDim;
            for (uint dim = 0; dim < headDim; dim++) {
                kTile[local_id.x * headDim + dim] = float(K[kBase + dim]);
                vTile[local_id.x * headDim + dim] = float(V[kBase + dim]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                for (uint dim = 0; dim < headDim; dim++)
                    dot += Q[qBase + dim] * kTile[kvIndex * headDim + dim];
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0f;
            float probs[16];
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                probs[kvIndex] = (scores[kvIndex] == -INFINITY) ? 0.0f : exp(scores[kvIndex] - nextMax);
                blockSum += probs[kvIndex];
            }
            runningSum = runningSum * correction + blockSum;

            for (uint dim = 0; dim < headDim; dim++) {
                float value = outputScratch[local_id.x * headDim + dim] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++)
                    value += probs[kvIndex] * vTile[kvIndex * headDim + dim];
                outputScratch[local_id.x * headDim + dim] = value;
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        for (uint dim = 0; dim < headDim; dim++)
            O[oBase + dim] = outputScratch[local_id.x * headDim + dim] * invSum;
    }
}
