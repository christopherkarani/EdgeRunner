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

    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;

    threadgroup float4 kTile[16 * maxHeadDim4];
    threadgroup float4 vTile[16 * maxHeadDim4];
    threadgroup float4 outputScratch[16 * maxHeadDim4];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            outputScratch[local_id.x * maxHeadDim4 + dim4] = float4(0.0f);
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
            const device float4 *kVec = reinterpret_cast<const device float4 *>(K + kBase);
            const device float4 *vVec = reinterpret_cast<const device float4 *>(V + kBase);
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = kVec[dim4];
                vTile[tileIndex] = vVec[dim4];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                uint tileBase = kvIndex * maxHeadDim4;
                for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                    dot += metal::dot(qVec[dim4], kTile[tileBase + dim4]);
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

            uint outBase = local_id.x * maxHeadDim4;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                float4 value = outputScratch[outBase + dim4] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                    value += probs[kvIndex] * vTile[kvIndex * maxHeadDim4 + dim4];
                }
                outputScratch[outBase + dim4] = value;
            }

            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        device float4 *oVec = reinterpret_cast<device float4 *>(O + oBase);
        uint outBase = local_id.x * maxHeadDim4;
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            oVec[dim4] = outputScratch[outBase + dim4] * invSum;
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

    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;

    threadgroup float4 kTile[16 * maxHeadDim4];
    threadgroup float4 vTile[16 * maxHeadDim4];
    threadgroup float4 outputScratch[16 * maxHeadDim4];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim4 = 0; dim4 < headDim4; dim4++)
            outputScratch[local_id.x * maxHeadDim4 + dim4] = float4(0.0f);
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
            const device half4 *kVec = reinterpret_cast<const device half4 *>(K + kBase);
            const device half4 *vVec = reinterpret_cast<const device half4 *>(V + kBase);
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = float4(kVec[dim4]);
                vTile[tileIndex] = float4(vVec[dim4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                uint tileBase = kvIndex * maxHeadDim4;
                for (uint dim4 = 0; dim4 < headDim4; dim4++)
                    dot += metal::dot(qVec[dim4], kTile[tileBase + dim4]);
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

            uint outBase = local_id.x * maxHeadDim4;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                float4 value = outputScratch[outBase + dim4] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++)
                    value += probs[kvIndex] * vTile[kvIndex * maxHeadDim4 + dim4];
                outputScratch[outBase + dim4] = value;
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        device float4 *oVec = reinterpret_cast<device float4 *>(O + oBase);
        uint outBase = local_id.x * maxHeadDim4;
        for (uint dim4 = 0; dim4 < headDim4; dim4++)
            oVec[dim4] = outputScratch[outBase + dim4] * invSum;
    }
}
