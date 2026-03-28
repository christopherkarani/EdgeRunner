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

struct ERPackKVDecodeCacheParams {
    uint tokenCount;
    uint numKVHeads;
    uint headDim;
    uint destinationStartToken;
};

kernel void pack_kv_decode_cache_f16(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant ERPackKVDecodeCacheParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint lane = gid.x;
    uint kvHead = gid.y;
    uint token = gid.z;
    if (lane >= 32 || kvHead >= params.numKVHeads || token >= params.tokenCount) return;

    uint srcBase = (token * params.numKVHeads + kvHead) * params.headDim;
    uint dstToken = params.destinationStartToken + token;
    uint dstBase = (dstToken * params.numKVHeads + kvHead) * params.headDim + lane * 4;

    destination[dstBase + 0] = source[srcBase + lane];
    destination[dstBase + 1] = source[srcBase + lane + 32];
    destination[dstBase + 2] = source[srcBase + lane + 64];
    destination[dstBase + 3] = source[srcBase + lane + 96];
}

kernel void pack_kv_decode_cache_pair_f16(
    device const half *keySource [[buffer(0)]],
    device const half *valueSource [[buffer(1)]],
    device half *keyDestination [[buffer(2)]],
    device half *valueDestination [[buffer(3)]],
    constant ERPackKVDecodeCacheParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint lane = gid.x;
    uint kvHead = gid.y;
    uint token = gid.z;
    if (lane >= 32 || kvHead >= params.numKVHeads || token >= params.tokenCount) return;

    uint srcBase = (token * params.numKVHeads + kvHead) * params.headDim;
    uint dstToken = params.destinationStartToken + token;
    uint dstBase = (dstToken * params.numKVHeads + kvHead) * params.headDim + lane * 4;

    keyDestination[dstBase + 0] = keySource[srcBase + lane];
    keyDestination[dstBase + 1] = keySource[srcBase + lane + 32];
    keyDestination[dstBase + 2] = keySource[srcBase + lane + 64];
    keyDestination[dstBase + 3] = keySource[srcBase + lane + 96];

    valueDestination[dstBase + 0] = valueSource[srcBase + lane];
    valueDestination[dstBase + 1] = valueSource[srcBase + lane + 32];
    valueDestination[dstBase + 2] = valueSource[srcBase + lane + 64];
    valueDestination[dstBase + 3] = valueSource[srcBase + lane + 96];
}

struct ERPackedDecodeGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint kvSeqLen;
    float scale;
};

kernel void gqa_decode_attention_packed_f16kv(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERPackedDecodeGQAParams &params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    if (lane >= 32 || headIndex >= params.numHeads) return;

    uint kvHeadIndex = headIndex / (params.numHeads / params.numKVHeads);
    uint qBase = headIndex * params.headDim;

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

    for (uint kv = 0; kv < params.kvSeqLen; ++kv) {
        uint kvBase = (kv * params.numKVHeads + kvHeadIndex) * params.headDim + lane * 4;
        half4 packedK = *reinterpret_cast<const device half4 *>(K + kvBase);
        float partial = q0 * float(packedK[0]) +
            q1 * float(packedK[1]) +
            q2 * float(packedK[2]) +
            q3 * float(packedK[3]);
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

        half4 packedV = *reinterpret_cast<const device half4 *>(V + kvBase);
        acc0 = acc0 * correction + prob * float(packedV[0]);
        acc1 = acc1 * correction + prob * float(packedV[1]);
        acc2 = acc2 * correction + prob * float(packedV[2]);
        acc3 = acc3 * correction + prob * float(packedV[3]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}
