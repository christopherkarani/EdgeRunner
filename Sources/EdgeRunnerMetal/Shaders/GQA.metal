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

static inline float gqa_attention_f32_wide_score(
    device const float *Q,
    device const float *K,
    constant ERGQAParams &params,
    uint qRow,
    uint headIndex,
    uint kvHeadIndex,
    uint kvPos,
    uint qStride,
    uint kvStride
) {
    const uint qBase = qRow * qStride + headIndex * params.headDim;
    const uint kBase = kvPos * kvStride + kvHeadIndex * params.headDim;
    float dot = 0.0f;
    for (uint dim = 0; dim < params.headDim; ++dim) {
        dot += Q[qBase + dim] * K[kBase + dim];
    }
    return dot * params.scale;
}

static inline void gqa_attention_f32_wide_impl(
    device const float *Q,
    device const float *K,
    device const float *V,
    device float *O,
    constant ERGQAParams &params,
    device const float *additiveMask,
    bool useAdditiveMask,
    uint outputIndex
) {
    const uint headDim = params.headDim;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint totalScalars = seqLen * numHeads * headDim;
    if (outputIndex >= totalScalars) {
        return;
    }

    const uint dim = outputIndex % headDim;
    const uint headIndex = (outputIndex / headDim) % numHeads;
    const uint qRow = outputIndex / (numHeads * headDim);
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint qStride = numHeads * headDim;
    const uint kvStride = numKVHeads * headDim;
    const uint causalLimit = qRow + params.qOffset;

    float maxScore = -INFINITY;
    for (uint kvPos = 0; kvPos < kvSeqLen; ++kvPos) {
        if (params.causal != 0 && kvPos > causalLimit) {
            continue;
        }
        float score = gqa_attention_f32_wide_score(
            Q,
            K,
            params,
            qRow,
            headIndex,
            kvHeadIndex,
            kvPos,
            qStride,
            kvStride
        );
        if (useAdditiveMask) {
            score += additiveMask[qRow * kvSeqLen + kvPos];
        }
        maxScore = max(maxScore, score);
    }

    if (maxScore == -INFINITY) {
        O[outputIndex] = 0.0f;
        return;
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (uint kvPos = 0; kvPos < kvSeqLen; ++kvPos) {
        if (params.causal != 0 && kvPos > causalLimit) {
            continue;
        }
        float score = gqa_attention_f32_wide_score(
            Q,
            K,
            params,
            qRow,
            headIndex,
            kvHeadIndex,
            kvPos,
            qStride,
            kvStride
        );
        if (useAdditiveMask) {
            score += additiveMask[qRow * kvSeqLen + kvPos];
        }
        if (score == -INFINITY) {
            continue;
        }
        float weight = exp(score - maxScore);
        uint vBase = kvPos * kvStride + kvHeadIndex * headDim;
        value += weight * V[vBase + dim];
        sum += weight;
    }

    O[outputIndex] = sum > 0.0f ? value / sum : 0.0f;
}

kernel void gqa_attention_f32_wide(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    gqa_attention_f32_wide_impl(Q, K, V, O, params, Q, false, outputIndex);
}

kernel void gqa_attention_f32_masked_wide(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    device const float *additiveMask [[buffer(5)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    gqa_attention_f32_wide_impl(Q, K, V, O, params, additiveMask, true, outputIndex);
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

constant uint gqa_q8_0_block_bytes = 34;
constant uint gqa_q8_0_weights_per_block = 32;

static inline float4 gqa_q8_0_load_float4(
    device const uchar *row,
    uint dim4
) {
    const uint scalarIndex = dim4 * 4;
    const uint blockIndex = scalarIndex / gqa_q8_0_weights_per_block;
    const uint inBlockIndex = scalarIndex % gqa_q8_0_weights_per_block;
    device const uchar *block = row + blockIndex * gqa_q8_0_block_bytes;
    const float scale = float(as_type<half>(*(device const ushort *) block));
    return scale * float4(
        float(as_type<char>(block[2 + inBlockIndex + 0])),
        float(as_type<char>(block[2 + inBlockIndex + 1])),
        float(as_type<char>(block[2 + inBlockIndex + 2])),
        float(as_type<char>(block[2 + inBlockIndex + 3]))
    );
}

kernel void gqa_attention_q8kv(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uchar *V [[buffer(2)]],
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
    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;
    const uint q8BlocksPerRow = headDim / gqa_q8_0_weights_per_block;
    const uint q8RowBytes = q8BlocksPerRow * gqa_q8_0_block_bytes;

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

        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint rowIndex = kvPos * numKVHeads + kvHeadIndex;
            device const uchar *kRow = K + rowIndex * q8RowBytes;
            device const uchar *vRow = V + rowIndex * q8RowBytes;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = gqa_q8_0_load_float4(kRow, dim4);
                vTile[tileIndex] = gqa_q8_0_load_float4(vRow, dim4);
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
                probs[kvIndex] = (scores[kvIndex] == -INFINITY) ? 0.0f : exp(scores[kvIndex] - nextMax);
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

struct ERPackedDecodeSplitKVParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint kvSeqLen;
    uint kvBlockSize;
    uint blockCount;
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

kernel void gqa_decode_attention_packed_f16kv_partial(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *partialMax [[buffer(3)]],
    device float *partialSum [[buffer(4)]],
    device float *partialAcc [[buffer(5)]],
    constant ERPackedDecodeSplitKVParams &params [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint blockIndex = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || blockIndex >= params.blockCount) return;

    uint groupSize = params.numHeads / params.numKVHeads;
    uint kvHeadIndex = headIndex / groupSize;
    uint qBase = headIndex * params.headDim;
    uint blockStart = blockIndex * params.kvBlockSize;
    uint blockEnd = min(blockStart + params.kvBlockSize, params.kvSeqLen);

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

    for (uint kv = blockStart; kv < blockEnd; ++kv) {
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

    uint partialIndex = blockIndex * params.numHeads + headIndex;
    uint partialBase = partialIndex * params.headDim;
    partialAcc[partialBase + lane] = acc0;
    partialAcc[partialBase + lane + 32] = acc1;
    partialAcc[partialBase + lane + 64] = acc2;
    partialAcc[partialBase + lane + 96] = acc3;
    if (lane == 0) {
        partialMax[partialIndex] = runMax;
        partialSum[partialIndex] = runSum;
    }
}

kernel void gqa_decode_attention_packed_f16kv_reduce(
    device const float *partialMax [[buffer(0)]],
    device const float *partialSum [[buffer(1)]],
    device const float *partialAcc [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERPackedDecodeSplitKVParams &params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    if (lane >= 32 || headIndex >= params.numHeads) return;

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint blockIndex = 0; blockIndex < params.blockCount; ++blockIndex) {
        uint partialIndex = blockIndex * params.numHeads + headIndex;
        uint partialBase = partialIndex * params.headDim;
        float blockMax = partialMax[partialIndex];
        float blockSum = partialSum[partialIndex];

        float nextRunMax = max(runMax, blockMax);
        float correction = exp(runMax - nextRunMax);
        float blockCorrection = exp(blockMax - nextRunMax);

        acc0 = acc0 * correction + partialAcc[partialBase + lane] * blockCorrection;
        acc1 = acc1 * correction + partialAcc[partialBase + lane + 32] * blockCorrection;
        acc2 = acc2 * correction + partialAcc[partialBase + lane + 64] * blockCorrection;
        acc3 = acc3 * correction + partialAcc[partialBase + lane + 96] * blockCorrection;
        runSum = runSum * correction + blockSum * blockCorrection;
        runMax = nextRunMax;
    }

    uint qBase = headIndex * params.headDim;
    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}
