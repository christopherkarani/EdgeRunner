#include <metal_stdlib>
using namespace metal;

struct ERGemma4RMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

struct ERGemma4ResidualRMSNormParams {
    uint count;
    float eps;
};

struct ERGemma4ResidualRMSNormRowsParams {
    uint rows;
    uint cols;
    float eps;
};

struct ERGemma4EmbeddingParams {
    uint rowWidth;
    uint tokenCount;
    uint rowStrideBytes;
    ulong tableByteOffset;
    float scale;
};

struct ERGemma4DecodeGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    uint headDim;
    uint kvStart;
    uint kvCount;
    uint kvCapacity;
    float scale;
};

static inline float gemma4_f16_at(device const uchar *ptr, uint offset) {
    ushort bits = ushort(ptr[offset]) | (ushort(ptr[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

static inline float gemma4_dequant_q6_k_value(device const uchar *block, uint inBlock) {
    const float d = gemma4_f16_at(block, 208);
    const uint halfBlock = inBlock / 128;
    const uint within = inBlock - halfBlock * 128;
    const uint lane = within & 31;
    const uint quarter = within / 32;
    const uint qlBase = halfBlock * 64;
    const uint qhBase = 128 + halfBlock * 32;
    const uint scaleBase = 192 + halfBlock * 8;

    const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
    const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
    const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
    const int q6 = int(lower4 | (upper2 << 4)) - 32;
    int scaleRaw = int(block[scaleBase + quarter * 2 + lane / 16]);
    if (scaleRaw >= 128) {
        scaleRaw -= 256;
    }
    return d * float(scaleRaw) * float(q6);
}

kernel void gemma4_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERGemma4RMSNormParams &params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];
    const uint offset = row * params.cols;

    float sum = 0.0f;
    for (uint col = tid; col < params.cols; col += threadCount) {
        const float value = input[offset + col];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.cols) + params.eps);
    for (uint col = tid; col < params.cols; col += threadCount) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}

kernel void gemma4_residual_rmsnorm_add_f32(
    device const float *residual [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERGemma4ResidualRMSNormParams &params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];

    float sum = 0.0f;
    for (uint index = tid; index < params.count; index += threadCount) {
        const float value = input[index];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.count) + params.eps);
    for (uint index = tid; index < params.count; index += threadCount) {
        output[index] = residual[index] + input[index] * scale * weight[index];
    }
}

kernel void gemma4_residual_rmsnorm_add_rows_f32(
    device const float *residual [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERGemma4ResidualRMSNormRowsParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];
    const uint offset = row * params.cols;

    float sum = 0.0f;
    for (uint col = tid; col < params.cols; col += threadCount) {
        const float value = input[offset + col];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.cols) + params.eps);
    for (uint col = tid; col < params.cols; col += threadCount) {
        output[offset + col] = residual[offset + col] + input[offset + col] * scale * weight[col];
    }
}

kernel void gemma4_store_f32_to_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        output[gid] = half(input[gid]);
    }
}

kernel void gemma4_mul_scalar_f32(
    device float *values [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        values[gid] *= scale;
    }
}

kernel void gemma4_gather_token_embedding_q6_k(
    device const uchar *table [[buffer(0)]],
    device const int *tokens [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERGemma4EmbeddingParams &params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint tokenIndex = gid.y;
    if (col >= params.rowWidth || tokenIndex >= params.tokenCount) {
        return;
    }

    const int tokenID = tokens[tokenIndex];
    const uint blockIndex = col / 256;
    const uint inBlock = col - blockIndex * 256;
    device const uchar *block = table
        + params.tableByteOffset
        + ulong(tokenID) * ulong(params.rowStrideBytes)
        + ulong(blockIndex) * 210ul;

    output[tokenIndex * params.rowWidth + col] = gemma4_dequant_q6_k_value(block, inBlock) * params.scale;
}

static inline float gemma4_decode_gqa_score(
    device const float *Q,
    device const half *K,
    constant ERGemma4DecodeGQAParams &params,
    uint head,
    uint kvHead,
    uint physicalPosition
) {
    const uint qBase = head * params.headDim;
    const uint kBase = (physicalPosition * params.numKVHeads + kvHead) * params.headDim;
    float dot = 0.0f;
    for (uint dim = 0; dim < params.headDim; ++dim) {
        dot += Q[qBase + dim] * float(K[kBase + dim]);
    }
    return dot * params.scale;
}

kernel void gemma4_decode_gqa_f16kv_windowed(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGemma4DecodeGQAParams &params [[buffer(4)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    const uint total = params.numHeads * params.headDim;
    if (outputIndex >= total) {
        return;
    }

    const uint dim = outputIndex % params.headDim;
    const uint head = outputIndex / params.headDim;
    const uint kvHead = head / params.groupSize;

    if (params.kvCount == 0 || params.kvCapacity == 0) {
        O[outputIndex] = 0.0f;
        return;
    }

    float maxScore = -INFINITY;
    for (uint kvIndex = 0; kvIndex < params.kvCount; ++kvIndex) {
        const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
        const float score = gemma4_decode_gqa_score(Q, K, params, head, kvHead, physical);
        maxScore = max(maxScore, score);
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (uint kvIndex = 0; kvIndex < params.kvCount; ++kvIndex) {
        const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
        const float score = gemma4_decode_gqa_score(Q, K, params, head, kvHead, physical);
        const float weight = exp(score - maxScore);
        const uint vBase = (physical * params.numKVHeads + kvHead) * params.headDim;
        value += weight * float(V[vBase + dim]);
        sum += weight;
    }

    O[outputIndex] = sum > 0.0f ? value / sum : 0.0f;
}

kernel void gemma4_decode_gqa_f16kv_windowed_fast(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGemma4DecodeGQAParams &params [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint maxWindow = 512;
    threadgroup float scores[maxWindow];
    threadgroup float sumShared;

    if (head >= params.numHeads) {
        return;
    }

    const uint kvHead = head / params.groupSize;
    const uint qBase = head * params.headDim;

    if (params.kvCount == 0 || params.kvCapacity == 0) {
        for (uint dim = tid; dim < params.headDim; dim += maxWindow) {
            O[qBase + dim] = 0.0f;
        }
        return;
    }

    if (tid < params.kvCount && tid < maxWindow) {
        const uint physical = (params.kvStart + tid) % params.kvCapacity;
        const uint kBase = (physical * params.numKVHeads + kvHead) * params.headDim;
        float dot = 0.0f;
        for (uint dim = 0; dim < params.headDim; ++dim) {
            dot += Q[qBase + dim] * float(K[kBase + dim]);
        }
        scores[tid] = dot * params.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maxScore = -INFINITY;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            maxScore = max(maxScore, scores[kvIndex]);
        }
        float sum = 0.0f;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            const float weight = exp(scores[kvIndex] - maxScore);
            scores[kvIndex] = weight;
            sum += weight;
        }
        sumShared = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normalizer = sumShared > 0.0f ? 1.0f / sumShared : 0.0f;
    for (uint dim = tid; dim < params.headDim; dim += maxWindow) {
        float value = 0.0f;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
            const uint vBase = (physical * params.numKVHeads + kvHead) * params.headDim;
            value += scores[kvIndex] * float(V[vBase + dim]);
        }
        O[qBase + dim] = value * normalizer;
    }
}
