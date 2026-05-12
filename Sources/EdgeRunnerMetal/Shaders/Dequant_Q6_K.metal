#include <metal_stdlib>
using namespace metal;

struct ERDequantQ6KParams {
    uint superBlockCount;
    uint outputOffset;
};

struct ERQ6KGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint Q6_K_BLOCK_BYTES = 210;
constant uint Q6_K_WEIGHTS_PER_BLOCK = 256;
constant uint Q6_K_GEMV_THREADS_PER_ROW = 256;
constant uint Q6_K_GEMV_PACKED_THREADS_PER_ROW = 64;

static inline float dequant_q6_k_value(device const uchar* block, uint inBlock, float d) {
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
    const char scale = as_type<char>(block[scaleBase + (quarter * 2) + lane / 16]);
    return d * float(scale) * float(q6);
}

kernel void dequant_q6_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ6KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q6_K_BLOCK_BYTES;

    // d at offset 208 (float16)
    device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
    float d = float(dPtr[0]);

    uint outBase = params.outputOffset + tid * Q6_K_WEIGHTS_PER_BLOCK;

    for (uint i = 0; i < 256; ++i) {
        output[outBase + i] = dequant_q6_k_value(block, i, d);
    }
}

kernel void q6_k_gemv_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ6KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScale[16];
    threadgroup float sharedD;
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q6_K_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q6_K_BLOCK_BYTES;

        if (local_id == 0) {
            device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
            sharedD = float(dPtr[0]);
        }
        if (local_id < 16) {
            sharedScale[local_id] = float(as_type<char>(block[192 + local_id]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q6_K_WEIGHTS_PER_BLOCK) {
            const uint halfBlock = inBlock / 128;
            const uint within = inBlock - halfBlock * 128;
            const uint lane = within & 31;
            const uint quarter = within / 32;
            const uint qlBase = halfBlock * 64;
            const uint qhBase = 128 + halfBlock * 32;
            const uint scaleIndex = halfBlock * 8 + quarter * 2 + lane / 16;
            const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
            const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
            const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
            const int q6 = int(lower4 | (upper2 << 4)) - 32;
            float weight = sharedD * sharedScale[scaleIndex] * float(q6);
            uint col = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q6_K_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q6_k_gemv_packed_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ6KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows || local_id >= Q6_K_GEMV_PACKED_THREADS_PER_ROW) {
        return;
    }

    float partial = 0.0f;
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q6_K_BLOCK_BYTES;
    const uint halfBlock = local_id >> 5;
    const uint lane = local_id & 31;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q6_K_BLOCK_BYTES;
        device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
        const float d = float(dPtr[0]);

        const uint qlBase = halfBlock * 64;
        const uint qhBase = 128 + halfBlock * 32;
        const uint scaleBase = 192 + halfBlock * 8;
        const uint scaleOffset = lane >> 4;
        const uchar ql0 = block[qlBase + lane];
        const uchar ql1 = block[qlBase + 32 + lane];
        const uchar qh = block[qhBase + lane];

        const int q1 = int((ql0 & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
        const int q2 = int((ql1 & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
        const int q3 = int((ql0 >> 4) | (((qh >> 4) & 0x03) << 4)) - 32;
        const int q4 = int((ql1 >> 4) | (((qh >> 6) & 0x03) << 4)) - 32;

        const float s1 = float(as_type<char>(block[scaleBase + scaleOffset + 0]));
        const float s2 = float(as_type<char>(block[scaleBase + scaleOffset + 2]));
        const float s3 = float(as_type<char>(block[scaleBase + scaleOffset + 4]));
        const float s4 = float(as_type<char>(block[scaleBase + scaleOffset + 6]));
        const uint colBase = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + halfBlock * 128;

        partial += d * s1 * float(q1) * x[colBase + lane];
        partial += d * s2 * float(q2) * x[colBase + 32 + lane];
        partial += d * s3 * float(q3) * x[colBase + 64 + lane];
        partial += d * s4 * float(q4) * x[colBase + 96 + lane];
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint numSimdgroups = (Q6_K_GEMV_PACKED_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q6_k_gemv_packed_4row_top1_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* partialValues [[buffer(2)]],
    device uint* partialIndices [[buffer(3)]],
    constant ERQ6KGEMVParams& params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rowsPerTile = 4;
    const uint rowBaseIndex = tile * rowsPerTile;
    if (rowBaseIndex + 3 >= params.rows || local_id >= Q6_K_GEMV_PACKED_THREADS_PER_ROW) {
        return;
    }

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    const uint halfBlock = local_id >> 5;
    const uint lane = local_id & 31;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        const uint colBase = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + halfBlock * 128;
        const float x1 = x[colBase + lane];
        const float x2 = x[colBase + 32 + lane];
        const float x3 = x[colBase + 64 + lane];
        const float x4 = x[colBase + 96 + lane];

        for (uint rowInTile = 0; rowInTile < rowsPerTile; ++rowInTile) {
            const uint row = rowBaseIndex + rowInTile;
            device const uchar* block =
                weights + (row * params.blocksPerRow + blockIndex) * Q6_K_BLOCK_BYTES;
            device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
            const float d = float(dPtr[0]);

            const uint qlBase = halfBlock * 64;
            const uint qhBase = 128 + halfBlock * 32;
            const uint scaleBase = 192 + halfBlock * 8;
            const uint scaleOffset = lane >> 4;
            const uchar ql0 = block[qlBase + lane];
            const uchar ql1 = block[qlBase + 32 + lane];
            const uchar qh = block[qhBase + lane];

            const int q1 = int((ql0 & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
            const int q2 = int((ql1 & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
            const int q3 = int((ql0 >> 4) | (((qh >> 4) & 0x03) << 4)) - 32;
            const int q4 = int((ql1 >> 4) | (((qh >> 6) & 0x03) << 4)) - 32;

            const float s1 = float(as_type<char>(block[scaleBase + scaleOffset + 0]));
            const float s2 = float(as_type<char>(block[scaleBase + scaleOffset + 2]));
            const float s3 = float(as_type<char>(block[scaleBase + scaleOffset + 4]));
            const float s4 = float(as_type<char>(block[scaleBase + scaleOffset + 6]));
            const float value =
                d * s1 * float(q1) * x1 +
                d * s2 * float(q2) * x2 +
                d * s3 * float(q3) * x3 +
                d * s4 * float(q4) * x4;
            if (rowInTile == 0) {
                partial0 += value;
            } else if (rowInTile == 1) {
                partial1 += value;
            } else if (rowInTile == 2) {
                partial2 += value;
            } else {
                partial3 += value;
            }
        }
    }

    partial0 = simd_sum(partial0);
    partial1 = simd_sum(partial1);
    partial2 = simd_sum(partial2);
    partial3 = simd_sum(partial3);

    threadgroup float sharedSums[128];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial0;
        sharedSums[32 + simd_group] = partial1;
        sharedSums[64 + simd_group] = partial2;
        sharedSums[96 + simd_group] = partial3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint numSimdgroups = (Q6_K_GEMV_PACKED_THREADS_PER_ROW + 31) / 32;
        float value0 = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        float value1 = simd_lane < numSimdgroups ? sharedSums[32 + simd_lane] : 0.0f;
        float value2 = simd_lane < numSimdgroups ? sharedSums[64 + simd_lane] : 0.0f;
        float value3 = simd_lane < numSimdgroups ? sharedSums[96 + simd_lane] : 0.0f;
        value0 = simd_sum(value0);
        value1 = simd_sum(value1);
        value2 = simd_sum(value2);
        value3 = simd_sum(value3);
        if (simd_lane == 0) {
            float bestValue = value0;
            uint bestIndex = rowBaseIndex;
            if (value1 > bestValue) {
                bestValue = value1;
                bestIndex = rowBaseIndex + 1;
            }
            if (value2 > bestValue) {
                bestValue = value2;
                bestIndex = rowBaseIndex + 2;
            }
            if (value3 > bestValue) {
                bestValue = value3;
                bestIndex = rowBaseIndex + 3;
            }
            partialValues[tile] = bestValue;
            partialIndices[tile] = bestIndex;
        }
    }
}

kernel void q6_k_top1_reduce(
    device const float* partialValues [[buffer(0)]],
    device const uint* partialIndices [[buffer(1)]],
    device uint* outputIndex [[buffer(2)]],
    constant uint& partialCount [[buffer(3)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float bestValue = -INFINITY;
    uint bestIndex = 0;
    for (uint index = local_id; index < partialCount; index += 256) {
        const float value = partialValues[index];
        const uint token = partialIndices[index];
        if (value > bestValue || (value == bestValue && token < bestIndex)) {
            bestValue = value;
            bestIndex = token;
        }
    }

    threadgroup float groupValues[32];
    threadgroup uint groupIndices[32];
    for (uint offset = 16; offset > 0; offset >>= 1) {
        const float otherValue = simd_shuffle_down(bestValue, offset);
        const uint otherIndex = simd_shuffle_down(bestIndex, offset);
        if (otherValue > bestValue || (otherValue == bestValue && otherIndex < bestIndex)) {
            bestValue = otherValue;
            bestIndex = otherIndex;
        }
    }
    if (simd_lane == 0) {
        groupValues[simd_group] = bestValue;
        groupIndices[simd_group] = bestIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint simdgroupCount = 8;
        bestValue = simd_lane < simdgroupCount ? groupValues[simd_lane] : -INFINITY;
        bestIndex = simd_lane < simdgroupCount ? groupIndices[simd_lane] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            const float otherValue = simd_shuffle_down(bestValue, offset);
            const uint otherIndex = simd_shuffle_down(bestIndex, offset);
            if (otherValue > bestValue || (otherValue == bestValue && otherIndex < bestIndex)) {
                bestValue = otherValue;
                bestIndex = otherIndex;
            }
        }
        if (simd_lane == 0) {
            outputIndex[0] = bestIndex;
        }
    }
}
