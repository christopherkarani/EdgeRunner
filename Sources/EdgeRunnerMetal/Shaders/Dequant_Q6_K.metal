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
