#include <metal_stdlib>
using namespace metal;

struct ERGEMVParams {
    uint M;
    uint K;
    uint lda;
};

// Each threadgroup handles one row.
// Threads within the group cooperatively reduce across K.
// Uses simd_sum for fast warp-level reduction.
constant uint GEMV_THREADS_PER_ROW = 256;

kernel void gemv_f32(
    device const float*      A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Each thread accumulates a partial dot product
    float partial = 0.0f;
    device const float* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += a_row[j] * x[j];
    }

    // Warp-level reduction
    partial = simd_sum(partial);

    // Cross-warp reduction via threadgroup memory
    threadgroup float shared_sums[32]; // max 32 simdgroups (1024/32)

    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp finalizes
    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = val;
        }
    }
}

kernel void gemv_f16(
    device const half*       A       [[buffer(0)]],
    device const half*       x       [[buffer(1)]],
    device half*             y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Accumulate in float for numerical stability
    float partial = 0.0f;
    device const half* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += float(a_row[j]) * float(x[j]);
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = half(val);
        }
    }
}

kernel void gemv_bf16_f32(
    device const ushort*     A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    float partial = 0.0f;
    device const ushort* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        float a = as_type<float>(uint(a_row[j]) << 16);
        partial += a * x[j];
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = val;
        }
    }
}

kernel void gemv_bf16_f32_batched(
    device const ushort*     A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint3 group_id     [[threadgroup_position_in_grid]],
    uint3 local_pos    [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.x;
    uint batch = group_id.y;
    uint local_id = local_pos.x;
    if (row >= params.M) return;

    float partial = 0.0f;
    device const ushort* a_row = A + row * params.lda;
    device const float* x_row = x + batch * params.K;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        float a = as_type<float>(uint(a_row[j]) << 16);
        partial += a * x_row[j];
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[batch * params.M + row] = val;
        }
    }
}
