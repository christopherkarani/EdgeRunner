#include <metal_stdlib>
using namespace metal;

struct ERElementwiseParams {
    uint elementCount;
};

kernel void elementwise_add_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}

kernel void elementwise_add_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}

// === Precision conversion kernels ===

kernel void convert_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ERElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        output[tid] = half(input[tid]);
    }
}

kernel void convert_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        output[tid] = float(input[tid]);
    }
}
