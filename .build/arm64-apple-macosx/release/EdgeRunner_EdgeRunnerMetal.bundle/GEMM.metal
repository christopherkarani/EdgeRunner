#include <metal_stdlib>
using namespace metal;

struct ERGEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) return;
    float sum = 0.0;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.lda + k] * B[k * params.ldb + col];
    }
    C[row * params.ldc + col] = sum;
}

kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) return;
    half sum = 0.0h;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.lda + k] * B[k * params.ldb + col];
    }
    C[row * params.ldc + col] = sum;
}
