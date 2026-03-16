#include <metal_stdlib>
using namespace metal;

struct ERTransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERTransposeParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    if (row >= params.rows || col >= params.cols) return;
    output[col * params.rows + row] = input[row * params.cols + col];
}
