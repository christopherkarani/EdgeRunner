#include <metal_stdlib>
using namespace metal;

// === PLE (Per-Layer Embedding) single-row Q8_0 gather kernel ===
//
// For Gemma 4 E4B: per_layer_token_embd has shape [vocab_size, num_layers * perLayerDim]
// stored as Q8_0 blocks (34 bytes per 32-element block). For each token in the batch and
// each layer, we gather that layer's slice from the token's row, dequantize, and scale by
// sqrt(perLayerDim).
//
// Output: [numTokens, num_layers, perLayerDim] as Float.

struct PLEGatherParams {
    uint perLayerDim;      // P
    uint numLayers;        // L
    uint numTokens;
    uint rowStrideBytes;   // bytes per (token, L*P row) in Q8_0 storage
};

constant uint kQ8BlockBytes = 34;
constant uint kQ8BlockElems = 32;

kernel void ple_gather_q8_0(
    device const uchar *q8Table        [[buffer(0)]],
    device const int *tokens           [[buffer(1)]],
    device float *out                  [[buffer(2)]],
    constant PLEGatherParams &params   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tIdx = gid.y;
    uint elem = gid.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    if (tIdx >= params.numTokens || elem >= totalElems) return;

    int tokenId = tokens[tIdx];
    uint rowBase = uint(tokenId) * params.rowStrideBytes;
    uint blockIndex = elem / kQ8BlockElems;
    uint inBlock = elem % kQ8BlockElems;
    device const uchar *blockPtr = q8Table + rowBase + blockIndex * kQ8BlockBytes;

    half scale = *reinterpret_cast<device const half *>(blockPtr);
    int8_t q = *reinterpret_cast<device const int8_t *>(blockPtr + 2 + inBlock);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = float(scale) * float(q) * sqrtP;
}
