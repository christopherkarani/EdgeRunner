#include <metal_stdlib>
using namespace metal;

// q8_0BlockBytes (34) and q8_0WeightsPerBlock (32) are declared in Dequant_Q8_0.metal
// and visible here because KernelRegistry concatenates all .metal files into one library.

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
    uint blockIndex = elem / q8_0WeightsPerBlock;
    uint inBlock = elem % q8_0WeightsPerBlock;
    device const uchar *blockPtr = q8Table + rowBase + blockIndex * q8_0BlockBytes;

    float scale = float(as_type<half>(*(device const ushort*)blockPtr));
    int8_t q = as_type<char>(blockPtr[2 + inBlock]);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = scale * float(q) * sqrtP;
}
