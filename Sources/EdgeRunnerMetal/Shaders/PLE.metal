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

// === PLE (Per-Layer Embedding) inputs builder kernel ===
//
// Combines the projected hidden state (RMSNorm-normalized) with the gathered
// PLE rows and mixes via scaleMix (typically 1/sqrt(2)). Per (batchSeq, layer)
// slice computes RMSNorm along the last dim (P) using Gemma's (1 + w) weight
// trick, then adds pleRows and multiplies by scaleMix.
//
// Inputs:
//   proj       [B*S, L*P]  — output of GEMV(Wproj, h), already scaled by 1/sqrt(H)
//   normW      [P]         — per_layer_proj_norm.weight (applied as (1 + w))
//   pleRows    [B*S, L, P] — output of ple_gather_q8_0 (already scaled by sqrt(P))
// Output:
//   out        [B*S, L, P] — per_layer_inputs

struct PLEInputsParams {
    uint hidden;       // H (not consumed by kernel; proj is pre-scaled)
    uint perLayerDim;  // P
    uint numLayers;    // L
    uint batchSeq;     // B*S
    float rmsEps;      // typically 1e-6
    float scaleMix;    // 1/sqrt(2)
};

kernel void ple_inputs_build(
    device const float *proj        [[buffer(0)]],
    device const float *normW       [[buffer(1)]],
    device const float *pleRows     [[buffer(2)]],
    device float *out               [[buffer(3)]],
    constant PLEInputsParams &p     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.y indexes (batchSeq * numLayers) combined; gid.x indexes perLayerDim
    uint layerIdx = gid.y % p.numLayers;
    uint batchSeq = gid.y / p.numLayers;
    uint pIdx = gid.x;
    if (batchSeq >= p.batchSeq || pIdx >= p.perLayerDim) return;

    uint sliceBase = batchSeq * p.numLayers * p.perLayerDim + layerIdx * p.perLayerDim;

    // Per-thread RMS reduction over P elements (naive — production version can use simdgroup reduce)
    float sumSq = 0;
    for (uint i = 0; i < p.perLayerDim; ++i) {
        float v = proj[sliceBase + i];
        sumSq += v * v;
    }
    float rms = sqrt(sumSq / float(p.perLayerDim) + p.rmsEps);

    float v = proj[sliceBase + pIdx];
    float w = 1.0f + normW[pIdx];        // Gemma (1 + w) trick
    float normed = (v / rms) * w;
    float ple = pleRows[sliceBase + pIdx];
    out[sliceBase + pIdx] = (normed + ple) * p.scaleMix;
}
