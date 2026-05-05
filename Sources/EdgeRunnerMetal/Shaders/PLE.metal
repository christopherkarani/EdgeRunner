#include <metal_stdlib>
using namespace metal;

constant uint pleQ8BlockBytes = 34;
constant uint pleQ8WeightsPerBlock = 32;
constant uint pleQ6KBlockBytes = 210;
constant uint pleQ6KWeightsPerBlock = 256;

static inline float ple_dequant_q6_k_value(device const uchar *block, uint inBlock) {
    device const half *dPtr = reinterpret_cast<device const half *>(block + 208);
    const float d = float(dPtr[0]);
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
    const char scale = as_type<char>(block[scaleBase + quarter * 2 + lane / 16]);
    return d * float(scale) * float(q6);
}

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
    ulong tableByteOffset;
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
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    uint blockIndex = elem / pleQ8WeightsPerBlock;
    uint inBlock = elem % pleQ8WeightsPerBlock;
    device const uchar *blockPtr = q8Table + rowBase + blockIndex * pleQ8BlockBytes;

    float scale = float(as_type<half>(*(device const ushort*)blockPtr));
    int8_t q = as_type<char>(blockPtr[2 + inBlock]);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = scale * float(q) * sqrtP;
}

kernel void ple_gather_q6_k(
    device const uchar *q6KTable       [[buffer(0)]],
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
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    uint blockIndex = elem / pleQ6KWeightsPerBlock;
    uint inBlock = elem % pleQ6KWeightsPerBlock;
    device const uchar *block = q6KTable + rowBase + ulong(blockIndex) * ulong(pleQ6KBlockBytes);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = ple_dequant_q6_k_value(block, inBlock) * sqrtP;
}

kernel void ple_gather_q6_k_blocked(
    device const uchar *q6KTable       [[buffer(0)]],
    device const int *tokens           [[buffer(1)]],
    device float *out                  [[buffer(2)]],
    constant PLEGatherParams &params   [[buffer(3)]],
    uint3 blockPos [[threadgroup_position_in_grid]],
    uint3 localPos [[thread_position_in_threadgroup]]
) {
    uint tIdx = blockPos.y;
    uint blockIndex = blockPos.x;
    uint lane = localPos.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    uint blocksPerRow = totalElems / pleQ6KWeightsPerBlock;
    if (tIdx >= params.numTokens || blockIndex >= blocksPerRow || lane >= pleQ6KWeightsPerBlock) return;

    int tokenId = tokens[tIdx];
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    device const uchar *block = q6KTable + rowBase + ulong(blockIndex) * ulong(pleQ6KBlockBytes);

    const float sqrtP = sqrt(float(params.perLayerDim));
    uint elem = blockIndex * pleQ6KWeightsPerBlock + lane;
    out[tIdx * totalElems + elem] = ple_dequant_q6_k_value(block, lane) * sqrtP;
}

// === PLE (Per-Layer Embedding) inputs builder kernel ===
//
// Combines the projected hidden state (RMSNorm-normalized) with the gathered
// PLE rows and mixes via scaleMix (typically 1/sqrt(2)). Per (batchSeq, layer)
// slice computes RMSNorm along the last dim (P) using Gemma 4's direct
// affine weight, then adds pleRows and multiplies by scaleMix.
//
// Inputs:
//   proj       [B*S, L*P]  — output of GEMV(Wproj, h), already scaled by 1/sqrt(H)
//   normW      [P]         — per_layer_proj_norm.weight
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
    float w = normW[pIdx];
    float normed = (v / rms) * w;
    float ple = pleRows[sliceBase + pIdx];
    out[sliceBase + pIdx] = (normed + ple) * p.scaleMix;
}

// === PLE side-channel finalize kernel ===
//
// Finalizes a single decoder layer's side-channel projection:
//   h = h + RMSNorm(proj, postNormW)
// using Gemma 4's direct RMSNorm weight convention.
//
// Inputs:
//   proj       [B*S, H] — output of side-channel down projection
//   postNormW  [H]      — post_per_layer_input_norm.weight
// In/out:
//   h          [B*S, H] — residual stream, updated in place

struct PLESideChannelParams {
    uint hidden;
    uint batchSeq;
    float rmsEps;
};

struct PLEGateParams {
    uint count;
};

kernel void ple_gate_gelu_mul_f32(
    device const float *gate          [[buffer(0)]],
    device const float *ple           [[buffer(1)]],
    device float *out                 [[buffer(2)]],
    constant PLEGateParams &p         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float g = gate[gid];
    float gelu;
    if (g > 10.0f) {
        gelu = g;
    } else if (g < -10.0f) {
        gelu = 0.0f;
    } else {
        float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        gelu = g * 0.5f * (1.0f + tanh(inner));
    }
    out[gid] = gelu * ple[gid];
}

kernel void ple_side_channel_finalize(
    device float *h                         [[buffer(0)]],
    device const float *proj                [[buffer(1)]],
    device const float *postNormW           [[buffer(2)]],
    constant PLESideChannelParams &p        [[buffer(3)]],
    uint batch [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    if (batch >= p.batchSeq) return;

    uint base = batch * p.hidden;
    threadgroup float partials[256];
    float sumSq = 0.0f;
    for (uint i = local_id; i < p.hidden; i += 256) {
        float v = proj[base + i];
        sumSq += v * v;
    }
    partials[local_id] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            partials[local_id] += partials[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(partials[0] / float(p.hidden) + p.rmsEps);
    for (uint hIdx = local_id; hIdx < p.hidden; hIdx += 256) {
        float normed = proj[base + hIdx] * scale * postNormW[hIdx];
        h[base + hIdx] += normed;
    }
}
