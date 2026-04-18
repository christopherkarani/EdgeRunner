#ifndef ROPE_PARAMS_H
#define ROPE_PARAMS_H

#include <stdint.h>

typedef struct {
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t startPos;
    float theta;
    float scalingFactor;
    // Fraction of head_dim (in [0, 1]) to rotate. 1.0 = full rotation (standard RoPE).
    // Values <1.0 enable pRoPE: channels beyond `headDim * partialRotaryFactor` pass
    // through unchanged. Required for Gemma 4 global-attention layers (partial=0.25).
    // Edge case: the number of rotated pairs is computed as
    //   rotatedPairs = uint(float(halfHeadDim) * partialRotaryFactor)
    // which rounds down. For the Gemma 4 case (headDim=512, partial=0.25) this is
    // 256 * 0.25 = 64 pairs => 128 rotated channels — clean integer.
    float partialRotaryFactor;
} ERRoPEParams;

#endif /* ROPE_PARAMS_H */
