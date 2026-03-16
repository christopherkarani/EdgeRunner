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
} ERRoPEParams;

#endif /* ROPE_PARAMS_H */
