#ifndef ATTENTION_PARAMS_H
#define ATTENTION_PARAMS_H

#include <stdint.h>

typedef struct {
    uint32_t seqLen;
    uint32_t headDim;
    float scale;
    uint32_t causal;
    uint32_t kvBlockSize;
    uint32_t qBlockSize;
} ERFlashAttentionParams;

typedef struct {
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t numHeads;
    uint32_t numKVHeads;
    uint32_t groupSize;
    float scale;
    uint32_t causal;
    uint32_t kvBlockSize;
    uint32_t qBlockSize;
    uint32_t kvSeqLen;    // K/V sequence length (0 = same as seqLen for backwards compat)
    uint32_t qOffset;     // offset into Q positions for causal mask (0 = default)
} ERGQAParams;

#endif /* ATTENTION_PARAMS_H */
