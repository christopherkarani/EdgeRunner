#ifndef KV_CACHE_PARAMS_H
#define KV_CACHE_PARAMS_H

#include <stdint.h>

typedef enum __attribute__((enum_extensibility(closed))) {
    ERKVPrecisionFloat32 = 0,
    ERKVPrecisionFloat16 = 1,
    ERKVPrecisionFloat8 = 2,
} ERKVPrecision;

typedef struct {
    uint32_t maxSeqLen;
    uint32_t currentLen;
    uint32_t writePos;
    uint32_t numKVHeads;
    uint32_t headDim;
    uint32_t precision;
} ERKVCacheParams;

#endif /* KV_CACHE_PARAMS_H */
