#ifndef DEQUANT_PARAMS_H
#define DEQUANT_PARAMS_H

#include <stdint.h>

typedef struct {
    uint32_t blockCount;
    uint32_t outputOffset;
} ERDequantParams;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t blocksPerRow;
} ERDequantGEMVParams;

typedef struct {
    uint32_t superBlockCount;
    uint32_t outputOffset;
} ERDequantQ4KParams;

#endif /* DEQUANT_PARAMS_H */
