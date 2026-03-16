#ifndef LAYERNORM_PARAMS_H
#define LAYERNORM_PARAMS_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} ERLayerNormParams;

#endif /* LAYERNORM_PARAMS_H */
