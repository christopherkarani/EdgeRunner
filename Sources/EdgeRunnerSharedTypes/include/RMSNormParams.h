#ifndef RMSNORM_PARAMS_H
#define RMSNORM_PARAMS_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} ERRMSNormParams;

#endif /* RMSNORM_PARAMS_H */
