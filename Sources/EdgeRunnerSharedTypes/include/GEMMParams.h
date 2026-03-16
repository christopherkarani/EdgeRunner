#ifndef GEMM_PARAMS_H
#define GEMM_PARAMS_H
#include <stdint.h>
typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
} ERGEMMParams;
#endif
