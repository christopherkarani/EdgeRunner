#ifndef GEMV_PARAMS_H
#define GEMV_PARAMS_H

#include <stdint.h>

/// Parameters for GEMV kernel: y[M] = A[M,K] * x[K].
typedef struct {
    uint32_t M;         // number of rows
    uint32_t K;         // number of columns (vector length)
    uint32_t lda;       // leading dimension of A (typically K)
} ERGEMVParams;

#endif /* GEMV_PARAMS_H */
