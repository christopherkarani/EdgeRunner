#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include <stdint.h>

typedef enum __attribute__((enum_extensibility(closed))) {
    ERDTypeFloat32 = 0,
    ERDTypeFloat16 = 1,
    ERDTypeInt8    = 2,
    ERDTypeUInt8   = 3,
} ERDType;

typedef struct {
    uint32_t elementCount;
} ERElementwiseParams;

typedef struct {
    uint32_t elementCount;
    uint32_t reductionSize;
    uint32_t outerSize;
} ERReductionParams;

typedef struct {
    uint32_t rows;
    uint32_t cols;
} ERTransposeParams;

#include "GEMMParams.h"
#include "GEMVParams.h"
#include "SoftmaxParams.h"

#endif /* SHADER_TYPES_H */
