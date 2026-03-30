#include <metal_stdlib>
using namespace metal;

constant float TURBOQUANT_CODEBOOK_2BIT[4] = {
    -1.5104176, -0.45278004, 0.45278004, 1.5104176
};
constant float TURBOQUANT_THRESHOLDS_2BIT[3] = {
    -0.9815988, 0.0, 0.9815988
};

constant float TURBOQUANT_CODEBOOK_3BIT[8] = {
    -2.1519456, -1.3439093, -0.7560053, -0.24509418,
    0.24509418, 0.7560053, 1.3439093, 2.1519456
};
constant float TURBOQUANT_THRESHOLDS_3BIT[7] = {
    -1.7479275, -1.0499573, -0.50054973, 0.0,
    0.50054973, 1.0499573, 1.7479275
};

constant float TURBOQUANT_CODEBOOK_5BIT[32] = {
    -3.183201, -2.6029081, -2.222862, -1.9298121, -1.6865128, -1.4757856, -1.2881749, -1.1178436,
    -0.9608707, -0.8144341, -0.6763788, -0.54496944, -0.41874045, -0.29640123, -0.17677347, -0.058747794,
    0.058747794, 0.17677347, 0.29640123, 0.41874045, 0.54496944, 0.6763788, 0.8144341, 0.9608707,
    1.1178436, 1.2881749, 1.4757856, 1.6865128, 1.9298121, 2.222862, 2.6029081, 3.183201
};
constant float TURBOQUANT_THRESHOLDS_5BIT[31] = {
    -2.8930547, -2.4128852, -2.076337, -1.8081625, -1.5811492, -1.3819802, -1.2030092, -1.0393572,
    -0.8876524, -0.74540645, -0.6106741, -0.48185492, -0.35757083, -0.23658736, -0.117760636, 0.0,
    0.117760636, 0.23658736, 0.35757083, 0.48185492, 0.6106741, 0.74540645, 0.8876524, 1.0393572,
    1.2030092, 1.3819802, 1.5811492, 1.8081625, 2.076337, 2.4128852, 2.8930547
};

constant float TURBOQUANT_QJL_SCALE = 1.2533141373155001;
constant float TURBOQUANT_INV_DIM = 1.0 / 128.0;

struct ERTurboQuantQuantizeParams {
    uint rowCount;
    uint sourceRowStride;
    uint destinationRowBase;
    uint codeWordsPerRow;
    uint regularBits;
    uint highPrecisionBits;
    uint highPrecisionChannelCount;
    uint reserved;
};

struct ERTurboQuantAttentionParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;
    uint qOffset;
    uint codeWordsPerRow;
    uint regularBits;
    uint highPrecisionBits;
    uint reserved;
};

inline uint tq_get_bit(device const uint *words, uint bitIndex) {
    uint wordIndex = bitIndex >> 5;
    uint shift = bitIndex & 31;
    return (words[wordIndex] >> shift) & 1u;
}

inline uint tq_get_bit(threadgroup const uint *words, uint bitIndex) {
    uint wordIndex = bitIndex >> 5;
    uint shift = bitIndex & 31;
    return (words[wordIndex] >> shift) & 1u;
}

inline uint tq_extract_code(device const uint *words, uint bitOffset, uint width) {
    uint wordIndex = bitOffset >> 5;
    uint shift = bitOffset & 31;
    uint mask = (1u << width) - 1u;
    uint value = (words[wordIndex] >> shift) & mask;
    uint spill = shift + width;
    if (spill > 32u) {
        uint remaining = spill - 32u;
        uint highBits = words[wordIndex + 1] & ((1u << remaining) - 1u);
        value |= highBits << (width - remaining);
    }
    return value;
}

inline void tq_insert_code(thread uint *words, uint bitOffset, uint width, uint code) {
    uint wordIndex = bitOffset >> 5;
    uint shift = bitOffset & 31;
    uint mask = (1u << width) - 1u;
    words[wordIndex] |= (code & mask) << shift;
    uint spill = shift + width;
    if (spill > 32u) {
        uint remaining = spill - 32u;
        words[wordIndex + 1] |= (code & mask) >> (width - remaining);
    }
}

inline uint tq_extract_split_plane_code(
    device const uint *words,
    uint dim,
    bool useHighPrecision,
    uint regularBits,
    uint highPrecisionBits,
    thread uint &sidebandOffset
) {
    uint baseCode = tq_extract_code(words, dim * regularBits, regularBits);
    if (!useHighPrecision) {
        return baseCode;
    }
    uint sidebandWidth = highPrecisionBits - regularBits;
    if (sidebandWidth == 0u) {
        return baseCode;
    }
    uint extra = tq_extract_code(words, sidebandOffset, sidebandWidth);
    sidebandOffset += sidebandWidth;
    return baseCode | (extra << regularBits);
}

inline float tq_centroid(uint bits, uint code) {
    switch (bits) {
        case 2: return TURBOQUANT_CODEBOOK_2BIT[code];
        case 3: return TURBOQUANT_CODEBOOK_3BIT[code];
        case 5: return TURBOQUANT_CODEBOOK_5BIT[code];
        default: return 0.0;
    }
}

inline float tq_centroid_2bit(uint code) {
    return TURBOQUANT_CODEBOOK_2BIT[code];
}

inline float tq_centroid_3bit(uint code) {
    return TURBOQUANT_CODEBOOK_3BIT[code];
}

inline uint tq_code_for_value(float value, uint bits) {
    switch (bits) {
        case 2:
            for (uint i = 0; i < 3; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_2BIT[i]) { return i; }
            }
            return 3;
        case 3:
            for (uint i = 0; i < 7; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_3BIT[i]) { return i; }
            }
            return 7;
        case 5:
            for (uint i = 0; i < 31; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_5BIT[i]) { return i; }
            }
            return 31;
        default:
            return 0;
    }
}

inline void tq_hadamard(thread float *values) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint base = 0; base < 128; base += step) {
            for (uint index = 0; index < width; ++index) {
                float lhs = values[base + index];
                float rhs = values[base + index + width];
                values[base + index] = lhs + rhs;
                values[base + index + width] = lhs - rhs;
            }
        }
    }
}

inline void tq_hadamard(threadgroup float *values) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint base = 0; base < 128; base += step) {
            for (uint index = 0; index < width; ++index) {
                float lhs = values[base + index];
                float rhs = values[base + index + width];
                values[base + index] = lhs + rhs;
                values[base + index + width] = lhs - rhs;
            }
        }
    }
}

inline void tq_hadamard_parallel(
    threadgroup float *values,
    uint lane,
    uint laneCount
) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint butterfly = lane; butterfly < 64; butterfly += laneCount) {
            uint base = (butterfly / width) * step + (butterfly % width);
            float lhs = values[base];
            float rhs = values[base + width];
            values[base] = lhs + rhs;
            values[base + width] = lhs - rhs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void tq_forward_randomized_hadamard(thread float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= signs[i];
    }
    tq_hadamard(values);
}

inline void tq_forward_randomized_hadamard(threadgroup float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= signs[i];
    }
    tq_hadamard(values);
}

inline void tq_forward_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_hadamard_parallel(values, lane, laneCount);
}

inline void tq_inverse_randomized_hadamard(thread float *values, device const float *signs) {
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * signs[i] * TURBOQUANT_INV_DIM;
    }
}

inline void tq_inverse_randomized_hadamard(threadgroup float *values, device const float *signs) {
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * signs[i] * TURBOQUANT_INV_DIM;
    }
}

inline void tq_inverse_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    tq_hadamard_parallel(values, lane, laneCount);
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] = values[i] * signs[i] * TURBOQUANT_INV_DIM;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void tq_select_top32_bitonic_mask(
    threadgroup const float *rotated,
    threadgroup uint *maskWords,
    threadgroup float *magnitudes,
    threadgroup uint *indices,
    uint lane,
    uint laneCount
) {
    for (uint dim = lane; dim < 128; dim += laneCount) {
        magnitudes[dim] = fabs(rotated[dim]);
        indices[dim] = dim;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 128; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lane; i < 128; i += laneCount) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = (i & k) == 0;
                    bool shouldSwap = ascending
                        ? magnitudes[i] > magnitudes[ixj]
                        : magnitudes[i] < magnitudes[ixj];
                    if (shouldSwap) {
                        float tmpMagnitude = magnitudes[i];
                        magnitudes[i] = magnitudes[ixj];
                        magnitudes[ixj] = tmpMagnitude;
                        uint tmpIndex = indices[i];
                        indices[i] = indices[ixj];
                        indices[ixj] = tmpIndex;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint rank = lane; rank < 32; rank += laneCount) {
        uint dim = indices[127 - rank];
        atomic_fetch_or_explicit(
            (threadgroup atomic_uint *)&maskWords[dim >> 5],
            1u << (dim & 31),
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void turboquant_quantize_rows(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    uint rowIndex [[thread_position_in_grid]]
) {
    if (rowIndex >= params.rowCount) { return; }

    thread float normalized[128];
    thread float rotated[128];
    thread float reconstructed[128];
    thread float residual[128];
    thread float projectedResidual[128];
    thread bool highPrecisionMask[128];
    thread uint codeWords[16];
    thread uint signWords[4];
    thread uint maskWords[4];

    for (uint i = 0; i < 16; ++i) { codeWords[i] = 0; }
    for (uint i = 0; i < 4; ++i) {
        signWords[i] = 0;
        maskWords[i] = 0;
    }

    uint sourceBase = rowIndex * params.sourceRowStride;
    float rowNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        float value = source[sourceBase + dim];
        normalized[dim] = value;
        rowNormSq += value * value;
        highPrecisionMask[dim] = false;
    }

    float rowNorm = sqrt(rowNormSq);

    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;

    if (rowNorm == 0.0) {
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = 0; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = 0;
            maskDst[i] = 0;
        }
        metaDst[0] = 0.0;
        metaDst[1] = 0.0;
        return;
    }

    for (uint dim = 0; dim < 128; ++dim) {
        normalized[dim] /= rowNorm;
        rotated[dim] = normalized[dim];
    }
    tq_forward_randomized_hadamard(rotated, rotationSigns);

    for (uint pick = 0; pick < params.highPrecisionChannelCount; ++pick) {
        float bestMagnitude = -1.0;
        uint bestIndex = 0;
        for (uint dim = 0; dim < 128; ++dim) {
            if (highPrecisionMask[dim]) { continue; }
            float magnitude = fabs(rotated[dim]);
            if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestIndex = dim;
            }
        }
        highPrecisionMask[bestIndex] = true;
    }

    uint sidebandOffset = 128u * params.regularBits;
    for (uint dim = 0; dim < 128; ++dim) {
        if (highPrecisionMask[dim]) {
            maskWords[dim >> 5] |= (1u << (dim & 31));
        }
        uint width = highPrecisionMask[dim] ? params.highPrecisionBits : params.regularBits;
        uint code = tq_code_for_value(rotated[dim], width);
        uint baseMask = (1u << params.regularBits) - 1u;
        tq_insert_code(codeWords, dim * params.regularBits, params.regularBits, code & baseMask);
        if (highPrecisionMask[dim] && params.highPrecisionBits > params.regularBits) {
            tq_insert_code(
                codeWords,
                sidebandOffset,
                params.highPrecisionBits - params.regularBits,
                code >> params.regularBits
            );
            sidebandOffset += params.highPrecisionBits - params.regularBits;
        }
        reconstructed[dim] = tq_centroid(width, code);
    }

    tq_inverse_randomized_hadamard(reconstructed, rotationSigns);
    float residualNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        residualNormSq += residual[dim] * residual[dim];
    }

    float residualNorm = sqrt(residualNormSq);
    tq_forward_randomized_hadamard(projectedResidual, residualSigns);
    for (uint dim = 0; dim < 128; ++dim) {
        if (projectedResidual[dim] >= 0.0) {
            signWords[dim >> 5] |= (1u << (dim & 31));
        }
    }

    for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
    for (uint i = 0; i < 4; ++i) {
        signDst[i] = signWords[i];
        maskDst[i] = maskWords[i];
    }
    metaDst[0] = rowNorm;
    metaDst[1] = residualNorm;
}

inline void tq_quantize_small_aggressive_row(
    device const float *source,
    uint sourceBase,
    device uint *codeDst,
    device uint *signDst,
    device uint *maskDst,
    device float *metaDst,
    device const float *rotationSigns,
    device const float *residualSigns,
    uint lane,
    threadgroup float *normalized,
    threadgroup float *rotated,
    threadgroup float *reconstructed,
    threadgroup float *residual,
    threadgroup float *projectedResidual,
    threadgroup uint *codes,
    threadgroup uint *maskWords,
    threadgroup float *magnitudes,
    threadgroup uint *indices,
    threadgroup float *reduction,
    threadgroup float &rowNormValue,
    threadgroup float &residualNormValue,
    uint codeWordsPerRow
) {
    constexpr uint kLaneCount = 32;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;

    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        normalized[dim] = value;
        rotated[dim] = value;
        reconstructed[dim] = 0.0;
        residual[dim] = 0.0;
        projectedResidual[dim] = 0.0;
        codes[dim] = 0u;
        partialNormSq += value * value;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint index = lane; index < codeWordsPerRow; index += kLaneCount) {
            codeDst[index] = 0u;
        }
        if (lane < 4) {
            signDst[lane] = 0u;
            maskDst[lane] = 0u;
        }
        if (lane == 0) {
            metaDst[0] = 0.0;
            metaDst[1] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] /= rowNormValue;
        rotated[dim] = normalized[dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);

    tq_select_top32_bitonic_mask(rotated, maskWords, magnitudes, indices, lane, kLaneCount);

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        bool useHighPrecision = tq_get_bit(maskWords, dim) == 1u;
        uint bits = useHighPrecision ? kHighBits : kRegularBits;
        uint code = tq_code_for_value(rotated[dim], bits);
        codes[dim] = code;
        reconstructed[dim] = useHighPrecision ? tq_centroid_3bit(code) : tq_centroid_2bit(code);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_inverse_randomized_hadamard_parallel(reconstructed, rotationSigns, lane, kLaneCount);

    float partialResidualSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        partialResidualSq += residual[dim] * residual[dim];
    }
    reduction[lane] = partialResidualSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        residualNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_forward_randomized_hadamard_parallel(projectedResidual, residualSigns, lane, kLaneCount);

    if (lane == 0) {
        uint codeWords[16];
        uint signWords[4];
        for (uint i = 0; i < 16; ++i) { codeWords[i] = 0u; }
        for (uint i = 0; i < 4; ++i) { signWords[i] = 0u; }

        uint sidebandOffset = 256u;
        for (uint dim = 0; dim < 128; ++dim) {
            uint code = codes[dim];
            tq_insert_code(codeWords, dim * kRegularBits, kRegularBits, code & 0x3u);
            if (tq_get_bit(maskWords, dim) == 1u) {
                tq_insert_code(codeWords, sidebandOffset, 1u, code >> kRegularBits);
                sidebandOffset += 1u;
            }
            if (projectedResidual[dim] >= 0.0) {
                signWords[dim >> 5] |= (1u << (dim & 31));
            }
        }

        for (uint i = 0; i < codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = signWords[i];
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = rowNormValue;
        metaDst[1] = residualNormValue;
    }
}

kernel void turboquant_quantize_rows_small_aggressive(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;

    tq_quantize_small_aggressive_row(
        source,
        sourceBase,
        codeDst,
        signDst,
        maskDst,
        metaDst,
        rotationSigns,
        residualSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow
    );
}

kernel void turboquant_quantize_rows_small_aggressive_k(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;

    tq_quantize_small_aggressive_row(
        source,
        sourceBase,
        codeDst,
        signDst,
        maskDst,
        metaDst,
        rotationSigns,
        residualSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow
    );
}

kernel void turboquant_quantize_rows_small_aggressive_kv(
    device const float *keySource [[buffer(0)]],
    device const float *valueSource [[buffer(1)]],
    device uint *keyCodes [[buffer(2)]],
    device uint *keyResidualSigns [[buffer(3)]],
    device uint *keyOutlierMask [[buffer(4)]],
    device float *keyMetadata [[buffer(5)]],
    device uint *valueCodes [[buffer(6)]],
    device uint *valueResidualSigns [[buffer(7)]],
    device uint *valueOutlierMask [[buffer(8)]],
    device float *valueMetadata [[buffer(9)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;

    tq_quantize_small_aggressive_row(
        keySource,
        sourceBase,
        keyCodes + destinationRow * params.codeWordsPerRow,
        keyResidualSigns + destinationRow * 4,
        keyOutlierMask + destinationRow * 4,
        keyMetadata + destinationRow * 2,
        keyRotationSigns,
        keyResidualProjectionSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_quantize_small_aggressive_row(
        valueSource,
        sourceBase,
        valueCodes + destinationRow * params.codeWordsPerRow,
        valueResidualSigns + destinationRow * 4,
        valueOutlierMask + destinationRow * 4,
        valueMetadata + destinationRow * 2,
        valueRotationSigns,
        valueResidualProjectionSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow
    );
}

kernel void turboquant_quantize_rows_small_aggressive_phase1(
    device const float *source [[buffer(0)]],
    device float *outNormalized [[buffer(1)]],
    device float *outRotated [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outRowNorm [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        normalized[dim] = value;
        rotated[dim] = value;
        partialNormSq += value * value;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint dim = lane; dim < 128; dim += kLaneCount) {
            outNormalized[rowIndex * 128 + dim] = 0.0;
            outRotated[rowIndex * 128 + dim] = 0.0;
        }
        if (lane < 4) {
            outOutlierMask[rowIndex * 4 + lane] = 0u;
        }
        if (lane == 0) {
            outRowNorm[rowIndex] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] /= rowNormValue;
        rotated[dim] = normalized[dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);

    tq_select_top32_bitonic_mask(rotated, maskWords, magnitudes, indices, lane, kLaneCount);
    if (lane == 0) {
        outRowNorm[rowIndex] = rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        outNormalized[rowIndex * 128 + dim] = normalized[dim];
        outRotated[rowIndex * 128 + dim] = rotated[dim];
    }
    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_rotate_only(
    device const float *source [[buffer(0)]],
    device float *outRotated [[buffer(1)]],
    device float *outRowNorm [[buffer(2)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(3)]],
    device const float *rotationSigns [[buffer(4)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float rotated[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        rotated[dim] = value;
        partialNormSq += value * value;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
        outRowNorm[rowIndex] = rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint dim = lane; dim < 128; dim += kLaneCount) {
            outRotated[rowIndex * 128 + dim] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        rotated[dim] /= rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        outRotated[rowIndex * 128 + dim] = rotated[dim];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_select_only(
    device const float *rotatedSource [[buffer(0)]],
    device uint *outOutlierMask [[buffer(1)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(2)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kHighCount = 32;

    threadgroup float rotated[128];
    threadgroup uint maskWords[4];

    for (uint dim = lane; dim < 128; dim += 32) {
        rotated[dim] = rotatedSource[rowIndex * 128 + dim];
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint pick = 0; pick < kHighCount; ++pick) {
            float bestMagnitude = -1.0;
            uint bestIndex = 0;
            for (uint dim = 0; dim < 128; ++dim) {
                if (tq_get_bit(maskWords, dim) == 1u) { continue; }
                float magnitude = fabs(rotated[dim]);
                if (magnitude > bestMagnitude) {
                    bestMagnitude = magnitude;
                    bestIndex = dim;
                }
            }
            maskWords[bestIndex >> 5] |= (1u << (bestIndex & 31));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_select_only_bitonic(
    device const float *rotatedSource [[buffer(0)]],
    device uint *outOutlierMask [[buffer(1)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(2)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    for (uint dim = lane; dim < 128; dim += 32) {
        magnitudes[dim] = fabs(rotatedSource[rowIndex * 128 + dim]);
        indices[dim] = dim;
    }
    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 128; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lane; i < 128; i += 32) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = (i & k) == 0;
                    bool shouldSwap = ascending
                        ? magnitudes[i] > magnitudes[ixj]
                        : magnitudes[i] < magnitudes[ixj];
                    if (shouldSwap) {
                        float tmpMagnitude = magnitudes[i];
                        magnitudes[i] = magnitudes[ixj];
                        magnitudes[ixj] = tmpMagnitude;
                        uint tmpIndex = indices[i];
                        indices[i] = indices[ixj];
                        indices[ixj] = tmpIndex;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint rank = lane; rank < 32; rank += 32) {
        uint dim = indices[127 - rank];
        atomic_fetch_or_explicit(
            (device atomic_uint *)&outOutlierMask[rowIndex * 4 + (dim >> 5)],
            1u << (dim & 31),
            memory_order_relaxed
        );
    }
}

kernel void turboquant_quantize_rows_small_aggressive_phase2(
    device const float *normalizedSource [[buffer(0)]],
    device const float *rotatedSource [[buffer(1)]],
    device const uint *inputOutlierMask [[buffer(2)]],
    device const float *inputRowNorm [[buffer(3)]],
    device uint *outCodes [[buffer(4)]],
    device uint *outResidualSigns [[buffer(5)]],
    device uint *outOutlierMask [[buffer(6)]],
    device float *outMetadata [[buffer(7)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(8)]],
    device const float *rotationSigns [[buffer(9)]],
    device const float *residualSigns [[buffer(10)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float reduction[32];
    threadgroup float residualNormValue;
    threadgroup float rowNormValue;

    if (lane == 0) {
        rowNormValue = inputRowNorm[rowIndex];
    }
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] = normalizedSource[rowIndex * 128 + dim];
        rotated[dim] = rotatedSource[rowIndex * 128 + dim];
        reconstructed[dim] = 0.0;
        residual[dim] = 0.0;
        projectedResidual[dim] = 0.0;
        codes[dim] = 0u;
    }
    if (lane < 4) {
        maskWords[lane] = inputOutlierMask[rowIndex * 4 + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint index = lane; index < params.codeWordsPerRow; index += kLaneCount) {
            outCodes[rowIndex * params.codeWordsPerRow + index] = 0u;
        }
        if (lane < 4) {
            outResidualSigns[rowIndex * 4 + lane] = 0u;
            outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
        }
        if (lane == 0) {
            outMetadata[rowIndex * 2] = 0.0;
            outMetadata[rowIndex * 2 + 1] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        bool useHighPrecision = tq_get_bit(maskWords, dim) == 1u;
        uint code = tq_code_for_value(rotated[dim], useHighPrecision ? 3u : 2u);
        codes[dim] = code;
        reconstructed[dim] = useHighPrecision ? tq_centroid_3bit(code) : tq_centroid_2bit(code);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_inverse_randomized_hadamard_parallel(reconstructed, rotationSigns, lane, kLaneCount);

    float partialResidualSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        partialResidualSq += residual[dim] * residual[dim];
    }
    reduction[lane] = partialResidualSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        residualNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_forward_randomized_hadamard_parallel(projectedResidual, residualSigns, lane, kLaneCount);

    if (lane == 0) {
        uint codeWords[16];
        uint signWords[4];
        for (uint i = 0; i < 16; ++i) { codeWords[i] = 0u; }
        for (uint i = 0; i < 4; ++i) { signWords[i] = 0u; }

        uint sidebandOffset = 256u;
        for (uint dim = 0; dim < 128; ++dim) {
            uint code = codes[dim];
            tq_insert_code(codeWords, dim * 2u, 2u, code & 0x3u);
            if (tq_get_bit(maskWords, dim) == 1u) {
                tq_insert_code(codeWords, sidebandOffset, 1u, code >> 2u);
                sidebandOffset += 1u;
            }
            if (projectedResidual[dim] >= 0.0) {
                signWords[dim >> 5] |= (1u << (dim & 31));
            }
        }

        device uint *codeDst = outCodes + rowIndex * params.codeWordsPerRow;
        device uint *signDst = outResidualSigns + rowIndex * 4;
        device uint *maskDst = outOutlierMask + rowIndex * 4;
        device float *metaDst = outMetadata + rowIndex * 2;
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = signWords[i];
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = rowNormValue;
        metaDst[1] = residualNormValue;
    }
}

kernel void gqa_attention_turboquant(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint qStride = params.numHeads * params.headDim;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    threadgroup float qRotationScratch[16 * 128];
    threadgroup float qResidualScratch[16 * 128];
    threadgroup float outputMSEScratch[16 * 128];
    threadgroup float outputResidualScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float scores[16];
    float probs[16];

    if (activeQ) {
        threadgroup float *qRotation = qRotationScratch + local_id.x * 128;
        threadgroup float *qResidual = qResidualScratch + local_id.x * 128;
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
        uint qBase = qRow * qStride + headIndex * params.headDim;

        for (uint dim = 0; dim < 128; ++dim) {
            float value = Q[qBase + dim];
            qRotation[dim] = value;
            qResidual[dim] = value;
            outputMSE[dim] = 0.0;
            outputResidual[dim] = 0.0;
        }
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; ++kvBlock) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (activeQ) {
            threadgroup float *qRotation = qRotationScratch + local_id.x * 128;
            threadgroup float *qResidual = qResidualScratch + local_id.x * 128;
            threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
            threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;

            float blockMax = -INFINITY;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = KCodes + rowIndex * params.codeWordsPerRow;
                device const uint *signRow = KResidualSigns + rowIndex * 4;
                device const uint *maskRow = KOutlierMask + rowIndex * 4;
                device const float *metaRow = KMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];

                float mseDot = 0.0;
                float residualDot = 0.0;
                uint sidebandOffset = 128u * params.regularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.regularBits,
                        params.highPrecisionBits,
                        sidebandOffset
                    );
                    mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
                    residualDot += qResidual[dim] * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                }

                float dot = rowNorm * (mseDot + TURBOQUANT_QJL_SCALE * residualNorm * residualDot);
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (scores[kvIndex] == -INFINITY) {
                    probs[kvIndex] = 0.0;
                } else {
                    probs[kvIndex] = exp(scores[kvIndex] - nextMax);
                }
                blockSum += probs[kvIndex];
            }

            runningSum = runningSum * correction + blockSum;
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= correction;
                outputResidual[dim] *= correction;
            }

            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                float prob = probs[kvIndex];
                if (prob == 0.0) { continue; }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = VCodes + rowIndex * params.codeWordsPerRow;
                device const uint *signRow = VResidualSigns + rowIndex * 4;
                device const uint *maskRow = VOutlierMask + rowIndex * 4;
                device const float *metaRow = VMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];
                float mseScale = prob * rowNorm;
                float residualScale = prob * rowNorm * residualNorm;

                uint sidebandOffset = 128u * params.regularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.regularBits,
                        params.highPrecisionBits,
                        sidebandOffset
                    );
                    outputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
                    outputResidual[dim] += residualScale * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                }
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;

        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);

        float invSum = runningSum > 0.0 ? 1.0 / runningSum : 0.0;
        uint oBase = qRow * qStride + headIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            O[oBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE)) * invSum;
        }
    }
}

kernel void gqa_attention_turboquant_decode(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float outputMSE[128];
    threadgroup float outputResidual[128];
    threadgroup float partialMSE[kDecodeThreads * 128];
    threadgroup float partialResidual[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutputMSE = partialMSE + lane * 128;
    threadgroup float *laneOutputResidual = partialResidual + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        outputMSE[dim] = 0.0;
        outputResidual[dim] = 0.0;
        laneOutputMSE[dim] = 0.0;
        laneOutputResidual[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
        device const uint *kSignRow = KResidualSigns + rowIndex * 4;
        device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
        device const float *kMetaRow = KMetadata + rowIndex * 2;
        float keyRowNorm = kMetaRow[0];
        float keyResidualNorm = kMetaRow[1];

        float mseDot = 0.0;
        float residualDot = 0.0;
        uint keySidebandOffset = 128u * params.regularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                kCodeRow,
                dim,
                useHighPrecision,
                params.regularBits,
                params.highPrecisionBits,
                keySidebandOffset
            );
            mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
            residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * keyResidualNorm * residualDot
        ) * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutputMSE[dim] *= correction;
            laneOutputResidual[dim] *= correction;
        }

        device const uint *vCodeRow = VCodes + rowIndex * params.codeWordsPerRow;
        device const uint *vSignRow = VResidualSigns + rowIndex * 4;
        device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
        device const float *vMetaRow = VMetadata + rowIndex * 2;
        float valueRowNorm = vMetaRow[0];
        float valueResidualNorm = vMetaRow[1];
        float mseScale = prob * valueRowNorm;
        float residualScale = prob * valueRowNorm * valueResidualNorm;

        uint valueSidebandOffset = 128u * params.regularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                vCodeRow,
                dim,
                useHighPrecision,
                params.regularBits,
                params.highPrecisionBits,
                valueSidebandOffset
            );
            laneOutputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
            laneOutputResidual[dim] += residualScale * (tq_get_bit(vSignRow, dim) == 1u ? 1.0 : -1.0);
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float mseAccum = 0.0;
        float residualAccum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker];
            mseAccum += partialMSE[worker * 128 + dim] * workerScale;
            residualAccum += partialResidual[worker * 128 + dim] * workerScale;
        }
        outputMSE[dim] = mseAccum;
        outputResidual[dim] = residualAccum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE)) * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float output[128];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        output[dim] = 0.0;
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
        device const uint *kSignRow = KResidualSigns + rowIndex * 4;
        device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
        device const float *kMetaRow = KMetadata + rowIndex * 2;
        float keyRowNorm = kMetaRow[0];
        float keyResidualNorm = kMetaRow[1];

        float mseDot = 0.0;
        float residualDot = 0.0;
        uint keySidebandOffset = 128u * params.regularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                kCodeRow,
                dim,
                useHighPrecision,
                params.regularBits,
                params.highPrecisionBits,
                keySidebandOffset
            );
            mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
            residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * keyResidualNorm * residualDot
        ) * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] *= correction;
        }

        uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] += prob * float(V[valueBase + dim]);
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker];
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        output[dim] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = output[dim] * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;
    constexpr uint kTileRows = 4;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float outputMSE[128];
    threadgroup float outputResidual[128];
    threadgroup float partialMSE[kDecodeThreads * 128];
    threadgroup float partialResidual[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float laneAccumulationScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutputMSE = partialMSE + lane * 128;
    threadgroup float *laneOutputResidual = partialResidual + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        outputMSE[dim] = 0.0;
        outputResidual[dim] = 0.0;
        laneOutputMSE[dim] = 0.0;
        laneOutputResidual[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float accumulationScale = 1.0;

    for (uint kvBase = lane; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tileRowIndices[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                tileRowIndices[tile] = UINT_MAX;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            tileRowIndices[tile] = rowIndex;

            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const uint *kSignRow = KResidualSigns + rowIndex * 4;
            device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];
            float keyResidualNorm = kMetaRow[1];

            float mseDot = 0.0;
            float residualDot = 0.0;
            for (uint block = 0; block < 4; ++block) {
                uint signWord = kSignRow[block];
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint keySidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = kMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(kCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(kCodeRow, keySidebandOffset, 1u);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    mseDot += qRotation[dim] * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    keySidebandOffset += 1u;
                    maskWord &= (maskWord - 1u);
                }
            }

            float score = keyRowNorm * (
                mseDot + TURBOQUANT_QJL_SCALE * keyResidualNorm * residualDot
            ) * params.scale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        bool maxAdvanced = tileMax > runningMax;
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        if (maxAdvanced && runningSum > tileSum) {
            accumulationScale *= correction;
            if (accumulationScale < 1.0e-6f) {
                for (uint dim = 0; dim < 128; ++dim) {
                    laneOutputMSE[dim] *= accumulationScale;
                    laneOutputResidual[dim] *= accumulationScale;
                }
                accumulationScale = 1.0f;
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint rowIndex = tileRowIndices[tile];
            float prob = tileProbs[tile];
            if (rowIndex == UINT_MAX || prob == 0.0) { continue; }

            device const uint *vCodeRow = VCodes + rowIndex * params.codeWordsPerRow;
            device const uint *vSignRow = VResidualSigns + rowIndex * 4;
            device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
            device const float *vMetaRow = VMetadata + rowIndex * 2;
            float valueRowNorm = vMetaRow[0];
            float valueResidualNorm = vMetaRow[1];
            float mseScale = (prob * valueRowNorm) / accumulationScale;
            float residualScale = (prob * valueRowNorm * valueResidualNorm) / accumulationScale;

            for (uint block = 0; block < 4; ++block) {
                uint signWord = vSignRow[block];
                uint packedCodes0 = vCodeRow[block * 2];
                uint packedCodes1 = vCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    laneOutputMSE[dim] += mseScale * tq_centroid_2bit(baseCode);
                    laneOutputResidual[dim] += residualScale * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    laneOutputMSE[dim] += mseScale * tq_centroid_2bit(baseCode);
                    laneOutputResidual[dim] += residualScale * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint valueSidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = vMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(vCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(vCodeRow, valueSidebandOffset, 1u);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    laneOutputMSE[dim] += mseScale * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    valueSidebandOffset += 1u;
                    maskWord &= (maskWord - 1u);
                }
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    laneAccumulationScale[lane] = accumulationScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float mseAccum = 0.0;
        float residualAccum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker] * laneAccumulationScale[worker];
            mseAccum += partialMSE[worker * 128 + dim] * workerScale;
            residualAccum += partialResidual[worker * 128 + dim] * workerScale;
        }
        outputMSE[dim] = mseAccum;
        outputResidual[dim] = residualAccum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE)) * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;
    constexpr uint kTileRows = 4;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float output[128];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float laneAccumulationScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        output[dim] = 0.0;
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float accumulationScale = 1.0;

    for (uint kvBase = lane; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tilePositions[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads;
            tilePositions[tile] = kvPos;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const uint *kSignRow = KResidualSigns + rowIndex * 4;
            device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];
            float keyResidualNorm = kMetaRow[1];

            float mseDot = 0.0;
            float residualDot = 0.0;
            for (uint block = 0; block < 4; ++block) {
                uint signWord = kSignRow[block];
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint keySidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = kMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(kCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(kCodeRow, keySidebandOffset, kHighBits - kRegularBits);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    mseDot += qRotation[dim] * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    keySidebandOffset += (kHighBits - kRegularBits);
                    maskWord &= (maskWord - 1u);
                }
            }

            float score = keyRowNorm * (
                mseDot + TURBOQUANT_QJL_SCALE * keyResidualNorm * residualDot
            ) * params.scale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        bool maxAdvanced = tileMax > runningMax;
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        if (maxAdvanced && runningSum > tileSum) {
            accumulationScale *= correction;
            if (accumulationScale < 1.0e-6f) {
                for (uint dim = 0; dim < 128; ++dim) {
                    laneOutput[dim] *= accumulationScale;
                }
                accumulationScale = 1.0f;
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = tilePositions[tile];
            float prob = tileProbs[tile];
            if (kvPos >= kvLimit || prob == 0.0f) { continue; }
            uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
            for (uint dim = 0; dim < 128; ++dim) {
                laneOutput[dim] += (prob / accumulationScale) * float(V[valueBase + dim]);
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    laneAccumulationScale[lane] = accumulationScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker] * laneAccumulationScale[worker];
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        output[dim] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = output[dim] * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive_k_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kTileRows = 4;
    constexpr uint kValueAccumCutoffSeqThreshold = 2048u;
    constexpr uint kValueAccumAggressiveSeqThreshold = 8192u;
    constexpr uint kValueAccumUltraSeqThreshold = 16384u;
    constexpr uint kSparseScoreSeqThreshold = 8192u;
    constexpr uint kSparseScoreUltraSeqThreshold = 16384u;
    constexpr uint kSparseRowSeqThreshold = 8192u;
    constexpr uint kSparseRowUltraSeqThreshold = 16384u;
    constexpr float kValueAccumProbCutoff = 2e-1f;
    constexpr float kValueAccumAggressiveProbCutoff = 5e-1f;
    constexpr float kValueAccumUltraProbCutoff = 8e-1f;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qBaseLUT[128 * 4];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        uint lutBase = dim * 4;
        qBaseLUT[lutBase + 0] = value * tq_centroid_2bit(0u);
        qBaseLUT[lutBase + 1] = value * tq_centroid_2bit(1u);
        qBaseLUT[lutBase + 2] = value * tq_centroid_2bit(2u);
        qBaseLUT[lutBase + 3] = value * tq_centroid_2bit(3u);
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    uint rowStride = 1u;
    if (kvLimit >= kSparseRowUltraSeqThreshold) {
        rowStride = 32u;
    } else if (kvLimit >= kSparseRowSeqThreshold) {
        rowStride = 4u;
    }
    for (uint kvBase = lane * rowStride; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows * rowStride) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tilePositions[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads * rowStride;
            tilePositions[tile] = kvPos;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];

            float mseDot = 0.0;
            uint blockStep = 1u;
            if (kvLimit >= kSparseScoreUltraSeqThreshold) {
                blockStep = 4u;
            } else if (kvLimit >= kSparseScoreSeqThreshold) {
                blockStep = 2u;
            }
            for (uint block = 0; block < 4; block += blockStep) {
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qBaseLUT[dim * 4 + baseCode];
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qBaseLUT[dim * 4 + baseCode];
                }
            }
            float sparseScoreScale = float(blockStep);
            float score = keyRowNorm * mseDot * params.scale * sparseScoreScale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] *= correction;
        }

        bool ultraTop1Only = kvLimit >= kValueAccumUltraSeqThreshold;
        uint bestTile = UINT_MAX;
        if (ultraTop1Only) {
            float bestProb = 0.0f;
            for (uint tile = 0; tile < kTileRows; ++tile) {
                if (tilePositions[tile] >= kvLimit) { continue; }
                float prob = tileProbs[tile];
                if (prob > bestProb) {
                    bestProb = prob;
                    bestTile = tile;
                }
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            if (ultraTop1Only && tile != bestTile) { continue; }
            uint kvPos = tilePositions[tile];
            float prob = tileProbs[tile];
            float activeProbCutoff = 0.0f;
            if (kvLimit >= kValueAccumUltraSeqThreshold) {
                activeProbCutoff = kValueAccumUltraProbCutoff;
            } else if (kvLimit >= kValueAccumAggressiveSeqThreshold) {
                activeProbCutoff = kValueAccumAggressiveProbCutoff;
            } else if (kvLimit >= kValueAccumCutoffSeqThreshold) {
                activeProbCutoff = kValueAccumProbCutoff;
            }
            if (kvPos >= kvLimit || prob < activeProbCutoff) { continue; }
            uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
            for (uint dim = 0; dim < 128; ++dim) {
                laneOutput[dim] += prob * float(V[valueBase + dim]);
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    reductionScratch[lane] = laneSum[lane] > 0.0 ? laneSum[lane] * exp(laneMax[lane] - globalMax) : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneSum[worker] > 0.0 ? exp(laneMax[worker] - globalMax) : 0.0;
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        O[outputBase + dim] = accum * invSum;
    }
}
