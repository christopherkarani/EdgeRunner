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

inline uint tq_prefix_high_precision_count(device const uint *maskWords, uint dim) {
    uint wordIndex = dim >> 5;
    uint bitIndex = dim & 31;
    uint total = 0;
    for (uint index = 0; index < wordIndex; ++index) {
        total += popcount(maskWords[index]);
    }
    if (bitIndex > 0) {
        uint lowerMask = (1u << bitIndex) - 1u;
        total += popcount(maskWords[wordIndex] & lowerMask);
    }
    return total;
}

inline uint tq_bit_offset_for_dimension(
    device const uint *maskWords,
    uint dim,
    uint regularBits,
    uint highPrecisionBits
) {
    uint prefixHighPrecisionCount = tq_prefix_high_precision_count(maskWords, dim);
    return dim * regularBits + prefixHighPrecisionCount * (highPrecisionBits - regularBits);
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

inline float tq_centroid(uint bits, uint code) {
    switch (bits) {
        case 2: return TURBOQUANT_CODEBOOK_2BIT[code];
        case 3: return TURBOQUANT_CODEBOOK_3BIT[code];
        case 5: return TURBOQUANT_CODEBOOK_5BIT[code];
        default: return 0.0;
    }
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

    uint bitOffset = 0;
    for (uint dim = 0; dim < 128; ++dim) {
        if (highPrecisionMask[dim]) {
            maskWords[dim >> 5] |= (1u << (dim & 31));
        }
        uint width = highPrecisionMask[dim] ? params.highPrecisionBits : params.regularBits;
        uint code = tq_code_for_value(rotated[dim], width);
        tq_insert_code(codeWords, bitOffset, width, code);
        reconstructed[dim] = tq_centroid(width, code);
        bitOffset += width;
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
                uint bitOffset = 0;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint width = useHighPrecision ? params.highPrecisionBits : params.regularBits;
                    uint code = tq_extract_code(codeRow, bitOffset, width);
                    mseDot += qRotation[dim] * tq_centroid(width, code);
                    residualDot += qResidual[dim] * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                    bitOffset += width;
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

                uint bitOffset = 0;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint width = useHighPrecision ? params.highPrecisionBits : params.regularBits;
                    uint code = tq_extract_code(codeRow, bitOffset, width);
                    outputMSE[dim] += mseScale * tq_centroid(width, code);
                    outputResidual[dim] += residualScale * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                    bitOffset += width;
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
        uint keyBitOffset = 0;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
            uint width = useHighPrecision ? params.highPrecisionBits : params.regularBits;
            uint code = tq_extract_code(kCodeRow, keyBitOffset, width);
            mseDot += qRotation[dim] * tq_centroid(width, code);
            residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
            keyBitOffset += width;
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

        uint valueBitOffset = 0;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint width = useHighPrecision ? params.highPrecisionBits : params.regularBits;
            uint code = tq_extract_code(vCodeRow, valueBitOffset, width);
            laneOutputMSE[dim] += mseScale * tq_centroid(width, code);
            laneOutputResidual[dim] += residualScale * (tq_get_bit(vSignRow, dim) == 1u ? 1.0 : -1.0);
            valueBitOffset += width;
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
