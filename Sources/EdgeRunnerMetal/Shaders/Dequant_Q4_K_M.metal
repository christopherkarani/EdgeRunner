#include <metal_stdlib>
using namespace metal;

struct ERDequantQ4KMParams {
    uint superBlockCount;
    uint outputOffset;
};

struct ERQ4KGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

struct ERQ4KGEMV3Params {
    uint rowsA;
    uint rowsB;
    uint rowsC;
    uint cols;
    uint blocksPerRow;
};

constant uint Q4_K_M_BLOCK_BYTES = 144;
constant uint Q4_K_M_WEIGHTS_PER_BLOCK = 256;
constant uint Q4_K_M_GEMV_THREADS_PER_ROW = 256;

kernel void dequant_q4_k_m(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ4KMParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q4_K_M_BLOCK_BYTES;
    device const half* masterScales = reinterpret_cast<device const half*>(block);
    float d = float(masterScales[0]);
    float dmin = float(masterScales[1]);

    float scales[8];
    float mins[8];
    for (uint subBlock = 0; subBlock < 4; ++subBlock) {
        uchar scaleByte = block[4 + subBlock];
        uchar minByte = block[8 + subBlock];
        uchar highBits = block[12 + subBlock];

        scales[subBlock] = d * float(scaleByte & 0x3F);
        scales[subBlock + 4] = d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
        mins[subBlock] = dmin * float(minByte & 0x3F);
        mins[subBlock + 4] = dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
    }

    uint outBase = params.outputOffset + tid * Q4_K_M_WEIGHTS_PER_BLOCK;
    for (uint subBlock = 0; subBlock < 8; ++subBlock) {
        float scale = scales[subBlock];
        float minValue = mins[subBlock];

        for (uint index = 0; index < 32; ++index) {
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            output[outBase + (subBlock * 32) + index] = (scale * float(nibble)) - minValue;
        }
    }
}

kernel void q4_k_gemv_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float weight = sharedScales[subBlock] * float(nibble) - sharedMins[subBlock];
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q4_k_gemv_dual_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* yA [[buffer(3)]],
    device float* yB [[buffer(4)]],
    constant ERQ4KGEMVParams& params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partialA = 0.0f;
    float partialB = 0.0f;
    threadgroup float sharedScalesA[8];
    threadgroup float sharedMinsA[8];
    threadgroup float sharedScalesB[8];
    threadgroup float sharedMinsB[8];
    device const uchar* rowBaseA = weightsA + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* rowBaseB = weightsB + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* blockA = rowBaseA + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* blockB = rowBaseB + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScalesA = reinterpret_cast<device const half*>(blockA);
            float dA = float(masterScalesA[0]);
            float dminA = float(masterScalesA[1]);
            uchar scaleByteA = blockA[4 + local_id];
            uchar minByteA = blockA[8 + local_id];
            uchar highBitsA = blockA[12 + local_id];

            sharedScalesA[local_id] = dA * float(scaleByteA & 0x3F);
            sharedScalesA[local_id + 4] =
                dA * float((highBitsA & 0x0F) | (((scaleByteA >> 6) & 0x03) << 4));
            sharedMinsA[local_id] = dminA * float(minByteA & 0x3F);
            sharedMinsA[local_id + 4] =
                dminA * float(((highBitsA >> 4) & 0x0F) | (((minByteA >> 6) & 0x03) << 4));

            device const half* masterScalesB = reinterpret_cast<device const half*>(blockB);
            float dB = float(masterScalesB[0]);
            float dminB = float(masterScalesB[1]);
            uchar scaleByteB = blockB[4 + local_id];
            uchar minByteB = blockB[8 + local_id];
            uchar highBitsB = blockB[12 + local_id];

            sharedScalesB[local_id] = dB * float(scaleByteB & 0x3F);
            sharedScalesB[local_id + 4] =
                dB * float((highBitsB & 0x0F) | (((scaleByteB >> 6) & 0x03) << 4));
            sharedMinsB[local_id] = dminB * float(minByteB & 0x3F);
            sharedMinsB[local_id + 4] =
                dminB * float(((highBitsB >> 4) & 0x0F) | (((minByteB >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packedA = blockA[byteIndex];
            uchar nibbleA = (subBlock & 1) == 0 ? (packedA & 0x0F) : ((packedA >> 4) & 0x0F);
            uchar packedB = blockB[byteIndex];
            uchar nibbleB = (subBlock & 1) == 0 ? (packedB & 0x0F) : ((packedB >> 4) & 0x0F);
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];
            partialA += (sharedScalesA[subBlock] * float(nibbleA) - sharedMinsA[subBlock]) * xValue;
            partialB += (sharedScalesB[subBlock] * float(nibbleB) - sharedMinsB[subBlock]) * xValue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partialA = simd_sum(partialA);
    partialB = simd_sum(partialB);

    threadgroup float sharedSumsA[32];
    threadgroup float sharedSumsB[32];
    if (simd_lane == 0) {
        sharedSumsA[simd_group] = partialA;
        sharedSumsB[simd_group] = partialB;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float valueA = simd_lane < numSimdgroups ? sharedSumsA[simd_lane] : 0.0f;
        float valueB = simd_lane < numSimdgroups ? sharedSumsB[simd_lane] : 0.0f;
        valueA = simd_sum(valueA);
        valueB = simd_sum(valueB);
        if (simd_lane == 0) {
            yA[row] = valueA;
            yB[row] = valueB;
        }
    }
}

kernel void q4_k_gemv_2row_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint rowPair [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = rowPair * 2;
    uint row1 = row0 + 1;
    if (row0 >= params.rows) {
        return;
    }

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    threadgroup float sharedScales0[8];
    threadgroup float sharedMins0[8];
    threadgroup float sharedScales1[8];
    threadgroup float sharedMins1[8];
    device const uchar* rowBase0 = weights + row0 * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* rowBase1 = weights + row1 * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    bool hasRow1 = row1 < params.rows;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block0 = rowBase0 + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* block1 = rowBase1 + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* master0 = reinterpret_cast<device const half*>(block0);
            float d0 = float(master0[0]);
            float dmin0 = float(master0[1]);
            uchar scaleByte0 = block0[4 + local_id];
            uchar minByte0 = block0[8 + local_id];
            uchar highBits0 = block0[12 + local_id];
            sharedScales0[local_id] = d0 * float(scaleByte0 & 0x3F);
            sharedScales0[local_id + 4] =
                d0 * float((highBits0 & 0x0F) | (((scaleByte0 >> 6) & 0x03) << 4));
            sharedMins0[local_id] = dmin0 * float(minByte0 & 0x3F);
            sharedMins0[local_id + 4] =
                dmin0 * float(((highBits0 >> 4) & 0x0F) | (((minByte0 >> 6) & 0x03) << 4));

            if (hasRow1) {
                device const half* master1 = reinterpret_cast<device const half*>(block1);
                float d1 = float(master1[0]);
                float dmin1 = float(master1[1]);
                uchar scaleByte1 = block1[4 + local_id];
                uchar minByte1 = block1[8 + local_id];
                uchar highBits1 = block1[12 + local_id];
                sharedScales1[local_id] = d1 * float(scaleByte1 & 0x3F);
                sharedScales1[local_id + 4] =
                    d1 * float((highBits1 & 0x0F) | (((scaleByte1 >> 6) & 0x03) << 4));
                sharedMins1[local_id] = dmin1 * float(minByte1 & 0x3F);
                sharedMins1[local_id + 4] =
                    dmin1 * float(((highBits1 >> 4) & 0x0F) | (((minByte1 >> 6) & 0x03) << 4));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];

            uchar packed0 = block0[byteIndex];
            uchar nibble0 = (subBlock & 1) == 0 ? (packed0 & 0x0F) : ((packed0 >> 4) & 0x0F);
            partial0 += (sharedScales0[subBlock] * float(nibble0) - sharedMins0[subBlock]) * xValue;

            if (hasRow1) {
                uchar packed1 = block1[byteIndex];
                uchar nibble1 = (subBlock & 1) == 0 ? (packed1 & 0x0F) : ((packed1 >> 4) & 0x0F);
                partial1 += (sharedScales1[subBlock] * float(nibble1) - sharedMins1[subBlock]) * xValue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial0 = simd_sum(partial0);
    partial1 = simd_sum(partial1);

    threadgroup float sharedSums0[32];
    threadgroup float sharedSums1[32];
    if (simd_lane == 0) {
        sharedSums0[simd_group] = partial0;
        sharedSums1[simd_group] = partial1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value0 = simd_lane < numSimdgroups ? sharedSums0[simd_lane] : 0.0f;
        float value1 = simd_lane < numSimdgroups ? sharedSums1[simd_lane] : 0.0f;
        value0 = simd_sum(value0);
        value1 = simd_sum(value1);
        if (simd_lane == 0) {
            y[row0] = value0;
            if (hasRow1) {
                y[row1] = value1;
            }
        }
    }
}

static inline float q4_k_gelu_tanh(float g) {
    if (g > 10.0f) {
        return g;
    }
    if (g < -10.0f) {
        return 0.0f;
    }
    const float c = 0.7978845608028654f;
    float inner = c * (g + 0.044715f * g * g * g);
    return g * 0.5f * (1.0f + tanh(inner));
}

kernel void q4_k_gemv_dual_geglu_f32(
    device const uchar* gateWeights [[buffer(0)]],
    device const uchar* upWeights [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    constant ERQ4KGEMVParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float gatePartial = 0.0f;
    float upPartial = 0.0f;
    threadgroup float gateScales[8];
    threadgroup float gateMins[8];
    threadgroup float upScales[8];
    threadgroup float upMins[8];
    device const uchar* gateRowBase = gateWeights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* upRowBase = upWeights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* gateBlock = gateRowBase + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* upBlock = upRowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* gateMaster = reinterpret_cast<device const half*>(gateBlock);
            float gateD = float(gateMaster[0]);
            float gateDMin = float(gateMaster[1]);
            uchar gateScaleByte = gateBlock[4 + local_id];
            uchar gateMinByte = gateBlock[8 + local_id];
            uchar gateHighBits = gateBlock[12 + local_id];

            gateScales[local_id] = gateD * float(gateScaleByte & 0x3F);
            gateScales[local_id + 4] =
                gateD * float((gateHighBits & 0x0F) | (((gateScaleByte >> 6) & 0x03) << 4));
            gateMins[local_id] = gateDMin * float(gateMinByte & 0x3F);
            gateMins[local_id + 4] =
                gateDMin * float(((gateHighBits >> 4) & 0x0F) | (((gateMinByte >> 6) & 0x03) << 4));

            device const half* upMaster = reinterpret_cast<device const half*>(upBlock);
            float upD = float(upMaster[0]);
            float upDMin = float(upMaster[1]);
            uchar upScaleByte = upBlock[4 + local_id];
            uchar upMinByte = upBlock[8 + local_id];
            uchar upHighBits = upBlock[12 + local_id];

            upScales[local_id] = upD * float(upScaleByte & 0x3F);
            upScales[local_id + 4] =
                upD * float((upHighBits & 0x0F) | (((upScaleByte >> 6) & 0x03) << 4));
            upMins[local_id] = upDMin * float(upMinByte & 0x3F);
            upMins[local_id + 4] =
                upDMin * float(((upHighBits >> 4) & 0x0F) | (((upMinByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar gatePacked = gateBlock[byteIndex];
            uchar gateNibble = (subBlock & 1) == 0 ? (gatePacked & 0x0F) : ((gatePacked >> 4) & 0x0F);
            uchar upPacked = upBlock[byteIndex];
            uchar upNibble = (subBlock & 1) == 0 ? (upPacked & 0x0F) : ((upPacked >> 4) & 0x0F);
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];
            gatePartial += (gateScales[subBlock] * float(gateNibble) - gateMins[subBlock]) * xValue;
            upPartial += (upScales[subBlock] * float(upNibble) - upMins[subBlock]) * xValue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gatePartial = simd_sum(gatePartial);
    upPartial = simd_sum(upPartial);

    threadgroup float gateSums[32];
    threadgroup float upSums[32];
    if (simd_lane == 0) {
        gateSums[simd_group] = gatePartial;
        upSums[simd_group] = upPartial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float gateValue = simd_lane < numSimdgroups ? gateSums[simd_lane] : 0.0f;
        float upValue = simd_lane < numSimdgroups ? upSums[simd_lane] : 0.0f;
        gateValue = simd_sum(gateValue);
        upValue = simd_sum(upValue);
        if (simd_lane == 0) {
            activated[row] = q4_k_gelu_tanh(gateValue) * upValue;
        }
    }
}

kernel void q4_k_gemv_three_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const uchar* weightsC [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* yA [[buffer(4)]],
    device float* yB [[buffer(5)]],
    device float* yC [[buffer(6)]],
    constant ERQ4KGEMV3Params& params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint totalRows = params.rowsA + params.rowsB + params.rowsC;
    if (row >= totalRows) {
        return;
    }

    device const uchar* selectedWeights = weightsA;
    device float* selectedOutput = yA;
    uint selectedRow = row;
    if (row >= params.rowsA + params.rowsB) {
        selectedWeights = weightsC;
        selectedOutput = yC;
        selectedRow = row - params.rowsA - params.rowsB;
    } else if (row >= params.rowsA) {
        selectedWeights = weightsB;
        selectedOutput = yB;
        selectedRow = row - params.rowsA;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = selectedWeights + selectedRow * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float weight = sharedScales[subBlock] * float(nibble) - sharedMins[subBlock];
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            selectedOutput[selectedRow] = value;
        }
    }
}
