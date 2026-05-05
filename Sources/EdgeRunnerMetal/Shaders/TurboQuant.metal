#include <metal_stdlib>
using namespace metal;

constant float TURBOQUANT_CODEBOOK_2BIT[4] = {
    -0.133462, -0.039994, 0.039994, 0.133462
};
constant float TURBOQUANT_THRESHOLDS_2BIT[3] = {
    -0.086728, 0.0, 0.086728
};

constant float TURBOQUANT_CODEBOOK_3BIT[8] = {
    -0.190685, -0.117832, -0.065717, -0.021460,
    0.021460, 0.065717, 0.117832, 0.190685
};
constant float TURBOQUANT_THRESHOLDS_3BIT[7] = {
    -0.1542585, -0.0917745, -0.0435885, 0.0,
    0.0435885, 0.0917745, 0.1542585
};

constant float TURBOQUANT_CODEBOOK_4BIT[16] = {
    -0.173926, -0.117195, -0.089527, -0.068756,
    -0.051262, -0.035597, -0.020989, -0.006938,
    0.006938, 0.020989, 0.035597, 0.051262,
    0.068756, 0.089527, 0.117195, 0.173926
};
constant float TURBOQUANT_THRESHOLDS_4BIT[15] = {
    -0.145560, -0.103361, -0.079142, -0.060009,
    -0.043430, -0.028293, -0.013963, 0.0,
    0.013963, 0.028293, 0.043430, 0.060009,
    0.079142, 0.103361, 0.145560
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

constant float TURBOQUANT_CODEBOOK_6BIT[64] = {
    -3.017255738, -2.423820320, -2.058930059, -1.797707297, -1.601922457, -1.448291610, -1.326356628, -1.225995089,
    -1.139137223, -1.062727519, -0.993135300, -0.928426027, -0.868524053, -0.811485776, -0.757007509, -0.704944748,
    -0.654529973, -0.606343034, -0.559378028, -0.513393457, -0.468229381, -0.424872197, -0.382620330, -0.340845414,
    -0.299256835, -0.258542997, -0.217923724, -0.178098846, -0.138179662, -0.098636745, -0.058835570, -0.018710570,
    0.021130944, 0.060815313, 0.100191218, 0.139750780, 0.179826144, 0.219826894, 0.260545027, 0.301575157,
    0.342993060, 0.384991868, 0.427889936, 0.471579352, 0.516263475, 0.561314135, 0.607624272, 0.655973460,
    0.705897157, 0.758131069, 0.812601781, 0.869977353, 0.930940401, 0.995699969, 1.065519595, 1.141443664,
    1.227592854, 1.328595886, 1.450287956, 1.602228166, 1.797860727, 2.058171284, 2.423209814, 3.012639275
};
constant float TURBOQUANT_THRESHOLDS_6BIT[63] = {
    -2.720538029, -2.241375189, -1.928318678, -1.699814877, -1.525107034, -1.387324119, -1.276175858, -1.182566156,
    -1.100932371, -1.027931409, -0.960780663, -0.898475040, -0.840004914, -0.784246643, -0.730976129, -0.679737361,
    -0.630436503, -0.582860531, -0.536385743, -0.490811419, -0.446550789, -0.403746264, -0.361732872, -0.320051124,
    -0.278899916, -0.238233360, -0.198011285, -0.158139254, -0.118408203, -0.078736157, -0.038773070, 0.001210187,
    0.040973129, 0.080503265, 0.119970999, 0.159788462, 0.199826519, 0.240185961, 0.281060092, 0.322284109,
    0.363992464, 0.406440902, 0.449734644, 0.493921414, 0.538788805, 0.584469203, 0.631798866, 0.680935309,
    0.732014113, 0.785366425, 0.841289567, 0.900458877, 0.963320185, 1.030609782, 1.103481629, 1.184518259,
    1.278094370, 1.389441921, 1.526258061, 1.700044446, 1.928016005, 2.240690549, 2.717924544
};

constant float TURBOQUANT_CODEBOOK_7BIT[128] = {
    -3.161821656, -2.593655236, -2.260366751, -2.036143901, -1.877254125, -1.756981236, -1.662470419, -1.584808772,
    -1.517156127, -1.456758653, -1.402001919, -1.350976044, -1.303012122, -1.257290771, -1.214683970, -1.174284348,
    -1.135820017, -1.099401736, -1.064814314, -1.031291885, -0.998516547, -0.966935452, -0.935911034, -0.905753606,
    -0.876445620, -0.847857376, -0.820004703, -0.792610167, -0.765753537, -0.739575621, -0.713990833, -0.688968586,
    -0.663956599, -0.639361813, -0.615405244, -0.591867693, -0.568442898, -0.545359109, -0.522723513, -0.500390633,
    -0.478618508, -0.457086748, -0.435321291, -0.413745130, -0.392190012, -0.371026980, -0.350001767, -0.328982207,
    -0.308341784, -0.287940557, -0.267649618, -0.247588727, -0.227361475, -0.207182363, -0.187185623, -0.167462390,
    -0.147365963, -0.127493547, -0.107718842, -0.088070616, -0.068404744, -0.048587333, -0.028918753, -0.009434275,
    0.010149665, 0.029802156, 0.049171144, 0.068828458, 0.088722840, 0.108229260, 0.128145472, 0.148108296,
    0.167952409, 0.187973966, 0.208235593, 0.228389448, 0.248464332, 0.268681237, 0.288765152, 0.309276338,
    0.329687599, 0.350446090, 0.371599662, 0.392955936, 0.414318992, 0.435806255, 0.457389054, 0.478953617,
    0.500847597, 0.523094315, 0.545609565, 0.568220557, 0.591509701, 0.615326460, 0.639434160, 0.663947113,
    0.688244905, 0.712990034, 0.738163980, 0.763943700, 0.790592008, 0.818028338, 0.846147718, 0.874858881,
    0.904436875, 0.934984934, 0.965949978, 0.997473628, 1.029677998, 1.063044768, 1.097693392, 1.134172843,
    1.172229797, 1.212077571, 1.254602382, 1.299288409, 1.346488593, 1.397246923, 1.452061588, 1.512432831,
    1.580646704, 1.659743219, 1.754504101, 1.873331421, 2.031510542, 2.255090788, 2.588195086, 3.143419937
};
constant float TURBOQUANT_THRESHOLDS_7BIT[127] = {
    -2.877738446, -2.427010993, -2.148255326, -1.956699013, -1.817117680, -1.709725828, -1.623639596, -1.550982450,
    -1.486957390, -1.429380286, -1.376488981, -1.326994083, -1.280151447, -1.235987371, -1.194484159, -1.155052183,
    -1.117610877, -1.082108025, -1.048053099, -1.014904216, -0.982725999, -0.951423243, -0.920832320, -0.891099613,
    -0.862151498, -0.833931039, -0.806307435, -0.779181852, -0.752664579, -0.726783227, -0.701479710, -0.676462593,
    -0.651659206, -0.627383529, -0.603636469, -0.580155296, -0.556901003, -0.534041311, -0.511557073, -0.489504570,
    -0.467852628, -0.446204019, -0.424533210, -0.402967571, -0.381608496, -0.360514373, -0.339491987, -0.318661995,
    -0.298141171, -0.277795088, -0.257619173, -0.237475101, -0.217271919, -0.197183993, -0.177324006, -0.157414176,
    -0.137429755, -0.117606194, -0.097894729, -0.078237680, -0.058496038, -0.038753043, -0.019176514, 0.000357695,
    0.019975910, 0.039486650, 0.058999801, 0.078775649, 0.098476050, 0.118187366, 0.138126884, 0.158030352,
    0.177963187, 0.198104780, 0.218312521, 0.238426890, 0.258572784, 0.278723194, 0.299020745, 0.319481968,
    0.340066844, 0.361022876, 0.382277799, 0.403637464, 0.425062623, 0.446597654, 0.468171335, 0.489900607,
    0.511970956, 0.534351940, 0.556915061, 0.579865129, 0.603418081, 0.627380310, 0.651690636, 0.676096009,
    0.700617469, 0.725577007, 0.751053840, 0.777267854, 0.804310173, 0.832088028, 0.860503300, 0.889647878,
    0.919710905, 0.950467456, 0.981711803, 1.013575813, 1.046361383, 1.080369080, 1.115933117, 1.153201320,
    1.192153684, 1.233339976, 1.276945396, 1.322888501, 1.371867758, 1.424654256, 1.482247210, 1.546539768,
    1.620194962, 1.707123660, 1.813917761, 1.952420982, 2.143300665, 2.421642937, 2.865807512
};

constant float TURBOQUANT_QJL_SCALE = 1.2533141373155001;
constant float TURBOQUANT_KEY_RESIDUAL_SCALE = 1.0;
constant uint TURBOQUANT_MAX_CODE_WORDS = 28u;
constant float TURBO_WHT_SIGNS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
constant float TURBO_WHT_SIGNS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

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
    float keyResidualScale;
    float valueResidualScale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;
    uint qOffset;
    uint codeWordsPerRow;
    uint regularBits;
    uint highPrecisionBits;
    uint valueCodeWordsPerRow;
    uint valueRegularBits;
    uint valueHighPrecisionBits;
    uint reserved;
};

struct ERTurboQuantDebugScoreTerms {
    float mseDot;
    float residualDot;
    float rowNorm;
    float residualNorm;
    float score;
};

constant uint TURBOQUANT_Q8_0_BLOCK_BYTES = 34u;
constant uint TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK = 32u;

inline float4 tq_q8_0_load_float4(
    device const uchar *row,
    uint dim4
) {
    const uint scalarIndex = dim4 * 4u;
    const uint blockIndex = scalarIndex / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint inBlockIndex = scalarIndex % TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    device const uchar *block = row + blockIndex * TURBOQUANT_Q8_0_BLOCK_BYTES;
    const float scale = float(as_type<half>(*(device const ushort *)block));
    return scale * float4(
        float(as_type<char>(block[2 + inBlockIndex + 0])),
        float(as_type<char>(block[2 + inBlockIndex + 1])),
        float(as_type<char>(block[2 + inBlockIndex + 2])),
        float(as_type<char>(block[2 + inBlockIndex + 3]))
    );
}

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
        case 4: return TURBOQUANT_CODEBOOK_4BIT[code];
        case 5: return TURBOQUANT_CODEBOOK_5BIT[code];
        case 6: return TURBOQUANT_CODEBOOK_6BIT[code];
        case 7: return TURBOQUANT_CODEBOOK_7BIT[code];
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
        case 4:
            for (uint i = 0; i < 15; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_4BIT[i]) { return i; }
            }
            return 15;
        case 5:
            for (uint i = 0; i < 31; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_5BIT[i]) { return i; }
            }
            return 31;
        case 6:
            for (uint i = 0; i < 63; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_6BIT[i]) { return i; }
            }
            return 63;
        case 7:
            for (uint i = 0; i < 127; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_7BIT[i]) { return i; }
            }
            return 127;
        default:
            return 0;
    }
}

inline float tq_quantization_benefit(float value, uint regularBits, uint highPrecisionBits) {
    uint regularCode = tq_code_for_value(value, regularBits);
    uint highCode = tq_code_for_value(value, highPrecisionBits);
    float regularDelta = value - tq_centroid(regularBits, regularCode);
    float highDelta = value - tq_centroid(highPrecisionBits, highCode);
    return (regularDelta * regularDelta) - (highDelta * highDelta);
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
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
}

inline void tq_forward_randomized_hadamard(threadgroup float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
}

inline void tq_forward_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_hadamard_parallel(values, lane, laneCount);
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void tq_inverse_randomized_hadamard(thread float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
}

inline void tq_inverse_randomized_hadamard(threadgroup float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
}

inline void tq_inverse_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_hadamard_parallel(values, lane, laneCount);
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void tq_forward_planar(thread float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) - (s * y);
        values[base + 1u] = (s * x) + (c * y);
    }
}

inline void tq_inverse_planar(thread float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) + (s * y);
        values[base + 1u] = (-s * x) + (c * y);
    }
}

inline void tq_forward_planar(threadgroup float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) - (s * y);
        values[base + 1u] = (s * x) + (c * y);
    }
}

inline void tq_inverse_planar(threadgroup float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) + (s * y);
        values[base + 1u] = (-s * x) + (c * y);
    }
}

inline void tq_forward_rotation(thread float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_forward_planar(values, rotationData);
    } else {
        tq_forward_randomized_hadamard(values, rotationData);
    }
}

inline void tq_inverse_rotation(thread float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_inverse_planar(values, rotationData);
    } else {
        tq_inverse_randomized_hadamard(values, rotationData);
    }
}

inline void tq_forward_rotation(threadgroup float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_forward_planar(values, rotationData);
    } else {
        tq_forward_randomized_hadamard(values, rotationData);
    }
}

inline void tq_inverse_rotation(threadgroup float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_inverse_planar(values, rotationData);
    } else {
        tq_inverse_randomized_hadamard(values, rotationData);
    }
}

inline void tq_select_top32_bitonic_mask(
    threadgroup const float *rotated,
    threadgroup uint *maskWords,
    threadgroup float *benefits,
    threadgroup uint *indices,
    uint regularBits,
    uint highPrecisionBits,
    bool useQuantizationBenefit,
    uint lane,
    uint laneCount
) {
    for (uint dim = lane; dim < 128; dim += laneCount) {
        benefits[dim] = useQuantizationBenefit
            ? tq_quantization_benefit(rotated[dim], regularBits, highPrecisionBits)
            : fabs(rotated[dim]);
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
                        ? benefits[i] > benefits[ixj]
                        : benefits[i] < benefits[ixj];
                    if (shouldSwap) {
                        float tmpBenefit = benefits[i];
                        benefits[i] = benefits[ixj];
                        benefits[ixj] = tmpBenefit;
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
    device const float *innerQScaleInv [[buffer(8)]],
    uint rowIndex [[thread_position_in_grid]]
) {
    if (rowIndex >= params.rowCount) { return; }
    const bool useInnerQScaling = ((params.reserved >> 1) & 1u) != 0u;
    const bool usePlanarRotation = ((params.reserved >> 2) & 1u) != 0u;

    thread float normalized[128];
    thread float rotated[128];
    thread float reconstructed[128];
    thread float residual[128];
    thread float projectedResidual[128];
    thread bool highPrecisionMask[128];
    thread uint codeWords[TURBOQUANT_MAX_CODE_WORDS];
    thread uint signWords[4];
    thread uint maskWords[4];

    for (uint i = 0; i < TURBOQUANT_MAX_CODE_WORDS; ++i) { codeWords[i] = 0; }
    for (uint i = 0; i < 4; ++i) {
        signWords[i] = 0;
        maskWords[i] = 0;
    }

    uint sourceBase = rowIndex * params.sourceRowStride;
    float rowNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        float value = source[sourceBase + dim];
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            value /= innerQScaleInv[dim];
        }
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
    const bool usesResidualPath =
        params.highPrecisionChannelCount > 0u || params.highPrecisionBits != params.regularBits;

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
    tq_forward_rotation(rotated, rotationSigns, usePlanarRotation);

    bool useQuantizationBenefit = (params.reserved & 1u) != 0u;
    for (uint pick = 0; pick < params.highPrecisionChannelCount; ++pick) {
        float bestBenefit = -INFINITY;
        uint bestIndex = 0;
        for (uint dim = 0; dim < 128; ++dim) {
            if (highPrecisionMask[dim]) { continue; }
            float benefit = useQuantizationBenefit
                ? tq_quantization_benefit(rotated[dim], params.regularBits, params.highPrecisionBits)
                : fabs(rotated[dim]);
            if (benefit > bestBenefit) {
                bestBenefit = benefit;
                bestIndex = dim;
            }
        }
        highPrecisionMask[bestIndex] = true;
    }

    uint sidebandOffset = 128u * params.regularBits;
    float reconstructedNormSq = 0.0;
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
        reconstructedNormSq += reconstructed[dim] * reconstructed[dim];
    }

    if (!usesResidualPath) {
        float reconstructedNorm = sqrt(reconstructedNormSq);
        float correctedNorm = reconstructedNorm > 0.0 ? (rowNorm / reconstructedNorm) : rowNorm;
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = 0u;
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = correctedNorm;
        metaDst[1] = 0.0;
        return;
    }

    tq_inverse_rotation(reconstructed, rotationSigns, usePlanarRotation);
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
    uint codeWordsPerRow,
    bool useQuantizationBenefit
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

    tq_select_top32_bitonic_mask(
        rotated,
        maskWords,
        magnitudes,
        indices,
        2u,
        3u,
        useQuantizationBenefit,
        lane,
        kLaneCount
    );

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
        params.codeWordsPerRow,
        (params.reserved & 1u) != 0u
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
        params.codeWordsPerRow,
        false
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
        params.codeWordsPerRow,
        false
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
        params.codeWordsPerRow,
        true
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

    tq_select_top32_bitonic_mask(
        rotated,
        maskWords,
        magnitudes,
        indices,
        2u,
        3u,
        false,
        lane,
        kLaneCount
    );
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
            float bestBenefit = -INFINITY;
            uint bestIndex = 0;
            for (uint dim = 0; dim < 128; ++dim) {
                if (tq_get_bit(maskWords, dim) == 1u) { continue; }
                float benefit = tq_quantization_benefit(rotated[dim], 2u, 3u);
                if (benefit > bestBenefit) {
                    bestBenefit = benefit;
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
        magnitudes[dim] = tq_quantization_benefit(rotatedSource[rowIndex * 128 + dim], 2u, 3u);
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
    device const float *innerQScaleInv [[buffer(15)]],
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
    const bool useInnerQScaling = (params.reserved & 1u) != 0u;
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool usePlanarValueRotation = (params.reserved & 4u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

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
            if (useInnerQScaling && innerQScaleInv != nullptr) {
                value *= innerQScaleInv[dim];
            }
            qRotation[dim] = value;
            qResidual[dim] = value;
            outputMSE[dim] = 0.0;
            outputResidual[dim] = 0.0;
        }
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
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
                    if (useKeyResidualPath) {
                        residualDot += qResidual[dim] * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                    }
                }

                float dot = rowNorm * (
                    mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * residualNorm * residualDot
                );
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
                device const uint *codeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
                device const uint *signRow = VResidualSigns + rowIndex * 4;
                device const uint *maskRow = VOutlierMask + rowIndex * 4;
                device const float *metaRow = VMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];
                float mseScale = prob * rowNorm;
                float residualScale = prob * rowNorm * residualNorm;

                uint sidebandOffset = 128u * params.valueRegularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.valueRegularBits,
                        params.valueHighPrecisionBits,
                        sidebandOffset
                    );
                    outputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
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

        tq_inverse_rotation(outputMSE, valueRotationSigns, usePlanarValueRotation);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);

        if (useInnerQScaling && innerQScaleInv != nullptr) {
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= innerQScaleInv[dim];
                outputResidual[dim] *= innerQScaleInv[dim];
            }
        }

        float invSum = runningSum > 0.0 ? 1.0 / runningSum : 0.0;
        uint oBase = qRow * qStride + headIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            O[oBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
        }
    }
}

kernel void gqa_attention_q8k_turboquant(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uint *VCodes [[buffer(2)]],
    device const uint *VResidualSigns [[buffer(3)]],
    device const uint *VOutlierMask [[buffer(4)]],
    device const float *VMetadata [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *valueRotationSigns [[buffer(8)]],
    device const float *valueResidualProjectionSigns [[buffer(9)]],
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
    const uint headDim4 = params.headDim / 4u;
    const uint q8BlocksPerRow = params.headDim / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint q8RowBytes = q8BlocksPerRow * TURBOQUANT_Q8_0_BLOCK_BYTES;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    threadgroup float outputMSEScratch[16 * 128];
    threadgroup float outputResidualScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float scores[16];
    float probs[16];

    if (activeQ) {
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
        for (uint dim = 0; dim < 128; ++dim) {
            outputMSE[dim] = 0.0;
            outputResidual[dim] = 0.0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; ++kvBlock) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (activeQ) {
            threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
            threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
            uint qBase = qRow * qStride + headIndex * params.headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);

            float blockMax = -INFINITY;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uchar *kRow = K + rowIndex * q8RowBytes;
                float dot = 0.0f;
                for (uint dim4 = 0; dim4 < headDim4; ++dim4) {
                    dot += metal::dot(qVec[dim4], tq_q8_0_load_float4(kRow, dim4));
                }
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0f;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (scores[kvIndex] == -INFINITY) {
                    probs[kvIndex] = 0.0f;
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
                if (prob == 0.0f) { continue; }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
                device const uint *signRow = VResidualSigns + rowIndex * 4;
                device const uint *maskRow = VOutlierMask + rowIndex * 4;
                device const float *metaRow = VMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];
                float mseScale = prob * rowNorm;
                float residualScale = prob * rowNorm * residualNorm;

                uint sidebandOffset = 128u * params.valueRegularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.valueRegularBits,
                        params.valueHighPrecisionBits,
                        sidebandOffset
                    );
                    outputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
                    outputResidual[dim] += residualScale * (tq_get_bit(signRow, dim) == 1u ? 1.0f : -1.0f);
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

        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            O[oBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
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
    device const float *innerQScaleInv [[buffer(15)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }
    const bool useInnerQScaling = (params.reserved & 1u) != 0u;
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool usePlanarValueRotation = (params.reserved & 4u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

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
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            value *= innerQScaleInv[dim];
        }
        qRotation[dim] = value;
        qResidual[dim] = value;
        outputMSE[dim] = 0.0;
        outputResidual[dim] = 0.0;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutputMSE[dim] = 0.0;
        laneOutputResidual[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
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
            if (useKeyResidualPath) {
                residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
            }
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
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

        device const uint *vCodeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
        device const uint *vSignRow = VResidualSigns + rowIndex * 4;
        device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
        device const float *vMetaRow = VMetadata + rowIndex * 2;
        float valueRowNorm = vMetaRow[0];
        float valueResidualNorm = vMetaRow[1];
        float mseScale = prob * valueRowNorm;
        float residualScale = prob * valueRowNorm * valueResidualNorm;

        uint valueSidebandOffset = 128u * params.valueRegularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                vCodeRow,
                dim,
                useHighPrecision,
                params.valueRegularBits,
                params.valueHighPrecisionBits,
                valueSidebandOffset
            );
            laneOutputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
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
        tq_inverse_rotation(outputMSE, valueRotationSigns, usePlanarValueRotation);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= innerQScaleInv[dim];
                outputResidual[dim] *= innerQScaleInv[dim];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
    }
}

kernel void gqa_attention_q8k_turboquant_decode(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uint *VCodes [[buffer(2)]],
    device const uint *VResidualSigns [[buffer(3)]],
    device const uint *VOutlierMask [[buffer(4)]],
    device const float *VMetadata [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *valueRotationSigns [[buffer(8)]],
    device const float *valueResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16u;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint headDim4 = params.headDim / 4u;
    const uint q8BlocksPerRow = params.headDim / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint q8RowBytes = q8BlocksPerRow * TURBOQUANT_Q8_0_BLOCK_BYTES;

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
    const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        outputMSE[dim] = 0.0f;
        outputResidual[dim] = 0.0f;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutputMSE[dim] = 0.0f;
        laneOutputResidual[dim] = 0.0f;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uchar *kRow = K + rowIndex * q8RowBytes;
        float dot = 0.0f;
        for (uint dim4 = 0; dim4 < headDim4; ++dim4) {
            dot += metal::dot(qVec[dim4], tq_q8_0_load_float4(kRow, dim4));
        }

        float score = dot * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0f : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutputMSE[dim] *= correction;
            laneOutputResidual[dim] *= correction;
        }

        device const uint *vCodeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
        device const uint *vSignRow = VResidualSigns + rowIndex * 4;
        device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
        device const float *vMetaRow = VMetadata + rowIndex * 2;
        float valueRowNorm = vMetaRow[0];
        float valueResidualNorm = vMetaRow[1];
        float mseScale = prob * valueRowNorm;
        float residualScale = prob * valueRowNorm * valueResidualNorm;

        uint valueSidebandOffset = 128u * params.valueRegularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                vCodeRow,
                dim,
                useHighPrecision,
                params.valueRegularBits,
                params.valueHighPrecisionBits,
                valueSidebandOffset
            );
            laneOutputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
            laneOutputResidual[dim] += residualScale * (tq_get_bit(vSignRow, dim) == 1u ? 1.0f : -1.0f);
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

    float localScale = runningSum > 0.0f ? exp(runningMax - globalMax) : 0.0f;
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
        float mseAccum = 0.0f;
        float residualAccum = 0.0f;
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

    float invSum = globalSum > 0.0f ? 1.0f / globalSum : 0.0f;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
    }
}

kernel void turboquant_debug_decode_score_terms(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device ERTurboQuantDebugScoreTerms *scoreTerms [[buffer(5)]],
    constant ERTurboQuantAttentionParams &params [[buffer(6)]],
    device const float *keyRotationSigns [[buffer(7)]],
    device const float *keyResidualProjectionSigns [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint kvPos = gid.x;
    uint headIndex = gid.y;
    if (headIndex >= params.numHeads) { return; }
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    if (kvPos >= kvLimit) { return; }

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint qBase = headIndex * params.headDim;
    uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;

    thread float qRotation[128];
    thread float qResidual[128];
    for (uint dim = 0; dim < 128; ++dim) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
    }
    tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
    if (useKeyResidualPath) {
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }

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
        if (useKeyResidualPath) {
            residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
        }
    }

    float score = keyRowNorm * (
        mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
    ) * params.scale;

    uint outputIndex = headIndex * kvLimit + kvPos;
    scoreTerms[outputIndex] = {
        mseDot,
        useKeyResidualPath ? residualDot : 0.0f,
        keyRowNorm,
        useKeyResidualPath ? keyResidualNorm : 0.0f,
        score
    };
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
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

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
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
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
            if (useKeyResidualPath) {
                residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
            }
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
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
    }
    for (uint dim = 0; dim < 128; ++dim) {
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
                mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
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
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
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
    }
    for (uint dim = 0; dim < 128; ++dim) {
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
                mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
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
    }
    for (uint dim = 0; dim < 128; ++dim) {
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
