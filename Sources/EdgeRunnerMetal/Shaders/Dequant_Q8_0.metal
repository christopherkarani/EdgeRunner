#include <metal_stdlib>
using namespace metal;

struct ERDequantQ8_0Params {
    uint blockCount;
    uint outputOffset;
};

constant uint q8_0BlockBytes = 34;
constant uint q8_0WeightsPerBlock = 32;

kernel void dequant_q8_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ8_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;
    device const uchar* block = input + (tid * q8_0BlockBytes);
    float scale = float(as_type<half>(*(device const ushort*)block));
    uint outputBase = params.outputOffset + (tid * q8_0WeightsPerBlock);
    for (uint index = 0; index < q8_0WeightsPerBlock; index++) {
        output[outputBase + index] = scale * float(as_type<char>(block[2 + index]));
    }
}

// === High-Performance Fused Q8_0 GEMV ===
// Architecture inspired by llama.cpp's ggml Metal kernel:
//   - 4 simdgroups (128 threads) per threadgroup
//   - 2 rows per threadgroup — Y-value caching across rows
//   - Each thread processes NQ=8 elements at a time
//   - Cross-simdgroup reduction via threadgroup memory
//
// y[row] = sum_k dequant(W_q8[row, k]) * x[k]

struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant constexpr short NQ = 8;        // elements per thread per iteration
constant constexpr short NR = 4;        // rows per threadgroup (4x Y-value reuse)
constant constexpr short NSG = 4;       // simdgroups per threadgroup (4×32=128 threads)
constant constexpr short NW = 32;       // simdwidth

kernel void dequant_q8_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = tgIndex * NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const short ix = tiisg / (NW / NQ);  // 0..7 (which block set)
    const short il = tiisg % (NW / NQ);  // 0..3 (which 8-element chunk within block)

    float sumf[NR] = { 0.f };

    // Pointers to weight rows
    device const uchar* ax[NR];
    for (short row = 0; row < NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    const short ib0 = sgitg * NQ + ix;
    device const float* yb = x + ib0 * q8_0WeightsPerBlock + il * NQ;

    // Main loop: each iteration processes NQ elements from one block
    for (short ib = ib0; ib < nb; ib += NSG * NQ) {
        // Cache x values in registers (reused across NR rows)
        float yl[NQ];
        for (short i = 0; i < NQ; i++) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2) + il * NQ;

            float sumq = 0.f;
            for (short i = 0; i < NQ; i++) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * q8_0WeightsPerBlock;
    }

    // Reduction: first within simdgroup, then across simdgroups
    for (short row = 0; row < NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float tg_sum[NR][NSG];

    if (tiisg == 0) {
        for (short row = 0; row < NR; row++) {
            tg_sum[row][sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread of first simdgroup writes final result
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < NR; row++) {
            if (row0 + row >= params.rows) break;
            float total = 0.f;
            for (short sg = 0; sg < NSG; sg++) {
                total += tg_sum[row][sg];
            }
            y[row0 + row] = total;
        }
    }
}

// === Float16 output variant — writes half directly to KV cache ===
// Eliminates separate f32→f16 conversion dispatch.
kernel void dequant_q8_0_gemv_f16out(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = tgIndex * NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);

    float sumf[NR] = { 0.f };

    device const uchar* ax[NR];
    for (short row = 0; row < NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    const short ib0 = sgitg * NQ + ix;
    device const float* yb = x + ib0 * q8_0WeightsPerBlock + il * NQ;

    for (short ib = ib0; ib < nb; ib += NSG * NQ) {
        float yl[NQ];
        for (short i = 0; i < NQ; i++) yl[i] = yb[i];

        for (short row = 0; row < NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2) + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; i++) sumq += float(qs[i]) * yl[i];
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * q8_0WeightsPerBlock;
    }

    for (short row = 0; row < NR; row++) sumf[row] = simd_sum(sumf[row]);

    threadgroup float tg_sum[NR][NSG];
    if (tiisg == 0) {
        for (short row = 0; row < NR; row++) tg_sum[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < NR; row++) {
            if (row0 + row >= params.rows) break;
            float total = 0.f;
            for (short sg = 0; sg < NSG; sg++) total += tg_sum[row][sg];
            y[row0 + row] = half(total);
        }
    }
}
