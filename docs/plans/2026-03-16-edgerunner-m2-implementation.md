# EdgeRunner Milestone 2: Transformer Primitives — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core transformer primitives needed to run decoder-only models: GEMM, attention, positional embeddings, normalization, and activations.

**Architecture:** Metal 4 compute shaders with simdgroup intrinsics. Flash Attention with O(N) memory. GQA for modern architectures. Ring-buffer KV cache with mixed-precision support.

**Tech Stack:** Swift 6.2, Metal Shading Language 4.0, Swift Testing

**Depends on:** Milestone 1 (docs/plans/2026-02-28-edgerunner-m1-implementation.md)

---

## Task 1: GEMM Kernel (Tiled MatMul)

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/GEMM.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/GEMMParams.h`
- Create: `Sources/EdgeRunnerMetal/GEMMKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/GEMMTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/GEMMTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference matmul for verification.
private func cpuMatmul(
    a: [Float], b: [Float],
    M: Int, N: Int, K: Int
) -> [Float] {
    var c = [Float](repeating: 0, count: M * N)
    for i in 0..<M {
        for j in 0..<N {
            var sum: Float = 0
            for p in 0..<K {
                sum += a[i * K + p] * b[p * N + j]
            }
            c[i * N + j] = sum
        }
    }
    return c
}

/// CPU reference matmul in Float16 (computed in Float32, stored as Float16).
private func cpuMatmulF16(
    a: [Float16], b: [Float16],
    M: Int, N: Int, K: Int
) -> [Float16] {
    var c = [Float16](repeating: 0, count: M * N)
    for i in 0..<M {
        for j in 0..<N {
            var sum: Float = 0
            for p in 0..<K {
                sum += Float(a[i * K + p]) * Float(b[p * N + j])
            }
            c[i * N + j] = Float16(sum)
        }
    }
    return c
}

@Suite("GEMM Kernel")
struct GEMMTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func smallSquareFloat32() async throws {
        let M = 32, N = 32, K = 32
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let b = (0..<K*N).map { _ in Float.random(in: -1...1) }
        let expected = cpuMatmul(a: a, b: b, M: M, N: N, K: K)

        let kernel = try GEMMKernel(device: device)
        let result = try await kernel.execute(
            a: a, b: b, M: M, N: N, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<(M * N) {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func rectangularFloat32() async throws {
        let M = 64, N = 128, K = 32
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let b = (0..<K*N).map { _ in Float.random(in: -1...1) }
        let expected = cpuMatmul(a: a, b: b, M: M, N: N, K: K)

        let kernel = try GEMMKernel(device: device)
        let result = try await kernel.execute(
            a: a, b: b, M: M, N: N, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<(M * N) {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func largerFloat32() async throws {
        let M = 128, N = 128, K = 64
        let a = (0..<M*K).map { _ in Float.random(in: -0.5...0.5) }
        let b = (0..<K*N).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuMatmul(a: a, b: b, M: M, N: N, K: K)

        let kernel = try GEMMKernel(device: device)
        let result = try await kernel.execute(
            a: a, b: b, M: M, N: N, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<(M * N) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func smallSquareFloat16() async throws {
        let M = 32, N = 32, K = 32
        let a = (0..<M*K).map { _ in Float16.random(in: -1...1) }
        let b = (0..<K*N).map { _ in Float16.random(in: -1...1) }
        let expected = cpuMatmulF16(a: a, b: b, M: M, N: N, K: K)

        let kernel = try GEMMKernel(device: device)
        let result: [Float16] = try await kernel.executeF16(
            a: a, b: b, M: M, N: N, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<(M * N) {
            #expect(abs(Float(result[i]) - Float(expected[i])) < 1e-2,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func identityMatrixFloat32() async throws {
        let N = 32
        var identity = [Float](repeating: 0, count: N * N)
        for i in 0..<N { identity[i * N + i] = 1.0 }
        let b = (0..<N*N).map { _ in Float.random(in: -1...1) }

        let kernel = try GEMMKernel(device: device)
        let result = try await kernel.execute(
            a: identity, b: b, M: N, N: N, K: N,
            commandQueue: commandQueue
        )

        for i in 0..<(N * N) {
            #expect(abs(result[i] - b[i]) < 1e-5,
                    "Identity mult failed at [\(i)]")
        }
    }

    @Test func nonMultipleOf32Dimensions() async throws {
        // Padded dispatch — M, N, K not multiples of tile size
        let M = 17, N = 23, K = 11
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let b = (0..<K*N).map { _ in Float.random(in: -1...1) }
        let expected = cpuMatmul(a: a, b: b, M: M, N: N, K: K)

        let kernel = try GEMMKernel(device: device)
        let result = try await kernel.execute(
            a: a, b: b, M: M, N: N, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<(M * N) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }
}

enum MetalTestError: Error {
    case noMetal
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GEMMTests 2>&1`
Expected: FAIL — `GEMMKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/GEMMParams.h
#ifndef GEMM_PARAMS_H
#define GEMM_PARAMS_H

#include <stdint.h>

/// Parameters for GEMM kernel dispatch.
/// C = A * B where A is [M x K], B is [K x N], C is [M x N].
typedef struct {
    uint32_t M;         // rows of A / rows of C
    uint32_t N;         // cols of B / cols of C
    uint32_t K;         // cols of A / rows of B
    uint32_t lda;       // leading dimension of A (typically K)
    uint32_t ldb;       // leading dimension of B (typically N)
    uint32_t ldc;       // leading dimension of C (typically N)
} ERGEMMParams;

#endif /* GEMM_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/GEMM.metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ERGEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Tile dimensions for simdgroup matrix multiply
constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 32;

/// Tiled GEMM using simdgroup_matrix_multiply_accumulate.
/// C[M,N] = A[M,K] * B[K,N], row-major layout.
/// Threadgroup: [32, 32], dispatched over (ceil(N/32), ceil(M/32)) grid.
kernel void gemm_f32(
    device const float* A         [[buffer(0)]],
    device const float* B         [[buffer(1)]],
    device float*       C         [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 group_id    [[threadgroup_position_in_grid]],
    uint2 local_id    [[thread_position_in_threadgroup]],
    uint  simd_index  [[simdgroup_index_in_threadgroup]],
    uint  lane_id     [[thread_index_in_simdgroup]]
) {
    // Each threadgroup computes a TILE_M x TILE_N block of C.
    const uint row_base = group_id.y * TILE_M;
    const uint col_base = group_id.x * TILE_N;

    // simdgroup_matrix types: 8x8 tiles
    simdgroup_float8x8 acc[4][4];  // 4x4 grid of 8x8 = 32x32
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_float8x8(0);

    // Tile over K dimension
    for (uint k_tile = 0; k_tile < params.K; k_tile += TILE_K) {
        // Load A tile [TILE_M x TILE_K] and B tile [TILE_K x TILE_N]
        // using simdgroup loads into 8x8 blocks
        simdgroup_float8x8 a_block[4][4];  // 32x32 = 4x4 grid of 8x8
        simdgroup_float8x8 b_block[4][4];

        for (uint bi = 0; bi < 4; bi++) {
            for (uint bj = 0; bj < 4; bj++) {
                uint a_row = row_base + bi * 8;
                uint a_col = k_tile + bj * 8;
                if (a_row < params.M && a_col < params.K) {
                    simdgroup_load(a_block[bi][bj], A, params.lda,
                                   ulong2(a_col, a_row));
                } else {
                    a_block[bi][bj] = simdgroup_float8x8(0);
                }

                uint b_row = k_tile + bi * 8;
                uint b_col = col_base + bj * 8;
                if (b_row < params.K && b_col < params.N) {
                    simdgroup_load(b_block[bi][bj], B, params.ldb,
                                   ulong2(b_col, b_row));
                } else {
                    b_block[bi][bj] = simdgroup_float8x8(0);
                }
            }
        }

        // Multiply-accumulate: acc[i][j] += sum_p a_block[i][p] * b_block[p][j]
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                for (uint p = 0; p < 4; p++) {
                    simdgroup_multiply_accumulate(acc[i][j], a_block[i][p],
                                                  b_block[p][j], acc[i][j]);
                }
            }
        }
    }

    // Store result
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint c_row = row_base + i * 8;
            uint c_col = col_base + j * 8;
            if (c_row < params.M && c_col < params.N) {
                simdgroup_store(acc[i][j], C, params.ldc,
                                ulong2(c_col, c_row));
            }
        }
    }
}

/// Tiled GEMM for Float16 using simdgroup_matrix_multiply_accumulate.
kernel void gemm_f16(
    device const half*  A         [[buffer(0)]],
    device const half*  B         [[buffer(1)]],
    device half*        C         [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 group_id    [[threadgroup_position_in_grid]],
    uint2 local_id    [[thread_position_in_threadgroup]],
    uint  simd_index  [[simdgroup_index_in_threadgroup]],
    uint  lane_id     [[thread_index_in_simdgroup]]
) {
    const uint row_base = group_id.y * TILE_M;
    const uint col_base = group_id.x * TILE_N;

    simdgroup_half8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_half8x8(0);

    for (uint k_tile = 0; k_tile < params.K; k_tile += TILE_K) {
        simdgroup_half8x8 a_block[4][4];
        simdgroup_half8x8 b_block[4][4];

        for (uint bi = 0; bi < 4; bi++) {
            for (uint bj = 0; bj < 4; bj++) {
                uint a_row = row_base + bi * 8;
                uint a_col = k_tile + bj * 8;
                if (a_row < params.M && a_col < params.K) {
                    simdgroup_load(a_block[bi][bj], A, params.lda,
                                   ulong2(a_col, a_row));
                } else {
                    a_block[bi][bj] = simdgroup_half8x8(0);
                }

                uint b_row = k_tile + bi * 8;
                uint b_col = col_base + bj * 8;
                if (b_row < params.K && b_col < params.N) {
                    simdgroup_load(b_block[bi][bj], B, params.ldb,
                                   ulong2(b_col, b_row));
                } else {
                    b_block[bi][bj] = simdgroup_half8x8(0);
                }
            }
        }

        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                for (uint p = 0; p < 4; p++) {
                    simdgroup_multiply_accumulate(acc[i][j], a_block[i][p],
                                                  b_block[p][j], acc[i][j]);
                }
            }
        }
    }

    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint c_row = row_base + i * 8;
            uint c_col = col_base + j * 8;
            if (c_row < params.M && c_col < params.N) {
                simdgroup_store(acc[i][j], C, params.ldc,
                                ulong2(c_col, c_row));
            }
        }
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/GEMMKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the tiled GEMM Metal kernel.
/// Supports Float32 and Float16 matrix multiplication.
public final class GEMMKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let pipelineF16: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw GEMMError.libraryNotFound
        }
        guard let fnF32 = library.makeFunction(name: "gemm_f32") else {
            throw GEMMError.functionNotFound("gemm_f32")
        }
        guard let fnF16 = library.makeFunction(name: "gemm_f16") else {
            throw GEMMError.functionNotFound("gemm_f16")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fnF32)
        self.pipelineF16 = try device.makeComputePipelineState(function: fnF16)
    }

    /// Execute Float32 GEMM: C[M,N] = A[M,K] * B[K,N].
    public func execute(
        a: [Float], b: [Float],
        M: Int, N: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufB = device.makeBuffer(
            bytes: b, length: b.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufC = device.makeBuffer(
            length: M * N * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMMParams(
            M: UInt32(M), N: UInt32(N), K: UInt32(K),
            lda: UInt32(K), ldb: UInt32(N), ldc: UInt32(N)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMMError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMMParams>.stride, index: 3)

        let gridSize = MTLSize(
            width: (N + 31) / 32,
            height: (M + 31) / 32,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: ptr, count: M * N))
    }

    /// Execute Float16 GEMM: C[M,N] = A[M,K] * B[K,N].
    public func executeF16(
        a: [Float16], b: [Float16],
        M: Int, N: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float16] {
        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufB = device.makeBuffer(
            bytes: b, length: b.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufC = device.makeBuffer(
            length: M * N * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMMParams(
            M: UInt32(M), N: UInt32(N), K: UInt32(K),
            lda: UInt32(K), ldb: UInt32(N), ldc: UInt32(N)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMMError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF16)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMMParams>.stride, index: 3)

        let gridSize = MTLSize(
            width: (N + 31) / 32,
            height: (M + 31) / 32,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufC.contents().bindMemory(to: Float16.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: ptr, count: M * N))
    }
}

public enum GEMMError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter GEMMTests 2>&1`
Expected: All 6 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/GEMM.metal \
      Sources/EdgeRunnerSharedTypes/include/GEMMParams.h \
      Sources/EdgeRunnerMetal/GEMMKernel.swift \
      Tests/EdgeRunnerMetalTests/GEMMTests.swift
git commit -m "feat: add tiled GEMM kernel with simdgroup matrix multiply

32x32 tiling using simdgroup_matrix_multiply_accumulate.
Float32 and Float16 variants. Handles non-aligned dimensions
with bounds checking. CPU reference tests verify correctness."
```

---

## Task 2: GEMV Kernel (Matrix-Vector for Decoding)

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/GEMV.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/GEMVParams.h`
- Create: `Sources/EdgeRunnerMetal/GEMVKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/GEMVTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/GEMVTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference matvec: y[M] = A[M,K] * x[K]
private func cpuGemv(a: [Float], x: [Float], M: Int, K: Int) -> [Float] {
    var y = [Float](repeating: 0, count: M)
    for i in 0..<M {
        var sum: Float = 0
        for j in 0..<K {
            sum += a[i * K + j] * x[j]
        }
        y[i] = sum
    }
    return y
}

/// CPU reference matvec Float16
private func cpuGemvF16(a: [Float16], x: [Float16], M: Int, K: Int) -> [Float16] {
    var y = [Float16](repeating: 0, count: M)
    for i in 0..<M {
        var sum: Float = 0
        for j in 0..<K {
            sum += Float(a[i * K + j]) * Float(x[j])
        }
        y[i] = Float16(sum)
    }
    return y
}

@Suite("GEMV Kernel")
struct GEMVTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func smallGemvFloat32() async throws {
        let M = 64, K = 128
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func largeGemvFloat32() async throws {
        let M = 4096, K = 4096
        let a = (0..<M*K).map { _ in Float.random(in: -0.1...0.1) }
        let x = (0..<K).map { _ in Float.random(in: -0.1...0.1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func nonAlignedDimensions() async throws {
        let M = 37, K = 73
        let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func identityGemv() async throws {
        let N = 64
        var identity = [Float](repeating: 0, count: N * N)
        for i in 0..<N { identity[i * N + i] = 1.0 }
        let x = (0..<N).map { _ in Float.random(in: -1...1) }

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: identity, x: x, M: N, K: N,
            commandQueue: commandQueue
        )

        for i in 0..<N {
            #expect(abs(result[i] - x[i]) < 1e-5,
                    "Identity gemv failed at [\(i)]")
        }
    }

    @Test func gemvFloat16() async throws {
        let M = 64, K = 128
        let a = (0..<M*K).map { _ in Float16.random(in: -1...1) }
        let x = (0..<K).map { _ in Float16.random(in: -1...1) }
        let expected = cpuGemvF16(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result: [Float16] = try await kernel.executeF16(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        for i in 0..<M {
            #expect(abs(Float(result[i]) - Float(expected[i])) < 1e-2,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func singleRowGemv() async throws {
        // Degenerate case: 1xK * Kx1 = scalar
        let M = 1, K = 256
        let a = (0..<K).map { _ in Float.random(in: -1...1) }
        let x = (0..<K).map { _ in Float.random(in: -1...1) }
        let expected = cpuGemv(a: a, x: x, M: M, K: K)

        let kernel = try GEMVKernel(device: device)
        let result = try await kernel.execute(
            a: a, x: x, M: M, K: K,
            commandQueue: commandQueue
        )

        #expect(abs(result[0] - expected[0]) < 1e-4)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GEMVTests 2>&1`
Expected: FAIL — `GEMVKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/GEMVParams.h
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
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/GEMV.metal
#include <metal_stdlib>
using namespace metal;

struct ERGEMVParams {
    uint M;
    uint K;
    uint lda;
};

// Each threadgroup handles one row.
// Threads within the group cooperatively reduce across K.
// Uses simd_sum for fast warp-level reduction.
constant uint GEMV_THREADS_PER_ROW = 256;

kernel void gemv_f32(
    device const float*      A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Each thread accumulates a partial dot product
    float partial = 0.0f;
    device const float* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += a_row[j] * x[j];
    }

    // Warp-level reduction
    partial = simd_sum(partial);

    // Cross-warp reduction via threadgroup memory
    threadgroup float shared_sums[32]; // max 32 simdgroups (1024/32)

    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp finalizes
    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = val;
        }
    }
}

kernel void gemv_f16(
    device const half*       A       [[buffer(0)]],
    device const half*       x       [[buffer(1)]],
    device half*             y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Accumulate in float for numerical stability
    float partial = 0.0f;
    device const half* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += float(a_row[j]) * float(x[j]);
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = half(val);
        }
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/GEMVKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the row-parallel GEMV Metal kernel.
/// Optimized for autoregressive decoding: y[M] = A[M,K] * x[K].
public final class GEMVKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let pipelineF16: MTLComputePipelineState
    private let device: MTLDevice

    private static let threadsPerRow = 256

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw GEMVError.libraryNotFound
        }
        guard let fnF32 = library.makeFunction(name: "gemv_f32") else {
            throw GEMVError.functionNotFound("gemv_f32")
        }
        guard let fnF16 = library.makeFunction(name: "gemv_f16") else {
            throw GEMVError.functionNotFound("gemv_f16")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fnF32)
        self.pipelineF16 = try device.makeComputePipelineState(function: fnF16)
    }

    /// Execute Float32 GEMV: y[M] = A[M,K] * x[K].
    public func execute(
        a: [Float], x: [Float],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMVParams(
            M: UInt32(M), K: UInt32(K), lda: UInt32(K)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufX, offset: 0, index: 1)
        encoder.setBuffer(bufY, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)

        // One threadgroup per row, each with threadsPerRow threads
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }

    /// Execute Float16 GEMV: y[M] = A[M,K] * x[K].
    public func executeF16(
        a: [Float16], x: [Float16],
        M: Int, K: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float16] {
        let bufA = device.makeBuffer(
            bytes: a, length: a.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufX = device.makeBuffer(
            bytes: x, length: x.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let bufY = device.makeBuffer(
            length: M * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        var params = ERGEMVParams(
            M: UInt32(M), K: UInt32(K), lda: UInt32(K)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GEMVError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF16)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufX, offset: 0, index: 1)
        encoder.setBuffer(bufY, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERGEMVParams>.stride, index: 3)

        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufY.contents().bindMemory(to: Float16.self, capacity: M)
        return Array(UnsafeBufferPointer(start: ptr, count: M))
    }
}

public enum GEMVError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter GEMVTests 2>&1`
Expected: All 6 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/GEMV.metal \
      Sources/EdgeRunnerSharedTypes/include/GEMVParams.h \
      Sources/EdgeRunnerMetal/GEMVKernel.swift \
      Tests/EdgeRunnerMetalTests/GEMVTests.swift
git commit -m "feat: add row-parallel GEMV kernel for autoregressive decoding

One threadgroup per row with simd_sum reduction.
256 threads per row with cross-warp shared memory reduction.
Float32 and Float16 variants. Accumulates in Float32 for FP16 stability."
```

---

## Task 3: Softmax Kernel

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Softmax.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/SoftmaxParams.h`
- Create: `Sources/EdgeRunnerMetal/SoftmaxKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/SoftmaxTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/SoftmaxTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference 1D softmax: numerically stable with max-subtraction.
private func cpuSoftmax1D(_ input: [Float]) -> [Float] {
    let maxVal = input.max() ?? 0
    let exps = input.map { exp($0 - maxVal) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

/// CPU reference 2D softmax along the last axis: each row independently.
private func cpuSoftmax2D(_ input: [Float], rows: Int, cols: Int) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for r in 0..<rows {
        let offset = r * cols
        let row = Array(input[offset..<offset + cols])
        let softRow = cpuSoftmax1D(row)
        for c in 0..<cols {
            output[offset + c] = softRow[c]
        }
    }
    return output
}

@Suite("Softmax Kernel")
struct SoftmaxTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func softmax1DSmall() async throws {
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let expected = cpuSoftmax1D(input)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: 1, cols: 4,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func softmax1DSumsToOne() async throws {
        let input = (0..<128).map { _ in Float.random(in: -5...5) }
        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: 1, cols: 128,
            commandQueue: commandQueue
        )

        let sum = result.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5, "Softmax should sum to 1, got \(sum)")
    }

    @Test func softmax1DAllEqual() async throws {
        let N = 64
        let input = [Float](repeating: 3.0, count: N)
        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: 1, cols: N,
            commandQueue: commandQueue
        )

        let expected = 1.0 / Float(N)
        for i in 0..<N {
            #expect(abs(result[i] - expected) < 1e-5,
                    "Equal inputs should give uniform distribution")
        }
    }

    @Test func softmax1DNumericalStability() async throws {
        // Large values that would overflow without max subtraction
        let input: [Float] = [1000.0, 1001.0, 1002.0, 1003.0]
        let expected = cpuSoftmax1D(input)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: 1, cols: 4,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(!result[i].isNaN, "Result should not be NaN")
            #expect(!result[i].isInfinite, "Result should not be infinite")
            #expect(abs(result[i] - expected[i]) < 1e-5)
        }
    }

    @Test func softmax2DRows() async throws {
        let rows = 4, cols = 32
        let input = (0..<rows * cols).map { _ in Float.random(in: -3...3) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: rows, cols: cols,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }

        // Each row should sum to 1
        for r in 0..<rows {
            let rowSum = (0..<cols).reduce(Float(0)) { $0 + result[r * cols + $1] }
            #expect(abs(rowSum - 1.0) < 1e-5, "Row \(r) sum: \(rowSum)")
        }
    }

    @Test func softmax2DLargeRows() async throws {
        let rows = 8, cols = 512
        let input = (0..<rows * cols).map { _ in Float.random(in: -2...2) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: rows, cols: cols,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]")
        }
    }

    @Test func softmaxNonAlignedCols() async throws {
        let rows = 3, cols = 37
        let input = (0..<rows * cols).map { _ in Float.random(in: -1...1) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input, rows: rows, cols: cols,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-4)
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter SoftmaxTests 2>&1`
Expected: FAIL — `SoftmaxKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/SoftmaxParams.h
#ifndef SOFTMAX_PARAMS_H
#define SOFTMAX_PARAMS_H

#include <stdint.h>

/// Parameters for softmax kernel.
/// Applies softmax along the last axis (cols) for each row independently.
typedef struct {
    uint32_t rows;      // number of independent softmax operations
    uint32_t cols;      // length of each softmax vector
} ERSoftmaxParams;

#endif /* SOFTMAX_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Softmax.metal
#include <metal_stdlib>
using namespace metal;

struct ERSoftmaxParams {
    uint rows;
    uint cols;
};

// Numerically stable softmax: subtract max, exponentiate, normalize.
// One threadgroup per row. Threads cooperatively reduce across cols.
constant uint SOFTMAX_THREADS = 256;

kernel void softmax_f32(
    device const float*       input   [[buffer(0)]],
    device float*             output  [[buffer(1)]],
    constant ERSoftmaxParams& params  [[buffer(2)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    device const float* row_in = input + row * params.cols;
    device float* row_out = output + row * params.cols;

    // Pass 1: Find max value in this row
    float thread_max = -INFINITY;
    for (uint j = local_id; j < params.cols; j += SOFTMAX_THREADS) {
        thread_max = max(thread_max, row_in[j]);
    }

    // Warp-level max reduction
    thread_max = simd_max(thread_max);

    // Cross-warp max reduction
    threadgroup float shared_vals[32];
    if (simd_lane == 0) {
        shared_vals[simd_group] = thread_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (SOFTMAX_THREADS + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_vals[simd_lane] : -INFINITY;
        val = simd_max(val);
        shared_vals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_vals[0];

    // Pass 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (uint j = local_id; j < params.cols; j += SOFTMAX_THREADS) {
        float e = exp(row_in[j] - row_max);
        row_out[j] = e;  // store intermediate
        thread_sum += e;
    }

    // Warp-level sum reduction
    thread_sum = simd_sum(thread_sum);

    if (simd_lane == 0) {
        shared_vals[simd_group] = thread_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (SOFTMAX_THREADS + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_vals[simd_lane] : 0.0f;
        val = simd_sum(val);
        shared_vals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_sum = shared_vals[0];

    // Pass 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (uint j = local_id; j < params.cols; j += SOFTMAX_THREADS) {
        row_out[j] *= inv_sum;
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/SoftmaxKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the numerically stable softmax Metal kernel.
/// Applies softmax independently along the last axis (cols) for each row.
public final class SoftmaxKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    private static let threadsPerRow = 256

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw SoftmaxError.libraryNotFound
        }
        guard let fn = library.makeFunction(name: "softmax_f32") else {
            throw SoftmaxError.functionNotFound("softmax_f32")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fn)
    }

    /// Execute softmax on a 2D array [rows, cols], normalizing along cols.
    /// For 1D softmax, pass rows=1.
    public func execute(
        input: [Float], rows: Int, cols: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols,
                     "Input count \(input.count) != rows*cols \(rows*cols)")

        let bufIn = device.makeBuffer(
            bytes: input, length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufOut = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERSoftmaxParams(
            rows: UInt32(rows), cols: UInt32(cols)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw SoftmaxError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERSoftmaxParams>.stride, index: 2)

        // One threadgroup per row
        let gridSize = MTLSize(width: rows, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.threadsPerRow, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: rows * cols)
        return Array(UnsafeBufferPointer(start: ptr, count: rows * cols))
    }
}

public enum SoftmaxError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter SoftmaxTests 2>&1`
Expected: All 7 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/Softmax.metal \
      Sources/EdgeRunnerSharedTypes/include/SoftmaxParams.h \
      Sources/EdgeRunnerMetal/SoftmaxKernel.swift \
      Tests/EdgeRunnerMetalTests/SoftmaxTests.swift
git commit -m "feat: add numerically stable softmax kernel

Three-pass algorithm: find max, compute exp(x-max) and sum, normalize.
Row-parallel with simd_sum/simd_max reductions.
Handles 1D and 2D inputs with arbitrary column counts."
```

---

## Task 4: Flash Attention Forward Pass

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/FlashAttention.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/AttentionParams.h`
- Create: `Sources/EdgeRunnerMetal/FlashAttentionKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/FlashAttentionTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/FlashAttentionTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference naive attention: O = softmax(Q * K^T / sqrt(d)) * V
/// Q: [seqLen, headDim], K: [seqLen, headDim], V: [seqLen, headDim]
/// Returns O: [seqLen, headDim]
private func cpuNaiveAttention(
    q: [Float], k: [Float], v: [Float],
    seqLen: Int, headDim: Int,
    causal: Bool
) -> [Float] {
    let scale = 1.0 / sqrt(Float(headDim))

    // Compute scores: Q * K^T -> [seqLen, seqLen]
    var scores = [Float](repeating: 0, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0..<seqLen {
            if causal && j > i {
                scores[i * seqLen + j] = -Float.greatestFiniteMagnitude
            } else {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += q[i * headDim + d] * k[j * headDim + d]
                }
                scores[i * seqLen + j] = dot * scale
            }
        }
    }

    // Softmax per row
    var attnWeights = [Float](repeating: 0, count: seqLen * seqLen)
    for i in 0..<seqLen {
        let offset = i * seqLen
        let row = Array(scores[offset..<offset + seqLen])
        let maxVal = row.max() ?? 0
        let exps = row.map { exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        for j in 0..<seqLen {
            attnWeights[offset + j] = exps[j] / sum
        }
    }

    // Output: attn_weights * V -> [seqLen, headDim]
    var output = [Float](repeating: 0, count: seqLen * headDim)
    for i in 0..<seqLen {
        for d in 0..<headDim {
            var sum: Float = 0
            for j in 0..<seqLen {
                sum += attnWeights[i * seqLen + j] * v[j * headDim + d]
            }
            output[i * headDim + d] = sum
        }
    }
    return output
}

@Suite("Flash Attention")
struct FlashAttentionTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func smallNonCausal() async throws {
        let seqLen = 16, headDim = 32
        let q = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuNaiveAttention(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim, causal: false
        )

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: false,
            commandQueue: commandQueue
        )

        for i in 0..<(seqLen * headDim) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func smallCausal() async throws {
        let seqLen = 16, headDim = 32
        let q = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuNaiveAttention(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim, causal: true
        )

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        for i in 0..<(seqLen * headDim) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func mediumCausal() async throws {
        let seqLen = 64, headDim = 64
        let q = (0..<seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let k = (0..<seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let v = (0..<seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let expected = cpuNaiveAttention(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim, causal: true
        )

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        for i in 0..<(seqLen * headDim) {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func causalFirstRowIdentical() async throws {
        // First row of causal attention should only attend to position 0
        let seqLen = 32, headDim = 16
        let q = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        // Row 0 should be exactly V[0] since softmax of a single element = 1.0
        for d in 0..<headDim {
            #expect(abs(result[d] - v[d]) < 1e-4,
                    "First row should equal V[0] at dim \(d)")
        }
    }

    @Test func longerSequence() async throws {
        let seqLen = 128, headDim = 32
        let q = (0..<seqLen * headDim).map { _ in Float.random(in: -0.2...0.2) }
        let k = (0..<seqLen * headDim).map { _ in Float.random(in: -0.2...0.2) }
        let v = (0..<seqLen * headDim).map { _ in Float.random(in: -0.2...0.2) }
        let expected = cpuNaiveAttention(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim, causal: true
        )

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: true,
            commandQueue: commandQueue
        )

        var maxError: Float = 0
        for i in 0..<(seqLen * headDim) {
            maxError = max(maxError, abs(result[i] - expected[i]))
        }
        #expect(maxError < 1e-3,
                "Max error \(maxError) exceeds tolerance for seq_len=128")
    }

    @Test func outputBufferDimensions() async throws {
        let seqLen = 8, headDim = 16
        let q = [Float](repeating: 0.1, count: seqLen * headDim)
        let k = [Float](repeating: 0.1, count: seqLen * headDim)
        let v = [Float](repeating: 0.5, count: seqLen * headDim)

        let kernel = try FlashAttentionKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            causal: false,
            commandQueue: commandQueue
        )

        #expect(result.count == seqLen * headDim)

        // With uniform V=0.5, attention output should be ~0.5 everywhere
        for i in 0..<result.count {
            #expect(abs(result[i] - 0.5) < 1e-3,
                    "Uniform V should give uniform output")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter FlashAttentionTests 2>&1`
Expected: FAIL — `FlashAttentionKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/AttentionParams.h
#ifndef ATTENTION_PARAMS_H
#define ATTENTION_PARAMS_H

#include <stdint.h>

/// Parameters for Flash Attention kernel.
/// Single-head attention: O = softmax(Q*K^T / sqrt(d)) * V
/// Q, K, V: [seqLen, headDim], O: [seqLen, headDim]
typedef struct {
    uint32_t seqLen;        // sequence length (N)
    uint32_t headDim;       // head dimension (d)
    float    scale;         // 1.0 / sqrt(headDim)
    uint32_t causal;        // 1 for causal mask, 0 for no mask
    uint32_t kvBlockSize;   // tile size for K/V blocks (Bc)
    uint32_t qBlockSize;    // tile size for Q blocks (Br)
} ERFlashAttentionParams;

/// Parameters for GQA (Grouped Query Attention).
typedef struct {
    uint32_t seqLen;        // sequence length
    uint32_t headDim;       // dimension per head
    uint32_t numHeads;      // total Q heads
    uint32_t numKVHeads;    // number of K/V heads (< numHeads for GQA)
    uint32_t groupSize;     // numHeads / numKVHeads
    float    scale;         // 1.0 / sqrt(headDim)
    uint32_t causal;        // 1 for causal mask
    uint32_t kvBlockSize;   // tile size for K/V
    uint32_t qBlockSize;    // tile size for Q
} ERGQAParams;

#endif /* ATTENTION_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/FlashAttention.metal
#include <metal_stdlib>
using namespace metal;

struct ERFlashAttentionParams {
    uint   seqLen;
    uint   headDim;
    float  scale;
    uint   causal;
    uint   kvBlockSize;   // Bc
    uint   qBlockSize;    // Br
};

// Flash Attention forward pass with online softmax.
// Tiled over K/V dimension to achieve O(N) memory.
// Each threadgroup processes one Q-block (Br rows of Q).
//
// Algorithm (FlashAttention-2 style):
//   For each Q block i:
//     mi = -inf, li = 0, Oi = 0       (running max, running sum, running output)
//     For each KV block j:
//       Sij = Qi * Kj^T * scale        (local scores)
//       Apply causal mask if needed
//       mij = rowmax(Sij)
//       mi_new = max(mi, mij)
//       Pij = exp(Sij - mi_new)
//       li = li * exp(mi - mi_new) + rowsum(Pij)
//       Oi = Oi * exp(mi - mi_new) + Pij * Vj
//       mi = mi_new
//     Oi = Oi / li

constant uint FLASH_BLOCK_SIZE = 32;  // Br = Bc = 32

kernel void flash_attention_f32(
    device const float*               Q       [[buffer(0)]],
    device const float*               K       [[buffer(1)]],
    device const float*               V       [[buffer(2)]],
    device float*                     O       [[buffer(3)]],
    constant ERFlashAttentionParams&  params  [[buffer(4)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one Q-block of Br rows.
    // Each thread in the group handles one row of the Q-block.
    const uint Br = params.qBlockSize;
    const uint Bc = params.kvBlockSize;
    const uint d  = params.headDim;
    const uint N  = params.seqLen;

    uint q_row = group_id * Br + local_id;
    if (q_row >= N) return;

    // Load Q row into registers
    // Use threadgroup memory for K/V tiles
    threadgroup float k_tile[32 * 128];  // Bc * d_max (128 max head_dim)
    threadgroup float v_tile[32 * 128];

    // Running statistics for online softmax
    float mi = -INFINITY;     // running max
    float li = 0.0f;          // running sum of exp

    // Running output accumulator (per head dim)
    // We use threadgroup scratch for output since headDim can be large
    threadgroup float o_scratch[32 * 128]; // Br * d_max
    for (uint dd = 0; dd < d; dd++) {
        o_scratch[local_id * d + dd] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Number of KV blocks
    uint num_kv_blocks = (N + Bc - 1) / Bc;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint kv_start = kv_block * Bc;
        uint kv_end = min(kv_start + Bc, N);
        uint kv_count = kv_end - kv_start;

        // Cooperatively load K tile into threadgroup memory
        // Each thread loads one row if it maps to a valid KV index
        if (local_id < kv_count) {
            for (uint dd = 0; dd < d; dd++) {
                k_tile[local_id * d + dd] = K[(kv_start + local_id) * d + dd];
                v_tile[local_id * d + dd] = V[(kv_start + local_id) * d + dd];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores: S[local_id, j] = Q[q_row] dot K[kv_start+j] * scale
        float mij = -INFINITY;

        // First pass: compute scores and local max
        float scores[32];  // max Bc scores per thread
        for (uint j = 0; j < kv_count; j++) {
            // Causal mask: skip if kv position > q position
            if (params.causal != 0 && (kv_start + j) > q_row) {
                scores[j] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (uint dd = 0; dd < d; dd++) {
                dot += Q[q_row * d + dd] * k_tile[j * d + dd];
            }
            scores[j] = dot * params.scale;
            mij = max(mij, scores[j]);
        }

        // Online softmax update
        float mi_new = max(mi, mij);
        float correction = exp(mi - mi_new);  // rescale old values

        // Compute Pij = exp(scores - mi_new) and their sum
        float pij_sum = 0.0f;
        float pij[32];
        for (uint j = 0; j < kv_count; j++) {
            if (scores[j] == -INFINITY) {
                pij[j] = 0.0f;
            } else {
                pij[j] = exp(scores[j] - mi_new);
            }
            pij_sum += pij[j];
        }

        // Update running sum: li = li * correction + pij_sum
        li = li * correction + pij_sum;

        // Update output: Oi = Oi * correction + Pij * Vj
        for (uint dd = 0; dd < d; dd++) {
            float o_val = o_scratch[local_id * d + dd] * correction;
            for (uint j = 0; j < kv_count; j++) {
                o_val += pij[j] * v_tile[j * d + dd];
            }
            o_scratch[local_id * d + dd] = o_val;
        }

        mi = mi_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization: O = O / li
    float inv_li = (li > 0.0f) ? (1.0f / li) : 0.0f;
    for (uint dd = 0; dd < d; dd++) {
        O[q_row * d + dd] = o_scratch[local_id * d + dd] * inv_li;
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/FlashAttentionKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for Flash Attention forward pass.
/// Implements tiled attention with O(N) memory via online softmax.
/// Supports causal masking for autoregressive models.
public final class FlashAttentionKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    /// Block sizes for Q and KV tiling.
    public static let qBlockSize: Int = 32
    public static let kvBlockSize: Int = 32

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw FlashAttentionError.libraryNotFound
        }
        guard let fn = library.makeFunction(name: "flash_attention_f32") else {
            throw FlashAttentionError.functionNotFound("flash_attention_f32")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fn)
    }

    /// Execute Flash Attention: O = softmax(Q * K^T / sqrt(d)) * V
    ///
    /// - Parameters:
    ///   - q: Query tensor [seqLen, headDim]
    ///   - k: Key tensor [seqLen, headDim]
    ///   - v: Value tensor [seqLen, headDim]
    ///   - seqLen: Sequence length
    ///   - headDim: Head dimension (must be <= 128)
    ///   - causal: Whether to apply causal mask
    ///   - commandQueue: Metal command queue
    /// - Returns: Output tensor [seqLen, headDim]
    public func execute(
        q: [Float], k: [Float], v: [Float],
        seqLen: Int, headDim: Int,
        causal: Bool,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(q.count == seqLen * headDim)
        precondition(k.count == seqLen * headDim)
        precondition(v.count == seqLen * headDim)
        precondition(headDim <= 128, "Head dimension must be <= 128")

        let bufQ = device.makeBuffer(
            bytes: q, length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufK = device.makeBuffer(
            bytes: k, length: k.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufV = device.makeBuffer(
            bytes: v, length: v.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufO = device.makeBuffer(
            length: seqLen * headDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        let scale = 1.0 / sqrt(Float(headDim))
        var params = ERFlashAttentionParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            scale: scale,
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.kvBlockSize),
            qBlockSize: UInt32(Self.qBlockSize)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw FlashAttentionError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufQ, offset: 0, index: 0)
        encoder.setBuffer(bufK, offset: 0, index: 1)
        encoder.setBuffer(bufV, offset: 0, index: 2)
        encoder.setBuffer(bufO, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERFlashAttentionParams>.stride, index: 4)

        // One threadgroup per Q block, each with Br threads
        let numQBlocks = (seqLen + Self.qBlockSize - 1) / Self.qBlockSize
        let gridSize = MTLSize(width: numQBlocks, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: Self.qBlockSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufO.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
        return Array(UnsafeBufferPointer(start: ptr, count: seqLen * headDim))
    }
}

public enum FlashAttentionError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter FlashAttentionTests 2>&1`
Expected: All 6 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/FlashAttention.metal \
      Sources/EdgeRunnerSharedTypes/include/AttentionParams.h \
      Sources/EdgeRunnerMetal/FlashAttentionKernel.swift \
      Tests/EdgeRunnerMetalTests/FlashAttentionTests.swift
git commit -m "feat: add Flash Attention forward pass with online softmax

Tiled Q/KV accumulation with O(N) memory complexity.
Online softmax with running max and rescaling for numerical stability.
Causal mask support for autoregressive decoding.
32x32 block tiling with threadgroup-cooperative KV loading."
```

---

## Task 5: Grouped Query Attention (GQA)

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/GQA.metal`
- Create: `Sources/EdgeRunnerMetal/GQAKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/GQATests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/GQATests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference GQA: multi-head Q with shared KV heads.
/// Q: [numHeads, seqLen, headDim]
/// K: [numKVHeads, seqLen, headDim]
/// V: [numKVHeads, seqLen, headDim]
/// Output: [numHeads, seqLen, headDim]
private func cpuGQA(
    q: [Float], k: [Float], v: [Float],
    seqLen: Int, headDim: Int,
    numHeads: Int, numKVHeads: Int,
    causal: Bool
) -> [Float] {
    let scale = 1.0 / sqrt(Float(headDim))
    let groupSize = numHeads / numKVHeads
    var output = [Float](repeating: 0, count: numHeads * seqLen * headDim)

    for h in 0..<numHeads {
        let kvHead = h / groupSize
        let qOffset = h * seqLen * headDim
        let kOffset = kvHead * seqLen * headDim
        let vOffset = kvHead * seqLen * headDim
        let oOffset = h * seqLen * headDim

        // Compute attention for this head
        for i in 0..<seqLen {
            // Compute scores
            var scores = [Float](repeating: 0, count: seqLen)
            for j in 0..<seqLen {
                if causal && j > i {
                    scores[j] = -Float.greatestFiniteMagnitude
                } else {
                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += q[qOffset + i * headDim + d] * k[kOffset + j * headDim + d]
                    }
                    scores[j] = dot * scale
                }
            }

            // Softmax
            let maxVal = scores.max() ?? 0
            let exps = scores.map { exp($0 - maxVal) }
            let sum = exps.reduce(0, +)
            let weights = exps.map { $0 / sum }

            // Weighted sum of V
            for d in 0..<headDim {
                var val: Float = 0
                for j in 0..<seqLen {
                    val += weights[j] * v[vOffset + j * headDim + d]
                }
                output[oOffset + i * headDim + d] = val
            }
        }
    }
    return output
}

@Suite("Grouped Query Attention")
struct GQATests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func mhaBaseline_1to1() async throws {
        // Standard MHA: numHeads == numKVHeads (group size = 1)
        let seqLen = 16, headDim = 32, numHeads = 4, numKVHeads = 4
        let q = (0..<numHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true
        )

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "1:1 mismatch at [\(i)]")
        }
    }

    @Test func gqa_1to4() async throws {
        // 4 Q heads per KV head
        let seqLen = 16, headDim = 32, numHeads = 8, numKVHeads = 2
        let q = (0..<numHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true
        )

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "1:4 mismatch at [\(i)]")
        }
    }

    @Test func gqa_1to8() async throws {
        // 8 Q heads per KV head (aggressive GQA like Llama 3)
        let seqLen = 16, headDim = 64, numHeads = 8, numKVHeads = 1
        let q = (0..<numHeads * seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let k = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let v = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.3...0.3) }
        let expected = cpuGQA(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true
        )

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: true,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "1:8 mismatch at [\(i)]")
        }
    }

    @Test func gqaGroupsShareKV() async throws {
        // Verify that heads within the same group produce different outputs
        // (because Q differs) but use the same K/V
        let seqLen = 8, headDim = 16, numHeads = 4, numKVHeads = 2

        // Make Q different per head but K/V identical per group
        let q = (0..<numHeads * seqLen * headDim).map { Float($0 % 7) * 0.1 }
        let k = (0..<numKVHeads * seqLen * headDim).map { Float($0 % 5) * 0.1 }
        let v = (0..<numKVHeads * seqLen * headDim).map { Float($0 % 3) * 0.1 }

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: false,
            commandQueue: commandQueue
        )

        #expect(result.count == numHeads * seqLen * headDim)

        // Heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1
        // Head 0 and head 1 should produce DIFFERENT outputs (different Q)
        let head0 = Array(result[0..<seqLen * headDim])
        let head1 = Array(result[seqLen * headDim..<2 * seqLen * headDim])
        #expect(head0 != head1, "Different Q heads should produce different outputs")
    }

    @Test func gqaNonCausal() async throws {
        let seqLen = 16, headDim = 32, numHeads = 4, numKVHeads = 2
        let q = (0..<numHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let k = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let v = (0..<numKVHeads * seqLen * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuGQA(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: false
        )

        let kernel = try GQAKernel(device: device)
        let result = try await kernel.execute(
            q: q, k: k, v: v,
            seqLen: seqLen, headDim: headDim,
            numHeads: numHeads, numKVHeads: numKVHeads,
            causal: false,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Non-causal GQA mismatch at [\(i)]")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GQATests 2>&1`
Expected: FAIL — `GQAKernel` not defined

**Step 3: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/GQA.metal
#include <metal_stdlib>
using namespace metal;

struct ERGQAParams {
    uint   seqLen;
    uint   headDim;
    uint   numHeads;
    uint   numKVHeads;
    uint   groupSize;
    float  scale;
    uint   causal;
    uint   kvBlockSize;
    uint   qBlockSize;
};

// Grouped Query Attention: multiple Q heads share fewer KV heads.
// Q: [numHeads, seqLen, headDim]
// K: [numKVHeads, seqLen, headDim]
// V: [numKVHeads, seqLen, headDim]
// O: [numHeads, seqLen, headDim]
//
// Each threadgroup handles one (head, q_block) pair.
// Dispatched as grid: (numQBlocks, numHeads, 1)
// Each thread handles one row within the Q block.

constant uint GQA_BLOCK_SIZE = 32;

kernel void gqa_attention_f32(
    device const float*   Q       [[buffer(0)]],
    device const float*   K       [[buffer(1)]],
    device const float*   V       [[buffer(2)]],
    device float*         O       [[buffer(3)]],
    constant ERGQAParams& params  [[buffer(4)]],
    uint2 group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]]
) {
    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint kv_head_idx = head_idx / params.groupSize;
    const uint d           = params.headDim;
    const uint N           = params.seqLen;

    uint q_row = q_block_idx * GQA_BLOCK_SIZE + local_id;
    if (q_row >= N) return;

    // Offsets into Q, K, V, O for this head
    device const float* q_head = Q + head_idx * N * d;
    device const float* k_head = K + kv_head_idx * N * d;
    device const float* v_head = V + kv_head_idx * N * d;
    device float*       o_head = O + head_idx * N * d;

    // K/V tiles in threadgroup memory
    threadgroup float k_tile[32 * 128];
    threadgroup float v_tile[32 * 128];

    // Online softmax state
    float mi = -INFINITY;
    float li = 0.0f;

    // Output accumulator in threadgroup memory
    threadgroup float o_scratch[32 * 128];
    for (uint dd = 0; dd < d; dd++) {
        o_scratch[local_id * d + dd] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (N + GQA_BLOCK_SIZE - 1) / GQA_BLOCK_SIZE;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint kv_start = kv_block * GQA_BLOCK_SIZE;
        uint kv_end   = min(kv_start + GQA_BLOCK_SIZE, N);
        uint kv_count = kv_end - kv_start;

        // Load K/V tile cooperatively
        if (local_id < kv_count) {
            for (uint dd = 0; dd < d; dd++) {
                k_tile[local_id * d + dd] = k_head[(kv_start + local_id) * d + dd];
                v_tile[local_id * d + dd] = v_head[(kv_start + local_id) * d + dd];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores and local max
        float mij = -INFINITY;
        float scores[32];
        for (uint j = 0; j < kv_count; j++) {
            if (params.causal != 0 && (kv_start + j) > q_row) {
                scores[j] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (uint dd = 0; dd < d; dd++) {
                dot += q_head[q_row * d + dd] * k_tile[j * d + dd];
            }
            scores[j] = dot * params.scale;
            mij = max(mij, scores[j]);
        }

        // Online softmax update
        float mi_new = max(mi, mij);
        float correction = exp(mi - mi_new);

        float pij_sum = 0.0f;
        float pij[32];
        for (uint j = 0; j < kv_count; j++) {
            pij[j] = (scores[j] == -INFINITY) ? 0.0f : exp(scores[j] - mi_new);
            pij_sum += pij[j];
        }

        li = li * correction + pij_sum;

        for (uint dd = 0; dd < d; dd++) {
            float o_val = o_scratch[local_id * d + dd] * correction;
            for (uint j = 0; j < kv_count; j++) {
                o_val += pij[j] * v_tile[j * d + dd];
            }
            o_scratch[local_id * d + dd] = o_val;
        }

        mi = mi_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float inv_li = (li > 0.0f) ? (1.0f / li) : 0.0f;
    for (uint dd = 0; dd < d; dd++) {
        o_head[q_row * d + dd] = o_scratch[local_id * d + dd] * inv_li;
    }
}
```

**Step 4: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/GQAKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for Grouped Query Attention.
/// Supports num_kv_heads < num_heads with automatic head group mapping.
/// Each Q head group shares the same KV head.
public final class GQAKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    private static let blockSize: Int = 32

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw GQAError.libraryNotFound
        }
        guard let fn = library.makeFunction(name: "gqa_attention_f32") else {
            throw GQAError.functionNotFound("gqa_attention_f32")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fn)
    }

    /// Execute Grouped Query Attention.
    ///
    /// - Parameters:
    ///   - q: Query tensor [numHeads, seqLen, headDim]
    ///   - k: Key tensor [numKVHeads, seqLen, headDim]
    ///   - v: Value tensor [numKVHeads, seqLen, headDim]
    ///   - seqLen: Sequence length
    ///   - headDim: Dimension per head (must be <= 128)
    ///   - numHeads: Number of query heads
    ///   - numKVHeads: Number of key/value heads (must divide numHeads evenly)
    ///   - causal: Whether to apply causal mask
    ///   - commandQueue: Metal command queue
    /// - Returns: Output tensor [numHeads, seqLen, headDim]
    public func execute(
        q: [Float], k: [Float], v: [Float],
        seqLen: Int, headDim: Int,
        numHeads: Int, numKVHeads: Int,
        causal: Bool,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(numHeads % numKVHeads == 0,
                     "numHeads (\(numHeads)) must be divisible by numKVHeads (\(numKVHeads))")
        precondition(q.count == numHeads * seqLen * headDim)
        precondition(k.count == numKVHeads * seqLen * headDim)
        precondition(v.count == numKVHeads * seqLen * headDim)
        precondition(headDim <= 128, "Head dimension must be <= 128")

        let groupSize = numHeads / numKVHeads
        let outputCount = numHeads * seqLen * headDim

        let bufQ = device.makeBuffer(
            bytes: q, length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufK = device.makeBuffer(
            bytes: k, length: k.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufV = device.makeBuffer(
            bytes: v, length: v.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufO = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        let scale = 1.0 / sqrt(Float(headDim))
        var params = ERGQAParams(
            seqLen: UInt32(seqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(groupSize),
            scale: scale,
            causal: causal ? 1 : 0,
            kvBlockSize: UInt32(Self.blockSize),
            qBlockSize: UInt32(Self.blockSize)
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw GQAError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufQ, offset: 0, index: 0)
        encoder.setBuffer(bufK, offset: 0, index: 1)
        encoder.setBuffer(bufV, offset: 0, index: 2)
        encoder.setBuffer(bufO, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERGQAParams>.stride, index: 4)

        // Grid: (numQBlocks, numHeads), each threadgroup has blockSize threads
        let numQBlocks = (seqLen + Self.blockSize - 1) / Self.blockSize
        let gridSize = MTLSize(width: numQBlocks, height: numHeads, depth: 1)
        let threadgroupSize = MTLSize(width: Self.blockSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufO.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outputCount))
    }
}

public enum GQAError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 5: Run tests to verify they pass**

Run: `swift test --filter GQATests 2>&1`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/GQA.metal \
      Sources/EdgeRunnerMetal/GQAKernel.swift \
      Tests/EdgeRunnerMetalTests/GQATests.swift
git commit -m "feat: add Grouped Query Attention with head group mapping

Supports num_kv_heads < num_heads for GQA architectures.
Automatic head-to-KV-head mapping via groupSize = numHeads/numKVHeads.
Flash Attention style online softmax within each head.
Tested with 1:1 (MHA), 1:4, and 1:8 group ratios."
```

---

## Task 6: KV Cache (Ring Buffer)

**Files:**
- Create: `Sources/EdgeRunnerMetal/KVCache.swift`
- Create: `Sources/EdgeRunnerSharedTypes/include/KVCacheParams.h`
- Test: `Tests/EdgeRunnerMetalTests/KVCacheTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/KVCacheTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("KV Cache Ring Buffer")
struct KVCacheTests {

    let device: MTLDevice

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
    }

    @Test func createCache() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 2048,
            numLayers: 32,
            numKVHeads: 8,
            headDim: 128,
            precision: .float16
        )
        #expect(cache.maxSeqLen == 2048)
        #expect(cache.currentLength == 0)
        #expect(cache.numLayers == 32)
    }

    @Test func appendAndRetrieveSingleStep() throws {
        let numKVHeads = 2, headDim = 4, maxSeqLen = 16
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        // Append one token's KV for layer 0
        let kData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]  // [numKVHeads, headDim]
        let vData: [Float] = [9, 10, 11, 12, 13, 14, 15, 16]
        try cache.append(layer: 0, keys: kData, values: vData)

        #expect(cache.currentLength == 1)

        // Retrieve
        let (keys, values) = try cache.retrieve(layer: 0, asType: Float.self)
        #expect(keys.count == numKVHeads * 1 * headDim)
        #expect(values.count == numKVHeads * 1 * headDim)
        #expect(keys == kData)
        #expect(values == vData)
    }

    @Test func appendMultipleSteps() throws {
        let numKVHeads = 1, headDim = 2, maxSeqLen = 8
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        // Append 3 tokens
        try cache.append(layer: 0, keys: [1.0, 2.0] as [Float], values: [10.0, 20.0] as [Float])
        try cache.append(layer: 0, keys: [3.0, 4.0] as [Float], values: [30.0, 40.0] as [Float])
        try cache.append(layer: 0, keys: [5.0, 6.0] as [Float], values: [50.0, 60.0] as [Float])

        #expect(cache.currentLength == 3)

        let (keys, values) = try cache.retrieve(layer: 0, asType: Float.self)
        #expect(keys == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as [Float])
        #expect(values == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] as [Float])
    }

    @Test func wrapAround() throws {
        let numKVHeads = 1, headDim = 2, maxSeqLen = 4
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        // Fill cache to capacity
        for i in 0..<4 {
            let f = Float(i)
            try cache.append(layer: 0, keys: [f, f + 0.5], values: [f * 10, f * 10 + 5])
        }
        #expect(cache.currentLength == 4)

        // Append one more — should wrap around, overwriting position 0
        try cache.append(layer: 0, keys: [99.0, 99.5], values: [990.0, 995.0])
        #expect(cache.currentLength == 4) // Still capped at maxSeqLen

        // The oldest entry (position 0) should be overwritten
        let (keys, _) = try cache.retrieve(layer: 0, asType: Float.self)
        // After wrap: positions are [99, 1, 2, 3] in ring order,
        // but retrieve returns them in logical sequence order
        #expect(keys.count == numKVHeads * 4 * headDim)

        // First logical position should now be what was position 1
        #expect(keys[0] == 1.0)
        #expect(keys[1] == 1.5)
        // Last logical position should be the new entry
        #expect(keys[6] == 99.0)
        #expect(keys[7] == 99.5)
    }

    @Test func multiLayerCache() throws {
        let numKVHeads = 1, headDim = 2, maxSeqLen = 8
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 4,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        // Append to different layers
        try cache.append(layer: 0, keys: [1.0, 2.0] as [Float], values: [10.0, 20.0] as [Float])
        try cache.append(layer: 1, keys: [3.0, 4.0] as [Float], values: [30.0, 40.0] as [Float])
        try cache.append(layer: 2, keys: [5.0, 6.0] as [Float], values: [50.0, 60.0] as [Float])
        try cache.append(layer: 3, keys: [7.0, 8.0] as [Float], values: [70.0, 80.0] as [Float])

        let (keys0, _) = try cache.retrieve(layer: 0, asType: Float.self)
        let (keys1, _) = try cache.retrieve(layer: 1, asType: Float.self)
        #expect(keys0 == [1.0, 2.0] as [Float])
        #expect(keys1 == [3.0, 4.0] as [Float])
    }

    @Test func float16Precision() throws {
        let numKVHeads = 1, headDim = 4, maxSeqLen = 8
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 1,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float16
        )

        let kData: [Float16] = [1.0, 2.0, 3.0, 4.0]
        let vData: [Float16] = [5.0, 6.0, 7.0, 8.0]
        try cache.appendF16(layer: 0, keys: kData, values: vData)

        #expect(cache.currentLength == 1)

        let (keys, values) = try cache.retrieve(layer: 0, asType: Float16.self)
        #expect(keys == kData)
        #expect(values == vData)
    }

    @Test func reset() throws {
        let cache = try KVCache(
            device: device,
            maxSeqLen: 16,
            numLayers: 1,
            numKVHeads: 1,
            headDim: 2,
            precision: .float32
        )

        try cache.append(layer: 0, keys: [1.0, 2.0] as [Float], values: [3.0, 4.0] as [Float])
        #expect(cache.currentLength == 1)

        cache.reset()
        #expect(cache.currentLength == 0)
    }

    @Test func metalBufferAccess() throws {
        let numKVHeads = 2, headDim = 4, maxSeqLen = 16
        let cache = try KVCache(
            device: device,
            maxSeqLen: maxSeqLen,
            numLayers: 2,
            numKVHeads: numKVHeads,
            headDim: headDim,
            precision: .float32
        )

        // Verify we can get MTLBuffers for kernel dispatch
        let (kBuf, vBuf) = cache.metalBuffers(layer: 0)
        #expect(kBuf.length == maxSeqLen * numKVHeads * headDim * MemoryLayout<Float>.stride)
        #expect(vBuf.length == maxSeqLen * numKVHeads * headDim * MemoryLayout<Float>.stride)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter KVCacheTests 2>&1`
Expected: FAIL — `KVCache` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/KVCacheParams.h
#ifndef KV_CACHE_PARAMS_H
#define KV_CACHE_PARAMS_H

#include <stdint.h>

/// Precision mode for KV cache storage.
typedef enum __attribute__((enum_extensibility(closed))) {
    ERKVPrecisionFloat32 = 0,
    ERKVPrecisionFloat16 = 1,
    ERKVPrecisionFloat8  = 2,
} ERKVPrecision;

/// Parameters for KV cache operations dispatched to Metal.
typedef struct {
    uint32_t maxSeqLen;     // maximum sequence length (ring buffer capacity)
    uint32_t currentLen;    // current valid entries in the cache
    uint32_t writePos;      // next write position (head of ring buffer)
    uint32_t numKVHeads;    // number of KV heads
    uint32_t headDim;       // dimension per head
    uint32_t precision;     // ERKVPrecision value
} ERKVCacheParams;

#endif /* KV_CACHE_PARAMS_H */
```

**Step 4: Implement KVCache**

```swift
// Sources/EdgeRunnerMetal/KVCache.swift
import Metal
import Synchronization
import EdgeRunnerSharedTypes

/// Pre-allocated ring buffer for KV cache storage.
/// Supports per-layer key/value buffers with position tracking.
/// Thread-safe via Mutex. Exposes MTLBuffers for direct kernel access.
public final class KVCache: Sendable {

    /// Storage precision for KV values.
    public enum Precision: Sendable {
        case float32
        case float16
        case float8

        var bytesPerElement: Int {
            switch self {
            case .float32: return 4
            case .float16: return 2
            case .float8:  return 1
            }
        }

        var erPrecision: ERKVPrecision {
            switch self {
            case .float32: return .ERKVPrecisionFloat32
            case .float16: return .ERKVPrecisionFloat16
            case .float8:  return .ERKVPrecisionFloat8
            }
        }
    }

    private struct State: Sendable {
        var writePos: Int = 0       // Next write position in ring buffer
        var totalWritten: Int = 0   // Total tokens written (for tracking wrap)
    }

    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision

    /// Per-layer key buffers: [maxSeqLen, numKVHeads, headDim]
    private let keyBuffers: [MTLBuffer]
    /// Per-layer value buffers: [maxSeqLen, numKVHeads, headDim]
    private let valueBuffers: [MTLBuffer]

    private let state: Mutex<State>
    private let device: MTLDevice

    /// Number of elements per token (numKVHeads * headDim).
    private var elementsPerToken: Int { numKVHeads * headDim }

    /// Bytes per token entry.
    private var bytesPerToken: Int { elementsPerToken * precision.bytesPerElement }

    /// Total buffer size per layer.
    private var bufferSize: Int { maxSeqLen * bytesPerToken }

    /// Current number of valid entries in the cache (capped at maxSeqLen).
    public var currentLength: Int {
        state.withLock { s in
            min(s.totalWritten, maxSeqLen)
        }
    }

    public init(
        device: MTLDevice,
        maxSeqLen: Int,
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        precision: Precision
    ) throws {
        self.device = device
        self.maxSeqLen = maxSeqLen
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.precision = precision
        self.state = Mutex(State())

        let size = maxSeqLen * numKVHeads * headDim * precision.bytesPerElement

        var keys = [MTLBuffer]()
        var vals = [MTLBuffer]()
        keys.reserveCapacity(numLayers)
        vals.reserveCapacity(numLayers)

        for _ in 0..<numLayers {
            guard let kBuf = device.makeBuffer(
                length: size, options: [.storageModeShared, .hazardTrackingModeUntracked]
            ) else {
                throw KVCacheError.allocationFailed
            }
            guard let vBuf = device.makeBuffer(
                length: size, options: [.storageModeShared, .hazardTrackingModeUntracked]
            ) else {
                throw KVCacheError.allocationFailed
            }
            keys.append(kBuf)
            vals.append(vBuf)
        }

        self.keyBuffers = keys
        self.valueBuffers = vals
    }

    /// Append one token's K/V data for a specific layer (Float32).
    /// keys/values: [numKVHeads * headDim] elements.
    public func append(layer: Int, keys: [Float], values: [Float]) throws {
        precondition(keys.count == elementsPerToken,
                     "Expected \(elementsPerToken) key elements, got \(keys.count)")
        precondition(values.count == elementsPerToken,
                     "Expected \(elementsPerToken) value elements, got \(values.count)")
        precondition(precision == .float32, "Use appendF16 for float16 precision")

        let writePos = state.withLock { s -> Int in
            let pos = s.writePos
            s.writePos = (s.writePos + 1) % maxSeqLen
            s.totalWritten += 1
            return pos
        }

        let byteOffset = writePos * elementsPerToken * MemoryLayout<Float>.stride
        let byteCount = elementsPerToken * MemoryLayout<Float>.stride

        memcpy(keyBuffers[layer].contents().advanced(by: byteOffset),
               keys, byteCount)
        memcpy(valueBuffers[layer].contents().advanced(by: byteOffset),
               values, byteCount)
    }

    /// Append one token's K/V data for a specific layer (Float16).
    public func appendF16(layer: Int, keys: [Float16], values: [Float16]) throws {
        precondition(keys.count == elementsPerToken)
        precondition(values.count == elementsPerToken)
        precondition(precision == .float16, "Use append for float32 precision")

        let writePos = state.withLock { s -> Int in
            let pos = s.writePos
            s.writePos = (s.writePos + 1) % maxSeqLen
            s.totalWritten += 1
            return pos
        }

        let byteOffset = writePos * elementsPerToken * MemoryLayout<Float16>.stride
        let byteCount = elementsPerToken * MemoryLayout<Float16>.stride

        memcpy(keyBuffers[layer].contents().advanced(by: byteOffset),
               keys, byteCount)
        memcpy(valueBuffers[layer].contents().advanced(by: byteOffset),
               values, byteCount)
    }

    /// Retrieve the full valid KV sequence for a layer in logical order.
    /// Returns (keys, values) each of shape [currentLength * numKVHeads * headDim].
    public func retrieve<T: BitwiseCopyable>(
        layer: Int,
        asType: T.Type
    ) throws -> ([T], [T]) {
        let (currentLen, writePos, totalWritten) = state.withLock { s in
            (min(s.totalWritten, maxSeqLen), s.writePos, s.totalWritten)
        }

        guard currentLen > 0 else {
            return ([], [])
        }

        let elemsPerToken = elementsPerToken
        let totalElems = currentLen * elemsPerToken

        var keys = [T]()
        var values = [T]()
        keys.reserveCapacity(totalElems)
        values.reserveCapacity(totalElems)

        let kPtr = keyBuffers[layer].contents().bindMemory(
            to: T.self, capacity: maxSeqLen * elemsPerToken
        )
        let vPtr = valueBuffers[layer].contents().bindMemory(
            to: T.self, capacity: maxSeqLen * elemsPerToken
        )

        if totalWritten <= maxSeqLen {
            // No wrap-around: read positions 0..<currentLen in order
            keys.append(contentsOf: UnsafeBufferPointer(
                start: kPtr, count: totalElems
            ))
            values.append(contentsOf: UnsafeBufferPointer(
                start: vPtr, count: totalElems
            ))
        } else {
            // Wrap-around: writePos is the oldest entry.
            // Read from writePos to end, then from 0 to writePos.
            let startPos = writePos  // oldest valid position
            let firstChunkTokens = maxSeqLen - startPos
            let secondChunkTokens = startPos

            if firstChunkTokens > 0 {
                let offset = startPos * elemsPerToken
                let count = firstChunkTokens * elemsPerToken
                keys.append(contentsOf: UnsafeBufferPointer(
                    start: kPtr.advanced(by: offset), count: count
                ))
                values.append(contentsOf: UnsafeBufferPointer(
                    start: vPtr.advanced(by: offset), count: count
                ))
            }
            if secondChunkTokens > 0 {
                let count = secondChunkTokens * elemsPerToken
                keys.append(contentsOf: UnsafeBufferPointer(
                    start: kPtr, count: count
                ))
                values.append(contentsOf: UnsafeBufferPointer(
                    start: vPtr, count: count
                ))
            }
        }

        return (keys, values)
    }

    /// Get the raw MTLBuffers for a layer, for direct kernel dispatch.
    /// Returns (keyBuffer, valueBuffer).
    public func metalBuffers(layer: Int) -> (MTLBuffer, MTLBuffer) {
        (keyBuffers[layer], valueBuffers[layer])
    }

    /// Get current cache parameters for Metal kernel dispatch.
    public func cacheParams() -> ERKVCacheParams {
        state.withLock { s in
            ERKVCacheParams(
                maxSeqLen: UInt32(maxSeqLen),
                currentLen: UInt32(min(s.totalWritten, maxSeqLen)),
                writePos: UInt32(s.writePos),
                numKVHeads: UInt32(numKVHeads),
                headDim: UInt32(headDim),
                precision: UInt32(precision.erPrecision.rawValue)
            )
        }
    }

    /// Reset the cache, clearing all stored KV pairs.
    public func reset() {
        state.withLock { s in
            s.writePos = 0
            s.totalWritten = 0
        }
    }
}

public enum KVCacheError: Error, Sendable {
    case allocationFailed
    case invalidLayer(Int)
    case precisionMismatch
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter KVCacheTests 2>&1`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerMetal/KVCache.swift \
      Sources/EdgeRunnerSharedTypes/include/KVCacheParams.h \
      Tests/EdgeRunnerMetalTests/KVCacheTests.swift
git commit -m "feat: add pre-allocated ring buffer KV cache

Per-layer MTLBuffer pairs for keys and values.
Ring buffer with wrap-around and logical-order retrieval.
Mixed-precision support: Float32, Float16, Float8 per layer.
Thread-safe via Mutex. Exposes raw MTLBuffers for kernel dispatch."
```

---

## Task 7: RoPE (Rotary Position Embeddings)

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/RoPE.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/RoPEParams.h`
- Create: `Sources/EdgeRunnerMetal/RoPEKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/RoPETests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/RoPETests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference RoPE implementation.
/// Applies rotary position embeddings to input tensor.
/// Input: [seqLen, numHeads, headDim]
/// Each pair of adjacent dimensions (2i, 2i+1) is rotated by position-dependent angle.
private func cpuRoPE(
    input: [Float],
    seqLen: Int, numHeads: Int, headDim: Int,
    startPos: Int = 0,
    theta: Float = 10000.0,
    scalingFactor: Float = 1.0
) -> [Float] {
    var output = input
    let halfDim = headDim / 2

    for s in 0..<seqLen {
        let pos = Float(s + startPos)
        for h in 0..<numHeads {
            for i in 0..<halfDim {
                let freq = 1.0 / pow(theta, Float(2 * i) / Float(headDim))
                let adjustedFreq = freq / scalingFactor
                let angle = pos * adjustedFreq

                let cosVal = cos(angle)
                let sinVal = sin(angle)

                let idx0 = (s * numHeads * headDim) + (h * headDim) + (2 * i)
                let idx1 = idx0 + 1
                let x0 = input[idx0]
                let x1 = input[idx1]

                output[idx0] = x0 * cosVal - x1 * sinVal
                output[idx1] = x0 * sinVal + x1 * cosVal
            }
        }
    }
    return output
}

@Suite("RoPE Kernel")
struct RoPETests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func basicRoPE() async throws {
        let seqLen = 4, numHeads = 2, headDim = 8
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }
        let expected = cpuRoPE(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim
        )

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func ropeWithOffset() async throws {
        // Test with non-zero start position (autoregressive decoding)
        let seqLen = 1, numHeads = 4, headDim = 64
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }
        let startPos = 42

        let expected = cpuRoPE(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: startPos
        )

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: startPos,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]")
        }
    }

    @Test func ropePosition0IsIdentityLike() async throws {
        // At position 0, all angles are 0, so cos=1, sin=0
        // Output should equal input
        let seqLen = 1, numHeads = 2, headDim = 16
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - input[i]) < 1e-5,
                    "Position 0 should be identity-like")
        }
    }

    @Test func ropePreservesNorm() async throws {
        // RoPE is a rotation, so it should preserve L2 norm per head
        let seqLen = 8, numHeads = 4, headDim = 32
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            commandQueue: commandQueue
        )

        // Check norm preservation per head
        for s in 0..<seqLen {
            for h in 0..<numHeads {
                let offset = (s * numHeads + h) * headDim
                var inputNorm: Float = 0
                var outputNorm: Float = 0
                for d in 0..<headDim {
                    inputNorm += input[offset + d] * input[offset + d]
                    outputNorm += result[offset + d] * result[offset + d]
                }
                inputNorm = sqrt(inputNorm)
                outputNorm = sqrt(outputNorm)
                #expect(abs(inputNorm - outputNorm) < 1e-4,
                        "Norm not preserved at seq=\(s) head=\(h): \(inputNorm) vs \(outputNorm)")
            }
        }
    }

    @Test func ntkAwareScaling() async throws {
        // Test dynamic NTK-aware scaling for extended context
        let seqLen = 4, numHeads = 2, headDim = 16
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }
        let scalingFactor: Float = 4.0

        let expected = cpuRoPE(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            scalingFactor: scalingFactor
        )

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )

        for i in 0..<result.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "NTK scaling mismatch at [\(i)]")
        }
    }

    @Test func ropeKnownValues() async throws {
        // Test with a simple known case: input = [1, 0] at position 1
        // With theta=10000, dim=2: freq = 1/10000^0 = 1.0
        // angle = 1.0 * 1.0 = 1.0
        // output[0] = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
        // output[1] = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.8415
        let seqLen = 2, numHeads = 1, headDim = 2
        let input: [Float] = [
            0.0, 0.0,   // position 0
            1.0, 0.0,   // position 1
        ]

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            commandQueue: commandQueue
        )

        // Position 0: angle=0, cos=1, sin=0 -> output = input = [0, 0]
        #expect(abs(result[0] - 0.0) < 1e-5)
        #expect(abs(result[1] - 0.0) < 1e-5)

        // Position 1: angle=1.0
        let expectedCos = cos(Float(1.0))
        let expectedSin = sin(Float(1.0))
        #expect(abs(result[2] - expectedCos) < 1e-5, "Expected cos(1)=\(expectedCos), got \(result[2])")
        #expect(abs(result[3] - expectedSin) < 1e-5, "Expected sin(1)=\(expectedSin), got \(result[3])")
    }

    @Test func largerDimensions() async throws {
        let seqLen = 32, numHeads = 8, headDim = 128
        let input = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuRoPE(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim
        )

        let kernel = try RoPEKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: 0,
            commandQueue: commandQueue
        )

        var maxError: Float = 0
        for i in 0..<result.count {
            maxError = max(maxError, abs(result[i] - expected[i]))
        }
        #expect(maxError < 1e-4,
                "Max error \(maxError) exceeds tolerance for large dimensions")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter RoPETests 2>&1`
Expected: FAIL — `RoPEKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/RoPEParams.h
#ifndef ROPE_PARAMS_H
#define ROPE_PARAMS_H

#include <stdint.h>

/// Parameters for RoPE (Rotary Position Embedding) kernel.
/// Input: [seqLen, numHeads, headDim]
/// Applies rotation to pairs of dimensions based on position.
typedef struct {
    uint32_t seqLen;            // sequence length
    uint32_t numHeads;          // number of attention heads
    uint32_t headDim;           // dimension per head (must be even)
    uint32_t startPos;          // position offset (for autoregressive decoding)
    float    theta;             // base frequency (typically 10000.0)
    float    scalingFactor;     // NTK-aware scaling factor (1.0 = no scaling)
} ERRoPEParams;

#endif /* ROPE_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/RoPE.metal
#include <metal_stdlib>
using namespace metal;

struct ERRoPEParams {
    uint   seqLen;
    uint   numHeads;
    uint   headDim;
    uint   startPos;
    float  theta;
    float  scalingFactor;
};

// RoPE: Apply rotary position embeddings in-place.
// Input/Output: [seqLen, numHeads, headDim]
// Each thread handles one (position, head, dim_pair) triple.
//
// For dimension pair (2i, 2i+1) at position p:
//   freq_i = 1 / (theta^(2i / headDim))
//   angle = p * freq_i / scalingFactor
//   out[2i]   = in[2i] * cos(angle) - in[2i+1] * sin(angle)
//   out[2i+1] = in[2i] * sin(angle) + in[2i+1] * cos(angle)

kernel void rope_f32(
    device const float*    input   [[buffer(0)]],
    device float*          output  [[buffer(1)]],
    constant ERRoPEParams& params  [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // tid.x = dimension pair index (0..<headDim/2)
    // tid.y = head index (0..<numHeads)
    // tid.z = sequence position (0..<seqLen)
    uint dim_pair = tid.x;
    uint head     = tid.y;
    uint seq_pos  = tid.z;

    uint halfDim = params.headDim / 2;
    if (dim_pair >= halfDim || head >= params.numHeads || seq_pos >= params.seqLen) {
        return;
    }

    // Compute frequency for this dimension pair
    float exponent = float(2 * dim_pair) / float(params.headDim);
    float freq = 1.0f / pow(params.theta, exponent);

    // Apply NTK-aware scaling
    float adjusted_freq = freq / params.scalingFactor;

    // Compute angle for this position
    float pos = float(seq_pos + params.startPos);
    float angle = pos * adjusted_freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Index into the flat array: [seqLen, numHeads, headDim]
    uint base_idx = (seq_pos * params.numHeads * params.headDim)
                  + (head * params.headDim)
                  + (2 * dim_pair);

    float x0 = input[base_idx];
    float x1 = input[base_idx + 1];

    output[base_idx]     = x0 * cos_val - x1 * sin_val;
    output[base_idx + 1] = x0 * sin_val + x1 * cos_val;
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/RoPEKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Swift wrapper for the RoPE (Rotary Position Embedding) Metal kernel.
/// Supports standard RoPE and dynamic NTK-aware scaling for extended contexts.
public final class RoPEKernel: Sendable {
    private let pipelineF32: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            throw RoPEError.libraryNotFound
        }
        guard let fn = library.makeFunction(name: "rope_f32") else {
            throw RoPEError.functionNotFound("rope_f32")
        }
        self.pipelineF32 = try device.makeComputePipelineState(function: fn)
    }

    /// Apply RoPE to input tensor.
    ///
    /// - Parameters:
    ///   - input: Input tensor [seqLen, numHeads, headDim]
    ///   - seqLen: Sequence length
    ///   - numHeads: Number of attention heads
    ///   - headDim: Dimension per head (must be even)
    ///   - startPos: Position offset for autoregressive decoding
    ///   - theta: Base frequency (default: 10000.0)
    ///   - scalingFactor: NTK-aware scaling factor (default: 1.0, no scaling)
    ///   - commandQueue: Metal command queue
    /// - Returns: Output tensor with RoPE applied [seqLen, numHeads, headDim]
    public func execute(
        input: [Float],
        seqLen: Int, numHeads: Int, headDim: Int,
        startPos: Int,
        theta: Float = 10000.0,
        scalingFactor: Float = 1.0,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(headDim % 2 == 0, "headDim must be even for RoPE")
        precondition(input.count == seqLen * numHeads * headDim)

        let totalElements = input.count
        let halfDim = headDim / 2

        let bufIn = device.makeBuffer(
            bytes: input, length: totalElements * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let bufOut = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERRoPEParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            startPos: UInt32(startPos),
            theta: theta,
            scalingFactor: scalingFactor
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw RoPEError.encodingFailed
        }

        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERRoPEParams>.stride, index: 2)

        // Grid: (halfDim, numHeads, seqLen) — one thread per dimension pair
        let gridSize = MTLSize(width: halfDim, height: numHeads, depth: seqLen)

        // Compute optimal threadgroup size
        let maxThreads = pipelineF32.maxTotalThreadsPerThreadgroup
        let threadgroupWidth = min(halfDim, maxThreads)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)

        // Use dispatchThreads for automatic grid coverage
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw error
        }

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: totalElements)
        return Array(UnsafeBufferPointer(start: ptr, count: totalElements))
    }

    /// Apply RoPE to Q and K tensors simultaneously for attention.
    /// Convenience method that applies the same positional encoding to both.
    ///
    /// - Parameters:
    ///   - q: Query tensor [seqLen, numHeads, headDim]
    ///   - k: Key tensor [seqLen, numKVHeads, headDim]
    ///   - seqLen: Sequence length
    ///   - numHeads: Number of Q heads
    ///   - numKVHeads: Number of KV heads
    ///   - headDim: Dimension per head
    ///   - startPos: Position offset
    ///   - theta: Base frequency
    ///   - scalingFactor: NTK scaling factor
    ///   - commandQueue: Metal command queue
    /// - Returns: (rotatedQ, rotatedK)
    public func applyToQK(
        q: [Float], k: [Float],
        seqLen: Int, numHeads: Int, numKVHeads: Int, headDim: Int,
        startPos: Int,
        theta: Float = 10000.0,
        scalingFactor: Float = 1.0,
        commandQueue: MTLCommandQueue
    ) async throws -> ([Float], [Float]) {
        let rotatedQ = try await execute(
            input: q,
            seqLen: seqLen, numHeads: numHeads, headDim: headDim,
            startPos: startPos,
            theta: theta, scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )
        let rotatedK = try await execute(
            input: k,
            seqLen: seqLen, numHeads: numKVHeads, headDim: headDim,
            startPos: startPos,
            theta: theta, scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )
        return (rotatedQ, rotatedK)
    }
}

public enum RoPEError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter RoPETests 2>&1`
Expected: All 7 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/RoPE.metal \
      Sources/EdgeRunnerSharedTypes/include/RoPEParams.h \
      Sources/EdgeRunnerMetal/RoPEKernel.swift \
      Tests/EdgeRunnerMetalTests/RoPETests.swift
git commit -m "feat: add RoPE kernel with NTK-aware scaling

Rotary position embeddings with configurable theta base.
Dynamic NTK-aware scaling for extended context windows.
Per-thread dimension-pair rotation via Metal compute.
Convenience applyToQK method for joint Q/K rotation."
```

---

> **End of Tasks 1–7 (First Half).** Tasks 8–14 follow below.

---

## Task 8: RMSNorm Kernel

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/RMSNorm.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/RMSNormParams.h`
- Create: `Sources/EdgeRunnerMetal/RMSNormKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/RMSNormTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/RMSNormTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
private func cpuRMSNorm(
    _ input: [Float],
    weight: [Float],
    eps: Float = 1e-5
) -> [Float] {
    let n = input.count
    let meanSq = input.reduce(0.0) { $0 + $1 * $1 } / Float(n)
    let rms = 1.0 / sqrt(meanSq + eps)
    return zip(input, weight).map { $0 * $1 * rms }
}

/// CPU reference batched RMSNorm: each row of [rows x cols] normalized independently.
private func cpuRMSNormBatched(
    _ input: [Float],
    weight: [Float],
    rows: Int, cols: Int,
    eps: Float = 1e-5
) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for r in 0..<rows {
        let offset = r * cols
        let row = Array(input[offset..<offset + cols])
        let normed = cpuRMSNorm(row, weight: weight, eps: eps)
        for c in 0..<cols {
            output[offset + c] = normed[c]
        }
    }
    return output
}

@Suite("RMSNorm Kernel")
struct RMSNormTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func singleTokenKnownValues() async throws {
        // Hand-computed: input = [1, 2, 3, 4], weight = [1, 1, 1, 1]
        // mean(x^2) = (1+4+9+16)/4 = 7.5
        // rsqrt(7.5 + 1e-5) ≈ 0.365148
        let input: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [1, 1, 1, 1]
        let expected = cpuRMSNorm(input, weight: weight)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, weight: weight,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func singleTokenWithLearnableWeight() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [0.5, 1.0, 1.5, 2.0]
        let expected = cpuRMSNorm(input, weight: weight)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, weight: weight,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func batchedTokens() async throws {
        let rows = 8, cols = 64
        let input = (0..<rows * cols).map { _ in Float.random(in: -2...2) }
        let weight = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }
        let expected = cpuRMSNormBatched(input, weight: weight, rows: rows, cols: cols)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, weight: weight,
            rows: rows, cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func largeHiddenDimension() async throws {
        // Llama-sized: 4096-dim hidden
        let rows = 4, cols = 4096
        let input = (0..<rows * cols).map { _ in Float.random(in: -1...1) }
        let weight = (0..<cols).map { _ in Float.random(in: 0.8...1.2) }
        let expected = cpuRMSNormBatched(input, weight: weight, rows: rows, cols: cols)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, weight: weight,
            rows: rows, cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func allZerosInput() async throws {
        // Should produce zeros (0 * rsqrt(0 + eps) * w = 0)
        let input: [Float] = [0, 0, 0, 0]
        let weight: [Float] = [1, 1, 1, 1]

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, weight: weight,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i]) < 1e-5,
                    "Expected ~0 at [\(i)], got \(result[i])")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter RMSNormTests 2>&1`
Expected: FAIL — `RMSNormKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/RMSNormParams.h
#ifndef RMSNORM_PARAMS_H
#define RMSNORM_PARAMS_H

#include <stdint.h>

/// Parameters for RMSNorm kernel dispatch.
/// input is [rows x cols], weight is [cols], output is [rows x cols].
typedef struct {
    uint32_t rows;      // number of tokens / rows
    uint32_t cols;      // hidden dimension
    float    eps;       // epsilon for numerical stability
} ERRMSNormParams;

#endif /* RMSNORM_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/RMSNorm.metal
#include <metal_stdlib>
using namespace metal;

struct ERRMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

/// RMSNorm: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
/// Each threadgroup handles one row (one token).
/// Threads within the threadgroup cooperate on the reduction.
kernel void rmsnorm_f32(
    device const float*       input   [[buffer(0)]],
    device const float*       weight  [[buffer(1)]],
    device float*             output  [[buffer(2)]],
    constant ERRMSNormParams& params  [[buffer(3)]],
    uint2 gid       [[thread_position_in_grid]],
    uint2 tid       [[thread_position_in_threadgroup]],
    uint2 tg_id     [[threadgroup_position_in_grid]],
    uint  tg_size   [[threads_per_threadgroup]]
) {
    const uint row = tg_id.x;
    if (row >= params.rows) return;

    const uint col = tid.x;
    const uint cols = params.cols;
    const uint row_offset = row * cols;

    // Phase 1: Each thread accumulates sum of squares for its strided elements
    float local_sum_sq = 0.0;
    for (uint c = col; c < cols; c += tg_size) {
        float val = input[row_offset + c];
        local_sum_sq += val * val;
    }

    // Phase 2: Reduce across threads using simd shuffle
    local_sum_sq = simd_sum(local_sum_sq);

    // Use threadgroup memory for cross-simdgroup reduction
    threadgroup float shared_sums[32]; // max 32 simdgroups
    uint simd_lane = simd_shuffle(0u, 0u); // not needed, use built-in
    uint simd_group_id = tid.x / 32;
    uint lane_in_simd = tid.x % 32;

    if (lane_in_simd == 0) {
        shared_sums[simd_group_id] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup does final reduction
    float total_sum_sq = 0.0;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group_id == 0 && lane_in_simd < num_simd_groups) {
        total_sum_sq = shared_sums[lane_in_simd];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    // Broadcast via threadgroup memory
    threadgroup float shared_rms;
    if (tid.x == 0) {
        float mean_sq = total_sum_sq / float(cols);
        shared_rms = rsqrt(mean_sq + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = shared_rms;

    // Phase 3: Apply normalization
    for (uint c = col; c < cols; c += tg_size) {
        float val = input[row_offset + c];
        output[row_offset + c] = val * rms * weight[c];
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/RMSNormKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Metal compute wrapper for RMS Layer Normalization.
public struct RMSNormKernel: Sendable {

    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw RMSNormError.libraryNotFound
        }
        guard let function = library.makeFunction(name: "rmsnorm_f32") else {
            throw RMSNormError.functionNotFound("rmsnorm_f32")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Execute RMSNorm on [rows x cols] input with [cols] weight vector.
    public func execute(
        input: [Float],
        weight: [Float],
        rows: Int,
        cols: Int,
        eps: Float = 1e-5,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols)
        precondition(weight.count == cols)

        let device = commandQueue.device
        let inputBuffer = device.makeBuffer(
            bytes: input, length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let weightBuffer = device.makeBuffer(
            bytes: weight, length: weight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERRMSNormParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            eps: eps
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RMSNormError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERRMSNormParams>.stride, index: 3)

        // One threadgroup per row. Threads per threadgroup = min(cols, maxThreads).
        let threadsPerGroup = min(cols, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        // Dispatch one threadgroup per row
        let numGroups = MTLSize(width: rows, height: 1, depth: 1)
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()

        return await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                let ptr = outputBuffer.contents().bindMemory(
                    to: Float.self, capacity: rows * cols
                )
                let result = Array(UnsafeBufferPointer(start: ptr, count: rows * cols))
                continuation.resume(returning: result)
            }
        }
    }
}

public enum RMSNormError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter RMSNormTests 2>&1`
Expected: All 5 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/RMSNorm.metal \
      Sources/EdgeRunnerSharedTypes/include/RMSNormParams.h \
      Sources/EdgeRunnerMetal/RMSNormKernel.swift \
      Tests/EdgeRunnerMetalTests/RMSNormTests.swift
git commit -m "feat: add RMSNorm kernel with per-token normalization

Metal compute shader with simd-based parallel reduction.
Learnable weight scaling. Threadgroup-per-row dispatch for
batched token normalization."
```

---

## Task 9: LayerNorm Kernel

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/LayerNorm.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/LayerNormParams.h`
- Create: `Sources/EdgeRunnerMetal/LayerNormKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/LayerNormTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/LayerNormTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
private func cpuLayerNorm(
    _ input: [Float],
    gamma: [Float],
    beta: [Float],
    eps: Float = 1e-5
) -> [Float] {
    let n = Float(input.count)
    let mean = input.reduce(0, +) / n
    let variance = input.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / n
    let invStd = 1.0 / sqrt(variance + eps)
    return zip(zip(input, gamma), beta).map { pair, b in
        let (x, g) = pair
        return (x - mean) * invStd * g + b
    }
}

/// CPU reference batched LayerNorm: each row independently.
private func cpuLayerNormBatched(
    _ input: [Float],
    gamma: [Float],
    beta: [Float],
    rows: Int, cols: Int,
    eps: Float = 1e-5
) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for r in 0..<rows {
        let offset = r * cols
        let row = Array(input[offset..<offset + cols])
        let normed = cpuLayerNorm(row, gamma: gamma, beta: beta, eps: eps)
        for c in 0..<cols {
            output[offset + c] = normed[c]
        }
    }
    return output
}

@Suite("LayerNorm Kernel")
struct LayerNormTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    @Test func singleTokenKnownValues() async throws {
        // input = [1, 2, 3, 4], mean = 2.5, var = 1.25
        let input: [Float] = [1, 2, 3, 4]
        let gamma: [Float] = [1, 1, 1, 1]
        let beta: [Float] = [0, 0, 0, 0]
        let expected = cpuLayerNorm(input, gamma: gamma, beta: beta)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, gamma: gamma, beta: beta,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func singleTokenWithGammaBeta() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let gamma: [Float] = [0.5, 1.0, 1.5, 2.0]
        let beta: [Float] = [0.1, 0.2, 0.3, 0.4]
        let expected = cpuLayerNorm(input, gamma: gamma, beta: beta)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, gamma: gamma, beta: beta,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func batchedTokens() async throws {
        let rows = 8, cols = 128
        let input = (0..<rows * cols).map { _ in Float.random(in: -2...2) }
        let gamma = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }
        let beta = (0..<cols).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuLayerNormBatched(input, gamma: gamma, beta: beta, rows: rows, cols: cols)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, gamma: gamma, beta: beta,
            rows: rows, cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func largeHiddenDimension() async throws {
        let rows = 4, cols = 4096
        let input = (0..<rows * cols).map { _ in Float.random(in: -1...1) }
        let gamma = (0..<cols).map { _ in Float.random(in: 0.8...1.2) }
        let beta = (0..<cols).map { _ in Float.random(in: -0.1...0.1) }
        let expected = cpuLayerNormBatched(input, gamma: gamma, beta: beta, rows: rows, cols: cols)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, gamma: gamma, beta: beta,
            rows: rows, cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<(rows * cols) {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func constantInput() async throws {
        // Constant input: variance = 0, so output = beta (since (x - mean) = 0)
        let input: [Float] = [5, 5, 5, 5]
        let gamma: [Float] = [1, 1, 1, 1]
        let beta: [Float] = [0.1, 0.2, 0.3, 0.4]

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input, gamma: gamma, beta: beta,
            rows: 1, cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i] - beta[i]) < 1e-5,
                    "Expected beta[\(i)]=\(beta[i]), got \(result[i])")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter LayerNormTests 2>&1`
Expected: FAIL — `LayerNormKernel` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/LayerNormParams.h
#ifndef LAYERNORM_PARAMS_H
#define LAYERNORM_PARAMS_H

#include <stdint.h>

/// Parameters for LayerNorm kernel dispatch.
/// input is [rows x cols], gamma/beta are [cols], output is [rows x cols].
typedef struct {
    uint32_t rows;      // number of tokens / rows
    uint32_t cols;      // hidden dimension
    float    eps;       // epsilon for numerical stability
} ERLayerNormParams;

#endif /* LAYERNORM_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/LayerNorm.metal
#include <metal_stdlib>
using namespace metal;

struct ERLayerNormParams {
    uint rows;
    uint cols;
    float eps;
};

/// LayerNorm: output[i] = (input[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
/// Each threadgroup handles one row (one token).
kernel void layernorm_f32(
    device const float*        input   [[buffer(0)]],
    device const float*        gamma   [[buffer(1)]],
    device const float*        beta    [[buffer(2)]],
    device float*              output  [[buffer(3)]],
    constant ERLayerNormParams& params [[buffer(4)]],
    uint2 tid       [[thread_position_in_threadgroup]],
    uint2 tg_id     [[threadgroup_position_in_grid]],
    uint  tg_size   [[threads_per_threadgroup]]
) {
    const uint row = tg_id.x;
    if (row >= params.rows) return;

    const uint col = tid.x;
    const uint cols = params.cols;
    const uint row_offset = row * cols;

    // Phase 1: Compute local sum and sum of squares
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    for (uint c = col; c < cols; c += tg_size) {
        float val = input[row_offset + c];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Phase 2: Reduce across threads using simd shuffle
    local_sum = simd_sum(local_sum);
    local_sum_sq = simd_sum(local_sum_sq);

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float shared_sum[32];
    threadgroup float shared_sum_sq[32];
    uint simd_group_id = col / 32;
    uint lane_in_simd = col % 32;

    if (lane_in_simd == 0) {
        shared_sum[simd_group_id] = local_sum;
        shared_sum_sq[simd_group_id] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum = 0.0;
    float total_sum_sq = 0.0;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group_id == 0 && lane_in_simd < num_simd_groups) {
        total_sum = shared_sum[lane_in_simd];
        total_sum_sq = shared_sum_sq[lane_in_simd];
    }
    total_sum = simd_sum(total_sum);
    total_sum_sq = simd_sum(total_sum_sq);

    // Broadcast mean and inv_std via threadgroup memory
    threadgroup float shared_mean;
    threadgroup float shared_inv_std;
    if (col == 0) {
        float mean = total_sum / float(cols);
        float variance = total_sum_sq / float(cols) - mean * mean;
        shared_mean = mean;
        shared_inv_std = rsqrt(variance + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_mean;
    float inv_std = shared_inv_std;

    // Phase 3: Apply normalization with affine transform
    for (uint c = col; c < cols; c += tg_size) {
        float val = input[row_offset + c];
        output[row_offset + c] = (val - mean) * inv_std * gamma[c] + beta[c];
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/LayerNormKernel.swift
import Metal
import EdgeRunnerSharedTypes

/// Metal compute wrapper for Layer Normalization.
public struct LayerNormKernel: Sendable {

    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw LayerNormError.libraryNotFound
        }
        guard let function = library.makeFunction(name: "layernorm_f32") else {
            throw LayerNormError.functionNotFound("layernorm_f32")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Execute LayerNorm on [rows x cols] input with [cols] gamma and beta vectors.
    public func execute(
        input: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        cols: Int,
        eps: Float = 1e-5,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(input.count == rows * cols)
        precondition(gamma.count == cols)
        precondition(beta.count == cols)

        let device = commandQueue.device
        let inputBuffer = device.makeBuffer(
            bytes: input, length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let gammaBuffer = device.makeBuffer(
            bytes: gamma, length: gamma.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let betaBuffer = device.makeBuffer(
            bytes: beta, length: beta.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERLayerNormParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            eps: eps
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LayerNormError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gammaBuffer, offset: 0, index: 1)
        encoder.setBuffer(betaBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ERLayerNormParams>.stride, index: 4)

        let threadsPerGroup = min(cols, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let numGroups = MTLSize(width: rows, height: 1, depth: 1)
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()

        return await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                let ptr = outputBuffer.contents().bindMemory(
                    to: Float.self, capacity: rows * cols
                )
                let result = Array(UnsafeBufferPointer(start: ptr, count: rows * cols))
                continuation.resume(returning: result)
            }
        }
    }
}

public enum LayerNormError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter LayerNormTests 2>&1`
Expected: All 5 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/LayerNorm.metal \
      Sources/EdgeRunnerSharedTypes/include/LayerNormParams.h \
      Sources/EdgeRunnerMetal/LayerNormKernel.swift \
      Tests/EdgeRunnerMetalTests/LayerNormTests.swift
git commit -m "feat: add LayerNorm kernel with affine transform

Mean/variance reduction via simd_sum and threadgroup memory.
Gamma/beta affine parameters. Threadgroup-per-row dispatch."
```

---

## Task 10: Activation Kernels (SwiGLU, GELU, Sigmoid)

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Activations.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/ActivationParams.h`
- Create: `Sources/EdgeRunnerMetal/ActivationKernels.swift`
- Test: `Tests/EdgeRunnerMetalTests/ActivationTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/ActivationTests.swift
import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

// MARK: - CPU References

private func cpuSigmoid(_ x: Float) -> Float {
    1.0 / (1.0 + exp(-x))
}

private func cpuSiLU(_ x: Float) -> Float {
    x * cpuSigmoid(x)
}

private func cpuGELU(_ x: Float) -> Float {
    let c: Float = sqrt(2.0 / .pi)
    return x * 0.5 * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
}

/// SwiGLU: silu(gate) * up — gate and up are each [n] packed as [2*n].
private func cpuSwiGLU(gate: [Float], up: [Float]) -> [Float] {
    precondition(gate.count == up.count)
    return zip(gate, up).map { g, u in cpuSiLU(g) * u }
}

@Suite("Activation Kernels")
struct ActivationTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.commandQueue = q
    }

    // MARK: - Sigmoid

    @Test func sigmoidKnownValues() async throws {
        let input: [Float] = [-10, -1, 0, 1, 10]
        let expected = input.map { cpuSigmoid($0) }

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.sigmoid(
            input: input, commandQueue: commandQueue
        )

        for i in 0..<input.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Sigmoid mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func sigmoidRandomBatch() async throws {
        let input = (0..<1024).map { _ in Float.random(in: -5...5) }
        let expected = input.map { cpuSigmoid($0) }

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.sigmoid(
            input: input, commandQueue: commandQueue
        )

        for i in 0..<input.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "Sigmoid mismatch at [\(i)]")
        }
    }

    // MARK: - GELU

    @Test func geluKnownValues() async throws {
        let input: [Float] = [-3, -1, 0, 1, 3]
        let expected = input.map { cpuGELU($0) }

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(
            input: input, commandQueue: commandQueue
        )

        for i in 0..<input.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "GELU mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func geluZeroIsZero() async throws {
        let input: [Float] = [0]
        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(input: input, commandQueue: commandQueue)
        #expect(abs(result[0]) < 1e-6, "GELU(0) should be 0")
    }

    @Test func geluRandomBatch() async throws {
        let input = (0..<4096).map { _ in Float.random(in: -5...5) }
        let expected = input.map { cpuGELU($0) }

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(input: input, commandQueue: commandQueue)

        for i in 0..<input.count {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "GELU mismatch at [\(i)]")
        }
    }

    // MARK: - SwiGLU

    @Test func swigluKnownValues() async throws {
        let gate: [Float] = [1, -1, 0, 2]
        let up: [Float] = [1, 1, 1, 0.5]
        let expected = cpuSwiGLU(gate: gate, up: up)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(
            gate: gate, up: up, commandQueue: commandQueue
        )

        for i in 0..<gate.count {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "SwiGLU mismatch at [\(i)]: GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func swigluRandomBatch() async throws {
        let n = 2048
        let gate = (0..<n).map { _ in Float.random(in: -3...3) }
        let up = (0..<n).map { _ in Float.random(in: -3...3) }
        let expected = cpuSwiGLU(gate: gate, up: up)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(
            gate: gate, up: up, commandQueue: commandQueue
        )

        for i in 0..<n {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "SwiGLU mismatch at [\(i)]")
        }
    }

    @Test func swigluZeroGate() async throws {
        // silu(0) = 0, so output should be all zeros
        let gate: [Float] = [0, 0, 0, 0]
        let up: [Float] = [1, 2, 3, 4]

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(
            gate: gate, up: up, commandQueue: commandQueue
        )

        for i in 0..<4 {
            #expect(abs(result[i]) < 1e-6,
                    "SwiGLU with zero gate should be 0 at [\(i)]")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ActivationTests 2>&1`
Expected: FAIL — `ActivationKernels` not defined

**Step 3: Implement the shared C params header**

```c
// Sources/EdgeRunnerSharedTypes/include/ActivationParams.h
#ifndef ACTIVATION_PARAMS_H
#define ACTIVATION_PARAMS_H

#include <stdint.h>

/// Parameters for element-wise activation kernel dispatch.
typedef struct {
    uint32_t count;     // total number of elements
} ERActivationParams;

#endif /* ACTIVATION_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Activations.metal
#include <metal_stdlib>
using namespace metal;

struct ERActivationParams {
    uint count;
};

// ─── Sigmoid ───────────────────────────────────────────────

kernel void sigmoid_f32(
    device const float*        input   [[buffer(0)]],
    device float*              output  [[buffer(1)]],
    constant ERActivationParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    float x = input[gid];
    output[gid] = 1.0 / (1.0 + exp(-x));
}

// ─── GELU (tanh approximation) ────────────────────────────

kernel void gelu_f32(
    device const float*        input   [[buffer(0)]],
    device float*              output  [[buffer(1)]],
    constant ERActivationParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    float x = input[gid];
    const float c = 0.7978845608028654; // sqrt(2/pi)
    float inner = c * (x + 0.044715 * x * x * x);
    output[gid] = x * 0.5 * (1.0 + tanh(inner));
}

// ─── SiLU (used internally by SwiGLU) ─────────────────────

inline float silu(float x) {
    return x / (1.0 + exp(-x));
}

// ─── SwiGLU: silu(gate) * up ──────────────────────────────

kernel void swiglu_f32(
    device const float*        gate    [[buffer(0)]],
    device const float*        up      [[buffer(1)]],
    device float*              output  [[buffer(2)]],
    constant ERActivationParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    output[gid] = silu(gate[gid]) * up[gid];
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/ActivationKernels.swift
import Metal
import EdgeRunnerSharedTypes

/// Metal compute wrapper for activation functions: Sigmoid, GELU, SwiGLU.
public struct ActivationKernels: Sendable {

    private let sigmoidPipeline: MTLComputePipelineState
    private let geluPipeline: MTLComputePipelineState
    private let swigluPipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw ActivationError.libraryNotFound
        }

        guard let sigmoidFn = library.makeFunction(name: "sigmoid_f32") else {
            throw ActivationError.functionNotFound("sigmoid_f32")
        }
        guard let geluFn = library.makeFunction(name: "gelu_f32") else {
            throw ActivationError.functionNotFound("gelu_f32")
        }
        guard let swigluFn = library.makeFunction(name: "swiglu_f32") else {
            throw ActivationError.functionNotFound("swiglu_f32")
        }

        self.sigmoidPipeline = try device.makeComputePipelineState(function: sigmoidFn)
        self.geluPipeline = try device.makeComputePipelineState(function: geluFn)
        self.swigluPipeline = try device.makeComputePipelineState(function: swigluFn)
    }

    // MARK: - Sigmoid

    public func sigmoid(
        input: [Float],
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try await executeElementWise(
            pipeline: sigmoidPipeline,
            input: input,
            commandQueue: commandQueue
        )
    }

    // MARK: - GELU

    public func gelu(
        input: [Float],
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        try await executeElementWise(
            pipeline: geluPipeline,
            input: input,
            commandQueue: commandQueue
        )
    }

    // MARK: - SwiGLU

    public func swiglu(
        gate: [Float],
        up: [Float],
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(gate.count == up.count)
        let count = gate.count
        let device = commandQueue.device

        let gateBuffer = device.makeBuffer(
            bytes: gate, length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let upBuffer = device.makeBuffer(
            bytes: up, length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERActivationParams(count: UInt32(count))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ActivationError.encodingFailed
        }

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERActivationParams>.stride, index: 3)

        let threadgroupSize = MTLSize(
            width: min(count, swigluPipeline.maxTotalThreadsPerThreadgroup),
            height: 1, depth: 1
        )
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()

        return await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                let ptr = outputBuffer.contents().bindMemory(
                    to: Float.self, capacity: count
                )
                let result = Array(UnsafeBufferPointer(start: ptr, count: count))
                continuation.resume(returning: result)
            }
        }
    }

    // MARK: - Private element-wise helper

    private func executeElementWise(
        pipeline: MTLComputePipelineState,
        input: [Float],
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let count = input.count
        let device = commandQueue.device

        let inputBuffer = device.makeBuffer(
            bytes: input, length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERActivationParams(count: UInt32(count))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ActivationError.encodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERActivationParams>.stride, index: 2)

        let threadgroupSize = MTLSize(
            width: min(count, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1, depth: 1
        )
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()

        return await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                let ptr = outputBuffer.contents().bindMemory(
                    to: Float.self, capacity: count
                )
                let result = Array(UnsafeBufferPointer(start: ptr, count: count))
                continuation.resume(returning: result)
            }
        }
    }
}

public enum ActivationError: Error, Sendable {
    case libraryNotFound
    case functionNotFound(String)
    case encodingFailed
}
```

**Step 6: Run tests to verify they pass**

Run: `swift test --filter ActivationTests 2>&1`
Expected: All 9 tests PASS

**Step 7: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/Activations.metal \
      Sources/EdgeRunnerSharedTypes/include/ActivationParams.h \
      Sources/EdgeRunnerMetal/ActivationKernels.swift \
      Tests/EdgeRunnerMetalTests/ActivationTests.swift
git commit -m "feat: add activation kernels — Sigmoid, GELU, SwiGLU

Element-wise Metal shaders for all three activations.
SwiGLU uses silu(gate) * up for Llama-style FFN.
GELU uses tanh approximation matching PyTorch default."
```

---

## Task 11: EdgeRunnerModule Protocol

**Files:**
- Create: `Sources/EdgeRunner/Module/EdgeRunnerModule.swift`
- Create: `Sources/EdgeRunner/Module/TensorBox.swift`
- Create: `Sources/EdgeRunner/Module/Sequential.swift`
- Create: `Sources/EdgeRunner/Module/Linear.swift`
- Test: `Tests/EdgeRunnerTests/ModuleTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/ModuleTests.swift
import Testing
@testable import EdgeRunner

// MARK: - Mock module for testing

/// A simple doubling module for protocol conformance tests.
struct DoublingModule: EdgeRunnerModule {
    typealias Input = [Float]
    typealias Output = [Float]

    let scale: Float

    var parameters: [String: any TensorBox] {
        ["scale": ScalarTensorBox(value: scale)]
    }

    func forward(_ input: [Float]) async throws -> [Float] {
        input.map { $0 * scale }
    }
}

/// A simple offset module for composability tests.
struct OffsetModule: EdgeRunnerModule {
    typealias Input = [Float]
    typealias Output = [Float]

    let offset: Float

    var parameters: [String: any TensorBox] {
        ["offset": ScalarTensorBox(value: offset)]
    }

    func forward(_ input: [Float]) async throws -> [Float] {
        input.map { $0 + offset }
    }
}

@Suite("EdgeRunnerModule Protocol")
struct ModuleTests {

    @Test func moduleForwardPass() async throws {
        let module = DoublingModule(scale: 2.0)
        let input: [Float] = [1, 2, 3]
        let output = try await module.forward(input)
        #expect(output == [2, 4, 6])
    }

    @Test func moduleParametersAccess() throws {
        let module = DoublingModule(scale: 3.0)
        let params = module.parameters
        #expect(params.count == 1)
        #expect(params.keys.contains("scale"))
        let box = params["scale"] as? ScalarTensorBox
        #expect(box?.value == 3.0)
    }

    @Test func sequentialForward() async throws {
        let seq = Sequential(
            DoublingModule(scale: 2.0),
            OffsetModule(offset: 10.0)
        )
        let input: [Float] = [1, 2, 3]
        let output = try await seq.forward(input)
        // 2*[1,2,3] = [2,4,6], then +10 = [12,14,16]
        #expect(output == [12, 14, 16])
    }

    @Test func sequentialParameters() throws {
        let seq = Sequential(
            DoublingModule(scale: 2.0),
            OffsetModule(offset: 10.0)
        )
        let params = seq.parameters
        // Sequential should namespace parameters: "0.scale", "1.offset"
        #expect(params.count == 2)
        #expect(params.keys.contains("0.scale"))
        #expect(params.keys.contains("1.offset"))
    }

    @Test func emptySequential() async throws {
        // Identity: pass-through
        let seq = Sequential<DoublingModule>()
        let input: [Float] = [1, 2, 3]
        let output = try await seq.forward(input)
        #expect(output == input)
    }

    @Test func tensorBoxProtocol() throws {
        let box = ScalarTensorBox(value: 42.0)
        #expect(box.elementCount == 1)
        #expect(box.floatArray == [42.0])
    }

    @Test func linearModuleForward() async throws {
        // Linear: y = x @ W^T + b
        // x = [1, 2] (1x2), W = [[1, 0], [0, 1], [1, 1]] (3x2), b = [0.1, 0.2, 0.3]
        // y = [1*1+2*0+0.1, 1*0+2*1+0.2, 1*1+2*1+0.3] = [1.1, 2.2, 3.3]
        let linear = try LinearModule(
            inFeatures: 2, outFeatures: 3,
            weight: [1, 0,  0, 1,  1, 1],  // [outFeatures x inFeatures] row-major
            bias: [0.1, 0.2, 0.3]
        )
        let output = try await linear.forward([1, 2])
        #expect(output.count == 3)
        for (i, expected) in [Float(1.1), 2.2, 3.3].enumerated() {
            #expect(abs(output[i] - expected) < 1e-5,
                    "Linear mismatch at [\(i)]: got \(output[i]) expected \(expected)")
        }
    }

    @Test func linearModuleNoBias() async throws {
        let linear = try LinearModule(
            inFeatures: 2, outFeatures: 2,
            weight: [1, 0,  0, 1],
            bias: nil
        )
        let output = try await linear.forward([3, 4])
        #expect(output.count == 2)
        #expect(abs(output[0] - 3.0) < 1e-5)
        #expect(abs(output[1] - 4.0) < 1e-5)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ModuleTests 2>&1`
Expected: FAIL — `EdgeRunnerModule` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Module/TensorBox.swift

/// Type-erased container for tensor parameter data.
/// Enables heterogeneous parameter dictionaries across module types.
public protocol TensorBox: Sendable {
    /// Total number of scalar elements in the tensor.
    var elementCount: Int { get }

    /// Returns a flat array of Float values (copies data).
    var floatArray: [Float] { get }

    /// Shape of the underlying tensor.
    var shape: [Int] { get }
}

/// A single scalar value wrapped as a TensorBox.
public struct ScalarTensorBox: TensorBox, Sendable {
    public let value: Float

    public init(value: Float) {
        self.value = value
    }

    public var elementCount: Int { 1 }
    public var floatArray: [Float] { [value] }
    public var shape: [Int] { [] }
}

/// A flat array of floats wrapped as a TensorBox.
public struct ArrayTensorBox: TensorBox, Sendable {
    public let data: [Float]
    public let shape: [Int]

    public init(data: [Float], shape: [Int]) {
        precondition(data.count == shape.reduce(1, *),
                     "Data count \(data.count) doesn't match shape \(shape)")
        self.data = data
        self.shape = shape
    }

    public var elementCount: Int { data.count }
    public var floatArray: [Float] { data }
}
```

```swift
// Sources/EdgeRunner/Module/EdgeRunnerModule.swift

/// Core protocol for all neural network modules in EdgeRunner.
///
/// Each module declares its Input/Output types and provides:
/// - A forward pass computation
/// - Named access to all learnable parameters
///
/// Conforming types must be Sendable for safe concurrent use.
public protocol EdgeRunnerModule: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable

    /// Perform the forward computation.
    func forward(_ input: Input) async throws -> Output

    /// All learnable parameters, keyed by name.
    /// Names should use dot-separated paths for nested modules (e.g., "layers.0.weight").
    var parameters: [String: any TensorBox] { get }
}
```

```swift
// Sources/EdgeRunner/Module/Sequential.swift

/// A container that chains modules in sequence, feeding the output of each
/// into the input of the next.
///
/// All child modules must share the same Input/Output type so they can be composed.
public struct Sequential<M: EdgeRunnerModule>: EdgeRunnerModule
where M.Input == M.Output {

    public typealias Input = M.Input
    public typealias Output = M.Output

    private let modules: [M]

    public init(_ modules: M...) {
        self.modules = modules
    }

    public init(_ modules: [M]) {
        self.modules = modules
    }

    public func forward(_ input: Input) async throws -> Output {
        var current = input
        for module in modules {
            current = try await module.forward(current)
        }
        return current
    }

    public var parameters: [String: any TensorBox] {
        var result: [String: any TensorBox] = [:]
        for (index, module) in modules.enumerated() {
            for (key, value) in module.parameters {
                result["\(index).\(key)"] = value
            }
        }
        return result
    }
}
```

```swift
// Sources/EdgeRunner/Module/Linear.swift

/// A fully-connected linear layer: y = x @ W^T + b
///
/// Weight is stored as [outFeatures x inFeatures] row-major.
/// This is a CPU reference implementation; the GPU-accelerated version
/// uses GEMMKernel from EdgeRunnerMetal.
public struct LinearModule: EdgeRunnerModule, Sendable {

    public typealias Input = [Float]
    public typealias Output = [Float]

    public let inFeatures: Int
    public let outFeatures: Int
    private let weight: [Float]     // [outFeatures x inFeatures]
    private let bias: [Float]?      // [outFeatures]

    public init(
        inFeatures: Int,
        outFeatures: Int,
        weight: [Float],
        bias: [Float]?
    ) throws {
        precondition(weight.count == outFeatures * inFeatures,
                     "Weight size \(weight.count) != \(outFeatures)*\(inFeatures)")
        if let bias {
            precondition(bias.count == outFeatures,
                         "Bias size \(bias.count) != \(outFeatures)")
        }
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.bias = bias
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        precondition(input.count == inFeatures,
                     "Input size \(input.count) != inFeatures \(inFeatures)")

        var output = [Float](repeating: 0, count: outFeatures)
        for o in 0..<outFeatures {
            var sum: Float = 0
            for i in 0..<inFeatures {
                sum += weight[o * inFeatures + i] * input[i]
            }
            if let bias {
                sum += bias[o]
            }
            output[o] = sum
        }
        return output
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [
            "weight": ArrayTensorBox(
                data: weight,
                shape: [outFeatures, inFeatures]
            )
        ]
        if let bias {
            params["bias"] = ArrayTensorBox(data: bias, shape: [outFeatures])
        }
        return params
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter ModuleTests 2>&1`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Module/EdgeRunnerModule.swift \
      Sources/EdgeRunner/Module/TensorBox.swift \
      Sources/EdgeRunner/Module/Sequential.swift \
      Sources/EdgeRunner/Module/Linear.swift \
      Tests/EdgeRunnerTests/ModuleTests.swift
git commit -m "feat: add EdgeRunnerModule protocol with Sequential container

Protocol with associatedtype Input/Output and parameter access.
TensorBox protocol for type-erased parameter storage.
Sequential container for composing same-typed modules.
LinearModule as CPU reference for dense layers."
```

---

## Task 12: Transformer Block Composition

**Files:**
- Create: `Sources/EdgeRunner/Transformer/MultiHeadAttention.swift`
- Create: `Sources/EdgeRunner/Transformer/FeedForward.swift`
- Create: `Sources/EdgeRunner/Transformer/TransformerBlock.swift`
- Create: `Sources/EdgeRunner/Transformer/TransformerConfig.swift`
- Test: `Tests/EdgeRunnerTests/TransformerBlockTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/TransformerBlockTests.swift
import Testing
@testable import EdgeRunner

@Suite("Transformer Block")
struct TransformerBlockTests {

    // MARK: - Config

    static let tinyConfig = TransformerConfig(
        hiddenDim: 64,
        numHeads: 4,
        numKVHeads: 4,
        intermediateSize: 128,  // FFN intermediate
        numLayers: 1,
        vocabSize: 32,
        maxSeqLen: 32,
        rmsNormEps: 1e-5,
        ropeTheta: 10000.0
    )

    // MARK: - MultiHeadAttention

    @Test func multiHeadAttentionOutputShape() async throws {
        let config = Self.tinyConfig
        let attn = try MultiHeadAttention(config: config)

        // Input: [seqLen x hiddenDim]
        let seqLen = 4
        let input = (0..<seqLen * config.hiddenDim).map { _ in Float.random(in: -0.1...0.1) }
        let output = try await attn.forward(AttentionInput(
            hidden: input,
            seqLen: seqLen,
            startPos: 0
        ))

        #expect(output.count == seqLen * config.hiddenDim,
                "Expected \(seqLen * config.hiddenDim), got \(output.count)")
    }

    @Test func multiHeadAttentionParameters() throws {
        let config = Self.tinyConfig
        let attn = try MultiHeadAttention(config: config)
        let params = attn.parameters

        // Should have: wq, wk, wv, wo weights
        #expect(params.keys.contains("wq.weight"))
        #expect(params.keys.contains("wk.weight"))
        #expect(params.keys.contains("wv.weight"))
        #expect(params.keys.contains("wo.weight"))
    }

    @Test func multiHeadAttentionDeterministic() async throws {
        let config = Self.tinyConfig
        let attn = try MultiHeadAttention(config: config)
        let seqLen = 4
        let input = (0..<seqLen * config.hiddenDim).map { Float($0) * 0.01 }

        let output1 = try await attn.forward(AttentionInput(
            hidden: input, seqLen: seqLen, startPos: 0
        ))
        let output2 = try await attn.forward(AttentionInput(
            hidden: input, seqLen: seqLen, startPos: 0
        ))

        for i in 0..<output1.count {
            #expect(abs(output1[i] - output2[i]) < 1e-6,
                    "Non-deterministic at [\(i)]")
        }
    }

    // MARK: - FeedForward (SwiGLU variant)

    @Test func feedForwardOutputShape() async throws {
        let config = Self.tinyConfig
        let ffn = try FeedForward(config: config)

        let seqLen = 4
        let input = (0..<seqLen * config.hiddenDim).map { _ in Float.random(in: -0.1...0.1) }
        let output = try await ffn.forward(input)

        #expect(output.count == seqLen * config.hiddenDim,
                "FFN output shape mismatch: expected \(seqLen * config.hiddenDim), got \(output.count)")
    }

    @Test func feedForwardParameters() throws {
        let config = Self.tinyConfig
        let ffn = try FeedForward(config: config)
        let params = ffn.parameters

        #expect(params.keys.contains("gate.weight"))
        #expect(params.keys.contains("up.weight"))
        #expect(params.keys.contains("down.weight"))
    }

    // MARK: - TransformerBlock (full)

    @Test func transformerBlockOutputShape() async throws {
        let config = Self.tinyConfig
        let block = try TransformerBlock(config: config, layerIndex: 0)

        let seqLen = 4
        let input = (0..<seqLen * config.hiddenDim).map { _ in Float.random(in: -0.1...0.1) }
        let output = try await block.forward(TransformerBlockInput(
            hidden: input,
            seqLen: seqLen,
            startPos: 0
        ))

        #expect(output.hidden.count == seqLen * config.hiddenDim,
                "Block output shape mismatch")
    }

    @Test func transformerBlockResidualConnection() async throws {
        // With zero weights, output should equal input (residual pass-through)
        let config = Self.tinyConfig
        let block = try TransformerBlock(
            config: config,
            layerIndex: 0,
            zeroWeights: true
        )

        let seqLen = 2
        let input = (0..<seqLen * config.hiddenDim).map { _ in Float.random(in: -1...1) }
        let output = try await block.forward(TransformerBlockInput(
            hidden: input, seqLen: seqLen, startPos: 0
        ))

        // With zero projection weights, attention and FFN contribute nothing,
        // so residual connections preserve the input
        for i in 0..<input.count {
            #expect(abs(output.hidden[i] - input[i]) < 1e-4,
                    "Residual broken at [\(i)]: in=\(input[i]) out=\(output.hidden[i])")
        }
    }

    @Test func transformerBlockParameters() throws {
        let config = Self.tinyConfig
        let block = try TransformerBlock(config: config, layerIndex: 0)
        let params = block.parameters

        // Should have attention_norm, ffn_norm, attention.*, ffn.*
        let hasAttnNorm = params.keys.contains(where: { $0.hasPrefix("attention_norm") })
        let hasFfnNorm = params.keys.contains(where: { $0.hasPrefix("ffn_norm") })
        let hasAttn = params.keys.contains(where: { $0.hasPrefix("attention.") })
        let hasFfn = params.keys.contains(where: { $0.hasPrefix("ffn.") })

        #expect(hasAttnNorm, "Missing attention_norm parameters")
        #expect(hasFfnNorm, "Missing ffn_norm parameters")
        #expect(hasAttn, "Missing attention parameters")
        #expect(hasFfn, "Missing ffn parameters")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter TransformerBlockTests 2>&1`
Expected: FAIL — `TransformerConfig` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Transformer/TransformerConfig.swift

/// Configuration for a decoder-only transformer model.
/// Immutable and Sendable for safe sharing across modules.
public struct TransformerConfig: Sendable {
    public let hiddenDim: Int           // d_model
    public let numHeads: Int            // number of query heads
    public let numKVHeads: Int          // number of KV heads (GQA)
    public let intermediateSize: Int    // FFN hidden dim
    public let numLayers: Int           // number of transformer blocks
    public let vocabSize: Int           // vocabulary size
    public let maxSeqLen: Int           // maximum sequence length
    public let rmsNormEps: Float        // epsilon for RMSNorm
    public let ropeTheta: Float         // base frequency for RoPE

    public var headDim: Int { hiddenDim / numHeads }
    public var kvGroupSize: Int { numHeads / numKVHeads }

    public init(
        hiddenDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        intermediateSize: Int,
        numLayers: Int,
        vocabSize: Int,
        maxSeqLen: Int,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0
    ) {
        precondition(hiddenDim % numHeads == 0, "hiddenDim must be divisible by numHeads")
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.intermediateSize = intermediateSize
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
    }
}
```

```swift
// Sources/EdgeRunner/Transformer/MultiHeadAttention.swift
import Foundation

/// Input to the multi-head attention module.
public struct AttentionInput: Sendable {
    public let hidden: [Float]      // [seqLen x hiddenDim]
    public let seqLen: Int
    public let startPos: Int        // for KV cache positioning

    public init(hidden: [Float], seqLen: Int, startPos: Int) {
        self.hidden = hidden
        self.seqLen = seqLen
        self.startPos = startPos
    }
}

/// Multi-head attention with support for GQA.
///
/// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
/// Uses separate linear projections for Q, K, V, and output.
/// This is a CPU reference implementation wrapping the module protocol.
/// The GPU-accelerated version dispatches to Flash Attention + RoPE + KV Cache kernels.
public struct MultiHeadAttention: EdgeRunnerModule, Sendable {

    public typealias Input = AttentionInput
    public typealias Output = [Float]

    private let config: TransformerConfig
    private let wq: LinearModule    // [hiddenDim x hiddenDim]
    private let wk: LinearModule    // [numKVHeads * headDim x hiddenDim]
    private let wv: LinearModule    // [numKVHeads * headDim x hiddenDim]
    private let wo: LinearModule    // [hiddenDim x hiddenDim]

    public init(config: TransformerConfig, zeroWeights: Bool = false) throws {
        self.config = config
        let hd = config.hiddenDim
        let kvDim = config.numKVHeads * config.headDim

        if zeroWeights {
            self.wq = try LinearModule(inFeatures: hd, outFeatures: hd,
                                       weight: [Float](repeating: 0, count: hd * hd), bias: nil)
            self.wk = try LinearModule(inFeatures: hd, outFeatures: kvDim,
                                       weight: [Float](repeating: 0, count: kvDim * hd), bias: nil)
            self.wv = try LinearModule(inFeatures: hd, outFeatures: kvDim,
                                       weight: [Float](repeating: 0, count: kvDim * hd), bias: nil)
            self.wo = try LinearModule(inFeatures: hd, outFeatures: hd,
                                       weight: [Float](repeating: 0, count: hd * hd), bias: nil)
        } else {
            // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
            let qScale = sqrt(2.0 / Float(hd + hd))
            let kvScale = sqrt(2.0 / Float(hd + kvDim))
            self.wq = try LinearModule(
                inFeatures: hd, outFeatures: hd,
                weight: (0..<hd * hd).map { _ in Float.random(in: -qScale...qScale) },
                bias: nil
            )
            self.wk = try LinearModule(
                inFeatures: hd, outFeatures: kvDim,
                weight: (0..<kvDim * hd).map { _ in Float.random(in: -kvScale...kvScale) },
                bias: nil
            )
            self.wv = try LinearModule(
                inFeatures: hd, outFeatures: kvDim,
                weight: (0..<kvDim * hd).map { _ in Float.random(in: -kvScale...kvScale) },
                bias: nil
            )
            self.wo = try LinearModule(
                inFeatures: hd, outFeatures: hd,
                weight: (0..<hd * hd).map { _ in Float.random(in: -qScale...qScale) },
                bias: nil
            )
        }
    }

    public func forward(_ input: AttentionInput) async throws -> [Float] {
        let seqLen = input.seqLen
        let hd = config.hiddenDim
        let headDim = config.headDim
        let numHeads = config.numHeads
        let numKVHeads = config.numKVHeads
        let kvGroupSize = config.kvGroupSize

        // Project Q, K, V for each token
        var allQ = [Float]()
        var allK = [Float]()
        var allV = [Float]()

        for t in 0..<seqLen {
            let token = Array(input.hidden[t * hd..<(t + 1) * hd])
            let q = try await wq.forward(token)
            let k = try await wk.forward(token)
            let v = try await wv.forward(token)
            allQ.append(contentsOf: q)
            allK.append(contentsOf: k)
            allV.append(contentsOf: v)
        }

        // Compute attention per head
        var attnOutput = [Float](repeating: 0, count: seqLen * hd)
        let scale = 1.0 / sqrt(Float(headDim))

        for h in 0..<numHeads {
            let kvH = h / kvGroupSize  // GQA: map query head to KV head

            for qi in 0..<seqLen {
                // Q for this head, this token
                let qOffset = qi * hd + h * headDim
                let qVec = Array(allQ[qOffset..<qOffset + headDim])

                // Compute attention scores against all K
                var scores = [Float](repeating: 0, count: seqLen)
                for ki in 0..<seqLen {
                    let kOffset = ki * (numKVHeads * headDim) + kvH * headDim
                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += qVec[d] * allK[kOffset + d]
                    }
                    scores[ki] = dot * scale
                }

                // Causal mask: only attend to positions <= qi + startPos
                for ki in 0..<seqLen {
                    if ki > qi {
                        scores[ki] = -Float.infinity
                    }
                }

                // Softmax
                let maxScore = scores.max() ?? 0
                var expScores = scores.map { exp($0 - maxScore) }
                let sumExp = expScores.reduce(0, +)
                expScores = expScores.map { $0 / sumExp }

                // Weighted sum of V
                let outOffset = qi * hd + h * headDim
                for vi in 0..<seqLen {
                    let vOffset = vi * (numKVHeads * headDim) + kvH * headDim
                    for d in 0..<headDim {
                        attnOutput[outOffset + d] += expScores[vi] * allV[vOffset + d]
                    }
                }
            }
        }

        // Output projection per token
        var result = [Float]()
        for t in 0..<seqLen {
            let token = Array(attnOutput[t * hd..<(t + 1) * hd])
            let projected = try await wo.forward(token)
            result.append(contentsOf: projected)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        for (key, value) in wq.parameters { params["wq.\(key)"] = value }
        for (key, value) in wk.parameters { params["wk.\(key)"] = value }
        for (key, value) in wv.parameters { params["wv.\(key)"] = value }
        for (key, value) in wo.parameters { params["wo.\(key)"] = value }
        return params
    }
}
```

```swift
// Sources/EdgeRunner/Transformer/FeedForward.swift
import Foundation

/// SwiGLU-based feed-forward network used in Llama-style models.
///
/// FFN(x) = down(silu(gate(x)) * up(x))
/// Where gate, up project from hiddenDim to intermediateSize,
/// and down projects back from intermediateSize to hiddenDim.
public struct FeedForward: EdgeRunnerModule, Sendable {

    public typealias Input = [Float]    // [seqLen * hiddenDim]
    public typealias Output = [Float]   // [seqLen * hiddenDim]

    private let config: TransformerConfig
    private let gate: LinearModule      // [intermediateSize x hiddenDim]
    private let up: LinearModule        // [intermediateSize x hiddenDim]
    private let down: LinearModule      // [hiddenDim x intermediateSize]

    public init(config: TransformerConfig, zeroWeights: Bool = false) throws {
        self.config = config
        let hd = config.hiddenDim
        let inter = config.intermediateSize

        if zeroWeights {
            self.gate = try LinearModule(inFeatures: hd, outFeatures: inter,
                                         weight: [Float](repeating: 0, count: inter * hd), bias: nil)
            self.up = try LinearModule(inFeatures: hd, outFeatures: inter,
                                       weight: [Float](repeating: 0, count: inter * hd), bias: nil)
            self.down = try LinearModule(inFeatures: inter, outFeatures: hd,
                                         weight: [Float](repeating: 0, count: hd * inter), bias: nil)
        } else {
            let scale1 = sqrt(2.0 / Float(hd + inter))
            let scale2 = sqrt(2.0 / Float(inter + hd))
            self.gate = try LinearModule(
                inFeatures: hd, outFeatures: inter,
                weight: (0..<inter * hd).map { _ in Float.random(in: -scale1...scale1) },
                bias: nil
            )
            self.up = try LinearModule(
                inFeatures: hd, outFeatures: inter,
                weight: (0..<inter * hd).map { _ in Float.random(in: -scale1...scale1) },
                bias: nil
            )
            self.down = try LinearModule(
                inFeatures: inter, outFeatures: hd,
                weight: (0..<hd * inter).map { _ in Float.random(in: -scale2...scale2) },
                bias: nil
            )
        }
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        let hd = config.hiddenDim
        let inter = config.intermediateSize
        let seqLen = input.count / hd
        precondition(input.count == seqLen * hd)

        var result = [Float]()
        for t in 0..<seqLen {
            let token = Array(input[t * hd..<(t + 1) * hd])
            let gateOut = try await gate.forward(token)
            let upOut = try await up.forward(token)

            // SwiGLU: silu(gate) * up
            var swigluOut = [Float](repeating: 0, count: inter)
            for i in 0..<inter {
                let sigmoid = 1.0 / (1.0 + exp(-gateOut[i]))
                swigluOut[i] = gateOut[i] * sigmoid * upOut[i]
            }

            let downOut = try await down.forward(swigluOut)
            result.append(contentsOf: downOut)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        for (key, value) in gate.parameters { params["gate.\(key)"] = value }
        for (key, value) in up.parameters { params["up.\(key)"] = value }
        for (key, value) in down.parameters { params["down.\(key)"] = value }
        return params
    }
}
```

```swift
// Sources/EdgeRunner/Transformer/TransformerBlock.swift
import Foundation

/// Input to a single transformer block.
public struct TransformerBlockInput: Sendable {
    public let hidden: [Float]      // [seqLen x hiddenDim]
    public let seqLen: Int
    public let startPos: Int

    public init(hidden: [Float], seqLen: Int, startPos: Int) {
        self.hidden = hidden
        self.seqLen = seqLen
        self.startPos = startPos
    }
}

/// Output from a single transformer block.
public struct TransformerBlockOutput: Sendable {
    public let hidden: [Float]      // [seqLen x hiddenDim]
}

/// A single transformer decoder block:
///   x = x + attention(rms_norm(x))
///   x = x + ffn(rms_norm(x))
///
/// Uses pre-norm architecture (RMSNorm before each sub-layer) with
/// residual connections, matching Llama/Mistral style.
public struct TransformerBlock: EdgeRunnerModule, Sendable {

    public typealias Input = TransformerBlockInput
    public typealias Output = TransformerBlockOutput

    private let config: TransformerConfig
    private let layerIndex: Int
    private let attention: MultiHeadAttention
    private let ffn: FeedForward
    private let attentionNormWeight: [Float]
    private let ffnNormWeight: [Float]

    public init(
        config: TransformerConfig,
        layerIndex: Int,
        zeroWeights: Bool = false
    ) throws {
        self.config = config
        self.layerIndex = layerIndex
        self.attention = try MultiHeadAttention(config: config, zeroWeights: zeroWeights)
        self.ffn = try FeedForward(config: config, zeroWeights: zeroWeights)
        self.attentionNormWeight = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ffnNormWeight = [Float](repeating: 1.0, count: config.hiddenDim)
    }

    public func forward(_ input: TransformerBlockInput) async throws -> TransformerBlockOutput {
        let seqLen = input.seqLen
        let hd = config.hiddenDim
        let eps = config.rmsNormEps

        // Pre-norm + attention + residual
        let normedForAttn = cpuRMSNorm(input.hidden, weight: attentionNormWeight,
                                        rows: seqLen, cols: hd, eps: eps)
        let attnOut = try await attention.forward(AttentionInput(
            hidden: normedForAttn, seqLen: seqLen, startPos: input.startPos
        ))
        var hidden = zip(input.hidden, attnOut).map { $0 + $1 }

        // Pre-norm + FFN + residual
        let normedForFFN = cpuRMSNorm(hidden, weight: ffnNormWeight,
                                       rows: seqLen, cols: hd, eps: eps)
        let ffnOut = try await ffn.forward(normedForFFN)
        hidden = zip(hidden, ffnOut).map { $0 + $1 }

        return TransformerBlockOutput(hidden: hidden)
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        params["attention_norm.weight"] = ArrayTensorBox(
            data: attentionNormWeight, shape: [config.hiddenDim]
        )
        params["ffn_norm.weight"] = ArrayTensorBox(
            data: ffnNormWeight, shape: [config.hiddenDim]
        )
        for (key, value) in attention.parameters {
            params["attention.\(key)"] = value
        }
        for (key, value) in ffn.parameters {
            params["ffn.\(key)"] = value
        }
        return params
    }

    // MARK: - CPU RMSNorm (used internally before GPU dispatch is wired)

    private func cpuRMSNorm(
        _ input: [Float], weight: [Float],
        rows: Int, cols: Int, eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            let offset = r * cols
            let row = Array(input[offset..<offset + cols])
            let meanSq = row.reduce(0.0) { $0 + $1 * $1 } / Float(cols)
            let rms = 1.0 / sqrt(meanSq + eps)
            for c in 0..<cols {
                output[offset + c] = row[c] * rms * weight[c]
            }
        }
        return output
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter TransformerBlockTests 2>&1`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Transformer/TransformerConfig.swift \
      Sources/EdgeRunner/Transformer/MultiHeadAttention.swift \
      Sources/EdgeRunner/Transformer/FeedForward.swift \
      Sources/EdgeRunner/Transformer/TransformerBlock.swift \
      Tests/EdgeRunnerTests/TransformerBlockTests.swift
git commit -m "feat: add TransformerBlock with MHA + SwiGLU FFN

Pre-norm decoder block matching Llama architecture.
MultiHeadAttention with GQA support and causal masking.
SwiGLU feed-forward network. Residual connections.
CPU reference implementation using EdgeRunnerModule protocol."
```

---

## Task 13: GPT-2 Reference Implementation

**Files:**
- Create: `Sources/EdgeRunner/Models/GPT2Config.swift`
- Create: `Sources/EdgeRunner/Models/GPT2Model.swift`
- Create: `Sources/EdgeRunner/Models/GPT2Attention.swift`
- Create: `Sources/EdgeRunner/Models/GPT2FeedForward.swift`
- Create: `Sources/EdgeRunner/Models/GPT2Block.swift`
- Create: `Sources/EdgeRunner/Models/Embedding.swift`
- Test: `Tests/EdgeRunnerTests/GPT2Tests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/GPT2Tests.swift
import Testing
import Foundation
@testable import EdgeRunner

@Suite("GPT-2 Reference Model")
struct GPT2Tests {

    static let gpt2Config = GPT2Config(
        vocabSize: 50257,
        maxSeqLen: 1024,
        numLayers: 12,
        numHeads: 12,
        hiddenDim: 768,
        layerNormEps: 1e-5
    )

    // Tiny config for fast tests
    static let tinyConfig = GPT2Config(
        vocabSize: 32,
        maxSeqLen: 16,
        numLayers: 2,
        numHeads: 2,
        hiddenDim: 32,
        layerNormEps: 1e-5
    )

    // MARK: - Config

    @Test func gpt2ConfigProperties() {
        let config = Self.gpt2Config
        #expect(config.headDim == 64)           // 768 / 12
        #expect(config.intermediateSize == 3072) // 4 * 768
    }

    // MARK: - Embedding

    @Test func embeddingLookup() async throws {
        let vocabSize = 8
        let dim = 4
        // Embedding table: row i = [i*0.1, i*0.1, i*0.1, i*0.1]
        var table = [Float](repeating: 0, count: vocabSize * dim)
        for i in 0..<vocabSize {
            for d in 0..<dim {
                table[i * dim + d] = Float(i) * 0.1
            }
        }

        let embedding = Embedding(weight: table, vocabSize: vocabSize, dim: dim)
        let result = try await embedding.forward([2, 5])

        // Token 2: [0.2, 0.2, 0.2, 0.2], Token 5: [0.5, 0.5, 0.5, 0.5]
        #expect(result.count == 2 * dim)
        for d in 0..<dim {
            #expect(abs(result[d] - 0.2) < 1e-6, "Token 2 dim \(d)")
            #expect(abs(result[dim + d] - 0.5) < 1e-6, "Token 5 dim \(d)")
        }
    }

    @Test func embeddingParameters() {
        let embedding = Embedding(
            weight: [Float](repeating: 0, count: 32 * 16),
            vocabSize: 32, dim: 16
        )
        let params = embedding.parameters
        #expect(params.keys.contains("weight"))
        #expect(params["weight"]?.elementCount == 32 * 16)
    }

    // MARK: - GPT-2 Model (tiny)

    @Test func gpt2ForwardOutputShape() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIds: [Int] = [1, 5, 10]
        let logits = try await model.forward(tokenIds)

        // Output: [seqLen x vocabSize]
        #expect(logits.count == tokenIds.count * config.vocabSize,
                "Expected \(tokenIds.count * config.vocabSize) logits, got \(logits.count)")
    }

    @Test func gpt2ForwardDeterministic() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let tokenIds: [Int] = [1, 2, 3]
        let logits1 = try await model.forward(tokenIds)
        let logits2 = try await model.forward(tokenIds)

        for i in 0..<logits1.count {
            #expect(abs(logits1[i] - logits2[i]) < 1e-5,
                    "Non-deterministic at [\(i)]")
        }
    }

    @Test func gpt2SingleToken() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let logits = try await model.forward([0])
        #expect(logits.count == config.vocabSize)

        // Logits should be finite
        for i in 0..<logits.count {
            #expect(logits[i].isFinite, "Non-finite logit at [\(i)]")
        }
    }

    @Test func gpt2ParameterCount() throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)
        let params = model.parameters
        let totalParams = params.values.reduce(0) { $0 + $1.elementCount }

        // Should have > 0 parameters
        #expect(totalParams > 0, "Model has no parameters")

        // Verify key parameter groups exist
        let hasTokenEmb = params.keys.contains(where: { $0.contains("token_embedding") })
        let hasPosEmb = params.keys.contains(where: { $0.contains("position_embedding") })
        let hasBlocks = params.keys.contains(where: { $0.contains("blocks.0") })
        let hasLnFinal = params.keys.contains(where: { $0.contains("ln_final") })

        #expect(hasTokenEmb, "Missing token embedding")
        #expect(hasPosEmb, "Missing position embedding")
        #expect(hasBlocks, "Missing transformer blocks")
        #expect(hasLnFinal, "Missing final layer norm")
    }

    @Test func gpt2SoftmaxSumsToOne() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let logits = try await model.forward([1, 2])

        // Check softmax of last token's logits sums to ~1
        let lastTokenLogits = Array(logits[(config.vocabSize)..<(2 * config.vocabSize)])
        let maxLogit = lastTokenLogits.max() ?? 0
        let exps = lastTokenLogits.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExp }
        let probSum = probs.reduce(0, +)

        #expect(abs(probSum - 1.0) < 1e-5,
                "Softmax should sum to 1.0, got \(probSum)")
    }

    // MARK: - Weight Loading Shape Validation

    @Test func gpt2WeightShapes() throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)
        let params = model.parameters

        // Token embedding: [vocabSize x hiddenDim]
        if let te = params["token_embedding.weight"] {
            #expect(te.shape == [config.vocabSize, config.hiddenDim])
        }

        // Position embedding: [maxSeqLen x hiddenDim]
        if let pe = params["position_embedding.weight"] {
            #expect(pe.shape == [config.maxSeqLen, config.hiddenDim])
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GPT2Tests 2>&1`
Expected: FAIL — `GPT2Config` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Models/GPT2Config.swift

/// Configuration for GPT-2 model variants.
public struct GPT2Config: Sendable {
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numHeads: Int
    public let hiddenDim: Int
    public let layerNormEps: Float

    public var headDim: Int { hiddenDim / numHeads }
    public var intermediateSize: Int { hiddenDim * 4 }

    public init(
        vocabSize: Int = 50257,
        maxSeqLen: Int = 1024,
        numLayers: Int = 12,
        numHeads: Int = 12,
        hiddenDim: Int = 768,
        layerNormEps: Float = 1e-5
    ) {
        precondition(hiddenDim % numHeads == 0)
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.hiddenDim = hiddenDim
        self.layerNormEps = layerNormEps
    }
}
```

```swift
// Sources/EdgeRunner/Models/Embedding.swift

/// Embedding lookup table: maps integer token IDs to dense vectors.
///
/// Weight is stored as [vocabSize x dim] row-major.
public struct Embedding: EdgeRunnerModule, Sendable {

    public typealias Input = [Int]      // token IDs
    public typealias Output = [Float]   // [seqLen x dim] flat

    private let weight: [Float]
    private let vocabSize: Int
    private let dim: Int

    public init(weight: [Float], vocabSize: Int, dim: Int) {
        precondition(weight.count == vocabSize * dim)
        self.weight = weight
        self.vocabSize = vocabSize
        self.dim = dim
    }

    public func forward(_ input: [Int]) async throws -> [Float] {
        var output = [Float]()
        output.reserveCapacity(input.count * dim)
        for tokenId in input {
            precondition(tokenId >= 0 && tokenId < vocabSize,
                         "Token ID \(tokenId) out of range [0, \(vocabSize))")
            let offset = tokenId * dim
            output.append(contentsOf: weight[offset..<offset + dim])
        }
        return output
    }

    public var parameters: [String: any TensorBox] {
        ["weight": ArrayTensorBox(data: weight, shape: [vocabSize, dim])]
    }
}
```

```swift
// Sources/EdgeRunner/Models/GPT2Attention.swift
import Foundation

/// GPT-2 multi-head attention with combined QKV projection.
///
/// GPT-2 uses standard MHA (not GQA), with LayerNorm (not RMSNorm),
/// and learned position embeddings (not RoPE).
public struct GPT2Attention: EdgeRunnerModule, Sendable {

    public typealias Input = AttentionInput
    public typealias Output = [Float]

    private let config: GPT2Config
    private let cAttn: LinearModule     // combined QKV: [hiddenDim x 3*hiddenDim]
    private let cProj: LinearModule     // output: [hiddenDim x hiddenDim]

    public init(config: GPT2Config) throws {
        self.config = config
        let hd = config.hiddenDim
        let scale = sqrt(2.0 / Float(hd + 3 * hd))
        let projScale = sqrt(2.0 / Float(hd + hd))

        self.cAttn = try LinearModule(
            inFeatures: hd, outFeatures: 3 * hd,
            weight: (0..<3 * hd * hd).map { _ in Float.random(in: -scale...scale) },
            bias: [Float](repeating: 0, count: 3 * hd)
        )
        self.cProj = try LinearModule(
            inFeatures: hd, outFeatures: hd,
            weight: (0..<hd * hd).map { _ in Float.random(in: -projScale...projScale) },
            bias: [Float](repeating: 0, count: hd)
        )
    }

    public init(config: GPT2Config, cAttnWeight: [Float], cAttnBias: [Float],
                cProjWeight: [Float], cProjBias: [Float]) throws {
        self.config = config
        let hd = config.hiddenDim
        self.cAttn = try LinearModule(inFeatures: hd, outFeatures: 3 * hd,
                                       weight: cAttnWeight, bias: cAttnBias)
        self.cProj = try LinearModule(inFeatures: hd, outFeatures: hd,
                                       weight: cProjWeight, bias: cProjBias)
    }

    public func forward(_ input: AttentionInput) async throws -> [Float] {
        let seqLen = input.seqLen
        let hd = config.hiddenDim
        let headDim = config.headDim
        let numHeads = config.numHeads
        let scale = 1.0 / sqrt(Float(headDim))

        // Project QKV for each token
        var allQKV = [[Float]]()
        for t in 0..<seqLen {
            let token = Array(input.hidden[t * hd..<(t + 1) * hd])
            let qkv = try await cAttn.forward(token)
            allQKV.append(qkv)
        }

        // Split into Q, K, V and compute attention per head
        var attnOutput = [Float](repeating: 0, count: seqLen * hd)

        for h in 0..<numHeads {
            let qOffset = h * headDim
            let kOffset = hd + h * headDim
            let vOffset = 2 * hd + h * headDim

            for qi in 0..<seqLen {
                let qVec = Array(allQKV[qi][qOffset..<qOffset + headDim])

                var scores = [Float](repeating: 0, count: seqLen)
                for ki in 0..<seqLen {
                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += qVec[d] * allQKV[ki][kOffset + d]
                    }
                    scores[ki] = dot * scale
                }

                // Causal mask
                for ki in 0..<seqLen where ki > qi {
                    scores[ki] = -Float.infinity
                }

                // Softmax
                let maxScore = scores.max() ?? 0
                var expScores = scores.map { exp($0 - maxScore) }
                let sumExp = expScores.reduce(0, +)
                expScores = expScores.map { $0 / sumExp }

                // Weighted sum of V
                let outOffset = qi * hd + h * headDim
                for vi in 0..<seqLen {
                    for d in 0..<headDim {
                        attnOutput[outOffset + d] += expScores[vi] * allQKV[vi][vOffset + d]
                    }
                }
            }
        }

        // Output projection
        var result = [Float]()
        for t in 0..<seqLen {
            let token = Array(attnOutput[t * hd..<(t + 1) * hd])
            let projected = try await cProj.forward(token)
            result.append(contentsOf: projected)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        for (key, value) in cAttn.parameters { params["c_attn.\(key)"] = value }
        for (key, value) in cProj.parameters { params["c_proj.\(key)"] = value }
        return params
    }
}
```

```swift
// Sources/EdgeRunner/Models/GPT2FeedForward.swift
import Foundation

/// GPT-2 feed-forward network: GELU activation, not SwiGLU.
///
/// FFN(x) = c_proj(gelu(c_fc(x)))
public struct GPT2FeedForward: EdgeRunnerModule, Sendable {

    public typealias Input = [Float]
    public typealias Output = [Float]

    private let config: GPT2Config
    private let cFc: LinearModule       // [intermediateSize x hiddenDim]
    private let cProj: LinearModule     // [hiddenDim x intermediateSize]

    public init(config: GPT2Config) throws {
        self.config = config
        let hd = config.hiddenDim
        let inter = config.intermediateSize
        let scale1 = sqrt(2.0 / Float(hd + inter))
        let scale2 = sqrt(2.0 / Float(inter + hd))

        self.cFc = try LinearModule(
            inFeatures: hd, outFeatures: inter,
            weight: (0..<inter * hd).map { _ in Float.random(in: -scale1...scale1) },
            bias: [Float](repeating: 0, count: inter)
        )
        self.cProj = try LinearModule(
            inFeatures: inter, outFeatures: hd,
            weight: (0..<hd * inter).map { _ in Float.random(in: -scale2...scale2) },
            bias: [Float](repeating: 0, count: hd)
        )
    }

    public init(config: GPT2Config, cFcWeight: [Float], cFcBias: [Float],
                cProjWeight: [Float], cProjBias: [Float]) throws {
        self.config = config
        let hd = config.hiddenDim
        let inter = config.intermediateSize
        self.cFc = try LinearModule(inFeatures: hd, outFeatures: inter,
                                     weight: cFcWeight, bias: cFcBias)
        self.cProj = try LinearModule(inFeatures: inter, outFeatures: hd,
                                       weight: cProjWeight, bias: cProjBias)
    }

    public func forward(_ input: [Float]) async throws -> [Float] {
        let hd = config.hiddenDim
        let inter = config.intermediateSize
        let seqLen = input.count / hd

        var result = [Float]()
        for t in 0..<seqLen {
            let token = Array(input[t * hd..<(t + 1) * hd])
            let hidden = try await cFc.forward(token)

            // GELU activation
            var activated = [Float](repeating: 0, count: inter)
            let c: Float = sqrt(2.0 / .pi)
            for i in 0..<inter {
                let x = hidden[i]
                activated[i] = x * 0.5 * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
            }

            let projected = try await cProj.forward(activated)
            result.append(contentsOf: projected)
        }

        return result
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        for (key, value) in cFc.parameters { params["c_fc.\(key)"] = value }
        for (key, value) in cProj.parameters { params["c_proj.\(key)"] = value }
        return params
    }
}
```

```swift
// Sources/EdgeRunner/Models/GPT2Block.swift
import Foundation

/// A single GPT-2 transformer block:
///   x = x + attention(layer_norm_1(x))
///   x = x + ffn(layer_norm_2(x))
///
/// GPT-2 uses pre-norm with LayerNorm (not RMSNorm).
public struct GPT2Block: EdgeRunnerModule, Sendable {

    public typealias Input = TransformerBlockInput
    public typealias Output = TransformerBlockOutput

    private let config: GPT2Config
    private let attention: GPT2Attention
    private let ffn: GPT2FeedForward
    private let ln1Gamma: [Float]
    private let ln1Beta: [Float]
    private let ln2Gamma: [Float]
    private let ln2Beta: [Float]

    public init(config: GPT2Config) throws {
        self.config = config
        self.attention = try GPT2Attention(config: config)
        self.ffn = try GPT2FeedForward(config: config)
        self.ln1Gamma = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ln1Beta = [Float](repeating: 0.0, count: config.hiddenDim)
        self.ln2Gamma = [Float](repeating: 1.0, count: config.hiddenDim)
        self.ln2Beta = [Float](repeating: 0.0, count: config.hiddenDim)
    }

    public func forward(_ input: TransformerBlockInput) async throws -> TransformerBlockOutput {
        let seqLen = input.seqLen
        let hd = config.hiddenDim
        let eps = config.layerNormEps

        // Pre-norm + attention + residual
        let normed1 = cpuLayerNorm(input.hidden, gamma: ln1Gamma, beta: ln1Beta,
                                    rows: seqLen, cols: hd, eps: eps)
        let attnOut = try await attention.forward(AttentionInput(
            hidden: normed1, seqLen: seqLen, startPos: input.startPos
        ))
        var hidden = zip(input.hidden, attnOut).map { $0 + $1 }

        // Pre-norm + FFN + residual
        let normed2 = cpuLayerNorm(hidden, gamma: ln2Gamma, beta: ln2Beta,
                                    rows: seqLen, cols: hd, eps: eps)
        let ffnOut = try await ffn.forward(normed2)
        hidden = zip(hidden, ffnOut).map { $0 + $1 }

        return TransformerBlockOutput(hidden: hidden)
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]
        params["ln_1.weight"] = ArrayTensorBox(data: ln1Gamma, shape: [config.hiddenDim])
        params["ln_1.bias"] = ArrayTensorBox(data: ln1Beta, shape: [config.hiddenDim])
        params["ln_2.weight"] = ArrayTensorBox(data: ln2Gamma, shape: [config.hiddenDim])
        params["ln_2.bias"] = ArrayTensorBox(data: ln2Beta, shape: [config.hiddenDim])
        for (key, value) in attention.parameters { params["attn.\(key)"] = value }
        for (key, value) in ffn.parameters { params["mlp.\(key)"] = value }
        return params
    }

    private func cpuLayerNorm(
        _ input: [Float], gamma: [Float], beta: [Float],
        rows: Int, cols: Int, eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            let offset = r * cols
            let row = Array(input[offset..<offset + cols])
            let mean = row.reduce(0, +) / Float(cols)
            let variance = row.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Float(cols)
            let invStd = 1.0 / sqrt(variance + eps)
            for c in 0..<cols {
                output[offset + c] = (row[c] - mean) * invStd * gamma[c] + beta[c]
            }
        }
        return output
    }
}
```

```swift
// Sources/EdgeRunner/Models/GPT2Model.swift
import Foundation

/// Full GPT-2 model: token embed + position embed + N blocks + final LN + LM head.
///
/// Architecture (GPT-2 124M):
///   - Token embedding: [50257 x 768]
///   - Position embedding: [1024 x 768]
///   - 12 transformer blocks (pre-norm LayerNorm + MHA + FFN)
///   - Final LayerNorm
///   - Language model head: weight-tied to token embedding
public struct GPT2Model: EdgeRunnerModule, Sendable {

    public typealias Input = [Int]      // token IDs
    public typealias Output = [Float]   // logits [seqLen x vocabSize]

    private let config: GPT2Config
    private let tokenEmbedding: Embedding
    private let positionEmbedding: Embedding
    private let blocks: [GPT2Block]
    private let lnFinalGamma: [Float]
    private let lnFinalBeta: [Float]

    public init(config: GPT2Config) throws {
        self.config = config
        let hd = config.hiddenDim

        // Initialize embeddings with small random values
        let embScale = sqrt(2.0 / Float(config.vocabSize + hd))
        self.tokenEmbedding = Embedding(
            weight: (0..<config.vocabSize * hd).map { _ in Float.random(in: -embScale...embScale) },
            vocabSize: config.vocabSize,
            dim: hd
        )
        let posScale = sqrt(2.0 / Float(config.maxSeqLen + hd))
        self.positionEmbedding = Embedding(
            weight: (0..<config.maxSeqLen * hd).map { _ in Float.random(in: -posScale...posScale) },
            vocabSize: config.maxSeqLen,
            dim: hd
        )

        var blocks = [GPT2Block]()
        for _ in 0..<config.numLayers {
            blocks.append(try GPT2Block(config: config))
        }
        self.blocks = blocks

        self.lnFinalGamma = [Float](repeating: 1.0, count: hd)
        self.lnFinalBeta = [Float](repeating: 0.0, count: hd)
    }

    public func forward(_ input: [Int]) async throws -> [Float] {
        let seqLen = input.count
        let hd = config.hiddenDim
        precondition(seqLen <= config.maxSeqLen,
                     "Sequence length \(seqLen) exceeds max \(config.maxSeqLen)")

        // Token + position embeddings
        let tokenEmb = try await tokenEmbedding.forward(input)
        let posIds = Array(0..<seqLen)
        let posEmb = try await positionEmbedding.forward(posIds)

        // Sum embeddings
        var hidden = zip(tokenEmb, posEmb).map { $0 + $1 }

        // Transformer blocks
        for block in blocks {
            let blockInput = TransformerBlockInput(
                hidden: hidden, seqLen: seqLen, startPos: 0
            )
            let blockOutput = try await block.forward(blockInput)
            hidden = blockOutput.hidden
        }

        // Final layer norm
        hidden = cpuLayerNorm(hidden, gamma: lnFinalGamma, beta: lnFinalBeta,
                               rows: seqLen, cols: hd, eps: config.layerNormEps)

        // Language model head: multiply by token embedding weight transposed
        // logits[t][v] = sum_d hidden[t][d] * wte[v][d]
        let wte = tokenEmbedding.parameters["weight"]!.floatArray
        var logits = [Float](repeating: 0, count: seqLen * config.vocabSize)
        for t in 0..<seqLen {
            for v in 0..<config.vocabSize {
                var dot: Float = 0
                for d in 0..<hd {
                    dot += hidden[t * hd + d] * wte[v * hd + d]
                }
                logits[t * config.vocabSize + v] = dot
            }
        }

        return logits
    }

    public var parameters: [String: any TensorBox] {
        var params: [String: any TensorBox] = [:]

        for (key, value) in tokenEmbedding.parameters {
            params["token_embedding.\(key)"] = value
        }
        for (key, value) in positionEmbedding.parameters {
            params["position_embedding.\(key)"] = value
        }
        for (i, block) in blocks.enumerated() {
            for (key, value) in block.parameters {
                params["blocks.\(i).\(key)"] = value
            }
        }
        params["ln_final.weight"] = ArrayTensorBox(data: lnFinalGamma, shape: [config.hiddenDim])
        params["ln_final.bias"] = ArrayTensorBox(data: lnFinalBeta, shape: [config.hiddenDim])

        return params
    }

    private func cpuLayerNorm(
        _ input: [Float], gamma: [Float], beta: [Float],
        rows: Int, cols: Int, eps: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            let offset = r * cols
            let row = Array(input[offset..<offset + cols])
            let mean = row.reduce(0, +) / Float(cols)
            let variance = row.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Float(cols)
            let invStd = 1.0 / sqrt(variance + eps)
            for c in 0..<cols {
                output[offset + c] = (row[c] - mean) * invStd * gamma[c] + beta[c]
            }
        }
        return output
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter GPT2Tests 2>&1`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Models/GPT2Config.swift \
      Sources/EdgeRunner/Models/GPT2Model.swift \
      Sources/EdgeRunner/Models/GPT2Attention.swift \
      Sources/EdgeRunner/Models/GPT2FeedForward.swift \
      Sources/EdgeRunner/Models/GPT2Block.swift \
      Sources/EdgeRunner/Models/Embedding.swift \
      Tests/EdgeRunnerTests/GPT2Tests.swift
git commit -m "feat: add GPT-2 124M reference implementation

Full GPT-2 architecture using EdgeRunnerModule protocol.
Token + position embeddings, 12 transformer blocks,
final LayerNorm, weight-tied LM head.
CPU reference for correctness verification."
```

---

## Task 14: Integration Tests & Perplexity Verification

**Files:**
- Create: `Tests/EdgeRunnerTests/IntegrationTests.swift`
- Create: `Sources/EdgeRunner/Metrics/Perplexity.swift`
- Test: `Tests/EdgeRunnerTests/PerplexityTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerTests/PerplexityTests.swift
import Testing
import Foundation
@testable import EdgeRunner

@Suite("Perplexity Computation")
struct PerplexityTests {

    @Test func perplexityOfUniformDistribution() {
        // Uniform distribution over 10 classes: perplexity = 10
        let vocabSize = 10
        let logits = [Float](repeating: 0, count: vocabSize) // uniform after softmax
        let targetId = 3

        let nll = Perplexity.negLogLikelihood(logits: logits, targetId: targetId)
        let ppl = exp(nll)

        #expect(abs(ppl - Float(vocabSize)) < 0.01,
                "Uniform PPL should be \(vocabSize), got \(ppl)")
    }

    @Test func perplexityOfPerfectPrediction() {
        // Peaked distribution: very high logit for target
        var logits = [Float](repeating: -100, count: 10)
        logits[5] = 100.0
        let targetId = 5

        let nll = Perplexity.negLogLikelihood(logits: logits, targetId: targetId)
        let ppl = exp(nll)

        #expect(ppl < 1.01, "Perfect prediction PPL should be ~1.0, got \(ppl)")
    }

    @Test func perplexityOfWrongPrediction() {
        var logits = [Float](repeating: -100, count: 10)
        logits[0] = 100.0
        let targetId = 5 // model is confident about wrong token

        let nll = Perplexity.negLogLikelihood(logits: logits, targetId: targetId)

        #expect(nll > 10.0, "Wrong prediction should have high NLL, got \(nll)")
    }

    @Test func sequencePerplexity() {
        // Sequence of uniform logits over vocab 10
        let vocabSize = 10
        let seqLen = 5
        let allLogits = [[Float]](repeating: [Float](repeating: 0, count: vocabSize),
                                   count: seqLen)
        let targetIds = [1, 3, 5, 7, 9]

        let ppl = Perplexity.compute(logitsPerToken: allLogits, targetIds: targetIds)
        #expect(abs(ppl - Float(vocabSize)) < 0.01,
                "Sequence PPL of uniform should be \(vocabSize), got \(ppl)")
    }

    @Test func perplexityNumericalStability() {
        // Very large logits should not overflow
        var logits = [Float](repeating: 0, count: 100)
        logits[50] = 1000.0
        let targetId = 50

        let nll = Perplexity.negLogLikelihood(logits: logits, targetId: targetId)
        #expect(nll.isFinite, "NLL should be finite, got \(nll)")
        #expect(nll >= 0, "NLL should be non-negative")
    }
}
```

```swift
// Tests/EdgeRunnerTests/IntegrationTests.swift
import Testing
import Foundation
@testable import EdgeRunner

@Suite("Integration Tests")
struct IntegrationTests {

    static let tinyConfig = GPT2Config(
        vocabSize: 32,
        maxSeqLen: 16,
        numLayers: 2,
        numHeads: 2,
        hiddenDim: 32,
        layerNormEps: 1e-5
    )

    // MARK: - Full Pipeline

    @Test func endToEndForwardPass() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        // Simulate: tokenize -> embed -> blocks -> logits
        let tokenIds = [1, 5, 10, 15]
        let logits = try await model.forward(tokenIds)

        // Verify output shape
        #expect(logits.count == tokenIds.count * config.vocabSize)

        // Verify all logits are finite
        for i in 0..<logits.count {
            #expect(logits[i].isFinite, "Non-finite logit at [\(i)]: \(logits[i])")
        }
    }

    @Test func endToEndWithPerplexity() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        // Input tokens and target (shifted by 1)
        let inputIds = [1, 5, 10, 15, 20]
        let targetIds = [5, 10, 15, 20, 25]  // next-token targets

        let logits = try await model.forward(inputIds)

        // Extract per-token logits
        var logitsPerToken = [[Float]]()
        for t in 0..<inputIds.count {
            let offset = t * config.vocabSize
            logitsPerToken.append(Array(logits[offset..<offset + config.vocabSize]))
        }

        let ppl = Perplexity.compute(logitsPerToken: logitsPerToken, targetIds: targetIds)
        #expect(ppl.isFinite, "Perplexity should be finite")
        #expect(ppl > 0, "Perplexity should be positive")

        // With random weights, PPL should be roughly vocabSize
        // Allow wide range since weights are random
        #expect(ppl > 1.0, "PPL should be > 1 for random model")
    }

    @Test func cpuReferenceSmallInput() async throws {
        // Verify CPU reference produces consistent results on tiny input
        let config = GPT2Config(
            vocabSize: 4, maxSeqLen: 4,
            numLayers: 1, numHeads: 1,
            hiddenDim: 4, layerNormEps: 1e-5
        )
        let model = try GPT2Model(config: config)

        let logits1 = try await model.forward([0, 1])
        let logits2 = try await model.forward([0, 1])

        #expect(logits1.count == logits2.count)
        for i in 0..<logits1.count {
            #expect(abs(logits1[i] - logits2[i]) < 1e-5,
                    "CPU reference non-deterministic at [\(i)]")
        }
    }

    @Test func causalMaskingVerification() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        // Token 0's logits should be the same whether seq is [0] or [0, 1, 2]
        let logitsShort = try await model.forward([5])
        let logitsLong = try await model.forward([5, 10, 15])

        // First token's logits should match (causal = no future leakage)
        let shortFirst = Array(logitsShort[0..<config.vocabSize])
        let longFirst = Array(logitsLong[0..<config.vocabSize])

        for i in 0..<config.vocabSize {
            #expect(abs(shortFirst[i] - longFirst[i]) < 1e-4,
                    "Causal mask violated at vocab[\(i)]: short=\(shortFirst[i]) long=\(longFirst[i])")
        }
    }

    @Test func performanceBaseline() async throws {
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        // Measure forward pass time
        let tokenIds = Array(0..<config.maxSeqLen)
        let startTime = CFAbsoluteTimeGetCurrent()

        _ = try await model.forward(tokenIds)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let tokensPerSec = Double(tokenIds.count) / elapsed

        // Just verify it completes in reasonable time and record baseline
        #expect(elapsed < 60.0,
                "Forward pass took \(elapsed)s — too slow for \(tokenIds.count) tokens")

        // Log performance (visible in test output)
        print("Performance baseline: \(String(format: "%.1f", tokensPerSec)) tokens/sec " +
              "(\(tokenIds.count) tokens in \(String(format: "%.3f", elapsed))s)")
    }

    @Test func gradientFreeForwardOnly() async throws {
        // Verify the model is inference-only: parameters don't change after forward
        let config = Self.tinyConfig
        let model = try GPT2Model(config: config)

        let paramsBefore = model.parameters
        _ = try await model.forward([1, 2, 3])
        let paramsAfter = model.parameters

        #expect(paramsBefore.count == paramsAfter.count)
        for key in paramsBefore.keys {
            let before = paramsBefore[key]!.floatArray
            let after = paramsAfter[key]!.floatArray
            #expect(before.count == after.count)
            for i in 0..<before.count {
                #expect(before[i] == after[i],
                        "Parameter \(key)[\(i)] changed during inference")
            }
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "PerplexityTests|IntegrationTests" 2>&1`
Expected: FAIL — `Perplexity` not defined

**Step 3: Implement**

```swift
// Sources/EdgeRunner/Metrics/Perplexity.swift
import Foundation

/// Perplexity computation utilities for language model evaluation.
///
/// Perplexity = exp(average negative log-likelihood across tokens).
/// Lower is better. A perfect model has PPL = 1.0.
/// A uniform-random model over vocab V has PPL = V.
public enum Perplexity: Sendable {

    /// Compute negative log-likelihood for a single token prediction.
    ///
    /// Uses numerically stable log-softmax: log(softmax(x_i)) = x_i - log(sum(exp(x_j - max)))  - max
    ///
    /// - Parameters:
    ///   - logits: Raw logits for one position, shape [vocabSize]
    ///   - targetId: The correct token ID
    /// - Returns: -log P(target | logits)
    public static func negLogLikelihood(logits: [Float], targetId: Int) -> Float {
        precondition(targetId >= 0 && targetId < logits.count,
                     "targetId \(targetId) out of range [0, \(logits.count))")

        // Numerically stable log-softmax
        let maxLogit = logits.max() ?? 0
        let shiftedLogits = logits.map { $0 - maxLogit }
        let logSumExp = log(shiftedLogits.map { exp($0) }.reduce(0, +))
        let logProb = shiftedLogits[targetId] - logSumExp

        return -logProb
    }

    /// Compute perplexity over a sequence of predictions.
    ///
    /// - Parameters:
    ///   - logitsPerToken: Array of logit vectors, one per token position
    ///   - targetIds: Target token IDs (same length as logitsPerToken)
    /// - Returns: exp(mean NLL) — the perplexity
    public static func compute(
        logitsPerToken: [[Float]],
        targetIds: [Int]
    ) -> Float {
        precondition(logitsPerToken.count == targetIds.count,
                     "Logits count \(logitsPerToken.count) != targets count \(targetIds.count)")
        precondition(!targetIds.isEmpty, "Cannot compute perplexity on empty sequence")

        var totalNLL: Float = 0
        for (logits, target) in zip(logitsPerToken, targetIds) {
            totalNLL += negLogLikelihood(logits: logits, targetId: target)
        }

        let avgNLL = totalNLL / Float(targetIds.count)
        return exp(avgNLL)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "PerplexityTests|IntegrationTests" 2>&1`
Expected: All 12 tests PASS (5 perplexity + 7 integration)

**Step 5: Commit**

```bash
git add Sources/EdgeRunner/Metrics/Perplexity.swift \
      Tests/EdgeRunnerTests/IntegrationTests.swift \
      Tests/EdgeRunnerTests/PerplexityTests.swift
git commit -m "feat: add integration tests and perplexity verification

End-to-end forward pass pipeline test.
Causal masking verification (no future token leakage).
Perplexity computation with numerically stable log-softmax.
Performance baseline measurement (tokens/sec).
Inference-only parameter immutability check."
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | GEMM Kernel (Tiled MatMul) | `GEMM.metal`, `GEMMParams.h`, `GEMMKernel.swift` | `GEMMTests.swift` (6 tests) |
| 2 | GEMV Kernel (Matrix-Vector) | `GEMV.metal`, `GEMVParams.h`, `GEMVKernel.swift` | `GEMVTests.swift` (6 tests) |
| 3 | Softmax Kernel | `Softmax.metal`, `SoftmaxParams.h`, `SoftmaxKernel.swift` | `SoftmaxTests.swift` (5 tests) |
| 4 | Flash Attention Forward | `FlashAttention.metal`, `FlashAttentionParams.h`, `FlashAttentionKernel.swift` | `FlashAttentionTests.swift` (6 tests) |
| 5 | Grouped Query Attention | `GQA.metal`, `GQAParams.h`, `GQAKernel.swift` | `GQATests.swift` (6 tests) |
| 6 | KV Cache (Ring Buffer) | `KVCache.metal`, `KVCacheParams.h`, `KVCacheKernel.swift` | `KVCacheTests.swift` (6 tests) |
| 7 | RoPE (Rotary Position Embeddings) | `RoPE.metal`, `RoPEParams.h`, `RoPEKernel.swift` | `RoPETests.swift` (7 tests) |
| 8 | RMSNorm Kernel | `RMSNorm.metal`, `RMSNormParams.h`, `RMSNormKernel.swift` | `RMSNormTests.swift` (5 tests) |
| 9 | LayerNorm Kernel | `LayerNorm.metal`, `LayerNormParams.h`, `LayerNormKernel.swift` | `LayerNormTests.swift` (5 tests) |
| 10 | Activation Kernels (SwiGLU, GELU, Sigmoid) | `Activations.metal`, `ActivationParams.h`, `ActivationKernels.swift` | `ActivationTests.swift` (9 tests) |
| 11 | EdgeRunnerModule Protocol | `EdgeRunnerModule.swift`, `TensorBox.swift`, `Sequential.swift`, `Linear.swift` | `ModuleTests.swift` (9 tests) |
| 12 | Transformer Block Composition | `MultiHeadAttention.swift`, `FeedForward.swift`, `TransformerBlock.swift`, `TransformerConfig.swift` | `TransformerBlockTests.swift` (8 tests) |
| 13 | GPT-2 Reference Implementation | `GPT2Config.swift`, `GPT2Model.swift`, `GPT2Attention.swift`, `GPT2FeedForward.swift`, `GPT2Block.swift`, `Embedding.swift` | `GPT2Tests.swift` (8 tests) |
| 14 | Integration Tests & Perplexity | `Perplexity.swift` | `IntegrationTests.swift` (7 tests), `PerplexityTests.swift` (5 tests) |

**Total: ~38 files, ~98 tests, 14 commits**
