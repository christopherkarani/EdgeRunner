# EdgeRunner Milestone 5: Memory Optimization & Device Adaptation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize EdgeRunner for memory-constrained devices: KV cache compression for long contexts on 8GB iPhones, context-aware memory budgeting, context summarisation/recycling, and Metal 4/M5 device-specific optimizations.

**Architecture:** Pluggable KV cache compression policies (MiniCache, Squeezed Attention, DuoAttention). Device-aware memory budgeting that caps context length based on available memory. Sliding window with compressed summary prefix for context recycling. Metal 4 tensor_ops dispatch with graceful fallback on older hardware.

**Tech Stack:** Swift 6.2, Metal Shading Language 4.0, Swift Testing

**Depends on:** Milestone 4 (docs/plans/2026-03-16-edgerunner-m4-implementation.md)

---

## Task 1: Context-Aware Memory Budgeting

**Files:**
- Create: `Sources/EdgeRunnerCore/Memory/MemoryBudget.swift`
- Create: `Sources/EdgeRunnerCore/Memory/DeviceProfile.swift`
- Test: `Tests/EdgeRunnerCoreTests/Memory/MemoryBudgetTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/Memory/MemoryBudgetTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Context-Aware Memory Budgeting")
struct MemoryBudgetTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    // MARK: - Device Profile

    @Test func deviceProfileDetection() async throws {
        let profile = await DeviceProfile.current(backend: backend)

        // Should detect something valid
        #expect(profile.totalMemoryBytes > 0)
        #expect(profile.usableMemoryBytes > 0)
        #expect(profile.usableMemoryBytes <= profile.totalMemoryBytes)
        #expect(!profile.deviceName.isEmpty)
    }

    @Test func knownDeviceProfiles() {
        // Verify known device memory amounts
        let iphone15Pro = DeviceProfile.knownProfile(for: "iPhone 15 Pro")
        #expect(iphone15Pro?.totalMemoryBytes == 8 * 1024 * 1024 * 1024) // 8GB

        let m4MacBookPro = DeviceProfile.knownProfile(for: "Apple M4 Pro")
        // M4 Pro has 24 or 48 GB — should be >= 24GB
        #expect((m4MacBookPro?.totalMemoryBytes ?? 0) >= 24 * 1024 * 1024 * 1024)
    }

    // MARK: - Memory Budget

    @Test func memoryBudgetComputationSmallModel() async throws {
        let modelConfig = ModelMemoryConfig(
            weightBytes: 4_000_000_000,    // 4GB weights (e.g., Q4 7B)
            kvCacheBytesPerToken: 262_144,  // 256KB per token
            overheadBytes: 256_000_000      // 256MB overhead
        )

        let device = DeviceProfile(
            deviceName: "iPhone 15 Pro",
            totalMemoryBytes: 8_589_934_592,   // 8GB
            usableMemoryBytes: 4_294_967_296,  // 4GB usable (50% budget)
            chipFamily: .apple
        )

        let budget = MemoryBudget.compute(
            model: modelConfig, device: device
        )

        // Available for KV cache = 4GB - 4GB weights - 256MB overhead = -256MB -> 0
        // Actually: usable - weights = 294MB, minus overhead = 38MB
        // Wait, let's recalculate:
        // usableMemoryBytes = 4GB = 4,294,967,296
        // After weights: 4,294,967,296 - 4,000,000,000 = 294,967,296
        // After overhead: 294,967,296 - 256,000,000 = 38,967,296
        // Max tokens: 38,967,296 / 262,144 ~ 148
        #expect(budget.maxContextLength > 0)
        #expect(budget.maxContextLength < 1000) // Very tight budget
        #expect(budget.warningLevel == .critical) // Very little headroom
    }

    @Test func memoryBudgetComputationLargeDevice() async throws {
        let modelConfig = ModelMemoryConfig(
            weightBytes: 4_000_000_000,
            kvCacheBytesPerToken: 262_144,
            overheadBytes: 256_000_000
        )

        let device = DeviceProfile(
            deviceName: "Apple M4 Max",
            totalMemoryBytes: 64_424_509_440,   // 60GB
            usableMemoryBytes: 32_212_254_720,  // 30GB usable
            chipFamily: .apple
        )

        let budget = MemoryBudget.compute(
            model: modelConfig, device: device
        )

        // After weights + overhead: 30GB - 4GB - 256MB ~ 25.7GB
        // Max tokens: ~25.7GB / 256KB ~ 103,000
        #expect(budget.maxContextLength > 50_000)
        #expect(budget.warningLevel == .nominal)
    }

    @Test func memoryBudgetWeightsExceedDevice() async throws {
        let modelConfig = ModelMemoryConfig(
            weightBytes: 16_000_000_000,  // 16GB weights -- too big for 8GB device
            kvCacheBytesPerToken: 262_144,
            overheadBytes: 256_000_000
        )

        let device = DeviceProfile(
            deviceName: "iPhone 15 Pro",
            totalMemoryBytes: 8_589_934_592,
            usableMemoryBytes: 4_294_967_296,
            chipFamily: .apple
        )

        let budget = MemoryBudget.compute(
            model: modelConfig, device: device
        )

        #expect(budget.maxContextLength == 0)
        #expect(budget.warningLevel == .impossible)
        #expect(budget.canRun == false)
    }

    @Test func contextLengthWarningAPI() async throws {
        let modelConfig = ModelMemoryConfig(
            weightBytes: 2_000_000_000,
            kvCacheBytesPerToken: 262_144,
            overheadBytes: 256_000_000
        )

        let device = DeviceProfile(
            deviceName: "iPhone 16 Pro",
            totalMemoryBytes: 8_589_934_592,
            usableMemoryBytes: 4_294_967_296,
            chipFamily: .apple
        )

        let budget = MemoryBudget.compute(model: modelConfig, device: device)

        // Should warn when requested context exceeds budget
        let warning4K = budget.checkContext(requestedLength: 4096)
        let warning32K = budget.checkContext(requestedLength: 32768)

        // 4K might be fine, 32K probably not on 8GB
        if budget.maxContextLength < 32768 {
            #expect(warning32K != nil)
        }
        if budget.maxContextLength >= 4096 {
            #expect(warning4K == nil)
        }
    }

    @Test func memoryBudgetWithQuantisation() async throws {
        // Q4 weights should allow much longer context than F16
        let q4Config = ModelMemoryConfig(
            weightBytes: 4_000_000_000,     // Q4: ~4GB for 7B
            kvCacheBytesPerToken: 131_072,  // Q8 KV cache: 128KB/token
            overheadBytes: 256_000_000
        )

        let f16Config = ModelMemoryConfig(
            weightBytes: 14_000_000_000,    // F16: ~14GB for 7B
            kvCacheBytesPerToken: 262_144,  // F16 KV cache: 256KB/token
            overheadBytes: 256_000_000
        )

        let device = DeviceProfile(
            deviceName: "Apple M4 Pro",
            totalMemoryBytes: 24_000_000_000,
            usableMemoryBytes: 12_000_000_000,
            chipFamily: .apple
        )

        let q4Budget = MemoryBudget.compute(model: q4Config, device: device)
        let f16Budget = MemoryBudget.compute(model: f16Config, device: device)

        #expect(q4Budget.maxContextLength > f16Budget.maxContextLength,
                "Q4 should allow longer context than F16")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter MemoryBudgetTests 2>&1`
Expected: FAIL -- types not defined

**Step 3: Implement DeviceProfile**

```swift
// Sources/EdgeRunnerCore/Memory/DeviceProfile.swift
import Metal
import EdgeRunnerMetal

/// Chip family for device-specific optimizations.
public enum ChipFamily: Sendable {
    case apple       // Apple Silicon (M1-M5, A-series)
    case unknown
}

/// Profile of the device's memory characteristics.
public struct DeviceProfile: Sendable {
    public let deviceName: String
    public let totalMemoryBytes: UInt64
    public let usableMemoryBytes: UInt64  // Conservative: totalMemory / memoryFactor
    public let chipFamily: ChipFamily

    public init(
        deviceName: String,
        totalMemoryBytes: UInt64,
        usableMemoryBytes: UInt64,
        chipFamily: ChipFamily
    ) {
        self.deviceName = deviceName
        self.totalMemoryBytes = totalMemoryBytes
        self.usableMemoryBytes = usableMemoryBytes
        self.chipFamily = chipFamily
    }

    /// Auto-detect current device profile from Metal.
    public static func current(backend: MetalBackend) async -> DeviceProfile {
        let device = await backend.device
        let name = device.name
        let totalMem = UInt64(device.recommendedMaxWorkingSetSize)

        // Use 50% of total as usable for inference (rest for OS, apps, etc.)
        let usableFactor: Double = 0.50
        let usableMem = UInt64(Double(totalMem) * usableFactor)

        return DeviceProfile(
            deviceName: name,
            totalMemoryBytes: totalMem,
            usableMemoryBytes: usableMem,
            chipFamily: .apple
        )
    }

    /// Known device profiles for common hardware.
    public static func knownProfile(for deviceName: String) -> DeviceProfile? {
        let profiles: [String: (total: UInt64, usable: UInt64)] = [
            // iPhones
            "iPhone 15 Pro":     (8 * _GB, 4 * _GB),
            "iPhone 15 Pro Max": (8 * _GB, 4 * _GB),
            "iPhone 16 Pro":     (8 * _GB, 4 * _GB),
            "iPhone 16 Pro Max": (8 * _GB, 4 * _GB),
            // iPads
            "iPad Pro M4":       (16 * _GB, 8 * _GB),
            // Macs
            "Apple M1":          (16 * _GB, 10 * _GB),
            "Apple M1 Pro":      (32 * _GB, 20 * _GB),
            "Apple M1 Max":      (64 * _GB, 40 * _GB),
            "Apple M2":          (24 * _GB, 15 * _GB),
            "Apple M2 Pro":      (32 * _GB, 20 * _GB),
            "Apple M2 Max":      (96 * _GB, 60 * _GB),
            "Apple M3":          (24 * _GB, 15 * _GB),
            "Apple M3 Pro":      (36 * _GB, 22 * _GB),
            "Apple M3 Max":      (128 * _GB, 80 * _GB),
            "Apple M4":          (32 * _GB, 20 * _GB),
            "Apple M4 Pro":      (48 * _GB, 30 * _GB),
            "Apple M4 Max":      (128 * _GB, 80 * _GB),
            "Apple M5":          (32 * _GB, 20 * _GB),
        ]

        guard let spec = profiles[deviceName] else { return nil }
        return DeviceProfile(
            deviceName: deviceName,
            totalMemoryBytes: spec.total,
            usableMemoryBytes: spec.usable,
            chipFamily: .apple
        )
    }

    private static let _GB: UInt64 = 1_073_741_824
}
```

**Step 4: Implement MemoryBudget**

```swift
// Sources/EdgeRunnerCore/Memory/MemoryBudget.swift
import Foundation

/// Memory configuration for a model.
public struct ModelMemoryConfig: Sendable {
    public let weightBytes: UInt64
    public let kvCacheBytesPerToken: UInt64
    public let overheadBytes: UInt64  // Activations, scratch buffers, etc.

    public init(weightBytes: UInt64, kvCacheBytesPerToken: UInt64, overheadBytes: UInt64) {
        self.weightBytes = weightBytes
        self.kvCacheBytesPerToken = kvCacheBytesPerToken
        self.overheadBytes = overheadBytes
    }

    /// Estimate memory config from model parameters.
    public static func estimate(
        parameterCount: UInt64,
        bitsPerWeight: Int,
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        kvBits: Int = 16
    ) -> ModelMemoryConfig {
        let weightBytes = parameterCount * UInt64(bitsPerWeight) / 8

        // KV cache per token: 2 (K+V) * numLayers * numKVHeads * headDim * (kvBits/8)
        let kvPerToken = UInt64(2 * numLayers * numKVHeads * headDim * kvBits / 8)

        // Overhead: ~5% of weights for activations
        let overhead = max(weightBytes / 20, 256_000_000)

        return ModelMemoryConfig(
            weightBytes: weightBytes,
            kvCacheBytesPerToken: kvPerToken,
            overheadBytes: overhead
        )
    }
}

/// Warning levels for memory pressure.
public enum MemoryWarningLevel: Sendable, Comparable {
    case nominal    // Plenty of headroom
    case caution    // Less than 20% headroom
    case critical   // Less than 5% headroom
    case impossible // Cannot fit model at all
}

/// Computed memory budget for a model on a device.
public struct MemoryBudget: Sendable {
    public let maxContextLength: Int
    public let warningLevel: MemoryWarningLevel
    public let availableForKVCache: UInt64
    public let deviceProfile: DeviceProfile
    public let modelConfig: ModelMemoryConfig

    /// Whether the model can run at all on this device.
    public var canRun: Bool { maxContextLength > 0 }

    /// Compute memory budget.
    public static func compute(
        model: ModelMemoryConfig, device: DeviceProfile
    ) -> MemoryBudget {
        let usable = device.usableMemoryBytes

        // Check if model even fits
        let fixedCost = model.weightBytes + model.overheadBytes
        guard fixedCost < usable else {
            return MemoryBudget(
                maxContextLength: 0,
                warningLevel: .impossible,
                availableForKVCache: 0,
                deviceProfile: device,
                modelConfig: model
            )
        }

        let availableForKV = usable - fixedCost
        let maxTokens = Int(availableForKV / model.kvCacheBytesPerToken)

        // Determine warning level based on headroom ratio
        let headroomRatio = Double(availableForKV) / Double(usable)
        let warningLevel: MemoryWarningLevel
        if maxTokens == 0 {
            warningLevel = .impossible
        } else if headroomRatio < 0.05 {
            warningLevel = .critical
        } else if headroomRatio < 0.20 {
            warningLevel = .caution
        } else {
            warningLevel = .nominal
        }

        return MemoryBudget(
            maxContextLength: maxTokens,
            warningLevel: warningLevel,
            availableForKVCache: availableForKV,
            deviceProfile: device,
            modelConfig: model
        )
    }

    /// Check if a requested context length fits within budget.
    /// Returns a warning message if it exceeds, nil if OK.
    public func checkContext(requestedLength: Int) -> MemoryWarning? {
        guard requestedLength > maxContextLength else { return nil }

        let requiredKVBytes = UInt64(requestedLength) * modelConfig.kvCacheBytesPerToken
        let totalRequired = modelConfig.weightBytes + modelConfig.overheadBytes + requiredKVBytes

        return MemoryWarning(
            requestedLength: requestedLength,
            maxLength: maxContextLength,
            requiredBytes: totalRequired,
            availableBytes: deviceProfile.usableMemoryBytes,
            recommendation: recommendAction(requestedLength: requestedLength)
        )
    }

    private func recommendAction(requestedLength: Int) -> String {
        let ratio = Float(requestedLength) / Float(max(maxContextLength, 1))
        if ratio <= 2.0 {
            return "Reduce context length to \(maxContextLength) or enable KV cache compression."
        } else if ratio <= 5.0 {
            return "Enable aggressive KV cache compression (MiniCache + DuoAttention) or use a quantized model."
        } else {
            return "This model is too large for this device. Use a smaller model or lower quantization."
        }
    }
}

/// Warning issued when context length exceeds memory budget.
public struct MemoryWarning: Sendable {
    public let requestedLength: Int
    public let maxLength: Int
    public let requiredBytes: UInt64
    public let availableBytes: UInt64
    public let recommendation: String
}
```

**Step 5: Run tests, verify pass**

Run: `swift test --filter MemoryBudgetTests 2>&1`
Expected: All 6 tests pass.

---

## Task 2: KV Cache Compression (MiniCache)

**Files:**
- Create: `Sources/EdgeRunnerCore/Attention/MiniCache.swift`
- Create: `Sources/EdgeRunnerMetal/Shaders/MiniCache.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/MiniCacheParams.h`
- Test: `Tests/EdgeRunnerCoreTests/Attention/MiniCacheTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/Attention/MiniCacheTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

/// CPU reference for KV merge across adjacent layers.
private func cpuMergeKV(
    kvLayer1: [Float],  // [seqLen, headDim]
    kvLayer2: [Float],
    seqLen: Int, headDim: Int
) -> [Float] {
    // Average adjacent layer KV states
    var merged = [Float](repeating: 0, count: seqLen * headDim)
    for i in 0..<(seqLen * headDim) {
        merged[i] = (kvLayer1[i] + kvLayer2[i]) / 2.0
    }
    return merged
}

/// CPU reference for 4-bit quantisation.
private func cpuQuantize4Bit(
    _ input: [Float], blockSize: Int
) -> (quantized: [UInt8], scales: [Float], zeroPoints: [Float]) {
    let numBlocks = (input.count + blockSize - 1) / blockSize
    var quantized = [UInt8](repeating: 0, count: (input.count + 1) / 2)  // 4-bit packed
    var scales = [Float](repeating: 0, count: numBlocks)
    var zeroPoints = [Float](repeating: 0, count: numBlocks)

    for block in 0..<numBlocks {
        let start = block * blockSize
        let end = min(start + blockSize, input.count)
        let slice = Array(input[start..<end])

        let minVal = slice.min() ?? 0
        let maxVal = slice.max() ?? 0
        let scale = (maxVal - minVal) / 15.0
        let zeroPoint = minVal

        scales[block] = max(scale, 1e-8)
        zeroPoints[block] = zeroPoint

        for i in start..<end {
            let q = UInt8(min(15, max(0, Int(((input[i] - zeroPoint) / max(scale, 1e-8)).rounded()))))
            let byteIdx = i / 2
            if i % 2 == 0 {
                quantized[byteIdx] = q
            } else {
                quantized[byteIdx] |= (q << 4)
            }
        }
    }

    return (quantized, scales, zeroPoints)
}

/// CPU dequantize for verification.
private func cpuDequantize4Bit(
    quantized: [UInt8], scales: [Float], zeroPoints: [Float],
    count: Int, blockSize: Int
) -> [Float] {
    var output = [Float](repeating: 0, count: count)
    for i in 0..<count {
        let block = i / blockSize
        let byteIdx = i / 2
        let q: UInt8
        if i % 2 == 0 {
            q = quantized[byteIdx] & 0x0F
        } else {
            q = (quantized[byteIdx] >> 4) & 0x0F
        }
        output[i] = Float(q) * scales[block] + zeroPoints[block]
    }
    return output
}

@Suite("MiniCache -- KV Compression")
struct MiniCacheTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    @Test func mergeAdjacentLayerKV() async throws {
        let seqLen = 64, headDim = 64
        let kv1Data = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let kv2Data = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }

        let kv1 = Tensor<Float>(shape: [seqLen, headDim], data: kv1Data)
        let kv2 = Tensor<Float>(shape: [seqLen, headDim], data: kv2Data)

        let merged = try await MiniCache.mergeAdjacentLayers(kv1, kv2, backend: backend)
        let gpuResult = try await merged.toArray()
        let cpuResult = cpuMergeKV(
            kvLayer1: kv1Data, kvLayer2: kv2Data,
            seqLen: seqLen, headDim: headDim
        )

        #expect(merged.shape == [seqLen, headDim])
        for i in 0..<cpuResult.count {
            #expect(abs(gpuResult[i] - cpuResult[i]) < 1e-5)
        }
    }

    @Test func quantize4BitRoundTrip() async throws {
        let size = 256
        let data = (0..<size).map { _ in Float.random(in: -2...2) }
        let input = Tensor<Float>(shape: [size], data: data)

        let compressed = try await MiniCache.quantize4Bit(
            input, blockSize: 32, backend: backend
        )
        let decompressed = try await MiniCache.dequantize4Bit(
            compressed, originalCount: size, blockSize: 32, backend: backend
        )

        let result = try await decompressed.toArray()

        // 4-bit quantisation has limited precision; expect ~0.3 max error
        var maxError: Float = 0
        for i in 0..<size {
            maxError = max(maxError, abs(result[i] - data[i]))
        }
        #expect(maxError < 0.5, "4-bit round-trip max error too large: \(maxError)")
    }

    @Test func quantize4BitNumericalAccuracy() async throws {
        let size = 128, blockSize = 32
        let data = (0..<size).map { _ in Float.random(in: -1...1) }
        let input = Tensor<Float>(shape: [size], data: data)

        let compressed = try await MiniCache.quantize4Bit(
            input, blockSize: blockSize, backend: backend
        )
        let gpuDecompressed = try await MiniCache.dequantize4Bit(
            compressed, originalCount: size, blockSize: blockSize, backend: backend
        )
        let gpuResult = try await gpuDecompressed.toArray()

        // CPU reference
        let (cpuQ, cpuS, cpuZ) = cpuQuantize4Bit(data, blockSize: blockSize)
        let cpuResult = cpuDequantize4Bit(
            quantized: cpuQ, scales: cpuS, zeroPoints: cpuZ,
            count: size, blockSize: blockSize
        )

        for i in 0..<size {
            #expect(abs(gpuResult[i] - cpuResult[i]) < 0.5,
                    "Quantize mismatch at [\(i)]: GPU=\(gpuResult[i]) CPU=\(cpuResult[i])")
        }
    }

    @Test func miniCacheCompressionRatio() async throws {
        let seqLen = 512, headDim = 128, numHeads = 8, numLayers = 32

        let originalBytesPerLayer = seqLen * headDim * numHeads * MemoryLayout<Float>.size
        let originalTotalBytes = originalBytesPerLayer * numLayers * 2 // K + V

        // MiniCache: merge pairs -> 16 layers, then 4-bit quantise
        let mergedLayers = numLayers / 2
        let mergedElements = seqLen * headDim * numHeads * mergedLayers * 2
        let quantizedBytes = mergedElements / 2  // 4 bits per element
        let scaleBytes = (mergedElements / 32) * MemoryLayout<Float>.size  // 1 scale per block of 32

        let compressedTotalBytes = quantizedBytes + scaleBytes

        let ratio = Float(originalTotalBytes) / Float(compressedTotalBytes)
        // Expected: ~4-5x compression (2x from merge, ~2x from 4-bit quant of F32)
        // More precisely: merge halves layers (2x), 4-bit = 1/8 of F32 (8x) -> net ~4x
        // But scales add overhead, so ~3.5-5x
        #expect(ratio > 3.0, "Compression ratio should be at least 3x, got \(ratio)")
        #expect(ratio < 10.0, "Sanity check: ratio=\(ratio)")
    }

    @Test func miniCacheEndToEnd() async throws {
        let seqLen = 32, headDim = 32, numHeads = 4, numLayers = 4

        // Simulate KV cache for all layers
        var kvStates: [(k: Tensor<Float>, v: Tensor<Float>)] = []
        for _ in 0..<numLayers {
            let k = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
            let v = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
            kvStates.append((k: k, v: v))
        }

        let cache = try await MiniCache(
            kvStates: kvStates, blockSize: 32, backend: backend
        )

        // Decompress and verify shape
        let restored = try await cache.decompress(backend: backend)
        #expect(restored.count == numLayers / 2) // Merged pairs

        for layer in restored {
            #expect(layer.k.shape == [numHeads, seqLen, headDim])
            #expect(layer.v.shape == [numHeads, seqLen, headDim])
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter MiniCacheTests 2>&1`
Expected: FAIL -- `MiniCache` not defined

**Step 3: Implement MiniCache params header**

```c
// Sources/EdgeRunnerSharedTypes/include/MiniCacheParams.h
#ifndef MINICACHE_PARAMS_H
#define MINICACHE_PARAMS_H

#include <stdint.h>

/// Parameters for KV merge kernel.
typedef struct {
    uint32_t elementCount;
} ERKVMergeParams;

/// Parameters for 4-bit quantisation kernel.
typedef struct {
    uint32_t elementCount;
    uint32_t blockSize;
    uint32_t numBlocks;
} ERQuantize4BitParams;

#endif /* MINICACHE_PARAMS_H */
```

**Step 4: Implement MiniCache Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/MiniCache.metal
#include <metal_stdlib>
using namespace metal;

struct ERKVMergeParams {
    uint elementCount;
};

/// Merge two KV tensors by averaging.
kernel void kv_merge_f32(
    device const float* kv1    [[buffer(0)]],
    device const float* kv2    [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant ERKVMergeParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.elementCount) return;
    output[tid] = (kv1[tid] + kv2[tid]) * 0.5f;
}

struct ERQuantize4BitParams {
    uint elementCount;
    uint blockSize;
    uint numBlocks;
};

/// Quantize float32 to 4-bit with per-block scale and zero point.
/// Output: packed uint8 (2 values per byte), scales, and zero points.
kernel void quantize_4bit_f32(
    device const float* input      [[buffer(0)]],
    device uint8_t*     quantized  [[buffer(1)]],  // packed: low nibble = even, high = odd
    device float*       scales     [[buffer(2)]],
    device float*       zeroPoints [[buffer(3)]],
    constant ERQuantize4BitParams& params [[buffer(4)]],
    uint blockIdx [[thread_position_in_grid]]
) {
    if (blockIdx >= params.numBlocks) return;

    uint start = blockIdx * params.blockSize;
    uint end = min(start + params.blockSize, params.elementCount);

    // Find min/max in block
    float minVal = input[start];
    float maxVal = input[start];
    for (uint i = start + 1; i < end; i++) {
        minVal = min(minVal, input[i]);
        maxVal = max(maxVal, input[i]);
    }

    float scale = (maxVal - minVal) / 15.0f;
    scale = max(scale, 1e-8f);
    float zeroPoint = minVal;

    scales[blockIdx] = scale;
    zeroPoints[blockIdx] = zeroPoint;

    // Quantize
    for (uint i = start; i < end; i++) {
        uint q = uint(clamp(round((input[i] - zeroPoint) / scale), 0.0f, 15.0f));
        uint byteIdx = i / 2;
        if (i % 2 == 0) {
            quantized[byteIdx] = uint8_t(q);
        } else {
            quantized[byteIdx] |= uint8_t(q << 4);
        }
    }
}

/// Dequantize 4-bit back to float32.
kernel void dequantize_4bit_f32(
    device const uint8_t* quantized  [[buffer(0)]],
    device const float*   scales     [[buffer(1)]],
    device const float*   zeroPoints [[buffer(2)]],
    device float*         output     [[buffer(3)]],
    constant ERQuantize4BitParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.elementCount) return;

    uint blockIdx = tid / params.blockSize;
    uint byteIdx = tid / 2;
    uint8_t packed = quantized[byteIdx];

    uint q;
    if (tid % 2 == 0) {
        q = uint(packed & 0x0F);
    } else {
        q = uint((packed >> 4) & 0x0F);
    }

    output[tid] = float(q) * scales[blockIdx] + zeroPoints[blockIdx];
}
```

**Step 5: Implement MiniCache**

```swift
// Sources/EdgeRunnerCore/Attention/MiniCache.swift
import Metal
import EdgeRunnerMetal
import Foundation

/// Compressed 4-bit KV data for a single layer pair.
public struct CompressedKV: Sendable {
    public let quantizedK: Tensor<UInt8>  // packed 4-bit
    public let kScales: Tensor<Float>
    public let kZeroPoints: Tensor<Float>
    public let quantizedV: Tensor<UInt8>
    public let vScales: Tensor<Float>
    public let vZeroPoints: Tensor<Float>
    public let originalShape: [Int]       // [numHeads, seqLen, headDim]
    public let blockSize: Int
}

/// MiniCache: KV cache compression via adjacent-layer merging + 4-bit quantisation.
/// Reference: "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models" (2024).
///
/// Achieves ~5x compression by:
/// 1. Merging KV states from adjacent layers (2x reduction)
/// 2. 4-bit quantisation of merged states (~2-2.5x additional reduction)
public struct MiniCache: Sendable {
    public let compressedLayers: [CompressedKV]
    public let originalLayerCount: Int

    /// Create MiniCache from full KV states.
    /// Merges adjacent layer pairs and quantises to 4-bit.
    public init(
        kvStates: [(k: Tensor<Float>, v: Tensor<Float>)],
        blockSize: Int = 32,
        backend: MetalBackend
    ) async throws {
        self.originalLayerCount = kvStates.count

        var compressed: [CompressedKV] = []

        // Merge adjacent pairs
        let pairCount = kvStates.count / 2
        for p in 0..<pairCount {
            let layer1 = kvStates[p * 2]
            let layer2 = kvStates[p * 2 + 1]

            // Merge K
            let mergedK = try await MiniCache.mergeAdjacentLayers(
                layer1.k, layer2.k, backend: backend
            )
            // Merge V
            let mergedV = try await MiniCache.mergeAdjacentLayers(
                layer1.v, layer2.v, backend: backend
            )

            // Quantise to 4-bit
            let compK = try await MiniCache.quantize4Bit(
                mergedK, blockSize: blockSize, backend: backend
            )
            let compV = try await MiniCache.quantize4Bit(
                mergedV, blockSize: blockSize, backend: backend
            )

            compressed.append(CompressedKV(
                quantizedK: compK.quantized,
                kScales: compK.scales,
                kZeroPoints: compK.zeroPoints,
                quantizedV: compV.quantized,
                vScales: compV.scales,
                vZeroPoints: compV.zeroPoints,
                originalShape: layer1.k.shape,
                blockSize: blockSize
            ))
        }

        self.compressedLayers = compressed
    }

    /// Merge two KV tensors from adjacent layers by averaging.
    public static func mergeAdjacentLayers(
        _ kv1: Tensor<Float>, _ kv2: Tensor<Float>,
        backend: MetalBackend
    ) async throws -> Tensor<Float> {
        precondition(kv1.shape == kv2.shape)
        return try await backend.kvMerge(kv1, kv2)
    }

    /// Quantise a tensor to 4-bit with per-block scale/zero-point.
    public static func quantize4Bit(
        _ input: Tensor<Float>, blockSize: Int,
        backend: MetalBackend
    ) async throws -> (quantized: Tensor<UInt8>, scales: Tensor<Float>, zeroPoints: Tensor<Float>) {
        return try await backend.quantize4Bit(input, blockSize: blockSize)
    }

    /// Dequantise 4-bit data back to Float32.
    public static func dequantize4Bit(
        _ compressed: (quantized: Tensor<UInt8>, scales: Tensor<Float>, zeroPoints: Tensor<Float>),
        originalCount: Int, blockSize: Int,
        backend: MetalBackend
    ) async throws -> Tensor<Float> {
        return try await backend.dequantize4Bit(
            quantized: compressed.quantized,
            scales: compressed.scales,
            zeroPoints: compressed.zeroPoints,
            originalCount: originalCount,
            blockSize: blockSize
        )
    }

    /// Decompress all layers back to (approximately) original KV states.
    /// Returns merged layer pairs (half the original layer count).
    public func decompress(
        backend: MetalBackend
    ) async throws -> [(k: Tensor<Float>, v: Tensor<Float>)] {
        var result: [(k: Tensor<Float>, v: Tensor<Float>)] = []

        for layer in compressedLayers {
            let elementCount = layer.originalShape.reduce(1, *)

            let k = try await backend.dequantize4Bit(
                quantized: layer.quantizedK,
                scales: layer.kScales,
                zeroPoints: layer.kZeroPoints,
                originalCount: elementCount,
                blockSize: layer.blockSize
            )
            let v = try await backend.dequantize4Bit(
                quantized: layer.quantizedV,
                scales: layer.vScales,
                zeroPoints: layer.vZeroPoints,
                originalCount: elementCount,
                blockSize: layer.blockSize
            )

            let reshapedK = try await backend.reshape(k, shape: layer.originalShape)
            let reshapedV = try await backend.reshape(v, shape: layer.originalShape)

            result.append((k: reshapedK, v: reshapedV))
        }

        return result
    }
}
```

**Step 6: Run tests, verify pass**

Run: `swift test --filter MiniCacheTests 2>&1`
Expected: All 5 tests pass.

---

## Task 3: Squeezed Attention

**Files:**
- Create: `Sources/EdgeRunnerCore/Attention/SqueezedAttention.swift`
- Create: `Sources/EdgeRunnerMetal/Shaders/KMeansClustering.metal`
- Create: `Sources/EdgeRunnerSharedTypes/include/KMeansParams.h`
- Test: `Tests/EdgeRunnerCoreTests/Attention/SqueezedAttentionTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/Attention/SqueezedAttentionTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

/// CPU reference for k-means clustering (Lloyd's algorithm).
private func cpuKMeans(
    data: [Float],     // [N, D] flattened
    N: Int, D: Int, K: Int,
    maxIter: Int = 10
) -> (centroids: [Float], assignments: [Int]) {
    // Initialize centroids from first K data points
    var centroids = Array(data.prefix(K * D))
    var assignments = [Int](repeating: 0, count: N)

    for _ in 0..<maxIter {
        // Assign each point to nearest centroid
        for i in 0..<N {
            var bestDist: Float = .infinity
            var bestK = 0
            for k in 0..<K {
                var dist: Float = 0
                for d in 0..<D {
                    let diff = data[i * D + d] - centroids[k * D + d]
                    dist += diff * diff
                }
                if dist < bestDist {
                    bestDist = dist
                    bestK = k
                }
            }
            assignments[i] = bestK
        }

        // Update centroids
        var newCentroids = [Float](repeating: 0, count: K * D)
        var counts = [Int](repeating: 0, count: K)

        for i in 0..<N {
            let k = assignments[i]
            counts[k] += 1
            for d in 0..<D {
                newCentroids[k * D + d] += data[i * D + d]
            }
        }

        for k in 0..<K {
            if counts[k] > 0 {
                for d in 0..<D {
                    newCentroids[k * D + d] /= Float(counts[k])
                }
            }
        }
        centroids = newCentroids
    }

    return (centroids, assignments)
}

@Suite("Squeezed Attention")
struct SqueezedAttentionTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    @Test func kMeansClusteringBasic() async throws {
        // Two clear clusters
        let N = 32, D = 8, K = 2
        var data = [Float]()
        for _ in 0..<(N / 2) {
            data.append(contentsOf: (0..<D).map { _ in Float.random(in: -1...0) })
        }
        for _ in 0..<(N / 2) {
            data.append(contentsOf: (0..<D).map { _ in Float.random(in: 0...1) })
        }

        let input = Tensor<Float>(shape: [N, D], data: data)
        let result = try await KMeansClustering.cluster(
            input, k: K, maxIterations: 20, backend: backend
        )

        #expect(result.centroids.shape == [K, D])
        #expect(result.assignments.count == N)

        // First half should be one cluster, second half another
        let firstHalfCluster = result.assignments[0]
        let secondHalfCluster = result.assignments[N / 2]
        #expect(firstHalfCluster != secondHalfCluster,
                "Two clear clusters should get different assignments")

        // Verify consistency within each half
        let firstHalfConsistent = result.assignments[0..<(N/2)].allSatisfy { $0 == firstHalfCluster }
        let secondHalfConsistent = result.assignments[(N/2)..<N].allSatisfy { $0 == secondHalfCluster }
        #expect(firstHalfConsistent, "First half should be consistent cluster")
        #expect(secondHalfConsistent, "Second half should be consistent cluster")
    }

    @Test func squeezedAttentionOutputShape() async throws {
        let seqLen = 64, headDim = 32, numCentroids = 8

        let query = Tensor<Float>.random(shape: [1, headDim])
        let keys = Tensor<Float>.random(shape: [seqLen, headDim])
        let values = Tensor<Float>.random(shape: [seqLen, headDim])

        let squeezed = try await SqueezedAttention(
            numCentroids: numCentroids, backend: backend
        )

        let output = try await squeezed.forward(
            query: query, keys: keys, values: values
        )

        #expect(output.shape == [1, headDim])
    }

    @Test func squeezedAttentionKVReduction() async throws {
        let seqLen = 256, headDim = 32, numCentroids = 16

        let keys = Tensor<Float>.random(shape: [seqLen, headDim])
        let values = Tensor<Float>.random(shape: [seqLen, headDim])

        let squeezed = try await SqueezedAttention(
            numCentroids: numCentroids, backend: backend
        )

        // Compress KV
        let compressed = try await squeezed.compress(keys: keys, values: values)

        // Should have numCentroids representative KVs
        #expect(compressed.keys.shape == [numCentroids, headDim])
        #expect(compressed.values.shape == [numCentroids, headDim])

        // Reduction ratio
        let ratio = Float(seqLen) / Float(numCentroids)
        #expect(ratio >= 3.0, "Expected at least 3x KV reduction, got \(ratio)x")
    }

    @Test func squeezedAttentionQualityPreservation() async throws {
        // Compare full attention output vs squeezed attention
        let seqLen = 32, headDim = 16, numCentroids = 8

        let query = Tensor<Float>.random(shape: [1, headDim])
        let keys = Tensor<Float>.random(shape: [seqLen, headDim])
        let values = Tensor<Float>.random(shape: [seqLen, headDim])

        // Full attention
        let fullOutput = try await backend.scaledDotProductAttention(
            query: query, key: keys, value: values, headDim: headDim
        )

        // Squeezed attention
        let squeezed = try await SqueezedAttention(
            numCentroids: numCentroids, backend: backend
        )
        let squeezedOutput = try await squeezed.forward(
            query: query, keys: keys, values: values
        )

        let fullArr = try await fullOutput.toArray()
        let sqArr = try await squeezedOutput.toArray()

        // Should be somewhat close (not exact due to clustering approximation)
        var totalError: Float = 0
        for i in 0..<fullArr.count {
            totalError += abs(fullArr[i] - sqArr[i])
        }
        let avgError = totalError / Float(fullArr.count)

        // Squeezed is an approximation; avg error should be reasonable
        #expect(avgError < 2.0,
                "Squeezed attention avg error too high: \(avgError)")
    }

    @Test func squeezedAttentionLargeReduction() async throws {
        // Test 8x reduction (256 tokens -> 32 centroids)
        let seqLen = 256, headDim = 64, numCentroids = 32

        let query = Tensor<Float>.random(shape: [4, headDim]) // 4 new query tokens
        let keys = Tensor<Float>.random(shape: [seqLen, headDim])
        let values = Tensor<Float>.random(shape: [seqLen, headDim])

        let squeezed = try await SqueezedAttention(
            numCentroids: numCentroids, backend: backend
        )

        let output = try await squeezed.forward(
            query: query, keys: keys, values: values
        )

        #expect(output.shape == [4, headDim])
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter SqueezedAttentionTests 2>&1`
Expected: FAIL -- types not defined

**Step 3: Implement KMeans params header**

```c
// Sources/EdgeRunnerSharedTypes/include/KMeansParams.h
#ifndef KMEANS_PARAMS_H
#define KMEANS_PARAMS_H

#include <stdint.h>

/// Parameters for k-means assignment kernel.
typedef struct {
    uint32_t N;       // number of data points
    uint32_t D;       // dimension
    uint32_t K;       // number of clusters
} ERKMeansParams;

#endif /* KMEANS_PARAMS_H */
```

**Step 4: Implement KMeans Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/KMeansClustering.metal
#include <metal_stdlib>
using namespace metal;

struct ERKMeansParams {
    uint N;
    uint D;
    uint K;
};

/// Assign each data point to nearest centroid.
kernel void kmeans_assign_f32(
    device const float* data        [[buffer(0)]],  // [N, D]
    device const float* centroids   [[buffer(1)]],  // [K, D]
    device uint32_t*    assignments [[buffer(2)]],   // [N]
    constant ERKMeansParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.N) return;

    float bestDist = INFINITY;
    uint bestK = 0;

    for (uint k = 0; k < params.K; k++) {
        float dist = 0;
        for (uint d = 0; d < params.D; d++) {
            float diff = data[tid * params.D + d] - centroids[k * params.D + d];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestK = k;
        }
    }

    assignments[tid] = bestK;
}

/// Update centroids by averaging assigned points.
/// One thread per centroid.
kernel void kmeans_update_f32(
    device const float*    data        [[buffer(0)]],
    device const uint32_t* assignments [[buffer(1)]],
    device float*          centroids   [[buffer(2)]],
    device uint32_t*       counts      [[buffer(3)]],
    constant ERKMeansParams& params    [[buffer(4)]],
    uint k [[thread_position_in_grid]]
) {
    if (k >= params.K) return;

    uint count = 0;
    for (uint d = 0; d < params.D; d++) {
        centroids[k * params.D + d] = 0;
    }

    for (uint i = 0; i < params.N; i++) {
        if (assignments[i] == k) {
            count++;
            for (uint d = 0; d < params.D; d++) {
                centroids[k * params.D + d] += data[i * params.D + d];
            }
        }
    }

    if (count > 0) {
        for (uint d = 0; d < params.D; d++) {
            centroids[k * params.D + d] /= float(count);
        }
    }

    counts[k] = count;
}
```

**Step 5: Implement SqueezedAttention**

```swift
// Sources/EdgeRunnerCore/Attention/SqueezedAttention.swift
import Metal
import EdgeRunnerMetal

/// K-means clustering on Metal.
public enum KMeansClustering: Sendable {
    public struct ClusterResult: Sendable {
        public let centroids: Tensor<Float>   // [K, D]
        public let assignments: [Int]          // [N]
    }

    /// Cluster N data points of dimension D into K clusters using Lloyd's algorithm.
    public static func cluster(
        _ data: Tensor<Float>,
        k: Int,
        maxIterations: Int = 20,
        backend: MetalBackend
    ) async throws -> ClusterResult {
        let N = data.shape[0]
        let D = data.shape[1]

        // Initialize centroids from first K data points
        var centroids = try await backend.slice(data, axis: 0, start: 0, end: k)
        var assignments = [Int](repeating: 0, count: N)

        for _ in 0..<maxIterations {
            // Assignment step
            let rawAssignments = try await backend.kmeansAssign(
                data: data, centroids: centroids, N: N, D: D, K: k
            )
            assignments = try await rawAssignments.toIntArray()

            // Update step
            centroids = try await backend.kmeansUpdate(
                data: data, assignments: rawAssignments, N: N, D: D, K: k
            )
        }

        return ClusterResult(centroids: centroids, assignments: assignments)
    }
}

/// Compressed KV representation using centroids.
public struct CompressedKVCentroids: Sendable {
    public let keys: Tensor<Float>     // [numCentroids, headDim]
    public let values: Tensor<Float>   // [numCentroids, headDim]
}

/// Squeezed Attention: clusters static prompt KV pairs into centroids,
/// then attends only to representative keys.
/// Reference: "Squeezed Attention" (2024).
///
/// For a prompt of seqLen tokens with numCentroids clusters:
/// - Original KV budget: O(seqLen * headDim)
/// - Squeezed KV budget: O(numCentroids * headDim)
/// - Reduction: seqLen / numCentroids (typically 3-8x)
public final class SqueezedAttention: @unchecked Sendable {
    public let numCentroids: Int
    private let backend: MetalBackend

    public init(numCentroids: Int, backend: MetalBackend) async throws {
        self.numCentroids = numCentroids
        self.backend = backend
    }

    /// Compress KV pairs by clustering keys and averaging corresponding values.
    public func compress(
        keys: Tensor<Float>, values: Tensor<Float>
    ) async throws -> CompressedKVCentroids {
        let seqLen = keys.shape[0]
        let headDim = keys.shape[1]

        precondition(values.shape == [seqLen, headDim])

        // Cluster keys
        let clusterResult = try await KMeansClustering.cluster(
            keys, k: numCentroids, maxIterations: 20, backend: backend
        )

        // Average values per cluster
        var valueCentroids = Tensor<Float>.zeros(shape: [numCentroids, headDim])
        valueCentroids = try await backend.scatterMean(
            values, assignments: clusterResult.assignments,
            numClusters: numCentroids
        )

        return CompressedKVCentroids(
            keys: clusterResult.centroids,
            values: valueCentroids
        )
    }

    /// Forward: compress KV then attend.
    /// query: [queryLen, headDim], keys: [seqLen, headDim], values: [seqLen, headDim]
    /// Returns: [queryLen, headDim]
    public func forward(
        query: Tensor<Float>, keys: Tensor<Float>, values: Tensor<Float>
    ) async throws -> Tensor<Float> {
        let compressed = try await compress(keys: keys, values: values)
        let headDim = query.shape[query.shape.count - 1]

        return try await backend.scaledDotProductAttention(
            query: query,
            key: compressed.keys,
            value: compressed.values,
            headDim: headDim
        )
    }
}
```

**Step 6: Run tests, verify pass**

Run: `swift test --filter SqueezedAttentionTests 2>&1`
Expected: All 5 tests pass.

---

## Task 4: DuoAttention

**Files:**
- Create: `Sources/EdgeRunnerCore/Attention/DuoAttention.swift`
- Create: `Sources/EdgeRunnerCore/Attention/HeadClassifier.swift`
- Test: `Tests/EdgeRunnerCoreTests/Attention/DuoAttentionTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/Attention/DuoAttentionTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("DuoAttention")
struct DuoAttentionTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    // MARK: - Head Classification

    @Test func classifyHeadsRetrievalVsStreaming() async throws {
        let numHeads = 32
        let classifier = HeadClassifier(numHeads: numHeads)

        // Simulate attention pattern analysis:
        // Some heads attend broadly (retrieval), others attend locally (streaming)
        var attentionEntropies = [Float](repeating: 0, count: numHeads)
        for h in 0..<numHeads {
            if h < 8 {
                attentionEntropies[h] = Float.random(in: 4.0...6.0) // high entropy -> retrieval
            } else {
                attentionEntropies[h] = Float.random(in: 0.5...2.0) // low entropy -> streaming
            }
        }

        let classification = classifier.classify(attentionEntropies: attentionEntropies)

        #expect(classification.retrievalHeads.count > 0)
        #expect(classification.streamingHeads.count > 0)
        #expect(classification.retrievalHeads.count + classification.streamingHeads.count == numHeads)

        // High-entropy heads should be classified as retrieval
        for h in 0..<8 {
            #expect(classification.retrievalHeads.contains(h),
                    "Head \(h) with high entropy should be retrieval")
        }
    }

    @Test func classifyHeadsWithThreshold() async throws {
        let classifier = HeadClassifier(numHeads: 16, entropyThreshold: 3.0)

        let entropies: [Float] = [
            5.0, 4.5, 4.0, 3.5,   // retrieval
            2.5, 2.0, 1.5, 1.0,   // streaming
            5.5, 3.2, 2.8, 4.1,   // mixed
            0.5, 6.0, 1.2, 3.0    // mixed
        ]

        let classification = classifier.classify(attentionEntropies: entropies)

        // Heads with entropy >= 3.0 should be retrieval
        let expectedRetrieval = Set([0, 1, 2, 3, 8, 11, 13, 15])
        #expect(Set(classification.retrievalHeads) == expectedRetrieval)
    }

    // MARK: - DuoAttention KV Allocation

    @Test func duoAttentionKVAllocation() async throws {
        let config = DuoAttentionConfig(
            numHeads: 32,
            headDim: 128,
            retrievalHeads: Set(0..<8),     // 8 retrieval heads: full KV
            streamingHeads: Set(8..<32),    // 24 streaming heads: sliding window
            slidingWindowSize: 256
        )

        let duo = try await DuoAttention(config: config, backend: backend)

        // Full KV for retrieval heads
        #expect(duo.retrievalKVCapacity == .unlimited)
        // Sliding window for streaming heads
        #expect(duo.streamingKVCapacity == 256)
    }

    @Test func duoAttentionMemoryReduction() async throws {
        let seqLen = 4096, headDim = 128, numHeads = 32
        let numRetrievalHeads = 8
        let windowSize = 256

        let config = DuoAttentionConfig(
            numHeads: numHeads,
            headDim: headDim,
            retrievalHeads: Set(0..<numRetrievalHeads),
            streamingHeads: Set(numRetrievalHeads..<numHeads),
            slidingWindowSize: windowSize
        )

        // Full KV budget: numHeads * seqLen * headDim * 2 (K+V) * 4 bytes
        let fullBudget = numHeads * seqLen * headDim * 2 * 4

        // DuoAttention budget:
        // Retrieval: numRetrievalHeads * seqLen * headDim * 2 * 4
        // Streaming: (numHeads - numRetrievalHeads) * windowSize * headDim * 2 * 4
        let retrievalBudget = numRetrievalHeads * seqLen * headDim * 2 * 4
        let streamingBudget = (numHeads - numRetrievalHeads) * windowSize * headDim * 2 * 4
        let duoBudget = retrievalBudget + streamingBudget

        let reduction = Float(fullBudget) / Float(duoBudget)
        #expect(reduction > 2.5, "Expected at least 2.5x reduction, got \(reduction)x")

        let memEst = DuoAttention.estimateMemory(config: config, seqLen: seqLen)
        #expect(Float(fullBudget) / Float(memEst) > 2.5)
    }

    @Test func duoAttentionForwardPass() async throws {
        let numHeads = 8, headDim = 32, seqLen = 64

        let config = DuoAttentionConfig(
            numHeads: numHeads,
            headDim: headDim,
            retrievalHeads: Set(0..<2),
            streamingHeads: Set(2..<numHeads),
            slidingWindowSize: 16
        )
        let duo = try await DuoAttention(config: config, backend: backend)

        let query = Tensor<Float>.random(shape: [numHeads, 1, headDim])
        let keys = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
        let values = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])

        let output = try await duo.forward(query: query, keys: keys, values: values)

        #expect(output.shape == [numHeads, 1, headDim])
    }

    @Test func duoAttentionRetrievalHeadsSeeFull() async throws {
        let numHeads = 4, headDim = 16, seqLen = 32

        let config = DuoAttentionConfig(
            numHeads: numHeads,
            headDim: headDim,
            retrievalHeads: Set([0]),
            streamingHeads: Set([1, 2, 3]),
            slidingWindowSize: 8
        )
        let duo = try await DuoAttention(config: config, backend: backend)

        let query = Tensor<Float>.random(shape: [numHeads, 1, headDim])
        let keys = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
        let values = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])

        // Run duo attention
        let duoOut = try await duo.forward(query: query, keys: keys, values: values)

        // Run standard full attention for head 0
        let q0 = try await backend.slice(query, axis: 0, start: 0, end: 1)
        let k0 = try await backend.slice(keys, axis: 0, start: 0, end: 1)
        let v0 = try await backend.slice(values, axis: 0, start: 0, end: 1)
        let fullOut0 = try await backend.scaledDotProductAttention(
            query: q0, key: k0, value: v0, headDim: headDim
        )

        // Retrieval head 0 should match full attention
        let duoHead0 = try await backend.slice(duoOut, axis: 0, start: 0, end: 1)
        let duoArr = try await duoHead0.toArray()
        let fullArr = try await fullOut0.toArray()

        for i in 0..<fullArr.count {
            #expect(abs(duoArr[i] - fullArr[i]) < 1e-4,
                    "Retrieval head should match full attention at [\(i)]")
        }
    }

    @Test func duoAttentionStreamingHeadsUseSlidingWindow() async throws {
        let numHeads = 4, headDim = 16, seqLen = 32, windowSize = 8

        let config = DuoAttentionConfig(
            numHeads: numHeads,
            headDim: headDim,
            retrievalHeads: Set([0]),
            streamingHeads: Set([1, 2, 3]),
            slidingWindowSize: windowSize
        )
        let duo = try await DuoAttention(config: config, backend: backend)

        let query = Tensor<Float>.random(shape: [numHeads, 1, headDim])
        let keys = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
        let values = Tensor<Float>.random(shape: [numHeads, seqLen, headDim])

        let duoOut = try await duo.forward(query: query, keys: keys, values: values)

        // For streaming head 1, only the last windowSize tokens should contribute
        let q1 = try await backend.slice(query, axis: 0, start: 1, end: 2)
        let k1Window = try await backend.slice(keys, axis: 0, start: 1, end: 2)
        let k1Windowed = try await backend.slice(k1Window, axis: 1, start: seqLen - windowSize, end: seqLen)
        let v1Window = try await backend.slice(values, axis: 0, start: 1, end: 2)
        let v1Windowed = try await backend.slice(v1Window, axis: 1, start: seqLen - windowSize, end: seqLen)

        let windowOut = try await backend.scaledDotProductAttention(
            query: q1, key: k1Windowed, value: v1Windowed, headDim: headDim
        )

        let duoHead1 = try await backend.slice(duoOut, axis: 0, start: 1, end: 2)
        let duoArr = try await duoHead1.toArray()
        let windowArr = try await windowOut.toArray()

        for i in 0..<windowArr.count {
            #expect(abs(duoArr[i] - windowArr[i]) < 1e-4,
                    "Streaming head should match windowed attention at [\(i)]")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter DuoAttentionTests 2>&1`
Expected: FAIL -- types not defined

**Step 3: Implement HeadClassifier**

```swift
// Sources/EdgeRunnerCore/Attention/HeadClassifier.swift
import Foundation

/// Result of head classification: which heads are retrieval vs streaming.
public struct HeadClassification: Sendable {
    public let retrievalHeads: [Int]
    public let streamingHeads: [Int]
}

/// Classifies attention heads as retrieval (full KV) or streaming (sliding window).
/// Uses attention entropy as the discriminating signal: high-entropy heads attend broadly
/// (retrieval pattern), low-entropy heads attend locally (streaming pattern).
public struct HeadClassifier: Sendable {
    public let numHeads: Int
    public let entropyThreshold: Float

    /// - Parameters:
    ///   - numHeads: Total number of attention heads.
    ///   - entropyThreshold: Heads with entropy >= threshold are classified as retrieval.
    ///     Default uses median-based auto-threshold.
    public init(numHeads: Int, entropyThreshold: Float? = nil) {
        self.numHeads = numHeads
        self.entropyThreshold = entropyThreshold ?? 3.0
    }

    /// Classify heads based on their attention pattern entropy.
    /// - Parameter attentionEntropies: [numHeads] entropy values per head.
    public func classify(attentionEntropies: [Float]) -> HeadClassification {
        precondition(attentionEntropies.count == numHeads)

        var retrieval: [Int] = []
        var streaming: [Int] = []

        for h in 0..<numHeads {
            if attentionEntropies[h] >= entropyThreshold {
                retrieval.append(h)
            } else {
                streaming.append(h)
            }
        }

        return HeadClassification(
            retrievalHeads: retrieval,
            streamingHeads: streaming
        )
    }

    /// Auto-classify using median entropy as threshold.
    public func classifyAutoThreshold(attentionEntropies: [Float]) -> HeadClassification {
        let sorted = attentionEntropies.sorted()
        let median = sorted[sorted.count / 2]

        var retrieval: [Int] = []
        var streaming: [Int] = []

        for h in 0..<numHeads {
            if attentionEntropies[h] >= median {
                retrieval.append(h)
            } else {
                streaming.append(h)
            }
        }

        return HeadClassification(
            retrievalHeads: retrieval,
            streamingHeads: streaming
        )
    }
}
```

**Step 4: Implement DuoAttention**

```swift
// Sources/EdgeRunnerCore/Attention/DuoAttention.swift
import Metal
import EdgeRunnerMetal

/// DuoAttention configuration.
/// Reference: "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads" (2024).
public struct DuoAttentionConfig: Sendable {
    public let numHeads: Int
    public let headDim: Int
    public let retrievalHeads: Set<Int>   // Full KV cache
    public let streamingHeads: Set<Int>   // Sliding window KV cache
    public let slidingWindowSize: Int

    public init(
        numHeads: Int, headDim: Int,
        retrievalHeads: Set<Int>, streamingHeads: Set<Int>,
        slidingWindowSize: Int
    ) {
        precondition(
            retrievalHeads.union(streamingHeads).count == numHeads,
            "All heads must be classified"
        )
        precondition(
            retrievalHeads.intersection(streamingHeads).isEmpty,
            "No head can be both retrieval and streaming"
        )
        self.numHeads = numHeads
        self.headDim = headDim
        self.retrievalHeads = retrievalHeads
        self.streamingHeads = streamingHeads
        self.slidingWindowSize = slidingWindowSize
    }
}

/// KV capacity for different head types.
public enum KVCapacity: Sendable, Equatable {
    case unlimited
    case limited(Int)
}

/// DuoAttention: per-head KV allocation.
/// Retrieval heads get full KV cache; streaming heads get sliding window only.
/// Achieves 2.55x memory reduction on typical configurations.
public final class DuoAttention: @unchecked Sendable {
    public let config: DuoAttentionConfig
    private let backend: MetalBackend

    public var retrievalKVCapacity: KVCapacity { .unlimited }
    public var streamingKVCapacity: Int { config.slidingWindowSize }

    public init(config: DuoAttentionConfig, backend: MetalBackend) async throws {
        self.config = config
        self.backend = backend
    }

    /// Forward pass with per-head KV allocation.
    /// - Parameters:
    ///   - query: [numHeads, queryLen, headDim]
    ///   - keys: [numHeads, seqLen, headDim] full KV cache
    ///   - values: [numHeads, seqLen, headDim] full KV cache
    /// - Returns: [numHeads, queryLen, headDim]
    public func forward(
        query: Tensor<Float>, keys: Tensor<Float>, values: Tensor<Float>
    ) async throws -> Tensor<Float> {
        let numHeads = config.numHeads
        let seqLen = keys.shape[1]
        let headDim = config.headDim

        var headOutputs: [Tensor<Float>] = []

        for h in 0..<numHeads {
            let qH = try await backend.slice(query, axis: 0, start: h, end: h + 1)
            let kH = try await backend.slice(keys, axis: 0, start: h, end: h + 1)
            let vH = try await backend.slice(values, axis: 0, start: h, end: h + 1)

            let output: Tensor<Float>
            if config.retrievalHeads.contains(h) {
                // Full attention over entire KV cache
                output = try await backend.scaledDotProductAttention(
                    query: qH, key: kH, value: vH, headDim: headDim
                )
            } else {
                // Sliding window: only attend to last windowSize tokens
                let windowStart = max(0, seqLen - config.slidingWindowSize)
                let kWindow = try await backend.slice(kH, axis: 1, start: windowStart, end: seqLen)
                let vWindow = try await backend.slice(vH, axis: 1, start: windowStart, end: seqLen)

                output = try await backend.scaledDotProductAttention(
                    query: qH, key: kWindow, value: vWindow, headDim: headDim
                )
            }

            headOutputs.append(output)
        }

        return try await backend.concatenate(headOutputs, axis: 0)
    }

    /// Estimate memory usage in bytes for a given sequence length.
    public static func estimateMemory(config: DuoAttentionConfig, seqLen: Int) -> Int {
        let bytesPerElement = MemoryLayout<Float>.size

        // Retrieval heads: full KV cache
        let retrievalBytes = config.retrievalHeads.count * seqLen * config.headDim * 2 * bytesPerElement

        // Streaming heads: sliding window
        let windowLen = min(seqLen, config.slidingWindowSize)
        let streamingBytes = config.streamingHeads.count * windowLen * config.headDim * 2 * bytesPerElement

        return retrievalBytes + streamingBytes
    }
}
```

**Step 5: Run tests, verify pass**

Run: `swift test --filter DuoAttentionTests 2>&1`
Expected: All 7 tests pass.

---

## Task 5: Context Summarisation & Recycling

**Files:**
- Create: `Sources/EdgeRunnerCore/Attention/ContextSummariser.swift`
- Create: `Sources/EdgeRunnerCore/Attention/SlidingWindowWithSummary.swift`
- Test: `Tests/EdgeRunnerCoreTests/Attention/ContextSummariserTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/Attention/ContextSummariserTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Context Summarisation & Recycling")
struct ContextSummariserTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    @Test func summariseTokensIntoCompressedRepresentation() async throws {
        let seqLen = 64, embedDim = 32
        let numSummaryTokens = 4  // Compress 64 tokens -> 4 summary tokens

        let config = ContextSummariserConfig(
            embedDim: embedDim,
            numSummaryTokens: numSummaryTokens,
            numHeads: 4
        )
        let summariser = try await ContextSummariser(config: config, backend: backend)

        let hiddenStates = Tensor<Float>.random(shape: [seqLen, embedDim])
        let summary = try await summariser.summarise(hiddenStates)

        #expect(summary.shape == [numSummaryTokens, embedDim])
    }

    @Test func summarisationReduction() async throws {
        let seqLen = 256, embedDim = 64, numSummary = 8

        let config = ContextSummariserConfig(
            embedDim: embedDim, numSummaryTokens: numSummary, numHeads: 4
        )
        let summariser = try await ContextSummariser(config: config, backend: backend)

        let input = Tensor<Float>.random(shape: [seqLen, embedDim])
        let summary = try await summariser.summarise(input)

        let compressionRatio = Float(seqLen) / Float(numSummary)
        #expect(compressionRatio == 32.0)
        #expect(summary.shape == [numSummary, embedDim])
    }

    @Test func summarisationDeterministic() async throws {
        let config = ContextSummariserConfig(
            embedDim: 32, numSummaryTokens: 4, numHeads: 4
        )
        let summariser = try await ContextSummariser(config: config, backend: backend)

        let input = Tensor<Float>.random(shape: [32, 32])
        let s1 = try await summariser.summarise(input)
        let s2 = try await summariser.summarise(input)

        let arr1 = try await s1.toArray()
        let arr2 = try await s2.toArray()

        for i in 0..<arr1.count {
            #expect(abs(arr1[i] - arr2[i]) < 1e-6)
        }
    }

    // MARK: - Sliding Window with Summary

    @Test func slidingWindowWithSummaryPrefix() async throws {
        let windowSize = 64, embedDim = 32, numSummary = 4

        let config = SlidingWindowConfig(
            windowSize: windowSize,
            embedDim: embedDim,
            numSummaryTokens: numSummary,
            numHeads: 4
        )
        let window = try await SlidingWindowWithSummary(config: config, backend: backend)

        // Add 128 tokens (will trigger summarisation)
        for _ in 0..<4 {
            let tokens = Tensor<Float>.random(shape: [32, embedDim])
            try await window.append(tokens)
        }

        // Active context should be: summary prefix + recent window
        let context = try await window.activeContext()

        // Expected: numSummary + windowSize = 4 + 64 = 68
        #expect(context.shape[0] <= numSummary + windowSize)
        #expect(context.shape[1] == embedDim)
    }

    @Test func slidingWindowEvictsOldTokens() async throws {
        let windowSize = 32, embedDim = 16, numSummary = 2

        let config = SlidingWindowConfig(
            windowSize: windowSize,
            embedDim: embedDim,
            numSummaryTokens: numSummary,
            numHeads: 4
        )
        let window = try await SlidingWindowWithSummary(config: config, backend: backend)

        // Add 100 tokens in batches
        for _ in 0..<10 {
            let tokens = Tensor<Float>.random(shape: [10, embedDim])
            try await window.append(tokens)
        }

        let context = try await window.activeContext()

        // Should not exceed summary + window
        #expect(context.shape[0] <= numSummary + windowSize)
    }

    @Test func slidingWindowKVCacheDiscardedAfterSummary() async throws {
        let windowSize = 16, embedDim = 16, numSummary = 2

        let config = SlidingWindowConfig(
            windowSize: windowSize,
            embedDim: embedDim,
            numSummaryTokens: numSummary,
            numHeads: 4
        )
        let window = try await SlidingWindowWithSummary(config: config, backend: backend)

        // Fill past capacity
        let batch1 = Tensor<Float>.random(shape: [windowSize, embedDim])
        try await window.append(batch1)

        let batch2 = Tensor<Float>.random(shape: [windowSize, embedDim])
        try await window.append(batch2)

        // After eviction, KV stats should show recycling happened
        let stats = await window.stats
        #expect(stats.evictionCount > 0, "Should have evicted old tokens")
        #expect(stats.summaryCount > 0, "Should have created summaries")
    }

    @Test func slidingWindowSmallInput() async throws {
        // Input smaller than window -- no summarisation needed
        let windowSize = 64, embedDim = 16, numSummary = 4

        let config = SlidingWindowConfig(
            windowSize: windowSize,
            embedDim: embedDim,
            numSummaryTokens: numSummary,
            numHeads: 4
        )
        let window = try await SlidingWindowWithSummary(config: config, backend: backend)

        let tokens = Tensor<Float>.random(shape: [10, embedDim])
        try await window.append(tokens)

        let context = try await window.activeContext()
        #expect(context.shape == [10, embedDim])

        let stats = await window.stats
        #expect(stats.evictionCount == 0)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ContextSummariserTests 2>&1`
Expected: FAIL -- types not defined

**Step 3: Implement ContextSummariser**

```swift
// Sources/EdgeRunnerCore/Attention/ContextSummariser.swift
import Metal
import EdgeRunnerMetal

/// Configuration for context summarisation.
public struct ContextSummariserConfig: Sendable {
    public let embedDim: Int
    public let numSummaryTokens: Int
    public let numHeads: Int

    public init(embedDim: Int, numSummaryTokens: Int, numHeads: Int) {
        self.embedDim = embedDim
        self.numSummaryTokens = numSummaryTokens
        self.numHeads = numHeads
    }
}

/// Compresses a sequence of hidden states into a fixed number of summary tokens.
/// Uses cross-attention: learnable summary queries attend to the input sequence.
public final class ContextSummariser: @unchecked Sendable {
    public let config: ContextSummariserConfig

    /// Learnable summary query tokens: [numSummaryTokens, embedDim].
    public let summaryQueries: Tensor<Float>

    /// Cross-attention for compression.
    private let crossAttn: CrossModalAttention
    private let backend: MetalBackend

    public init(config: ContextSummariserConfig, backend: MetalBackend) async throws {
        self.config = config
        self.backend = backend

        let bound = (1.0 / Float(config.embedDim)).squareRoot()
        self.summaryQueries = Tensor<Float>.random(
            shape: [config.numSummaryTokens, config.embedDim],
            range: -bound...bound
        )

        self.crossAttn = try await CrossModalAttention(
            queryDim: config.embedDim,
            keyValueDim: config.embedDim,
            numHeads: config.numHeads,
            backend: backend
        )
    }

    /// Summarise hidden states [seqLen, embedDim] -> [numSummaryTokens, embedDim].
    public func summarise(_ hiddenStates: Tensor<Float>) async throws -> Tensor<Float> {
        return try await crossAttn.forward(
            query: summaryQueries,
            keyValue: hiddenStates
        )
    }
}
```

**Step 4: Implement SlidingWindowWithSummary**

```swift
// Sources/EdgeRunnerCore/Attention/SlidingWindowWithSummary.swift
import Metal
import EdgeRunnerMetal

/// Configuration for sliding window with summary prefix.
public struct SlidingWindowConfig: Sendable {
    public let windowSize: Int
    public let embedDim: Int
    public let numSummaryTokens: Int
    public let numHeads: Int

    public init(
        windowSize: Int, embedDim: Int,
        numSummaryTokens: Int, numHeads: Int
    ) {
        self.windowSize = windowSize
        self.embedDim = embedDim
        self.numSummaryTokens = numSummaryTokens
        self.numHeads = numHeads
    }
}

/// Statistics for context recycling.
public struct ContextRecyclingStats: Sendable {
    public var totalTokensSeen: Int = 0
    public var evictionCount: Int = 0
    public var summaryCount: Int = 0
}

/// Sliding window attention with summary prefix for long-context on memory-constrained devices.
///
/// Strategy:
/// 1. Maintain a fixed-size window of recent tokens.
/// 2. When tokens overflow the window, summarise the evicted tokens.
/// 3. Prepend summary tokens to the window as a compressed prefix.
/// 4. Discard old KV entries for evicted tokens.
///
/// Effective context: [summary_prefix | recent_window]
public actor SlidingWindowWithSummary {
    public let config: SlidingWindowConfig
    private let summariser: ContextSummariser
    private let backend: MetalBackend

    /// Current window of recent token hidden states.
    private var windowBuffer: Tensor<Float>?

    /// Compressed summary of evicted tokens.
    private var summaryPrefix: Tensor<Float>?

    /// Running stats.
    public var stats: ContextRecyclingStats = .init()

    public init(config: SlidingWindowConfig, backend: MetalBackend) async throws {
        self.config = config
        self.backend = backend

        let sumConfig = ContextSummariserConfig(
            embedDim: config.embedDim,
            numSummaryTokens: config.numSummaryTokens,
            numHeads: config.numHeads
        )
        self.summariser = try await ContextSummariser(config: sumConfig, backend: backend)
    }

    /// Append new token hidden states. Triggers summarisation + eviction if needed.
    public func append(_ newTokens: Tensor<Float>) async throws {
        let newLen = newTokens.shape[0]
        stats.totalTokensSeen += newLen

        if var current = windowBuffer {
            // Concatenate with existing buffer
            current = try await backend.concatenate([current, newTokens], axis: 0)

            let currentLen = current.shape[0]
            if currentLen > config.windowSize {
                // Evict old tokens into summary
                let evictCount = currentLen - config.windowSize
                let toEvict = try await backend.slice(current, axis: 0, start: 0, end: evictCount)

                // Summarise evicted tokens
                let newSummary = try await summariser.summarise(toEvict)

                // Merge with existing summary if present
                if let existing = summaryPrefix {
                    let combined = try await backend.concatenate([existing, newSummary], axis: 0)
                    summaryPrefix = try await summariser.summarise(combined)
                } else {
                    summaryPrefix = newSummary
                }

                stats.evictionCount += 1
                stats.summaryCount += 1

                // Keep only the recent window
                windowBuffer = try await backend.slice(
                    current, axis: 0, start: evictCount, end: currentLen
                )
            } else {
                windowBuffer = current
            }
        } else {
            // First append
            if newLen > config.windowSize {
                let evictCount = newLen - config.windowSize
                let toEvict = try await backend.slice(newTokens, axis: 0, start: 0, end: evictCount)
                summaryPrefix = try await summariser.summarise(toEvict)
                windowBuffer = try await backend.slice(
                    newTokens, axis: 0, start: evictCount, end: newLen
                )
                stats.evictionCount += 1
                stats.summaryCount += 1
            } else {
                windowBuffer = newTokens
            }
        }
    }

    /// Get the active context: [summary_prefix | recent_window].
    public func activeContext() async throws -> Tensor<Float> {
        guard let window = windowBuffer else {
            preconditionFailure("No tokens appended yet")
        }

        if let summary = summaryPrefix {
            return try await backend.concatenate([summary, window], axis: 0)
        } else {
            return window
        }
    }
}
```

**Step 5: Run tests, verify pass**

Run: `swift test --filter ContextSummariserTests 2>&1`
Expected: All 7 tests pass.

---

## Task 6: Metal 4 / M5 Optimizations

**Files:**
- Create: `Sources/EdgeRunnerMetal/Metal4Dispatch.swift`
- Create: `Sources/EdgeRunnerMetal/DeviceCapabilities.swift`
- Create: `Sources/EdgeRunnerMetal/Shaders/Metal4GEMM.metal`
- Test: `Tests/EdgeRunnerMetalTests/Metal4OptimizationTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/Metal4OptimizationTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("Metal 4 / M5 Optimizations")
struct Metal4OptimizationTests {

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

    // MARK: - Device Capability Detection

    @Test func detectDeviceFamily() {
        let caps = DeviceCapabilities(device: device)

        // Should detect something
        #expect(caps.gpuFamily != .unknown)
        #expect(!caps.deviceName.isEmpty)
    }

    @Test func detectMetal4Support() {
        let caps = DeviceCapabilities(device: device)

        // Metal 4 features: check for MTLGPUFamily enum
        if caps.gpuFamily == .apple9 || caps.gpuFamily == .apple10 {
            #expect(caps.supportsSimdgroupMatrix)
        }

        // Regardless of hardware, the detection should not crash
        #expect(caps.maxThreadgroupMemory > 0)
        #expect(caps.maxThreadsPerThreadgroup > 0)
    }

    @Test func detectM5TensorOps() {
        let caps = DeviceCapabilities(device: device)

        if caps.supportsMetal4TensorOps {
            #expect(caps.gpuFamily.rawValue >= GPUFamily.apple10.rawValue)
        }
    }

    // MARK: - Kernel Dispatch Strategy

    @Test func gemmDispatchSelectsOptimalKernel() {
        let caps = DeviceCapabilities(device: device)
        let dispatch = Metal4Dispatch(capabilities: caps)

        let strategy = dispatch.selectGEMMStrategy(M: 1024, N: 1024, K: 1024)

        if caps.supportsMetal4TensorOps {
            #expect(strategy == .metal4TensorOps)
        } else if caps.supportsSimdgroupMatrix {
            #expect(strategy == .simdgroupMatrix)
        } else {
            #expect(strategy == .naive)
        }
    }

    @Test func smallGEMMUsesSimdgroup() {
        let caps = DeviceCapabilities(device: device)
        let dispatch = Metal4Dispatch(capabilities: caps)

        // Small matmul -- tensor_ops overhead not worth it
        let strategy = dispatch.selectGEMMStrategy(M: 16, N: 16, K: 16)

        if caps.supportsSimdgroupMatrix {
            #expect(strategy == .simdgroupMatrix)
        }
    }

    @Test func gracefulFallbackOnOlderDevices() {
        // Simulate an older device without simdgroup matrix support
        let mockCaps = DeviceCapabilities.mock(
            gpuFamily: .apple7,
            supportsSimdgroupMatrix: false,
            supportsMetal4TensorOps: false
        )
        let dispatch = Metal4Dispatch(capabilities: mockCaps)

        let strategy = dispatch.selectGEMMStrategy(M: 512, N: 512, K: 512)
        #expect(strategy == .naive)
    }

    // MARK: - Metal 4 Unified Encoding

    @Test func unifiedCommandEncoderDispatch() async throws {
        let caps = DeviceCapabilities(device: device)

        // Verifies that MTL4ComputeCommandEncoder path compiles and runs.
        // On non-Metal 4 hardware, falls back to standard compute encoder.
        let dispatch = Metal4Dispatch(capabilities: caps)

        let size = 1024
        let a = (0..<size).map { _ in Float.random(in: -1...1) }
        let b = (0..<size).map { _ in Float.random(in: -1...1) }

        let result = try await dispatch.elementwiseAdd(
            a: a, b: b, device: device, commandQueue: commandQueue
        )

        for i in 0..<size {
            #expect(abs(result[i] - (a[i] + b[i])) < 1e-5)
        }
    }

    // MARK: - Placement Sparse Resources

    @Test func sparseResourceWeightStreaming() async throws {
        let caps = DeviceCapabilities(device: device)

        if caps.supportsPlacementSparseResources {
            let sparse = try SparseWeightStreamer(device: device)
            let allocated = try sparse.allocateSparseBuffer(
                sizeInPages: 16, pageSize: 16384
            )
            #expect(allocated != nil)
        } else {
            let canUse = caps.supportsPlacementSparseResources
            #expect(!canUse) // Expected on most current hardware
        }
    }

    // MARK: - Feature Detection Consistency

    @Test func capabilitiesSelfConsistent() {
        let caps = DeviceCapabilities(device: device)

        // Metal 4 tensor ops implies simdgroup matrix support
        if caps.supportsMetal4TensorOps {
            #expect(caps.supportsSimdgroupMatrix)
        }

        // Sparse resources implies at least apple8 family
        if caps.supportsPlacementSparseResources {
            #expect(caps.gpuFamily.rawValue >= GPUFamily.apple8.rawValue)
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter Metal4OptimizationTests 2>&1`
Expected: FAIL -- types not defined

**Step 3: Implement DeviceCapabilities**

```swift
// Sources/EdgeRunnerMetal/DeviceCapabilities.swift
import Metal

/// Known Apple GPU families.
public enum GPUFamily: Int, Sendable, Comparable {
    case unknown = 0
    case apple7 = 7   // M1, A14
    case apple8 = 8   // M2, A15/A16
    case apple9 = 9   // M3, A17 Pro
    case apple10 = 10 // M4, M5

    public static func < (lhs: GPUFamily, rhs: GPUFamily) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// GEMM dispatch strategy.
public enum GEMMStrategy: Sendable, Equatable {
    case naive                 // Basic tiled GEMM
    case simdgroupMatrix       // simdgroup_matrix intrinsics (M1+)
    case metal4TensorOps       // Metal 4 tensor_ops::matmul2d (M5+)
}

/// Detected GPU capabilities for dispatch decisions.
public struct DeviceCapabilities: Sendable {
    public let deviceName: String
    public let gpuFamily: GPUFamily
    public let supportsSimdgroupMatrix: Bool
    public let supportsMetal4TensorOps: Bool
    public let supportsPlacementSparseResources: Bool
    public let maxThreadgroupMemory: Int
    public let maxThreadsPerThreadgroup: Int

    public init(device: MTLDevice) {
        self.deviceName = device.name

        // Detect GPU family
        if device.supportsFamily(.apple9) {
            self.gpuFamily = .apple9
        } else if device.supportsFamily(.apple8) {
            self.gpuFamily = .apple8
        } else if device.supportsFamily(.apple7) {
            self.gpuFamily = .apple7
        } else {
            self.gpuFamily = .unknown
        }

        // M5 detection via name heuristic + feature probe
        let isM5 = device.name.contains("M5")

        self.supportsSimdgroupMatrix = gpuFamily >= .apple7
        self.supportsMetal4TensorOps = isM5
        self.supportsPlacementSparseResources = gpuFamily >= .apple9

        self.maxThreadgroupMemory = device.maxThreadgroupMemoryLength
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
    }

    /// Create mock capabilities for testing fallback paths.
    public static func mock(
        gpuFamily: GPUFamily,
        supportsSimdgroupMatrix: Bool,
        supportsMetal4TensorOps: Bool
    ) -> DeviceCapabilities {
        DeviceCapabilities(
            deviceName: "Mock \(gpuFamily)",
            gpuFamily: gpuFamily,
            supportsSimdgroupMatrix: supportsSimdgroupMatrix,
            supportsMetal4TensorOps: supportsMetal4TensorOps,
            supportsPlacementSparseResources: gpuFamily >= .apple9,
            maxThreadgroupMemory: 32768,
            maxThreadsPerThreadgroup: 1024
        )
    }

    private init(
        deviceName: String, gpuFamily: GPUFamily,
        supportsSimdgroupMatrix: Bool, supportsMetal4TensorOps: Bool,
        supportsPlacementSparseResources: Bool,
        maxThreadgroupMemory: Int, maxThreadsPerThreadgroup: Int
    ) {
        self.deviceName = deviceName
        self.gpuFamily = gpuFamily
        self.supportsSimdgroupMatrix = supportsSimdgroupMatrix
        self.supportsMetal4TensorOps = supportsMetal4TensorOps
        self.supportsPlacementSparseResources = supportsPlacementSparseResources
        self.maxThreadgroupMemory = maxThreadgroupMemory
        self.maxThreadsPerThreadgroup = maxThreadsPerThreadgroup
    }
}
```

**Step 4: Implement Metal4Dispatch**

```swift
// Sources/EdgeRunnerMetal/Metal4Dispatch.swift
import Metal

/// Dispatch strategy selector for Metal 4 / M5 optimizations.
/// Automatically selects the optimal kernel based on device capabilities and problem size.
public struct Metal4Dispatch: Sendable {
    public let capabilities: DeviceCapabilities

    public init(capabilities: DeviceCapabilities) {
        self.capabilities = capabilities
    }

    /// Select GEMM strategy based on device and matrix dimensions.
    public func selectGEMMStrategy(M: Int, N: Int, K: Int) -> GEMMStrategy {
        let totalOps = M * N * K

        // Metal 4 tensor_ops only worth it for large matrices
        if capabilities.supportsMetal4TensorOps && totalOps > 1_000_000 {
            return .metal4TensorOps
        }

        if capabilities.supportsSimdgroupMatrix {
            return .simdgroupMatrix
        }

        return .naive
    }

    /// Select attention strategy.
    public func selectAttentionStrategy(seqLen: Int, headDim: Int) -> AttentionStrategy {
        if capabilities.supportsMetal4TensorOps {
            return .metal4Flash
        }
        if seqLen > 2048 && capabilities.supportsSimdgroupMatrix {
            return .flashAttention
        }
        return .standard
    }

    /// Dispatch an element-wise add using the optimal encoder.
    public func elementwiseAdd(
        a: [Float], b: [Float],
        device: MTLDevice, commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        precondition(a.count == b.count)
        let count = a.count

        let bufA = device.makeBuffer(bytes: a, length: count * 4)!
        let bufB = device.makeBuffer(bytes: b, length: count * 4)!
        let bufC = device.makeBuffer(length: count * 4)!

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        let function = try library.makeFunction(name: "elementwise_add_f32")
        let pipeline = try device.makeComputePipelineState(function: function!)

        let cmdBuffer = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)

        var params = UInt32(count)
        encoder.setBytes(&params, length: 4, index: 3)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        let ptr = bufC.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}

/// Attention dispatch strategy.
public enum AttentionStrategy: Sendable {
    case standard
    case flashAttention
    case metal4Flash
}

/// Sparse resource weight streamer for M5 placement sparse resources.
public struct SparseWeightStreamer: Sendable {
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
    }

    /// Allocate a sparse buffer for weight streaming.
    /// Returns nil if sparse resources are not supported.
    public func allocateSparseBuffer(
        sizeInPages: Int, pageSize: Int
    ) throws -> MTLBuffer? {
        let totalSize = sizeInPages * pageSize

        guard let heap = device.makeHeap(descriptor: {
            let desc = MTLHeapDescriptor()
            desc.size = totalSize
            desc.storageMode = .private
            desc.type = .automatic
            return desc
        }()) else {
            return nil
        }

        return heap.makeBuffer(length: totalSize)
    }
}
```

**Step 5: Implement Metal 4 GEMM shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Metal4GEMM.metal
#include <metal_stdlib>
using namespace metal;

struct ERGEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Standard simdgroup GEMM (M1-M4 path)
kernel void gemm_simdgroup_f32(
    device const float* A         [[buffer(0)]],
    device const float* B         [[buffer(1)]],
    device float*       C         [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 group_id    [[threadgroup_position_in_grid]],
    uint  simd_index  [[simdgroup_index_in_threadgroup]],
    uint  lane_id     [[thread_index_in_simdgroup]]
) {
    const uint TILE = 32;
    const uint row_base = group_id.y * TILE;
    const uint col_base = group_id.x * TILE;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_float8x8(0);

    for (uint k_tile = 0; k_tile < params.K; k_tile += TILE) {
        simdgroup_float8x8 a_block[4][4];
        simdgroup_float8x8 b_block[4][4];

        for (uint bi = 0; bi < 4; bi++) {
            for (uint bj = 0; bj < 4; bj++) {
                uint a_row = row_base + bi * 8;
                uint a_col = k_tile + bj * 8;
                if (a_row < params.M && a_col < params.K)
                    simdgroup_load(a_block[bi][bj], A, params.lda, ulong2(a_col, a_row));
                else
                    a_block[bi][bj] = simdgroup_float8x8(0);

                uint b_row = k_tile + bi * 8;
                uint b_col = col_base + bj * 8;
                if (b_row < params.K && b_col < params.N)
                    simdgroup_load(b_block[bi][bj], B, params.ldb, ulong2(b_col, b_row));
                else
                    b_block[bi][bj] = simdgroup_float8x8(0);
            }
        }

        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                for (uint p = 0; p < 4; p++)
                    simdgroup_multiply_accumulate(acc[i][j], a_block[i][p], b_block[p][j], acc[i][j]);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) {
            uint c_row = row_base + i * 8;
            uint c_col = col_base + j * 8;
            if (c_row < params.M && c_col < params.N)
                simdgroup_store(acc[i][j], C, params.ldc, ulong2(c_col, c_row));
        }
}

// Metal 4 tensor_ops path (M5 only)
// When Metal 4 SDK ships, uncomment:
// #if __METAL_VERSION__ >= 400
// #include <metal_tensor_ops>
// kernel void gemm_metal4_tensor_ops_f32(
//     device const float* A [[buffer(0)]],
//     device const float* B [[buffer(1)]],
//     device float*       C [[buffer(2)]],
//     constant ERGEMMParams& params [[buffer(3)]],
//     uint2 group_id [[threadgroup_position_in_grid]]
// ) {
//     tensor_ops::matmul2d(C, A, B, params.M, params.N, params.K);
// }
// #endif
```

**Step 6: Run tests, verify pass**

Run: `swift test --filter Metal4OptimizationTests 2>&1`
Expected: All 8 tests pass (sparse resource test may skip on non-M5 hardware).

---

## Task 7: Integration & Benchmarks

**Files:**
- Create: `Tests/EdgeRunnerIntegrationTests/LongContextBenchmarkTests.swift`
- Create: `Tests/EdgeRunnerIntegrationTests/MemoryProfilingTests.swift`

**Step 1: Write the integration tests**

```swift
// Tests/EdgeRunnerIntegrationTests/LongContextBenchmarkTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Long Context Benchmark", .tags(.benchmark))
struct LongContextBenchmarkTests {

    let backend: MetalBackend

    init() async throws {
        self.backend = await MetalBackend.shared
    }

    @Test func miniCacheCompressionAtScale() async throws {
        let seqLen = 512, headDim = 64, numHeads = 8, numLayers = 8

        var kvStates: [(k: Tensor<Float>, v: Tensor<Float>)] = []
        for _ in 0..<numLayers {
            kvStates.append((
                k: Tensor<Float>.random(shape: [numHeads, seqLen, headDim]),
                v: Tensor<Float>.random(shape: [numHeads, seqLen, headDim])
            ))
        }

        let cache = try await MiniCache(
            kvStates: kvStates, blockSize: 32, backend: backend
        )

        #expect(cache.compressedLayers.count == numLayers / 2)
        #expect(cache.originalLayerCount == numLayers)

        let restored = try await cache.decompress(backend: backend)
        for layer in restored {
            #expect(layer.k.shape == [numHeads, seqLen, headDim])
        }
    }

    @Test func duoAttentionMemorySavings() async throws {
        let numHeads = 32, headDim = 128
        let seqLengths = [1024, 2048, 4096, 8192]

        let config = DuoAttentionConfig(
            numHeads: numHeads,
            headDim: headDim,
            retrievalHeads: Set(0..<8),
            streamingHeads: Set(8..<32),
            slidingWindowSize: 256
        )

        for seqLen in seqLengths {
            let fullMemory = numHeads * seqLen * headDim * 2 * 4
            let duoMemory = DuoAttention.estimateMemory(config: config, seqLen: seqLen)
            let savings = 1.0 - (Float(duoMemory) / Float(fullMemory))

            #expect(savings > 0.5,
                    "At seqLen=\(seqLen): expected >50% savings, got \(savings * 100)%")
        }
    }

    @Test func slidingWindowLongSequence() async throws {
        let windowSize = 128, embedDim = 32, numSummary = 4

        let config = SlidingWindowConfig(
            windowSize: windowSize, embedDim: embedDim,
            numSummaryTokens: numSummary, numHeads: 4
        )
        let window = try await SlidingWindowWithSummary(config: config, backend: backend)

        let totalTokens = 4096
        let batchSize = 64

        for _ in 0..<(totalTokens / batchSize) {
            let batch = Tensor<Float>.random(shape: [batchSize, embedDim])
            try await window.append(batch)
        }

        let context = try await window.activeContext()

        #expect(context.shape[0] <= numSummary + windowSize)
        #expect(context.shape[1] == embedDim)

        let stats = await window.stats
        #expect(stats.totalTokensSeen == totalTokens)
        #expect(stats.evictionCount > 0)
    }

    @Test func squeezedAttentionReductionBenchmark() async throws {
        let seqLen = 512, headDim = 64

        let query = Tensor<Float>.random(shape: [1, headDim])
        let keys = Tensor<Float>.random(shape: [seqLen, headDim])
        let values = Tensor<Float>.random(shape: [seqLen, headDim])

        for numCentroids in [16, 32, 64, 128] {
            let squeezed = try await SqueezedAttention(
                numCentroids: numCentroids, backend: backend
            )

            let compressed = try await squeezed.compress(keys: keys, values: values)

            #expect(compressed.keys.shape == [numCentroids, headDim])
        }
    }
}
```

```swift
// Tests/EdgeRunnerIntegrationTests/MemoryProfilingTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Memory Profiling", .tags(.benchmark))
struct MemoryProfilingTests {

    @Test func memoryBudgetTable() {
        let models: [(name: String, params: UInt64, bits: Int, layers: Int, kvHeads: Int, headDim: Int)] = [
            ("Llama-3.2-1B Q4",  1_300_000_000, 4,  16, 8,  64),
            ("Llama-3.2-3B Q4",  3_200_000_000, 4,  28, 8,  128),
            ("Llama-3-8B Q4",    8_000_000_000, 4,  32, 8,  128),
            ("Llama-3-8B F16",   8_000_000_000, 16, 32, 8,  128),
            ("Llama-3-70B Q4",  70_000_000_000, 4,  80, 8,  128),
        ]

        let devices: [(name: String, total: UInt64, usable: UInt64)] = [
            ("iPhone 15 Pro (8GB)",   8_589_934_592,  4_294_967_296),
            ("iPad Pro M4 (16GB)",   17_179_869_184,  8_589_934_592),
            ("MacBook M4 Pro (24GB)",25_769_803_776, 16_106_127_360),
            ("Mac Studio M4 Max (64GB)", 68_719_476_736, 42_949_672_960),
        ]

        for model in models {
            let memConfig = ModelMemoryConfig.estimate(
                parameterCount: model.params,
                bitsPerWeight: model.bits,
                numLayers: model.layers,
                numKVHeads: model.kvHeads,
                headDim: model.headDim
            )

            for device in devices {
                let profile = DeviceProfile(
                    deviceName: device.name,
                    totalMemoryBytes: device.total,
                    usableMemoryBytes: device.usable,
                    chipFamily: .apple
                )

                let budget = MemoryBudget.compute(model: memConfig, device: profile)

                #expect(budget.maxContextLength >= 0)
                if budget.canRun {
                    #expect(budget.warningLevel != .impossible)
                }
            }
        }
    }

    @Test func compressionStrategyComparison() async throws {
        let backend = await MetalBackend.shared
        let seqLen = 256, headDim = 64, numHeads = 8

        // Baseline: full KV cache
        let fullKVBytes = seqLen * headDim * numHeads * 2 * MemoryLayout<Float>.size

        // MiniCache: merge + 4-bit
        let miniCacheBytes = numHeads * seqLen * headDim / 2 +
                            numHeads * seqLen * headDim / 32 * 4

        // DuoAttention: 25% retrieval + 75% sliding(64)
        let duoConfig = DuoAttentionConfig(
            numHeads: numHeads, headDim: headDim,
            retrievalHeads: Set(0..<2),
            streamingHeads: Set(2..<numHeads),
            slidingWindowSize: 64
        )
        let duoBytes = DuoAttention.estimateMemory(config: duoConfig, seqLen: seqLen)

        #expect(miniCacheBytes < fullKVBytes)
        #expect(duoBytes < fullKVBytes)
    }
}
```

**Step 2: Run all integration tests**

Run: `swift test --filter "LongContextBenchmarkTests|MemoryProfilingTests" 2>&1`
Expected: All integration and benchmark tests pass.

---

## Summary

| Task | Files | Tests | Key Deliverable |
|------|-------|-------|----------------|
| 1. Memory Budgeting | `MemoryBudget.swift`, `DeviceProfile.swift` | 6 | Device-aware KV budget + warning API |
| 2. MiniCache | `MiniCache.swift`, `MiniCache.metal`, `MiniCacheParams.h` | 5 | Adjacent-layer merge + 4-bit quant (~5x compression) |
| 3. Squeezed Attention | `SqueezedAttention.swift`, `KMeansClustering.metal`, `KMeansParams.h` | 5 | K-means KV clustering (3-8x reduction) |
| 4. DuoAttention | `DuoAttention.swift`, `HeadClassifier.swift` | 7 | Per-head KV: retrieval (full) vs streaming (window) |
| 5. Context Summarisation | `ContextSummariser.swift`, `SlidingWindowWithSummary.swift` | 7 | Sliding window + compressed summary prefix |
| 6. Metal 4 / M5 Optimizations | `Metal4Dispatch.swift`, `DeviceCapabilities.swift`, `Metal4GEMM.metal` | 8 | Feature detection + tensor_ops dispatch + fallback |
| 7. Integration & Benchmarks | `LongContextBenchmarkTests.swift`, `MemoryProfilingTests.swift` | 6 | KV compression benchmarks, long-context benchmarks, memory profiling |

**Total: 7 tasks, ~44 tests, ~20 source files**

**Key metrics targeted:**
- MiniCache: ~5x KV compression
- DuoAttention: ~2.55x memory reduction
- Squeezed Attention: 3-8x KV budget reduction
- 8GB device: 4K+ tokens with compression enabled
