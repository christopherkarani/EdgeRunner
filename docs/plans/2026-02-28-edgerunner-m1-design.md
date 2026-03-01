# EdgeRunner Milestone 1: Core Tensor & Metal Infrastructure — Design Document

**Date:** 2026-02-28
**Author:** Christopher Karani + Claude
**Status:** Approved

## Goal

Build the foundational tensor computation layer for EdgeRunner: a Metal 4-native, Swift 6.2 inference engine. This milestone delivers `Tensor<T>` with copy-on-write semantics, a lazy evaluation graph with 3-tier kernel fusion, and a production-quality Metal backend with buffer caching, residency management, and command batching.

## Architecture Summary

EdgeRunner Milestone 1 uses a **hybrid MTLBuffer + MTLTensor** architecture:
- **Storage**: `MTLBuffer` (shared storage mode) for CPU access, memory mapping, and buffer cache recycling
- **Dispatch**: Buffer-backed `MTLTensor` views with explicit strides for GPU kernel dispatch via Metal 4 APIs
- **Fusion**: 3-tier strategy — function constants (hot), function stitching (warm), JIT compilation (cold)

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment targets | iOS 26+ / macOS 26+ | Metal 4 only, no dual codepaths |
| Package structure | Multi-target from start | Clean separation of Metal, Core, and public API |
| Metal compilation | Custom BuildToolPlugin | SwiftPM auto-compile doesn't pass `-I` flags for shared headers |
| Tensor generic | `Tensor<T: TensorScalar>` protocol-constrained | Type safety, extensibility |
| Buffer management | LRU cache (primary) + small MTLHeap | Validated by MLX architecture |
| Hazard tracking | Manual (`HazardTrackingModeUntracked`) | Better performance, MLX validates |
| Fusion | 3-tier: function constants + stitching + JIT | Function stitching is the key insight — composes pre-compiled ops at AIR level |
| Concurrency | Actor-based `MetalBackend`, `Mutex<T>` for caches | No `@unchecked Sendable`, no global locks |
| Residency | Single `MTLResidencySet`, queue-attached | Apple recommends few large sets, max 32 per queue |
| Command batching | 20-50 dispatches per command buffer, per-chip | MLX-validated pattern |

## Package Structure

```
EdgeRunner/
├── Package.swift                          # Swift 6.2, iOS 26+ / macOS 26+
├── Sources/
│   ├── EdgeRunnerSharedTypes/             # C target: Metal ↔ Swift type bridging
│   │   ├── include/ShaderTypes.h          # Dispatch params, dtype enums, tensor metadata
│   │   └── ShaderTypes.c                  # Required stub for SwiftPM C target
│   │
│   ├── EdgeRunnerMetal/                   # GPU backend (depends on SharedTypes)
│   │   ├── Shaders/
│   │   │   ├── Elementwise.metal          # add, sub, mul, div, broadcast
│   │   │   ├── Reduction.metal            # sum, mean, max
│   │   │   ├── Transpose.metal            # Tiled matrix transpose
│   │   │   ├── StitchableOps.metal        # [[stitchable]] op vocabulary for fusion
│   │   │   └── FusedPatterns.metal        # Function-constant hot-path kernels
│   │   ├── MetalBackend.swift             # Actor: MTLDevice, queue, allocator pool
│   │   ├── KernelRegistry.swift           # Pipeline cache + stitched library cache
│   │   ├── CommandBatcher.swift           # Per-chip batch limits (20-50 dispatches)
│   │   ├── BufferCache.swift              # LRU buffer cache keyed by size
│   │   ├── ResidencyManager.swift         # Single MTLResidencySet, queue-attached
│   │   └── BarrierTracker.swift           # Manual hazard tracking
│   │
│   ├── EdgeRunnerCore/                    # Tensor & graph (depends on Metal)
│   │   ├── Tensor.swift                   # Tensor<T: TensorScalar> with COW
│   │   ├── TensorScalar.swift             # Protocol + Float/Float16/Int8/UInt8
│   │   ├── TensorStorage.swift            # MTLBuffer + MTLTensor view factory
│   │   ├── Shape.swift                    # Shape, strides, broadcast, contiguity
│   │   ├── Graph/
│   │   │   ├── TensorOp.swift             # DAG nodes (operation + inputs)
│   │   │   ├── ComputeGraph.swift         # DAG build, topological sort, BFS eval
│   │   │   └── FusionEngine.swift         # 3-tier: constants → stitch → JIT
│   │   └── AutoTuner.swift                # Threadgroup/tile config per device
│   │
│   └── EdgeRunner/                        # Public API facade (re-exports)
│       └── EdgeRunner.swift
│
├── Tests/
│   ├── EdgeRunnerMetalTests/              # GPU vs CPU reference tests
│   └── EdgeRunnerCoreTests/               # Shape, graph, fusion logic tests
│
├── Benchmarks/
│   └── EdgeRunnerBenchmarks/main.swift    # Auto-tuner + performance regression tracking
│
└── Plugins/
    ├── MetalShaderPlugin/                 # BuildToolPlugin: .metal → .metallib
    │   └── MetalShaderPlugin.swift
    └── Sources/MetalCompilerTool/         # Executable: xcrun metal/metallib
        └── MetalCompilerTool.swift
```

### Target Dependencies

```
EdgeRunnerSharedTypes (C)
        │
        ▼
EdgeRunnerMetal (Swift + Metal)
        │
        ▼
EdgeRunnerCore (Swift)
        │
        ▼
EdgeRunner (Swift, public facade)
```

## Component Designs

### 1. EdgeRunnerSharedTypes (C Target)

Shared C header defining structs that must have identical layout in Metal and Swift:

```c
// ShaderTypes.h
#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include <simd/simd.h>

typedef enum {
    ERDTypeFloat32 = 0,
    ERDTypeFloat16 = 1,
    ERDTypeInt8    = 2,
    ERDTypeUInt8   = 3,
} ERDType;

typedef struct {
    uint32_t elementCount;
    uint32_t offset;       // byte offset into buffer
    ERDType  dtype;
} ERElementwiseParams;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t rowStride;
    uint32_t colStride;
} ERTransposeParams;

typedef struct {
    uint32_t elementCount;
    uint32_t reductionDim;
    uint32_t outerSize;
    uint32_t innerSize;
} ERReductionParams;

#endif
```

### 2. TensorScalar Protocol

```swift
public protocol TensorScalar: Sendable, BitwiseCopyable {
    static var metalDataType: MTLDataType { get }
    static var byteSize: Int { get }
    static var erDType: ERDType { get }
}

extension Float: TensorScalar {
    public static let metalDataType: MTLDataType = .float
    public static let byteSize = 4
    public static let erDType: ERDType = .ERDTypeFloat32
}

extension Float16: TensorScalar {
    public static let metalDataType: MTLDataType = .half
    public static let byteSize = 2
    public static let erDType: ERDType = .ERDTypeFloat16
}

extension Int8: TensorScalar {
    public static let metalDataType: MTLDataType = .char
    public static let byteSize = 1
    public static let erDType: ERDType = .ERDTypeInt8
}

extension UInt8: TensorScalar {
    public static let metalDataType: MTLDataType = .uchar
    public static let byteSize = 1
    public static let erDType: ERDType = .ERDTypeUInt8
}
```

### 3. Tensor<T> with COW

```swift
public struct Tensor<T: TensorScalar>: Sendable {
    // COW storage — shared reference to MTLBuffer
    private var _storage: TensorStorage

    // Metadata
    public let shape: Shape
    public let strides: Strides

    // Lazy graph node (nil if already realized)
    private let graphNode: TensorOp?

    // Whether data has been computed
    public var isRealized: Bool { graphNode == nil }

    // COW mutation
    private mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&_storage) {
            _storage = _storage.copy()
        }
    }

    // Materialize the computation graph
    public func realize() async throws -> Tensor<T> {
        guard let node = graphNode else { return self }
        let backend = MetalBackend.shared
        return try await backend.evaluate(node)
    }

    // Element-wise operators create new graph nodes
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor { ... }
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor { ... }
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor { ... }
    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor { ... }
    public func relu() -> Tensor { ... }
    public func sigmoid() -> Tensor { ... }
    public func gelu() -> Tensor { ... }
    public func sum(axis: Int?) -> Tensor { ... }
    public func mean(axis: Int?) -> Tensor { ... }
    public func max(axis: Int?) -> Tensor { ... }
    public func transpose() -> Tensor { ... }
    public func reshape(_ newShape: Shape) -> Tensor { ... }
}
```

### 4. TensorStorage

```swift
final class TensorStorage: @unchecked Sendable {
    // Note: @unchecked because MTLBuffer is not Sendable but is thread-safe
    // when accessed through actor-isolated MetalBackend
    let buffer: MTLBuffer
    let byteCount: Int

    // Create MTLTensor view for GPU dispatch
    func makeTensorView(
        shape: [Int],
        strides: [Int],
        dataType: MTLDataType
    ) -> MTLTensor {
        let descriptor = MTLTensorDescriptor()
        descriptor.dataType = dataType
        descriptor.dimensions = shape.map { NSNumber(value: $0) }
        descriptor.strides = strides.map { NSNumber(value: $0) }
        return buffer.makeTensor(descriptor: descriptor, offset: 0)
    }

    func copy() -> TensorStorage {
        // Allocate new buffer from cache, copy contents
        ...
    }
}
```

### 5. MetalBackend (Actor)

```swift
public actor MetalBackend {
    public static let shared = MetalBackend()

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Triple-buffered command allocators
    private var allocators: [MTL4CommandAllocator]
    private var allocatorIndex = 0

    // Sub-systems
    private let bufferCache: BufferCache         // LRU, Mutex-protected
    private let kernelRegistry: KernelRegistry   // Pipeline cache
    private let residencyManager: ResidencyManager
    private let commandBatcher: CommandBatcher
    private let barrierTracker: BarrierTracker

    // Evaluate a computation graph
    func evaluate<T: TensorScalar>(_ root: TensorOp) async throws -> Tensor<T> {
        // 1. Topological sort the DAG
        // 2. Run FusionEngine to identify fusible groups
        // 3. For each group, select tier (constants/stitch/JIT)
        // 4. Dispatch via CommandBatcher
        // 5. Return realized tensor
    }
}
```

### 6. BufferCache (LRU)

```swift
struct BufferCache: Sendable {
    private let state: Mutex<CacheState>

    struct CacheState {
        var cache: [Int: [MTLBuffer]]  // size -> available buffers (LRU order)
        var totalCachedBytes: Int
        let maxCachedBytes: Int        // e.g., 1.5x recommendedMaxWorkingSetSize
    }

    // Find a buffer of matching size (within 2x or size + 2 pages)
    func reuse(size: Int) -> MTLBuffer? { ... }

    // Return a buffer to the cache
    func recycle(_ buffer: MTLBuffer) { ... }

    // Evict LRU buffers when under memory pressure
    func evict(targetBytes: Int) { ... }
}
```

### 7. 3-Tier Fusion Engine

```swift
struct FusionEngine {
    private let kernelRegistry: KernelRegistry

    // Limits (validated by MLX research)
    static let maxFusionDepth = 11
    static let maxBufferArgs = 31

    enum FusionTier {
        case functionConstants(String)    // Pre-compiled kernel name + constant values
        case stitched([StitchableOp])     // Compose [[stitchable]] ops at AIR level
        case jit(String)                  // MSL source string for makeLibrary(source:)
    }

    // Analyze a subgraph and select fusion strategy
    func selectTier(for ops: [TensorOp]) -> FusionTier {
        // 1. Check if ops match a known hot-path pattern (e.g., bias+relu)
        //    → FunctionConstants
        // 2. If all ops have [[stitchable]] equivalents and depth ≤ 11
        //    → Stitched
        // 3. Otherwise, generate MSL source
        //    → JIT with MTLBinaryArchive caching
    }

    // Compile or retrieve cached pipeline for a fusion group
    func pipeline(for tier: FusionTier) throws -> MTLComputePipelineState { ... }
}
```

### 8. Shape & Broadcasting

```swift
public struct Shape: Sendable, Equatable, Hashable {
    public let dimensions: [Int]
    public var rank: Int { dimensions.count }
    public var elementCount: Int { dimensions.reduce(1, *) }

    public func broadcastCompatible(with other: Shape) -> Bool { ... }
    public func broadcastedShape(with other: Shape) throws -> Shape { ... }
}

public struct Strides: Sendable, Equatable {
    public let values: [Int]

    public var isContiguous: Bool { ... }      // C-contiguous (row-major)
    public var isColumnMajor: Bool { ... }

    public static func contiguous(for shape: Shape) -> Strides { ... }
}
```

### 9. ComputeGraph

```swift
struct ComputeGraph {
    private var nodes: [TensorOp]

    // Build topologically sorted evaluation order
    func topologicalSort(from root: TensorOp) -> [TensorOp] { ... }

    // Identify fusible groups (element-wise, same shape, within limits)
    func identifyFusionGroups(_ sorted: [TensorOp]) -> [[TensorOp]] {
        // Walk sorted ops, greedily group consecutive element-wise ops
        // that share shape and don't exceed depth/arg limits
    }

    // BFS evaluation with fusion
    func evaluate(root: TensorOp, backend: MetalBackend) async throws -> TensorStorage {
        let sorted = topologicalSort(from: root)
        let groups = identifyFusionGroups(sorted)

        for group in groups {
            let tier = fusionEngine.selectTier(for: group)
            let pipeline = try fusionEngine.pipeline(for: tier)
            try await backend.dispatch(pipeline, for: group)
        }

        return root.storage!
    }
}
```

### 10. AutoTuner

```swift
struct AutoTuner {
    // Benchmark different threadgroup sizes for each kernel type
    func tune(kernel: String, device: MTLDevice) async -> ThreadgroupConfig {
        let candidates: [MTLSize] = [
            MTLSize(width: 64, height: 1, depth: 1),
            MTLSize(width: 128, height: 1, depth: 1),
            MTLSize(width: 256, height: 1, depth: 1),
            MTLSize(width: 512, height: 1, depth: 1),
        ]
        // Benchmark each, select fastest
        // Save to JSON keyed by device family
    }

    // Load saved config for current device
    func loadConfig(for device: MTLDevice) -> [String: ThreadgroupConfig]? { ... }
}
```

## Testing Strategy

| Layer | Tests | Dependencies |
|-------|-------|-------------|
| Shape/Strides | Broadcasting, contiguity, reshape, slice | None (pure Swift) |
| TensorScalar | Protocol conformance, byte sizes | None (pure Swift) |
| Graph/Fusion | DAG construction, topological sort, fusion pattern matching, tier selection | None (pure Swift, mock dispatch) |
| Kernel Correctness | Each Metal kernel vs CPU reference within tolerance | Metal device (`#if canImport(Metal)`) |
| Fusion Correctness | Fused result == sequential unfused result | Metal device |
| Buffer Cache | Allocation, reuse, eviction under pressure | Metal device |
| Performance | Throughput, memory usage per device family | Benchmark target |

**Tolerance thresholds:**
- Float32: 1e-6
- Float16: 1e-3
- Int8/UInt8: exact match

## Research-Validated Constraints

| Constraint | Value | Source |
|-----------|-------|--------|
| Max buffer arguments per kernel | 31 | Metal specification |
| Max fusion depth | 11 | MLX validated |
| Command buffer batch size | 20-50 per chip | MLX per-architecture tuning |
| Metal dispatch overhead | ~120us per empty kernel | Apple Developer Forums benchmarks |
| matmul2d K dimension | Must be multiple of 32 | Community testing (Metal 26.1) |
| Residency sets per queue | Max 32 | Apple documentation |
| Function stitching availability | iOS 15+ / macOS 12+ | Apple WWDC21 |
| JIT compilation latency | 100-500ms (cold), cached after | MLX benchmarks, Apple docs |

## False Positives Identified and Avoided

1. **SwiftPM `include/` for Metal**: SwiftPM does NOT pass header search paths to the Metal compiler. Solved via dedicated C target + BuildToolPlugin with `-I` flag.
2. **Placement sparse resources for inference**: Not useful when tensor sizes are known at load time. Designed for streaming/LOD scenarios.
3. **MTL4ArgumentTable pooling**: Unnecessary — state is copied per dispatch. Single reusable table suffices.
4. **MTLHeap as primary allocator**: MLX proves LRU buffer cache is the primary optimization. Heap is secondary for tiny allocations only.
5. **Device-allocated MTLTensor for intermediates**: Contradicts buffer cache pattern. Intermediates use buffer-backed MTLTensors via the LRU cache.

## Open Questions for Future Milestones

- Metal 4 `tensor_ops::matmul2d` vs custom `simdgroup_matrix_multiply_accumulate` for GEMM (M2)
- `MTL4MachineLearningCommandEncoder` integration for whole-network dispatch (M5)
- Memory-mapped GGUF weights → MTLBuffer → MTLTensor pipeline (M3)
- `@Generable` macro integration for structured output (M4)
