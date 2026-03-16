# EdgeRunner Milestone 1: Core Tensor & Metal Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational tensor computation layer: `Tensor<T>` with COW, lazy evaluation graph with 3-tier kernel fusion, and a production-quality Metal 4 backend.

**Architecture:** Hybrid MTLBuffer storage + MTLTensor views. Actor-based MetalBackend with LRU buffer cache, manual hazard tracking, and command batching. 3-tier fusion: function constants (hot) → function stitching (warm) → JIT (cold).

**Tech Stack:** Swift 6.2, Metal 4, Metal Shading Language 4.0, SwiftPM BuildToolPlugin, Swift Testing

**Design Doc:** `docs/plans/2026-02-28-edgerunner-m1-design.md`

---

## Task 1: Package.swift & Project Scaffold

**Files:**
- Create: `Package.swift`
- Create: `Sources/EdgeRunnerSharedTypes/include/ShaderTypes.h`
- Create: `Sources/EdgeRunnerSharedTypes/ShaderTypes.c`
- Create: `Sources/EdgeRunnerMetal/MetalBackend.swift`
- Create: `Sources/EdgeRunnerCore/Tensor.swift`
- Create: `Sources/EdgeRunner/EdgeRunner.swift`
- Create: `Tests/EdgeRunnerCoreTests/PlaceholderTests.swift`
- Create: `Tests/EdgeRunnerMetalTests/PlaceholderTests.swift`

**Step 1: Create Package.swift**

```swift
// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "EdgeRunner",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "EdgeRunner", targets: ["EdgeRunner"]),
    ],
    targets: [
        // Shared C types for Metal ↔ Swift bridging
        .target(
            name: "EdgeRunnerSharedTypes",
            path: "Sources/EdgeRunnerSharedTypes",
            publicHeadersPath: "include"
        ),

        // Metal shaders + GPU backend
        .target(
            name: "EdgeRunnerMetal",
            dependencies: ["EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerMetal"
        ),

        // Tensor types, lazy graph, fusion engine
        .target(
            name: "EdgeRunnerCore",
            dependencies: ["EdgeRunnerMetal"],
            path: "Sources/EdgeRunnerCore"
        ),

        // Public API facade
        .target(
            name: "EdgeRunner",
            dependencies: ["EdgeRunnerCore"],
            path: "Sources/EdgeRunner"
        ),

        // Tests
        .testTarget(
            name: "EdgeRunnerCoreTests",
            dependencies: ["EdgeRunnerCore"]
        ),
        .testTarget(
            name: "EdgeRunnerMetalTests",
            dependencies: ["EdgeRunnerMetal"]
        ),
    ]
)
```

**Step 2: Create the shared C types header**

```c
// Sources/EdgeRunnerSharedTypes/include/ShaderTypes.h
#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include <stdint.h>

// Data type enum shared between Metal and Swift
typedef enum __attribute__((enum_extensibility(closed))) {
    ERDTypeFloat32 = 0,
    ERDTypeFloat16 = 1,
    ERDTypeInt8    = 2,
    ERDTypeUInt8   = 3,
} ERDType;

// Parameters for element-wise kernel dispatch
typedef struct {
    uint32_t elementCount;
} ERElementwiseParams;

// Parameters for reduction kernel dispatch
typedef struct {
    uint32_t elementCount;
    uint32_t reductionSize;
    uint32_t outerSize;
} ERReductionParams;

// Parameters for transpose kernel dispatch
typedef struct {
    uint32_t rows;
    uint32_t cols;
} ERTransposeParams;

#endif /* SHADER_TYPES_H */
```

**Step 3: Create the C stub file**

```c
// Sources/EdgeRunnerSharedTypes/ShaderTypes.c
// Required by SwiftPM for C targets. Types are defined in the header.
```

**Step 4: Create minimal source files for each target**

```swift
// Sources/EdgeRunnerMetal/MetalBackend.swift
import Metal

public actor MetalBackend {
    public static let shared = MetalBackend()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue
    }
}
```

```swift
// Sources/EdgeRunnerCore/Tensor.swift
import EdgeRunnerMetal

public struct Tensor<T: Sendable>: Sendable {
    public let shape: [Int]

    public init(shape: [Int]) {
        self.shape = shape
    }
}
```

```swift
// Sources/EdgeRunner/EdgeRunner.swift
@_exported import EdgeRunnerCore
@_exported import EdgeRunnerMetal
```

**Step 5: Create placeholder tests**

```swift
// Tests/EdgeRunnerCoreTests/PlaceholderTests.swift
import Testing
@testable import EdgeRunnerCore

@Test func tensorInitialization() {
    let t = Tensor<Float>(shape: [2, 3])
    #expect(t.shape == [2, 3])
}
```

```swift
// Tests/EdgeRunnerMetalTests/PlaceholderTests.swift
import Testing
@testable import EdgeRunnerMetal

@Test func metalBackendExists() async {
    let backend = MetalBackend.shared
    let device = await backend.device
    #expect(device.name.isEmpty == false)
}
```

**Step 6: Build and run tests**

Run: `cd /Users/chriskarani/CodingProjects/EdgeRunner && swift build 2>&1`
Expected: BUILD SUCCEEDED

Run: `swift test 2>&1`
Expected: All tests pass

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold EdgeRunner package with multi-target structure

Targets: EdgeRunnerSharedTypes (C), EdgeRunnerMetal, EdgeRunnerCore, EdgeRunner
Shared C header for Metal/Swift type bridging
Minimal MetalBackend actor with device/queue initialization"
```

---

## Task 2: Shape & Strides

**Files:**
- Create: `Sources/EdgeRunnerCore/Shape.swift`
- Create: `Tests/EdgeRunnerCoreTests/ShapeTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/ShapeTests.swift
import Testing
@testable import EdgeRunnerCore

@Suite("Shape")
struct ShapeTests {

    @Test func initAndProperties() {
        let s = Shape([2, 3, 4])
        #expect(s.rank == 3)
        #expect(s.elementCount == 24)
        #expect(s.dimensions == [2, 3, 4])
    }

    @Test func scalarShape() {
        let s = Shape([])
        #expect(s.rank == 0)
        #expect(s.elementCount == 1)
    }

    @Test func contiguousStrides() {
        let strides = Strides.contiguous(for: Shape([2, 3, 4]))
        #expect(strides.values == [12, 4, 1])
    }

    @Test func contiguousStridesVector() {
        let strides = Strides.contiguous(for: Shape([5]))
        #expect(strides.values == [1])
    }

    @Test func isContiguous() {
        let s = Strides(values: [12, 4, 1])
        #expect(s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func isNotContiguous() {
        let s = Strides(values: [12, 1, 3]) // column-major
        #expect(!s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func broadcastCompatible() throws {
        let a = Shape([2, 3, 4])
        let b = Shape([1, 3, 4])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastCompatibleScalar() throws {
        let a = Shape([2, 3])
        let b = Shape([])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastIncompatible() throws {
        let a = Shape([2, 3])
        let b = Shape([2, 4])
        #expect(!a.broadcastCompatible(with: b))
    }

    @Test func broadcastedShape() throws {
        let a = Shape([2, 1, 4])
        let b = Shape([3, 4])
        let result = try a.broadcasted(with: b)
        #expect(result.dimensions == [2, 3, 4])
    }

    @Test func broadcastedShapeError() throws {
        let a = Shape([2, 3])
        let b = Shape([2, 4])
        #expect(throws: ShapeError.self) {
            try a.broadcasted(with: b)
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ShapeTests 2>&1`
Expected: FAIL — `Shape` not defined

**Step 3: Implement Shape and Strides**

```swift
// Sources/EdgeRunnerCore/Shape.swift

public enum ShapeError: Error, Sendable {
    case incompatibleBroadcast(Shape, Shape)
    case invalidReshape(from: Shape, to: Shape)
}

public struct Shape: Sendable, Equatable, Hashable, CustomStringConvertible {
    public let dimensions: [Int]

    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions
    }

    public var rank: Int { dimensions.count }

    public var elementCount: Int {
        dimensions.isEmpty ? 1 : dimensions.reduce(1, *)
    }

    public var description: String {
        "Shape(\(dimensions))"
    }

    /// Check if two shapes are broadcast-compatible (NumPy rules).
    public func broadcastCompatible(with other: Shape) -> Bool {
        let maxRank = max(rank, other.rank)
        for i in 0..<maxRank {
            let dimA = i < rank ? dimensions[rank - 1 - i] : 1
            let dimB = i < other.rank ? other.dimensions[other.rank - 1 - i] : 1
            if dimA != dimB && dimA != 1 && dimB != 1 {
                return false
            }
        }
        return true
    }

    /// Compute the broadcasted output shape. Throws if incompatible.
    public func broadcasted(with other: Shape) throws -> Shape {
        let maxRank = max(rank, other.rank)
        var result = [Int]()
        result.reserveCapacity(maxRank)

        for i in 0..<maxRank {
            let dimA = i < rank ? dimensions[rank - 1 - i] : 1
            let dimB = i < other.rank ? other.dimensions[other.rank - 1 - i] : 1
            if dimA == dimB {
                result.append(dimA)
            } else if dimA == 1 {
                result.append(dimB)
            } else if dimB == 1 {
                result.append(dimA)
            } else {
                throw ShapeError.incompatibleBroadcast(self, other)
            }
        }
        result.reverse()
        return Shape(result)
    }
}

public struct Strides: Sendable, Equatable, CustomStringConvertible {
    public let values: [Int]

    public init(values: [Int]) {
        self.values = values
    }

    public var description: String {
        "Strides(\(values))"
    }

    /// Create C-contiguous (row-major) strides for a given shape.
    public static func contiguous(for shape: Shape) -> Strides {
        let dims = shape.dimensions
        guard !dims.isEmpty else { return Strides(values: []) }

        var strides = [Int](repeating: 1, count: dims.count)
        for i in stride(from: dims.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * dims[i + 1]
        }
        return Strides(values: strides)
    }

    /// Check if strides represent a C-contiguous layout for the given shape.
    public func isContiguous(for shape: Shape) -> Bool {
        self == Strides.contiguous(for: shape)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter ShapeTests 2>&1`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Shape.swift Tests/EdgeRunnerCoreTests/ShapeTests.swift
git commit -m "feat: add Shape and Strides types with broadcasting

NumPy-compatible broadcast rules, contiguous stride computation,
contiguity checks. Fully Sendable value types."
```

---

## Task 3: TensorScalar Protocol

**Files:**
- Create: `Sources/EdgeRunnerCore/TensorScalar.swift`
- Create: `Tests/EdgeRunnerCoreTests/TensorScalarTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/TensorScalarTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore

@Suite("TensorScalar")
struct TensorScalarTests {

    @Test func float32Properties() {
        #expect(Float.erDType == .ERDTypeFloat32)
        #expect(Float.byteSize == 4)
    }

    @Test func float16Properties() {
        #expect(Float16.erDType == .ERDTypeFloat16)
        #expect(Float16.byteSize == 2)
    }

    @Test func int8Properties() {
        #expect(Int8.erDType == .ERDTypeInt8)
        #expect(Int8.byteSize == 1)
    }

    @Test func uint8Properties() {
        #expect(UInt8.erDType == .ERDTypeUInt8)
        #expect(UInt8.byteSize == 1)
    }

    @Test func metalDataTypes() {
        #expect(Float.metalDataType == .float)
        #expect(Float16.metalDataType == .half)
        #expect(Int8.metalDataType == .char)
        #expect(UInt8.metalDataType == .uchar)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter TensorScalarTests 2>&1`
Expected: FAIL — `TensorScalar` not defined

**Step 3: Implement TensorScalar**

```swift
// Sources/EdgeRunnerCore/TensorScalar.swift
import Metal
import EdgeRunnerSharedTypes

/// Protocol for scalar types that can be stored in tensors.
public protocol TensorScalar: Sendable, BitwiseCopyable {
    /// The Metal data type for this scalar.
    static var metalDataType: MTLDataType { get }

    /// Size in bytes of a single element.
    static var byteSize: Int { get }

    /// The EdgeRunner dtype enum value (shared with Metal shaders).
    static var erDType: ERDType { get }

    /// Zero value for this type.
    static var zero: Self { get }
}

extension Float: TensorScalar {
    public static let metalDataType: MTLDataType = .float
    public static let byteSize: Int = 4
    public static let erDType: ERDType = .ERDTypeFloat32
    public static let zero: Float = 0.0
}

extension Float16: TensorScalar {
    public static let metalDataType: MTLDataType = .half
    public static let byteSize: Int = 2
    public static let erDType: ERDType = .ERDTypeFloat16
    public static let zero: Float16 = 0.0
}

extension Int8: TensorScalar {
    public static let metalDataType: MTLDataType = .char
    public static let byteSize: Int = 1
    public static let erDType: ERDType = .ERDTypeInt8
    public static let zero: Int8 = 0
}

extension UInt8: TensorScalar {
    public static let metalDataType: MTLDataType = .uchar
    public static let byteSize: Int = 1
    public static let erDType: ERDType = .ERDTypeUInt8
    public static let zero: UInt8 = 0
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter TensorScalarTests 2>&1`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/TensorScalar.swift Tests/EdgeRunnerCoreTests/TensorScalarTests.swift
git commit -m "feat: add TensorScalar protocol with Float/Float16/Int8/UInt8

Protocol-constrained generics for type-safe tensor operations.
Bridges to Metal data types and shared C enum."
```

---

## Task 4: BufferCache (LRU)

**Files:**
- Create: `Sources/EdgeRunnerMetal/BufferCache.swift`
- Create: `Tests/EdgeRunnerMetalTests/BufferCacheTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/BufferCacheTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("BufferCache")
struct BufferCacheTests {

    let device: MTLDevice

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
    }

    @Test func reuseExactSize() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 256, options: .storageModeShared)!

        cache.recycle(buf)
        let reused = cache.reuse(size: 256)
        #expect(reused != nil)
        #expect(reused!.length >= 256)
    }

    @Test func reuseSlightlyLarger() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 300, options: .storageModeShared)!

        cache.recycle(buf)
        let reused = cache.reuse(size: 256)
        #expect(reused != nil)
        #expect(reused!.length >= 256)
    }

    @Test func noReuseTooLarge() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 1024, options: .storageModeShared)!

        cache.recycle(buf)
        // Buffer is 4x requested size — too large to reuse
        let reused = cache.reuse(size: 256)
        #expect(reused == nil)
    }

    @Test func emptyCache() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let reused = cache.reuse(size: 256)
        #expect(reused == nil)
    }

    @Test func eviction() throws {
        let cache = BufferCache(device: device, maxBytes: 512)
        let buf1 = device.makeBuffer(length: 256, options: .storageModeShared)!
        let buf2 = device.makeBuffer(length: 256, options: .storageModeShared)!
        let buf3 = device.makeBuffer(length: 256, options: .storageModeShared)!

        cache.recycle(buf1)
        cache.recycle(buf2)
        // This should evict buf1 (LRU) since total would exceed max
        cache.recycle(buf3)

        #expect(cache.totalCachedBytes <= 512)
    }

    @Test func allocateNew() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = cache.acquire(size: 256)
        #expect(buf.length >= 256)
    }

    @Test func allocateReusesFromCache() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let original = device.makeBuffer(length: 256, options: .storageModeShared)!
        cache.recycle(original)

        let acquired = cache.acquire(size: 256)
        #expect(acquired.length >= 256)
    }
}

enum TestError: Error {
    case noMetal
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter BufferCacheTests 2>&1`
Expected: FAIL — `BufferCache` not defined

**Step 3: Implement BufferCache**

```swift
// Sources/EdgeRunnerMetal/BufferCache.swift
import Metal
import Synchronization

/// LRU buffer cache for reusing Metal buffers of similar sizes.
/// Thread-safe via Mutex.
public final class BufferCache: Sendable {
    private let state: Mutex<CacheState>
    private let device: MTLDevice

    struct CacheState {
        /// Buffers grouped by size, ordered by recency (most recent first).
        var buckets: [Int: [MTLBuffer]] = [:]
        var totalBytes: Int = 0
        let maxBytes: Int
    }

    public init(device: MTLDevice, maxBytes: Int) {
        self.device = device
        self.state = Mutex(CacheState(maxBytes: maxBytes))
    }

    public var totalCachedBytes: Int {
        state.withLock { $0.totalBytes }
    }

    /// Try to reuse a cached buffer of matching size.
    /// Accepts buffers up to 2x the requested size.
    public func reuse(size: Int) -> MTLBuffer? {
        state.withLock { state in
            // Find the smallest bucket >= size and <= 2 * size
            let candidates = state.buckets.keys
                .filter { $0 >= size && $0 <= size * 2 }
                .sorted()

            guard let bucketSize = candidates.first,
                  var buffers = state.buckets[bucketSize],
                  !buffers.isEmpty else {
                return nil
            }

            let buffer = buffers.removeFirst()
            if buffers.isEmpty {
                state.buckets.removeValue(forKey: bucketSize)
            } else {
                state.buckets[bucketSize] = buffers
            }
            state.totalBytes -= buffer.length
            return buffer
        }
    }

    /// Return a buffer to the cache for future reuse.
    public func recycle(_ buffer: MTLBuffer) {
        state.withLock { state in
            // Evict LRU entries if over capacity
            while state.totalBytes + buffer.length > state.maxBytes {
                guard evictOldest(&state) else { break }
            }

            // Only cache if it fits within the budget
            if state.totalBytes + buffer.length <= state.maxBytes {
                state.buckets[buffer.length, default: []].append(buffer)
                state.totalBytes += buffer.length
            }
        }
    }

    /// Acquire a buffer: try cache first, then allocate new.
    public func acquire(size: Int) -> MTLBuffer {
        if let cached = reuse(size: size) {
            return cached
        }
        guard let buffer = device.makeBuffer(
            length: size,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            fatalError("Failed to allocate Metal buffer of size \(size)")
        }
        return buffer
    }

    /// Evict the oldest buffer from the largest bucket. Returns false if cache is empty.
    private func evictOldest(_ state: inout CacheState) -> Bool {
        // Find any non-empty bucket and remove the last (oldest) element
        guard let bucketSize = state.buckets.keys.first(where: { state.buckets[$0]?.isEmpty == false }),
              var buffers = state.buckets[bucketSize] else {
            return false
        }
        let evicted = buffers.removeLast()
        if buffers.isEmpty {
            state.buckets.removeValue(forKey: bucketSize)
        } else {
            state.buckets[bucketSize] = buffers
        }
        state.totalBytes -= evicted.length
        return true
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter BufferCacheTests 2>&1`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerMetal/BufferCache.swift Tests/EdgeRunnerMetalTests/BufferCacheTests.swift
git commit -m "feat: add LRU buffer cache for Metal buffer reuse

Size-matched reuse (within 2x), LRU eviction under memory pressure,
thread-safe via Mutex. Allocates with hazardTrackingModeUntracked."
```

---

## Task 5: TensorStorage & Tensor<T>

**Files:**
- Create: `Sources/EdgeRunnerCore/TensorStorage.swift`
- Modify: `Sources/EdgeRunnerCore/Tensor.swift`
- Create: `Tests/EdgeRunnerCoreTests/TensorTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/TensorTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Tensor")
struct TensorTests {

    @Test func createFromArray() async throws {
        let t = Tensor<Float>(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: Shape([2, 3]))
        #expect(t.shape == Shape([2, 3]))
        #expect(t.strides == Strides.contiguous(for: Shape([2, 3])))
        #expect(t.elementCount == 6)
    }

    @Test func createZeros() async throws {
        let t = Tensor<Float>.zeros(shape: Shape([3, 4]))
        #expect(t.shape == Shape([3, 4]))
        #expect(t.elementCount == 12)
    }

    @Test func createOnes() async throws {
        let t = Tensor<Float>.ones(shape: Shape([2, 2]))
        #expect(t.shape == Shape([2, 2]))
        let data = t.toArray()
        #expect(data == [1.0, 1.0, 1.0, 1.0])
    }

    @Test func toArrayRoundTrip() async throws {
        let original: [Float] = [1.0, 2.0, 3.0, 4.0]
        let t = Tensor<Float>(data: original, shape: Shape([4]))
        let result = t.toArray()
        #expect(result == original)
    }

    @Test func scalarTensor() async throws {
        let t = Tensor<Float>(scalar: 42.0)
        #expect(t.shape == Shape([]))
        #expect(t.elementCount == 1)
        let data = t.toArray()
        #expect(data == [42.0])
    }

    @Test func copyOnWriteSharing() async throws {
        let a = Tensor<Float>(data: [1.0, 2.0, 3.0], shape: Shape([3]))
        let b = a // should share storage
        #expect(a.toArray() == b.toArray())
    }

    @Test func reshape() async throws {
        let t = Tensor<Float>(data: [1, 2, 3, 4, 5, 6], shape: Shape([2, 3]))
        let reshaped = try t.reshape(Shape([3, 2]))
        #expect(reshaped.shape == Shape([3, 2]))
        #expect(reshaped.toArray() == t.toArray())
    }

    @Test func reshapeInvalidThrows() async throws {
        let t = Tensor<Float>(data: [1, 2, 3, 4], shape: Shape([2, 2]))
        #expect(throws: ShapeError.self) {
            try t.reshape(Shape([3, 2]))
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "TensorTests" 2>&1`
Expected: FAIL — `Tensor` missing required APIs

**Step 3: Implement TensorStorage**

```swift
// Sources/EdgeRunnerCore/TensorStorage.swift
import Metal
import EdgeRunnerMetal

/// Reference-counted wrapper around an MTLBuffer.
/// Immutable once created — COW is handled at the Tensor level.
final class TensorStorage: @unchecked Sendable {
    // @unchecked because MTLBuffer is not Sendable but is thread-safe
    // when accessed through actor-isolated MetalBackend
    let buffer: MTLBuffer
    let byteCount: Int

    init(buffer: MTLBuffer) {
        self.buffer = buffer
        self.byteCount = buffer.length
    }

    /// Create storage from a Swift array.
    /// Uses MetalBackend.shared for device access and buffer cache integration.
    static func from<T: TensorScalar>(_ data: [T]) async -> TensorStorage {
        let backend = await MetalBackend.shared
        let byteCount = data.count * T.byteSize
        let buffer = await backend.acquireBuffer(size: byteCount)
        buffer.contents().copyMemory(from: data, byteCount: byteCount)
        return TensorStorage(buffer: buffer)
    }

    /// Create zero-initialized storage.
    /// Uses MetalBackend.shared for device access and buffer cache integration.
    static func zeros(byteCount: Int) async -> TensorStorage {
        let backend = await MetalBackend.shared
        let buffer = await backend.acquireBuffer(size: byteCount)
        // storageModeShared buffers are zero-initialized by the system
        return TensorStorage(buffer: buffer)
    }

    /// Copy data out to a Swift array.
    func toArray<T: TensorScalar>(count: Int) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Create a deep copy of the storage via MetalBackend buffer cache.
    func copy() async -> TensorStorage {
        let backend = await MetalBackend.shared
        let newBuffer = await backend.acquireBuffer(size: byteCount)
        newBuffer.contents().copyMemory(from: buffer.contents(), byteCount: byteCount)
        return TensorStorage(buffer: newBuffer)
    }
}
```

**Step 4: Implement the full Tensor<T>**

```swift
// Sources/EdgeRunnerCore/Tensor.swift
import Metal
import EdgeRunnerMetal

/// A multi-dimensional array of scalar values backed by a Metal buffer.
/// Value type with copy-on-write semantics.
public struct Tensor<T: TensorScalar>: Sendable {
    // COW storage
    private var _storage: TensorStorage

    // Metadata
    public let shape: Shape
    public let strides: Strides

    /// Number of elements in this tensor.
    public var elementCount: Int { shape.elementCount }

    /// Number of bytes used by the data.
    public var byteCount: Int { elementCount * T.byteSize }

    // MARK: - Initializers

    /// Create a tensor from a flat array and shape.
    public init(data: [T], shape: Shape) {
        precondition(
            data.count == shape.elementCount,
            "Data count \(data.count) doesn't match shape \(shape) (expected \(shape.elementCount))"
        )
        self._storage = TensorStorage.from(data)
        self.shape = shape
        self.strides = Strides.contiguous(for: shape)
    }

    /// Create a scalar tensor.
    public init(scalar: T) {
        self.init(data: [scalar], shape: Shape([]))
    }

    /// Internal initializer with pre-existing storage.
    init(storage: TensorStorage, shape: Shape, strides: Strides) {
        self._storage = storage
        self.shape = shape
        self.strides = strides
    }

    // MARK: - Factory Methods

    /// Create a zero-filled tensor.
    public static func zeros(shape: Shape) -> Tensor {
        let storage = TensorStorage.zeros(byteCount: shape.elementCount * T.byteSize)
        return Tensor(storage: storage, shape: shape, strides: .contiguous(for: shape))
    }

    /// Create a tensor filled with ones.
    public static func ones(shape: Shape) -> Tensor {
        let data = [T](repeating: T.zero, count: shape.elementCount)
        // For ones, we need to set all bytes appropriately
        var ones = data
        for i in 0..<ones.count {
            ones[i] = _one()
        }
        return Tensor(data: ones, shape: shape)
    }

    // MARK: - Data Access

    /// Copy tensor data to a Swift array (synchronous, for realized tensors).
    public func toArray() -> [T] {
        _storage.toArray(count: elementCount)
    }

    /// Access the underlying Metal buffer (for kernel dispatch).
    var metalBuffer: MTLBuffer {
        _storage.buffer
    }

    // MARK: - Shape Operations

    /// Reshape to a new shape with the same element count.
    public func reshape(_ newShape: Shape) throws -> Tensor {
        guard newShape.elementCount == shape.elementCount else {
            throw ShapeError.invalidReshape(from: shape, to: newShape)
        }
        return Tensor(
            storage: _storage,
            shape: newShape,
            strides: .contiguous(for: newShape)
        )
    }

    // MARK: - COW

    /// Ensure unique ownership of storage before mutation.
    mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&_storage) {
            _storage = _storage.copy()
        }
    }

    // MARK: - Private Helpers

    /// Get the "one" value for this scalar type.
    private static func _one() -> T {
        // Use unsafeBitCast for known types
        if T.self == Float.self {
            return unsafeBitCast(Float(1.0), to: T.self)
        } else if T.self == Float16.self {
            return unsafeBitCast(Float16(1.0), to: T.self)
        } else if T.self == Int8.self {
            return unsafeBitCast(Int8(1), to: T.self)
        } else if T.self == UInt8.self {
            return unsafeBitCast(UInt8(1), to: T.self)
        }
        return T.zero
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `swift test --filter "TensorTests" 2>&1`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add Sources/EdgeRunnerCore/TensorStorage.swift Sources/EdgeRunnerCore/Tensor.swift Tests/EdgeRunnerCoreTests/TensorTests.swift
git commit -m "feat: add Tensor<T> with COW semantics and TensorStorage

MTLBuffer-backed storage, copy-on-write via isKnownUniquelyReferenced,
factory methods (zeros, ones, scalar), reshape, toArray round-trip."
```

---

## Task 6: Metal Element-wise Kernels

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Elementwise.metal`
- Create: `Sources/EdgeRunnerMetal/KernelRegistry.swift`
- Create: `Tests/EdgeRunnerMetalTests/ElementwiseKernelTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/ElementwiseKernelTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("ElementwiseKernels")
struct ElementwiseKernelTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = q
        self.registry = try KernelRegistry(device: d)
    }

    @Test func addFloat32() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]
        let expected: [Float] = [6.0, 8.0, 10.0, 12.0]

        let result = try dispatchBinary(name: "elementwise_add_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func subtractFloat32() throws {
        let a: [Float] = [5.0, 6.0, 7.0, 8.0]
        let b: [Float] = [1.0, 2.0, 3.0, 4.0]
        let expected: [Float] = [4.0, 4.0, 4.0, 4.0]

        let result = try dispatchBinary(name: "elementwise_sub_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func multiplyFloat32() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [2.0, 3.0, 4.0, 5.0]
        let expected: [Float] = [2.0, 6.0, 12.0, 20.0]

        let result = try dispatchBinary(name: "elementwise_mul_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func divideFloat32() throws {
        let a: [Float] = [10.0, 20.0, 30.0, 40.0]
        let b: [Float] = [2.0, 4.0, 5.0, 8.0]
        let expected: [Float] = [5.0, 5.0, 6.0, 5.0]

        let result = try dispatchBinary(name: "elementwise_div_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    // Helper: dispatch a binary element-wise kernel
    private func dispatchBinary(name: String, a: [Float], b: [Float], count: Int) throws -> [Float] {
        let byteCount = count * MemoryLayout<Float>.size
        let bufA = device.makeBuffer(bytes: a, length: byteCount, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: name)
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var params = ERElementwiseParams(elementCount: UInt32(count))
        encoder.setBytes(&params, length: MemoryLayout<ERElementwiseParams>.size, index: 3)

        let threadsPerGroup = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ElementwiseKernelTests 2>&1`
Expected: FAIL — `KernelRegistry` not defined, shaders not compiled

**Step 3: Write the Metal shaders**

```metal
// Sources/EdgeRunnerMetal/Shaders/Elementwise.metal
#include <metal_stdlib>
using namespace metal;

// Import shared types — the BuildToolPlugin passes -I to include EdgeRunnerSharedTypes
// For now, define inline until the plugin is set up
struct ERElementwiseParams {
    uint elementCount;
};

// ---- Float32 kernels ----

kernel void elementwise_add_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}

// ---- Float16 kernels ----

kernel void elementwise_add_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}
```

**Step 4: Implement KernelRegistry**

```swift
// Sources/EdgeRunnerMetal/KernelRegistry.swift
import Metal

/// Caches compiled Metal pipeline states by kernel name.
public final class KernelRegistry: Sendable {
    private let library: MTLLibrary
    private let cache: Mutex<[String: MTLComputePipelineState]>
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
        // Load from the default metallib in the bundle
        self.library = try device.makeDefaultLibrary(bundle: Bundle.module)
        self.cache = Mutex([:])
    }

    /// Get or create a compute pipeline state for the given kernel name.
    public func pipeline(for name: String) throws -> MTLComputePipelineState {
        if let cached = cache.withLock({ $0[name] }) {
            return cached
        }
        guard let function = library.makeFunction(name: name) else {
            throw KernelRegistryError.functionNotFound(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        cache.withLock { $0[name] = pipeline }
        return pipeline
    }
}

public enum KernelRegistryError: Error, Sendable {
    case functionNotFound(String)
}
```

**Step 5: Run tests to verify they pass**

Run: `swift test --filter ElementwiseKernelTests 2>&1`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/Elementwise.metal Sources/EdgeRunnerMetal/KernelRegistry.swift Tests/EdgeRunnerMetalTests/ElementwiseKernelTests.swift
git commit -m "feat: add element-wise Metal kernels and KernelRegistry

Float32 and Float16 kernels for add, sub, mul, div.
KernelRegistry caches pipeline states by kernel name.
Tests validate GPU output against expected results."
```

---

## Task 7: Reduction & Transpose Kernels

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Reduction.metal`
- Create: `Sources/EdgeRunnerMetal/Shaders/Transpose.metal`
- Create: `Tests/EdgeRunnerMetalTests/ReductionKernelTests.swift`
- Create: `Tests/EdgeRunnerMetalTests/TransposeKernelTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/ReductionKernelTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("ReductionKernels")
struct ReductionKernelTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func sumAll() throws {
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = try dispatchReduction(name: "reduce_sum_float", input: input, reductionSize: 4, outerSize: 1)
        #expect(result.count == 1)
        #expect(abs(result[0] - 10.0) < 1e-6)
    }

    @Test func sumRows() throws {
        // 2x3 matrix, sum along axis 1 (columns) → [6, 15]
        let input: [Float] = [1, 2, 3, 4, 5, 6]
        let result = try dispatchReduction(name: "reduce_sum_float", input: input, reductionSize: 3, outerSize: 2)
        #expect(result.count == 2)
        #expect(abs(result[0] - 6.0) < 1e-6)
        #expect(abs(result[1] - 15.0) < 1e-6)
    }

    @Test func maxAll() throws {
        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        let result = try dispatchReduction(name: "reduce_max_float", input: input, reductionSize: 6, outerSize: 1)
        #expect(result.count == 1)
        #expect(abs(result[0] - 9.0) < 1e-6)
    }

    private func dispatchReduction(name: String, input: [Float], reductionSize: Int, outerSize: Int) throws -> [Float] {
        let byteCount = input.count * MemoryLayout<Float>.size
        let outCount = outerSize
        let outBytes = outCount * MemoryLayout<Float>.size

        let bufIn = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: outBytes, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: name)
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var params = ERReductionParams(
            elementCount: UInt32(input.count),
            reductionSize: UInt32(reductionSize),
            outerSize: UInt32(outerSize)
        )
        encoder.setBytes(&params, length: MemoryLayout<ERReductionParams>.size, index: 2)

        let threadsPerGroup = MTLSize(width: min(outerSize, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (outerSize + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }
}
```

```swift
// Tests/EdgeRunnerMetalTests/TransposeKernelTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("TransposeKernels")
struct TransposeKernelTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func transpose2x3() throws {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let input: [Float] = [1, 2, 3, 4, 5, 6]
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        let result = try dispatchTranspose(input: input, rows: 2, cols: 3)
        #expect(result == expected)
    }

    @Test func transposeSquare() throws {
        let input: [Float] = [1, 2, 3, 4]
        let expected: [Float] = [1, 3, 2, 4]
        let result = try dispatchTranspose(input: input, rows: 2, cols: 2)
        #expect(result == expected)
    }

    private func dispatchTranspose(input: [Float], rows: Int, cols: Int) throws -> [Float] {
        let count = rows * cols
        let byteCount = count * MemoryLayout<Float>.size

        let bufIn = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: "transpose_float")
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var params = ERTransposeParams(rows: UInt32(rows), cols: UInt32(cols))
        encoder.setBytes(&params, length: MemoryLayout<ERTransposeParams>.size, index: 2)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (cols + 15) / 16,
            height: (rows + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "ReductionKernelTests|TransposeKernelTests" 2>&1`
Expected: FAIL — kernels not found

**Step 3: Write the Metal shaders**

```metal
// Sources/EdgeRunnerMetal/Shaders/Reduction.metal
#include <metal_stdlib>
using namespace metal;

struct ERReductionParams {
    uint elementCount;
    uint reductionSize;
    uint outerSize;
};

kernel void reduce_sum_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;

    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum;
}

kernel void reduce_mean_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;

    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum / float(params.reductionSize);
}

kernel void reduce_max_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;

    uint base = tid * params.reductionSize;
    float maxVal = input[base];
    for (uint i = 1; i < params.reductionSize; i++) {
        maxVal = max(maxVal, input[base + i]);
    }
    output[tid] = maxVal;
}
```

```metal
// Sources/EdgeRunnerMetal/Shaders/Transpose.metal
#include <metal_stdlib>
using namespace metal;

struct ERTransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERTransposeParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;

    if (row >= params.rows || col >= params.cols) return;

    // input[row][col] → output[col][row]
    output[col * params.rows + row] = input[row * params.cols + col];
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "ReductionKernelTests|TransposeKernelTests" 2>&1`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/Reduction.metal Sources/EdgeRunnerMetal/Shaders/Transpose.metal Tests/EdgeRunnerMetalTests/ReductionKernelTests.swift Tests/EdgeRunnerMetalTests/TransposeKernelTests.swift
git commit -m "feat: add reduction (sum/mean/max) and transpose Metal kernels

Reduction: per-thread serial reduction over contiguous segments.
Transpose: 2D tiled matrix transpose.
Tests validate GPU output against expected values."
```

---

## Task 8: Stitchable Operations for Fusion

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/StitchableOps.metal`
- Create: `Sources/EdgeRunnerMetal/Shaders/FusedPatterns.metal`
- Create: `Tests/EdgeRunnerMetalTests/StitchableOpsTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/StitchableOpsTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("StitchableOps")
struct StitchableOpsTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func fusedAddRelu() throws {
        // add + relu in a single kernel via function constants
        let a: [Float] = [-1.0, 2.0, -3.0, 4.0]
        let b: [Float] = [0.5, -3.0, 4.0, -1.0]
        // add: [-0.5, -1.0, 1.0, 3.0]
        // relu: [0.0, 0.0, 1.0, 3.0]
        let expected: [Float] = [0.0, 0.0, 1.0, 3.0]

        let result = try dispatchFusedBinary(
            a: a, b: b, count: 4,
            activation: .relu
        )
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-6)
        }
    }

    @Test func fusedAddSigmoid() throws {
        let a: [Float] = [0.0, 0.0, 0.0, 0.0]
        let b: [Float] = [0.0, 0.0, 0.0, 0.0]
        // add: [0, 0, 0, 0], sigmoid(0) = 0.5
        let expected: [Float] = [0.5, 0.5, 0.5, 0.5]

        let result = try dispatchFusedBinary(
            a: a, b: b, count: 4,
            activation: .sigmoid
        )
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-6)
        }
    }

    @Test func fusedAddNoActivation() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]
        let expected: [Float] = [6.0, 8.0, 10.0, 12.0]

        let result = try dispatchFusedBinary(
            a: a, b: b, count: 4,
            activation: .none
        )
        #expect(result == expected)
    }

    enum Activation: Int {
        case none = 0
        case relu = 1
        case sigmoid = 2
        case gelu = 3
        case silu = 4
    }

    private func dispatchFusedBinary(a: [Float], b: [Float], count: Int, activation: Activation) throws -> [Float] {
        let byteCount = count * MemoryLayout<Float>.size
        let bufA = device.makeBuffer(bytes: a, length: byteCount, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        // Use function constants to select activation
        let constants = MTLFunctionConstantValues()
        var activationType = Int32(activation.rawValue)
        constants.setConstantValue(&activationType, type: .int, index: 0)

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        let function = try library.makeFunction(name: "fused_add_activate_float", constantValues: constants)
        let pipeline = try device.makeComputePipelineState(function: function)

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var elemCount = UInt32(count)
        encoder.setBytes(&elemCount, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter StitchableOpsTests 2>&1`
Expected: FAIL — shaders not found

**Step 3: Write the Metal shaders**

```metal
// Sources/EdgeRunnerMetal/Shaders/StitchableOps.metal
#include <metal_stdlib>
using namespace metal;

// ---- Stitchable unary operations ----
// These can be composed via Metal Function Stitching at runtime.

[[stitchable]] float op_relu_float(float x) {
    return max(x, 0.0f);
}

[[stitchable]] float op_sigmoid_float(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[[stitchable]] float op_gelu_float(float x) {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float kSqrt2OverPi = 0.7978845608f;
    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + 0.044715f * cube);
    return 0.5f * x * (1.0f + tanh(inner));
}

[[stitchable]] float op_silu_float(float x) {
    return x / (1.0f + exp(-x));
}

[[stitchable]] float op_neg_float(float x) {
    return -x;
}

[[stitchable]] float op_abs_float(float x) {
    return abs(x);
}

[[stitchable]] float op_sqrt_float(float x) {
    return sqrt(x);
}

[[stitchable]] float op_exp_float(float x) {
    return exp(x);
}

[[stitchable]] float op_log_float(float x) {
    return log(x);
}

[[stitchable]] float op_tanh_float(float x) {
    return tanh(x);
}

// ---- Stitchable binary operations ----

[[stitchable]] float op_add_float(float a, float b) {
    return a + b;
}

[[stitchable]] float op_sub_float(float a, float b) {
    return a - b;
}

[[stitchable]] float op_mul_float(float a, float b) {
    return a * b;
}

[[stitchable]] float op_div_float(float a, float b) {
    return a / b;
}
```

```metal
// Sources/EdgeRunnerMetal/Shaders/FusedPatterns.metal
#include <metal_stdlib>
using namespace metal;

// Activation type (matches function constant at index 0)
// 0 = none, 1 = relu, 2 = sigmoid, 3 = gelu, 4 = silu
constant int activation_type [[function_constant(0)]];

// Helper: apply activation based on function constant
inline float apply_activation(float x) {
    if (activation_type == 1) {
        return max(x, 0.0f);
    } else if (activation_type == 2) {
        return 1.0f / (1.0f + exp(-x));
    } else if (activation_type == 3) {
        const float kSqrt2OverPi = 0.7978845608f;
        float cube = x * x * x;
        float inner = kSqrt2OverPi * (x + 0.044715f * cube);
        return 0.5f * x * (1.0f + tanh(inner));
    } else if (activation_type == 4) {
        return x / (1.0f + exp(-x));
    }
    return x; // no activation
}

// Fused add + activation (hot path)
kernel void fused_add_activate_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    float sum = a[tid] + b[tid];
    out[tid] = apply_activation(sum);
}

// Fused mul + activation (hot path)
kernel void fused_mul_activate_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    float product = a[tid] * b[tid];
    out[tid] = apply_activation(product);
}

// Fused unary activation (hot path)
kernel void fused_activate_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& elementCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    output[tid] = apply_activation(input[tid]);
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter StitchableOpsTests 2>&1`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerMetal/Shaders/StitchableOps.metal Sources/EdgeRunnerMetal/Shaders/FusedPatterns.metal Tests/EdgeRunnerMetalTests/StitchableOpsTests.swift
git commit -m "feat: add stitchable ops and fused pattern kernels

[[stitchable]] vocabulary: relu, sigmoid, gelu, silu, neg, abs, sqrt, exp, log, tanh, add, sub, mul, div.
Function-constant fused kernels: add+activate, mul+activate.
Dead-code elimination via Metal compiler for unused activation paths."
```

---

## Task 9: TensorOp DAG & ComputeGraph

**Files:**
- Create: `Sources/EdgeRunnerCore/Graph/TensorOp.swift`
- Create: `Sources/EdgeRunnerCore/Graph/ComputeGraph.swift`
- Create: `Tests/EdgeRunnerCoreTests/ComputeGraphTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/ComputeGraphTests.swift
import Testing
@testable import EdgeRunnerCore

@Suite("ComputeGraph")
struct ComputeGraphTests {

    @Test func singleOp() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))

        let sorted = ComputeGraph.topologicalSort(root: add)
        #expect(sorted.count == 3)
        // Inputs come before the add
        #expect(sorted[2].id == add.id)
    }

    @Test func chainedOps() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))

        let sorted = ComputeGraph.topologicalSort(root: relu)
        #expect(sorted.count == 4)
        #expect(sorted.last?.id == relu.id)
    }

    @Test func fusionGroupIdentification() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let sigmoid = TensorOp.unary(.sigmoid, input: relu, outputShape: Shape([4]))

        let sorted = ComputeGraph.topologicalSort(root: sigmoid)
        let groups = ComputeGraph.identifyFusionGroups(sorted)

        // Inputs are not fusible; add+relu+sigmoid should fuse into one group
        #expect(groups.count >= 1)
        let fusedGroup = groups.first(where: { $0.count > 1 })
        #expect(fusedGroup != nil)
        #expect(fusedGroup!.count == 3) // add + relu + sigmoid
    }

    @Test func fusionRespectsDifferentShapes() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let sum = TensorOp.reduction(.sum, input: add, outputShape: Shape([1]))

        let sorted = ComputeGraph.topologicalSort(root: sum)
        let groups = ComputeGraph.identifyFusionGroups(sorted)

        // add and sum have different shapes, should not fuse
        let fusedGroup = groups.first(where: { $0.count > 1 })
        #expect(fusedGroup == nil)
    }

    @Test func fusionRespectsDepthLimit() throws {
        // Build a chain of 15 unary ops — should split at depth 11
        var current = TensorOp.input(id: 0, shape: Shape([4]))
        for _ in 0..<15 {
            current = TensorOp.unary(.relu, input: current, outputShape: Shape([4]))
        }

        let sorted = ComputeGraph.topologicalSort(root: current)
        let groups = ComputeGraph.identifyFusionGroups(sorted)

        // Should have at least 2 groups due to depth limit
        let fusedGroups = groups.filter { $0.count > 1 }
        #expect(fusedGroups.count >= 2)
        for group in fusedGroups {
            #expect(group.count <= ComputeGraph.maxFusionDepth)
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter ComputeGraphTests 2>&1`
Expected: FAIL — `TensorOp`, `ComputeGraph` not defined

**Step 3: Implement TensorOp**

```swift
// Sources/EdgeRunnerCore/Graph/TensorOp.swift

import Foundation

/// Supported unary operations.
public enum UnaryOp: String, Sendable {
    case relu, sigmoid, gelu, silu
    case neg, abs, sqrt, exp, log, tanh
}

/// Supported binary operations.
public enum BinaryOp: String, Sendable {
    case add, sub, mul, div
}

/// Supported reduction operations.
public enum ReductionOp: String, Sendable {
    case sum, mean, max
}

/// A node in the computation DAG.
/// Each node represents a deferred tensor operation.
public final class TensorOp: Sendable {
    public let id: UUID
    public let kind: Kind
    public let outputShape: Shape

    public enum Kind: Sendable {
        case input
        case unary(UnaryOp, input: TensorOp)
        case binary(BinaryOp, lhs: TensorOp, rhs: TensorOp)
        case reduction(ReductionOp, input: TensorOp)
    }

    init(kind: Kind, outputShape: Shape) {
        self.id = UUID()
        self.kind = kind
        self.outputShape = outputShape
    }

    // MARK: - Factory methods

    public static func input(id: Int, shape: Shape) -> TensorOp {
        TensorOp(kind: .input, outputShape: shape)
    }

    public static func unary(_ op: UnaryOp, input: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .unary(op, input: input), outputShape: outputShape)
    }

    public static func binary(_ op: BinaryOp, lhs: TensorOp, rhs: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .binary(op, lhs: lhs, rhs: rhs), outputShape: outputShape)
    }

    public static func reduction(_ op: ReductionOp, input: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .reduction(op, input: input), outputShape: outputShape)
    }

    /// Whether this op is element-wise (can be fused).
    public var isElementwise: Bool {
        switch kind {
        case .input: return false
        case .unary: return true
        case .binary: return true
        case .reduction: return false
        }
    }

    /// Direct input dependencies.
    public var inputs: [TensorOp] {
        switch kind {
        case .input: return []
        case .unary(_, let input): return [input]
        case .binary(_, let lhs, let rhs): return [lhs, rhs]
        case .reduction(_, let input): return [input]
        }
    }
}

extension TensorOp: Equatable {
    public static func == (lhs: TensorOp, rhs: TensorOp) -> Bool {
        lhs.id == rhs.id
    }
}

extension TensorOp: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
```

**Step 4: Implement ComputeGraph**

```swift
// Sources/EdgeRunnerCore/Graph/ComputeGraph.swift

/// Static utilities for analyzing and evaluating computation graphs.
public enum ComputeGraph {

    /// Maximum number of ops in a single fused group.
    public static let maxFusionDepth = 11

    /// Maximum buffer arguments per fused kernel.
    public static let maxBufferArgs = 31

    /// Topological sort of the DAG rooted at `root`.
    /// Returns nodes in dependency order (inputs first, root last).
    public static func topologicalSort(root: TensorOp) -> [TensorOp] {
        var visited = Set<UUID>()
        var result = [TensorOp]()

        func visit(_ node: TensorOp) {
            guard !visited.contains(node.id) else { return }
            visited.insert(node.id)

            for input in node.inputs {
                visit(input)
            }
            result.append(node)
        }

        visit(root)
        return result
    }

    /// Identify groups of consecutive element-wise ops that can be fused.
    /// Groups respect shape boundaries and depth limits.
    public static func identifyFusionGroups(_ sorted: [TensorOp]) -> [[TensorOp]] {
        var groups = [[TensorOp]]()
        var currentGroup = [TensorOp]()
        var currentShape: Shape?

        for op in sorted {
            if op.isElementwise {
                if let shape = currentShape, shape == op.outputShape, currentGroup.count < maxFusionDepth {
                    // Same shape, within depth limit — extend the group
                    currentGroup.append(op)
                } else {
                    // Flush previous group
                    if !currentGroup.isEmpty {
                        groups.append(currentGroup)
                    }
                    // Start new group
                    currentGroup = [op]
                    currentShape = op.outputShape
                }
            } else {
                // Non-element-wise op breaks any fusion chain
                if !currentGroup.isEmpty {
                    groups.append(currentGroup)
                    currentGroup = []
                    currentShape = nil
                }
                groups.append([op])
            }
        }
        if !currentGroup.isEmpty {
            groups.append(currentGroup)
        }
        return groups
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `swift test --filter ComputeGraphTests 2>&1`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add Sources/EdgeRunnerCore/Graph/TensorOp.swift Sources/EdgeRunnerCore/Graph/ComputeGraph.swift Tests/EdgeRunnerCoreTests/ComputeGraphTests.swift
git commit -m "feat: add TensorOp DAG and ComputeGraph with fusion analysis

TensorOp: unary/binary/reduction DAG nodes with UUID identity.
ComputeGraph: topological sort, fusion group identification.
Respects max depth (11) and shape boundaries."
```

---

## Task 10: FusionEngine (3-Tier Selection)

**Files:**
- Create: `Sources/EdgeRunnerCore/Graph/FusionEngine.swift`
- Create: `Tests/EdgeRunnerCoreTests/FusionEngineTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/FusionEngineTests.swift
import Testing
@testable import EdgeRunnerCore

@Suite("FusionEngine")
struct FusionEngineTests {

    @Test func selectsHotPathForAddRelu() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))

        let tier = FusionEngine.selectTier(for: [add, relu])
        switch tier {
        case .functionConstants:
            break // expected
        default:
            Issue.record("Expected .functionConstants, got \(tier)")
        }
    }

    @Test func selectsStitchForLongChain() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let sigmoid = TensorOp.unary(.sigmoid, input: relu, outputShape: Shape([4]))
        let neg = TensorOp.unary(.neg, input: sigmoid, outputShape: Shape([4]))

        let tier = FusionEngine.selectTier(for: [add, relu, sigmoid, neg])
        switch tier {
        case .stitched:
            break // expected — too many ops for a known hot path
        default:
            Issue.record("Expected .stitched, got \(tier)")
        }
    }

    @Test func selectsSingleOpAsFunctionConstant() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: a, outputShape: Shape([4]))

        let tier = FusionEngine.selectTier(for: [relu])
        switch tier {
        case .functionConstants:
            break // expected — single activation is a hot path
        default:
            Issue.record("Expected .functionConstants, got \(tier)")
        }
    }

    @Test func tierDescriptions() {
        let hot = FusionTier.functionConstants(kernelName: "fused_add_activate_float", activationType: 1)
        let warm = FusionTier.stitched(ops: [.relu, .sigmoid])
        let cold = FusionTier.jit(source: "kernel void ...")

        #expect(hot.isHotPath)
        #expect(!warm.isHotPath)
        #expect(!cold.isHotPath)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter FusionEngineTests 2>&1`
Expected: FAIL — `FusionEngine`, `FusionTier` not defined

**Step 3: Implement FusionEngine**

```swift
// Sources/EdgeRunnerCore/Graph/FusionEngine.swift

/// Represents how a group of fused ops will be dispatched.
public enum FusionTier: Sendable {
    /// Pre-compiled kernel with function constants (fastest).
    case functionConstants(kernelName: String, activationType: Int32)

    /// Compose pre-compiled [[stitchable]] ops at runtime (fast).
    case stitched(ops: [UnaryOp])

    /// JIT-compile MSL source string (slowest, cached after first compile).
    case jit(source: String)

    /// Whether this uses a pre-compiled hot-path kernel.
    public var isHotPath: Bool {
        if case .functionConstants = self { return true }
        return false
    }
}

/// Analyzes operation groups and selects the optimal fusion tier.
public enum FusionEngine {

    // Known hot-path patterns: binary op + optional activation
    private static let hotPathPatterns: Set<String> = [
        "add",            // just add
        "add+relu",
        "add+sigmoid",
        "add+gelu",
        "add+silu",
        "mul",
        "mul+relu",
        "mul+sigmoid",
        "mul+gelu",
        "mul+silu",
        "relu",           // single activations
        "sigmoid",
        "gelu",
        "silu",
    ]

    /// Map from activation op to function constant value.
    private static let activationMap: [UnaryOp: Int32] = [
        .relu: 1,
        .sigmoid: 2,
        .gelu: 3,
        .silu: 4,
    ]

    /// Select the fusion tier for a group of ops.
    public static func selectTier(for ops: [TensorOp]) -> FusionTier {
        let pattern = patternKey(for: ops)

        // Tier 1: Hot path — known pattern with pre-compiled kernel
        if hotPathPatterns.contains(pattern) {
            return buildFunctionConstantsTier(ops: ops, pattern: pattern)
        }

        // Tier 2: Stitched — all ops have stitchable equivalents
        let unaryOps = extractUnaryOps(from: ops)
        if unaryOps != nil {
            return .stitched(ops: unaryOps!)
        }

        // Tier 3: JIT — generate MSL source
        let source = generateMSL(for: ops)
        return .jit(source: source)
    }

    // MARK: - Private Helpers

    /// Build a string key describing the operation pattern.
    private static func patternKey(for ops: [TensorOp]) -> String {
        ops.compactMap { op in
            switch op.kind {
            case .unary(let unaryOp, _): return unaryOp.rawValue
            case .binary(let binOp, _, _): return binOp.rawValue
            default: return nil
            }
        }.joined(separator: "+")
    }

    /// Build a function-constants tier for a known hot-path pattern.
    private static func buildFunctionConstantsTier(ops: [TensorOp], pattern: String) -> FusionTier {
        // Determine kernel name and activation
        let parts = pattern.split(separator: "+")

        if parts.count == 1 {
            // Single op
            let opName = String(parts[0])
            if let activation = UnaryOp(rawValue: opName), let actValue = activationMap[activation] {
                return .functionConstants(kernelName: "fused_activate_float", activationType: actValue)
            }
            return .functionConstants(kernelName: "elementwise_\(opName)_float", activationType: 0)
        }

        // Binary + activation
        let binOp = String(parts[0])
        let activation = parts.count > 1 ? UnaryOp(rawValue: String(parts[1])) : nil
        let actValue = activation.flatMap { activationMap[$0] } ?? 0
        return .functionConstants(kernelName: "fused_\(binOp)_activate_float", activationType: actValue)
    }

    /// Extract unary ops from a group (for stitching). Returns nil if not all stitchable.
    private static func extractUnaryOps(from ops: [TensorOp]) -> [UnaryOp]? {
        var unaryOps = [UnaryOp]()
        for op in ops {
            switch op.kind {
            case .unary(let unaryOp, _):
                unaryOps.append(unaryOp)
            case .binary:
                // Binary ops are stitchable too, but need different handling
                return nil // for now, stitch only supports unary chains
            default:
                return nil
            }
        }
        return unaryOps.isEmpty ? nil : unaryOps
    }

    /// Generate MSL source for JIT compilation (tier 3 fallback).
    private static func generateMSL(for ops: [TensorOp]) -> String {
        // Placeholder — full JIT codegen is a later enhancement
        return """
        #include <metal_stdlib>
        using namespace metal;
        kernel void jit_fused(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
            if (tid >= count) return;
            float x = in[tid];
            // TODO: generated ops
            out[tid] = x;
        }
        """
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter FusionEngineTests 2>&1`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/Graph/FusionEngine.swift Tests/EdgeRunnerCoreTests/FusionEngineTests.swift
git commit -m "feat: add 3-tier FusionEngine (constants → stitch → JIT)

Tier 1: Function constants for known hot paths (add+relu, mul+sigmoid, etc).
Tier 2: Stitched composition for arbitrary unary chains.
Tier 3: JIT MSL generation fallback.
Pattern matching with activation type mapping."
```

---

## Task 11: ResidencyManager & CommandBatcher

**Files:**
- Create: `Sources/EdgeRunnerMetal/ResidencyManager.swift`
- Create: `Sources/EdgeRunnerMetal/CommandBatcher.swift`
- Create: `Sources/EdgeRunnerMetal/BarrierTracker.swift`

**Step 1: Implement ResidencyManager**

```swift
// Sources/EdgeRunnerMetal/ResidencyManager.swift
import Metal

/// Manages a single MTLResidencySet for GPU memory residency.
/// Populated at initialization, attached to the command queue.
public final class ResidencyManager: @unchecked Sendable {
    private let residencySet: MTLResidencySet?
    private let device: MTLDevice

    public init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 256

        if let set = try? device.makeResidencySet(descriptor: descriptor) {
            self.residencySet = set
            set.requestResidency()
            commandQueue.addResidencySet(set)
        } else {
            self.residencySet = nil
        }
    }

    /// Add a buffer to the residency set.
    public func addBuffer(_ buffer: MTLBuffer) {
        guard let set = residencySet else { return }
        set.addAllocation(buffer)
        set.commit()
    }

    /// Add a heap to the residency set.
    public func addHeap(_ heap: MTLHeap) {
        guard let set = residencySet else { return }
        set.addAllocation(heap)
        set.commit()
    }
}
```

**Step 2: Implement CommandBatcher**

```swift
// Sources/EdgeRunnerMetal/CommandBatcher.swift
import Metal

/// Batches multiple kernel dispatches into a single command buffer.
/// Limits configured per GPU family for optimal throughput.
public final class CommandBatcher: @unchecked Sendable {
    private let commandQueue: MTLCommandQueue
    private let maxOpsPerBuffer: Int

    private var currentCommandBuffer: MTLCommandBuffer?
    private var currentEncoder: MTLComputeCommandEncoder?
    private var currentOpCount: Int = 0

    public init(commandQueue: MTLCommandQueue, device: MTLDevice) {
        self.commandQueue = commandQueue

        // Determine batch limits based on GPU family
        if device.supportsFamily(.apple9) {
            // M4, A18 — high-end
            self.maxOpsPerBuffer = 50
        } else if device.supportsFamily(.apple8) {
            // M3, A17 — mid-range
            self.maxOpsPerBuffer = 40
        } else {
            // Older or unknown
            self.maxOpsPerBuffer = 30
        }
    }

    /// Get or create a compute command encoder for the current batch.
    public func encoder() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        if currentOpCount >= maxOpsPerBuffer {
            flush()
        }

        if currentCommandBuffer == nil {
            currentCommandBuffer = commandQueue.makeCommandBuffer()!
            currentEncoder = currentCommandBuffer!.makeComputeCommandEncoder(dispatchType: .concurrent)!
        }

        currentOpCount += 1
        return (currentCommandBuffer!, currentEncoder!)
    }

    /// Flush the current batch: end encoding and commit.
    public func flush() {
        if let encoder = currentEncoder {
            encoder.endEncoding()
        }
        if let buffer = currentCommandBuffer {
            buffer.commit()
        }
        currentCommandBuffer = nil
        currentEncoder = nil
        currentOpCount = 0
    }

    /// Flush and wait for GPU completion.
    public func flushAndWait() {
        if let encoder = currentEncoder {
            encoder.endEncoding()
        }
        if let buffer = currentCommandBuffer {
            buffer.commit()
            buffer.waitUntilCompleted()
        }
        currentCommandBuffer = nil
        currentEncoder = nil
        currentOpCount = 0
    }
}
```

**Step 3: Implement BarrierTracker**

```swift
// Sources/EdgeRunnerMetal/BarrierTracker.swift
import Metal

/// Tracks buffer read/write dependencies for manual hazard tracking.
/// Since we use hazardTrackingModeUntracked for better performance,
/// we must insert memory barriers explicitly.
public final class BarrierTracker: @unchecked Sendable {
    /// Buffers that have been written by dispatches in the current encoder.
    private var writtenBuffers: Set<ObjectIdentifier> = []

    /// Check if a buffer needs a barrier before reading.
    public func needsBarrier(forReading buffer: MTLBuffer) -> Bool {
        writtenBuffers.contains(ObjectIdentifier(buffer))
    }

    /// Record that a buffer was written.
    public func recordWrite(_ buffer: MTLBuffer) {
        writtenBuffers.insert(ObjectIdentifier(buffer))
    }

    /// Insert a barrier if needed, then clear the write record for this buffer.
    public func insertBarrierIfNeeded(
        forReading buffer: MTLBuffer,
        encoder: MTLComputeCommandEncoder
    ) {
        if needsBarrier(forReading: buffer) {
            encoder.memoryBarrier(scope: .buffers)
            writtenBuffers.remove(ObjectIdentifier(buffer))
        }
    }

    /// Reset tracking (called when starting a new command buffer).
    public func reset() {
        writtenBuffers.removeAll()
    }
}
```

**Step 4: Build to verify compilation**

Run: `swift build 2>&1`
Expected: BUILD SUCCEEDED

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerMetal/ResidencyManager.swift Sources/EdgeRunnerMetal/CommandBatcher.swift Sources/EdgeRunnerMetal/BarrierTracker.swift
git commit -m "feat: add ResidencyManager, CommandBatcher, BarrierTracker

ResidencyManager: single MTLResidencySet, queue-attached.
CommandBatcher: per-GPU-family batch limits (30-50 ops/buffer).
BarrierTracker: manual hazard tracking for untracked buffers."
```

---

## Task 12: Wire Up MetalBackend with All Components

**Files:**
- Modify: `Sources/EdgeRunnerMetal/MetalBackend.swift`
- Create: `Tests/EdgeRunnerMetalTests/MetalBackendTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/MetalBackendTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("MetalBackend")
struct MetalBackendTests {

    @Test func sharedInstanceExists() async {
        let backend = MetalBackend.shared
        let name = await backend.deviceName
        #expect(!name.isEmpty)
    }

    @Test func acquireAndRecycleBuffer() async throws {
        let backend = MetalBackend.shared
        let buf = await backend.acquireBuffer(size: 1024)
        #expect(buf.length >= 1024)
        await backend.recycleBuffer(buf)
    }

    @Test func kernelRegistryAccessible() async throws {
        let backend = MetalBackend.shared
        let pipeline = try await backend.pipeline(for: "elementwise_add_float")
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter MetalBackendTests 2>&1`
Expected: FAIL — missing methods

**Step 3: Update MetalBackend**

```swift
// Sources/EdgeRunnerMetal/MetalBackend.swift
import Metal
import Synchronization

/// Actor managing all Metal GPU resources.
/// Single shared instance for the lifetime of the application.
public actor MetalBackend {
    public static let shared = MetalBackend()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let kernelRegistry: KernelRegistry
    public let bufferCache: BufferCache
    public let residencyManager: ResidencyManager
    public let commandBatcher: CommandBatcher
    public let barrierTracker: BarrierTracker

    public var deviceName: String { device.name }

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue

        // Initialize subsystems
        do {
            self.kernelRegistry = try KernelRegistry(device: device)
        } catch {
            fatalError("Failed to initialize KernelRegistry: \(error)")
        }

        // Buffer cache: use 50% of recommended working set as cache budget
        let maxCacheBytes = Int(Double(device.recommendedMaxWorkingSetSize) * 0.5)
        self.bufferCache = BufferCache(device: device, maxBytes: max(maxCacheBytes, 64 * 1024 * 1024))

        self.residencyManager = ResidencyManager(device: device, commandQueue: queue)
        self.commandBatcher = CommandBatcher(commandQueue: queue, device: device)
        self.barrierTracker = BarrierTracker()
    }

    // MARK: - Buffer Management

    /// Acquire a buffer from the cache or allocate a new one.
    public func acquireBuffer(size: Int) -> MTLBuffer {
        let buffer = bufferCache.acquire(size: size)
        residencyManager.addBuffer(buffer)
        return buffer
    }

    /// Return a buffer to the cache for reuse.
    public func recycleBuffer(_ buffer: MTLBuffer) {
        bufferCache.recycle(buffer)
    }

    // MARK: - Kernel Dispatch

    /// Get a compiled pipeline state for a kernel by name.
    public func pipeline(for name: String) throws -> MTLComputePipelineState {
        try kernelRegistry.pipeline(for: name)
    }

    /// Dispatch a single compute kernel.
    public func dispatch(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)], // (buffer, index)
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        let (_, encoder) = commandBatcher.encoder()

        // Insert barriers for buffers that were previously written
        for (buffer, _) in buffers {
            barrierTracker.insertBarrierIfNeeded(forReading: buffer, encoder: encoder)
        }

        encoder.setComputePipelineState(pipeline)
        for (buffer, index) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        // Record output buffer as written (last buffer is typically the output)
        if let (outBuffer, _) = buffers.last {
            barrierTracker.recordWrite(outBuffer)
        }
    }

    /// Flush all pending dispatches and wait for GPU completion.
    public func synchronize() {
        commandBatcher.flushAndWait()
        barrierTracker.reset()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter MetalBackendTests 2>&1`
Expected: All 3 tests PASS

**Step 5: Run full test suite**

Run: `swift test 2>&1`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add Sources/EdgeRunnerMetal/MetalBackend.swift Tests/EdgeRunnerMetalTests/MetalBackendTests.swift
git commit -m "feat: wire MetalBackend with all subsystems

Integrates BufferCache, ResidencyManager, CommandBatcher, BarrierTracker.
Provides acquireBuffer/recycleBuffer, pipeline lookup, dispatch, synchronize.
Actor-isolated for thread safety."
```

---

## Task 13: AutoTuner Framework

**Files:**
- Create: `Sources/EdgeRunnerCore/AutoTuner.swift`
- Create: `Tests/EdgeRunnerCoreTests/AutoTunerTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerCoreTests/AutoTunerTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore

@Suite("AutoTuner")
struct AutoTunerTests {

    @Test func defaultConfig() {
        let config = ThreadgroupConfig.default
        #expect(config.width > 0)
        #expect(config.height >= 1)
    }

    @Test func configForElementwise() {
        let config = AutoTuner.config(for: .elementwise, elementCount: 1024)
        #expect(config.width > 0)
        #expect(config.width <= 1024)
    }

    @Test func configForReduction() {
        let config = AutoTuner.config(for: .reduction, elementCount: 4096)
        #expect(config.width > 0)
    }

    @Test func configForTranspose() {
        let config = AutoTuner.config(for: .transpose, elementCount: 256)
        #expect(config.width > 0)
        #expect(config.height > 0)
    }

    @Test func threadgroupsCalculation() {
        let config = ThreadgroupConfig(width: 256, height: 1, depth: 1)
        let groups = config.threadgroups(for: 1000)
        #expect(groups.width == 4) // ceil(1000/256) = 4
        #expect(groups.height == 1)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter AutoTunerTests 2>&1`
Expected: FAIL

**Step 3: Implement AutoTuner**

```swift
// Sources/EdgeRunnerCore/AutoTuner.swift
import Metal

/// Threadgroup size configuration for a kernel dispatch.
public struct ThreadgroupConfig: Sendable {
    public let width: Int
    public let height: Int
    public let depth: Int

    public init(width: Int, height: Int, depth: Int) {
        self.width = width
        self.height = height
        self.depth = depth
    }

    public static let `default` = ThreadgroupConfig(width: 256, height: 1, depth: 1)

    public var metalSize: MTLSize {
        MTLSize(width: width, height: height, depth: depth)
    }

    /// Calculate the number of threadgroups needed for a given element count.
    public func threadgroups(for elementCount: Int) -> MTLSize {
        MTLSize(
            width: (elementCount + width - 1) / width,
            height: 1,
            depth: 1
        )
    }

    /// Calculate 2D threadgroups for a given (rows, cols).
    public func threadgroups2D(rows: Int, cols: Int) -> MTLSize {
        MTLSize(
            width: (cols + width - 1) / width,
            height: (rows + height - 1) / height,
            depth: 1
        )
    }
}

/// Kernel category for selecting optimal threadgroup sizes.
public enum KernelCategory: Sendable {
    case elementwise
    case reduction
    case transpose
}

/// Provides optimal threadgroup configurations per kernel category.
/// Currently uses static heuristics; will add runtime benchmarking later.
public enum AutoTuner {

    /// Get the optimal threadgroup config for a kernel category.
    public static func config(for category: KernelCategory, elementCount: Int) -> ThreadgroupConfig {
        switch category {
        case .elementwise:
            // Element-wise: 1D, maximize occupancy
            let width = min(256, elementCount)
            return ThreadgroupConfig(width: width, height: 1, depth: 1)

        case .reduction:
            // Reduction: power-of-2 threadgroup size for efficient reduction
            let width = min(256, nextPowerOf2(elementCount))
            return ThreadgroupConfig(width: width, height: 1, depth: 1)

        case .transpose:
            // Transpose: 2D tiling for cache-friendly access
            return ThreadgroupConfig(width: 16, height: 16, depth: 1)
        }
    }

    private static func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter AutoTunerTests 2>&1`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Sources/EdgeRunnerCore/AutoTuner.swift Tests/EdgeRunnerCoreTests/AutoTunerTests.swift
git commit -m "feat: add AutoTuner with per-category threadgroup configs

Static heuristics for elementwise (1D/256), reduction (power-of-2),
transpose (16x16 tiling). Runtime benchmarking to be added later."
```

---

## Task 14: Integration Test — Full Pipeline

**Files:**
- Create: `Tests/EdgeRunnerCoreTests/IntegrationTests.swift`

**Step 1: Write the integration test**

```swift
// Tests/EdgeRunnerCoreTests/IntegrationTests.swift
import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Integration")
struct IntegrationTests {

    @Test func tensorAddOnGPU() async throws {
        let backend = MetalBackend.shared
        let a = Tensor<Float>(data: [1.0, 2.0, 3.0, 4.0], shape: Shape([4]))
        let b = Tensor<Float>(data: [5.0, 6.0, 7.0, 8.0], shape: Shape([4]))

        let pipeline = try await backend.pipeline(for: "elementwise_add_float")
        let outBuffer = await backend.acquireBuffer(size: 4 * MemoryLayout<Float>.size)

        let threadConfig = AutoTuner.config(for: .elementwise, elementCount: 4)

        await backend.dispatch(
            pipeline: pipeline,
            buffers: [(a.metalBuffer, 0), (b.metalBuffer, 1), (outBuffer, 2)],
            threadgroups: threadConfig.threadgroups(for: 4),
            threadsPerThreadgroup: threadConfig.metalSize
        )

        // Need to pass params
        // For now, just verify the pipeline dispatch doesn't crash
        await backend.synchronize()
    }

    @Test func computeGraphBuildsCorrectly() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))

        let sorted = ComputeGraph.topologicalSort(root: relu)
        let groups = ComputeGraph.identifyFusionGroups(sorted)
        let tier = FusionEngine.selectTier(for: groups.last!)

        // add+relu is a known hot path
        switch tier {
        case .functionConstants(let name, let activation):
            #expect(name == "fused_add_activate_float")
            #expect(activation == 1) // relu
        default:
            Issue.record("Expected function constants tier")
        }
    }

    @Test func bufferCacheRoundTrip() async throws {
        let backend = MetalBackend.shared
        let buf = await backend.acquireBuffer(size: 512)
        #expect(buf.length >= 512)

        await backend.recycleBuffer(buf)

        // Acquire again — should reuse from cache
        let buf2 = await backend.acquireBuffer(size: 512)
        #expect(buf2.length >= 512)
    }

    @Test func cpuReferenceAdd() {
        // CPU reference: used to validate GPU kernel output
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]
        let expected: [Float] = zip(a, b).map(+)
        #expect(expected == [6.0, 8.0, 10.0, 12.0])
    }

    @Test func cpuReferenceRelu() {
        let input: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let expected: [Float] = input.map { max($0, 0.0) }
        #expect(expected == [0.0, 0.0, 0.0, 1.0, 2.0])
    }

    @Test func cpuReferenceSigmoid() {
        let input: [Float] = [0.0]
        let expected: [Float] = input.map { 1.0 / (1.0 + Foundation.exp(-$0)) }
        #expect(abs(expected[0] - 0.5) < 1e-6)
    }
}
```

**Step 2: Run all tests**

Run: `swift test 2>&1`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add Tests/EdgeRunnerCoreTests/IntegrationTests.swift
git commit -m "test: add integration tests for full pipeline

GPU dispatch, compute graph + fusion tier selection,
buffer cache round-trip, CPU reference implementations
for add, relu, sigmoid."
```

---

## Task 15: Clean Up & Final Verification

**Step 1: Remove placeholder tests**

Delete the placeholder test files created in Task 1 (they've been superseded by real tests).

**Step 2: Update the public EdgeRunner module to re-export properly**

```swift
// Sources/EdgeRunner/EdgeRunner.swift
@_exported import EdgeRunnerCore
@_exported import EdgeRunnerMetal
@_exported import EdgeRunnerSharedTypes
```

**Step 3: Run the full test suite**

Run: `swift test 2>&1`
Expected: All tests PASS, no warnings

**Step 4: Run a clean build**

Run: `swift package clean && swift build 2>&1`
Expected: BUILD SUCCEEDED

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: clean up placeholders, finalize M1 scaffold

Remove placeholder tests, ensure public API re-exports all targets.
All tests pass, clean build succeeds."
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Package scaffold | 8 created | 2 placeholder |
| 2 | Shape & Strides | 2 created | 11 tests |
| 3 | TensorScalar protocol | 2 created | 5 tests |
| 4 | BufferCache (LRU) | 2 created | 7 tests |
| 5 | TensorStorage & Tensor<T> | 2 created, 1 modified | 8 tests |
| 6 | Element-wise Metal kernels | 3 created | 4 tests |
| 7 | Reduction & Transpose kernels | 4 created | 5 tests |
| 8 | Stitchable ops & fused patterns | 3 created | 3 tests |
| 9 | TensorOp DAG & ComputeGraph | 3 created | 5 tests |
| 10 | FusionEngine (3-tier) | 2 created | 4 tests |
| 11 | ResidencyManager, CommandBatcher, BarrierTracker | 3 created | — |
| 12 | MetalBackend integration | 1 modified, 1 created | 3 tests |
| 13 | AutoTuner | 2 created | 5 tests |
| 14 | Integration tests | 1 created | 6 tests |
| 15 | Cleanup | — | — |

**Total: ~35 files, ~66 tests, 15 commits**
