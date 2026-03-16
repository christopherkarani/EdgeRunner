# Phase 0: EdgeRunner Package Scaffold & Foundation Types

## Role

You are a senior Swift 6.2 / Metal 4 engineer implementing the foundational scaffold for **EdgeRunner**, a Metal-native inference engine for running LLMs on Apple Silicon. You write production-quality code — no placeholders, no shortcuts, no "TODO" comments. You implement, test, verify, and commit. You do not propose solutions or ask for permission to continue.

---

## Context

### What EdgeRunner Is

EdgeRunner is a Swift 6.2 framework for neural-network inference on Apple Silicon using Metal 4 compute shaders. It targets **iOS 26+ / macOS 26+** exclusively — no Metal 3 fallback, no dual codepaths. Key differentiators:

- **Native GGUF weight format support** — loads community-quantised models directly without conversion
- **Full SwiftPM CLI compatibility** — all Metal shaders compile via SwiftPM, no Xcode project required
- **Swift 6 strict concurrency** — `Sendable` everywhere, actor-isolated GPU resources, zero `@unchecked Sendable`
- **Hybrid MTLBuffer + MTLTensor architecture** — `MTLBuffer` (shared storage) for CPU access and buffer cache recycling; buffer-backed `MTLTensor` views with explicit strides for GPU kernel dispatch via Metal 4 APIs
- **3-tier kernel fusion** — function constants (hot) → function stitching (warm) → JIT compilation (cold)
- **iPhone-first memory design** — memory-mapped weights, lazy loading, tiered quantisation fallback, 4-8K context windows on 8 GB devices

### Repository State

- **Location**: `/Users/chriskarani/CodingProjects/EdgeRunner`
- **Current state**: Greenfield — only `docs/plans/`, `docs/prompts/`, and `Plan review summary.docx` exist. No `Package.swift`, no `Sources/`, no `Tests/`.
- **Git**: Initialized on `main` branch with 3 commits (design doc, implementation plan, phase-0 prompt).
- **Design doc**: `docs/plans/2026-02-28-edgerunner-m1-design.md` — contains full architectural context, component designs, and research-validated constraints.
- **Implementation plan**: `docs/plans/2026-02-28-edgerunner-m1-implementation.md` — you are executing the first 3 tasks from this plan.

### Architecture Decisions (Non-Negotiable)

These decisions are final. Do not deviate, question, or propose alternatives:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Swift tools version | 6.2 | Required for latest language features |
| Platforms | `.iOS(.v26)`, `.macOS(.v26)` | Metal 4 only |
| Package structure | Multi-target: `EdgeRunnerSharedTypes` (C) → `EdgeRunnerMetal` (Swift) → `EdgeRunnerCore` (Swift) → `EdgeRunner` (facade) | Clean separation of Metal backend, core types, public API |
| Shared Metal/Swift types | Dedicated C target with `publicHeadersPath: "include"` | Identical memory layout in Metal and Swift; SwiftPM auto-bridges C headers |
| Concurrency model | `MetalBackend` is an `actor`; caches use `Mutex<T>` from `Synchronization` | Swift 6 strict concurrency, no global locks |
| Buffer allocation | `MTLResourceOptions: [.storageModeShared, .hazardTrackingModeUntracked]` | Unified memory for CPU-GPU sharing; manual hazard tracking for performance |
| Test framework | Swift Testing (`import Testing`, `@Test`, `@Suite`, `#expect`) | Modern test framework, NOT XCTest |
| Tensor generic | `Tensor<T: TensorScalar>` protocol-constrained | Type safety with `BitwiseCopyable` + `Sendable` |
| Buffer management | LRU cache (primary) + small MTLHeap (secondary) | Validated by MLX architecture research |
| Hazard tracking | Manual (`HazardTrackingModeUntracked`) | Better GPU performance, validated by MLX |

### Target Dependency Graph

```
EdgeRunnerSharedTypes (C)    ← shared structs/enums for Metal ↔ Swift
        │
        ▼
EdgeRunnerMetal (Swift)      ← Metal device, command queue, buffer cache
        │
        ▼
EdgeRunnerCore (Swift)       ← Tensor<T>, Shape, Strides, TensorScalar
        │
        ▼
EdgeRunner (Swift)           ← thin public facade, re-exports Core + Metal
```

---

## Instructions

Continue until ALL tasks below are fully complete and ALL verification steps pass. Do not stop early, do not ask for permission to continue between tasks, do not propose solutions — implement them.

Execute the following tasks **in order**. Each task follows strict TDD: write failing tests first, then implement until tests pass, then commit. Parallelize nothing — these tasks have sequential dependencies.

Check prerequisites before each task. If `swift build` fails, read the error, fix it, and rebuild before proceeding. If `swift test` fails, diagnose the failure and fix it — do not skip tests or mark them as expected failures.

Gate any destructive operation (deleting files, force-pushing) with explicit confirmation. All other actions proceed autonomously.

---

### Task 1: Package.swift & Directory Structure

**Create these files:**

#### 1.1 `Package.swift`

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

        // Metal GPU backend
        .target(
            name: "EdgeRunnerMetal",
            dependencies: ["EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerMetal"
        ),

        // Tensor types, shape, lazy graph
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

**Zero external dependencies.** Do not add any.

#### 1.2 `Sources/EdgeRunnerSharedTypes/include/ShaderTypes.h`

```c
#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include <stdint.h>

// Data type enum shared between Metal shaders and Swift.
// __attribute__((enum_extensibility(closed))) ensures Swift bridges this
// as a frozen enum, enabling exhaustive switching.
typedef enum __attribute__((enum_extensibility(closed))) {
    ERDTypeFloat32 = 0,
    ERDTypeFloat16 = 1,
    ERDTypeInt8    = 2,
    ERDTypeUInt8   = 3,
} ERDType;

// Parameters for element-wise kernel dispatch.
typedef struct {
    uint32_t elementCount;
} ERElementwiseParams;

// Parameters for reduction kernel dispatch.
typedef struct {
    uint32_t elementCount;
    uint32_t reductionSize;
    uint32_t outerSize;
} ERReductionParams;

// Parameters for transpose kernel dispatch.
typedef struct {
    uint32_t rows;
    uint32_t cols;
} ERTransposeParams;

#endif /* SHADER_TYPES_H */
```

#### 1.3 `Sources/EdgeRunnerSharedTypes/ShaderTypes.c`

```c
// Required by SwiftPM — C targets must contain at least one .c source file.
// All types are defined in include/ShaderTypes.h.
```

#### 1.4 `Sources/EdgeRunnerMetal/MetalBackend.swift`

```swift
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

#### 1.5 `Sources/EdgeRunnerCore/Tensor.swift`

Placeholder — will be replaced in Task 3 when `TensorScalar` exists.

```swift
import EdgeRunnerMetal

public struct Tensor<T: Sendable>: Sendable {
    public let shape: [Int]

    public init(shape: [Int]) {
        self.shape = shape
    }
}
```

#### 1.6 `Sources/EdgeRunner/EdgeRunner.swift`

```swift
@_exported import EdgeRunnerCore
@_exported import EdgeRunnerMetal
```

#### 1.7 `Tests/EdgeRunnerCoreTests/TensorPlaceholderTests.swift`

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("Tensor Placeholder")
struct TensorPlaceholderTests {

    @Test func tensorInitialization() {
        let t = Tensor<Float>(shape: [2, 3])
        #expect(t.shape == [2, 3])
    }

    @Test func tensorEmptyShape() {
        let t = Tensor<Float>(shape: [])
        #expect(t.shape.isEmpty)
    }
}
```

#### 1.8 `Tests/EdgeRunnerMetalTests/MetalBackendTests.swift`

```swift
import Testing
@testable import EdgeRunnerMetal

@Suite("MetalBackend")
struct MetalBackendTests {

    @Test func metalBackendInitializes() async {
        let backend = MetalBackend.shared
        let device = await backend.device
        #expect(device.name.isEmpty == false)
    }

    @Test func commandQueueExists() async {
        let backend = MetalBackend.shared
        let queue = await backend.commandQueue
        #expect(queue.device.name.isEmpty == false)
    }
}
```

**Verification:**

```bash
cd /Users/chriskarani/CodingProjects/EdgeRunner && swift build 2>&1
# Expected: Build complete! No errors.

swift test 2>&1
# Expected: All 4 tests pass.

swift package describe --type json 2>&1 | head -5
# Expected: Valid JSON, no dependency cycle errors.
```

**Commit:**

```bash
git add Package.swift Sources/ Tests/
git commit -m "feat: scaffold EdgeRunner package with multi-target structure

Targets: EdgeRunnerSharedTypes (C), EdgeRunnerMetal, EdgeRunnerCore, EdgeRunner
Shared C header for Metal/Swift type bridging with ERDType and dispatch param structs.
Minimal MetalBackend actor with MTLDevice/MTLCommandQueue initialization.
Placeholder Tensor<T> and tests using Swift Testing framework."
```

---

### Task 2: Shape & Strides

**Create these files:**
- `Sources/EdgeRunnerCore/Shape.swift`
- `Tests/EdgeRunnerCoreTests/ShapeTests.swift`

#### 2.1 Write failing tests FIRST

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

    @Test func contiguousStridesScalar() {
        let strides = Strides.contiguous(for: Shape([]))
        #expect(strides.values == [])
    }

    @Test func isContiguous() {
        let s = Strides(values: [12, 4, 1])
        #expect(s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func isNotContiguous() {
        let s = Strides(values: [12, 1, 3])
        #expect(!s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func broadcastCompatible() {
        let a = Shape([2, 3, 4])
        let b = Shape([1, 3, 4])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastCompatibleScalar() {
        let a = Shape([2, 3])
        let b = Shape([])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastCompatibleDifferentRank() {
        let a = Shape([2, 3, 4])
        let b = Shape([4])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastIncompatible() {
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

    @Test func broadcastedShapeIdentical() throws {
        let a = Shape([2, 3])
        let result = try a.broadcasted(with: a)
        #expect(result.dimensions == [2, 3])
    }

    @Test func broadcastedShapeError() {
        let a = Shape([2, 3])
        let b = Shape([2, 4])
        #expect(throws: ShapeError.self) {
            try a.broadcasted(with: b)
        }
    }

    @Test func reshapeCompatible() {
        let a = Shape([2, 3, 4])
        #expect(a.reshapeCompatible(with: Shape([6, 4])))
        #expect(a.reshapeCompatible(with: Shape([24])))
        #expect(a.reshapeCompatible(with: Shape([2, 12])))
    }

    @Test func reshapeIncompatible() {
        let a = Shape([2, 3, 4])
        #expect(!a.reshapeCompatible(with: Shape([25])))
        #expect(!a.reshapeCompatible(with: Shape([5, 5])))
    }
}
```

#### 2.2 Run tests — confirm they fail

```bash
swift test --filter ShapeTests 2>&1
# Expected: FAIL — Shape is not defined
```

#### 2.3 Implement Shape and Strides

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

    public func reshapeCompatible(with other: Shape) -> Bool {
        elementCount == other.elementCount
    }
}

public struct Strides: Sendable, Equatable, Hashable, CustomStringConvertible {
    public let values: [Int]

    public init(values: [Int]) {
        self.values = values
    }

    public var description: String {
        "Strides(\(values))"
    }

    public static func contiguous(for shape: Shape) -> Strides {
        let dims = shape.dimensions
        guard !dims.isEmpty else { return Strides(values: []) }

        var strides = [Int](repeating: 1, count: dims.count)
        for i in stride(from: dims.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * dims[i + 1]
        }
        return Strides(values: strides)
    }

    public func isContiguous(for shape: Shape) -> Bool {
        self == Strides.contiguous(for: shape)
    }
}
```

#### 2.4 Run tests — confirm they pass

```bash
swift test --filter ShapeTests 2>&1
# Expected: All 16 tests PASS
```

#### 2.5 Commit

```bash
git add Sources/EdgeRunnerCore/Shape.swift Tests/EdgeRunnerCoreTests/ShapeTests.swift
git commit -m "feat: add Shape and Strides types with broadcasting

NumPy-compatible broadcast rules, contiguous stride computation,
contiguity checks, reshape compatibility. Fully Sendable value types."
```

---

### Task 3: TensorScalar Protocol

**Create these files:**
- `Sources/EdgeRunnerCore/TensorScalar.swift`
- `Tests/EdgeRunnerCoreTests/TensorScalarTests.swift`
- **Modify**: `Sources/EdgeRunnerCore/Tensor.swift` — upgrade from `Sendable` to `TensorScalar` constraint

#### 3.1 Write failing tests FIRST

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
        #expect(Float.zero == 0.0)
    }

    @Test func float16Properties() {
        #expect(Float16.erDType == .ERDTypeFloat16)
        #expect(Float16.byteSize == 2)
        #expect(Float16.zero == 0.0)
    }

    @Test func int8Properties() {
        #expect(Int8.erDType == .ERDTypeInt8)
        #expect(Int8.byteSize == 1)
        #expect(Int8.zero == 0)
    }

    @Test func uint8Properties() {
        #expect(UInt8.erDType == .ERDTypeUInt8)
        #expect(UInt8.byteSize == 1)
        #expect(UInt8.zero == 0)
    }

    @Test func metalDataTypes() {
        #expect(Float.metalDataType == .float)
        #expect(Float16.metalDataType == .half)
        #expect(Int8.metalDataType == .char)
        #expect(UInt8.metalDataType == .uchar)
    }
}
```

#### 3.2 Run tests — confirm they fail

```bash
swift test --filter TensorScalarTests 2>&1
# Expected: FAIL — TensorScalar not defined
```

#### 3.3 Implement TensorScalar

```swift
// Sources/EdgeRunnerCore/TensorScalar.swift
import Metal
import EdgeRunnerSharedTypes

public protocol TensorScalar: Sendable, BitwiseCopyable {
    static var metalDataType: MTLDataType { get }
    static var byteSize: Int { get }
    static var erDType: ERDType { get }
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

#### 3.4 Upgrade Tensor.swift to use TensorScalar

Replace the entire contents of `Sources/EdgeRunnerCore/Tensor.swift` with:

```swift
import EdgeRunnerMetal

public struct Tensor<T: TensorScalar>: Sendable {
    public let shape: Shape
    public let strides: Strides

    public init(shape: Shape) {
        self.shape = shape
        self.strides = Strides.contiguous(for: shape)
    }

    public init(shape: [Int]) {
        self.init(shape: Shape(shape))
    }

    public var rank: Int { shape.rank }
    public var elementCount: Int { shape.elementCount }
    public var isContiguous: Bool { strides.isContiguous(for: shape) }
}
```

#### 3.5 Update placeholder test to match new API

Replace `Tests/EdgeRunnerCoreTests/TensorPlaceholderTests.swift`:

```swift
import Testing
@testable import EdgeRunnerCore

@Suite("Tensor")
struct TensorTests {

    @Test func initWithShapeArray() {
        let t = Tensor<Float>(shape: [2, 3])
        #expect(t.shape.dimensions == [2, 3])
        #expect(t.rank == 2)
        #expect(t.elementCount == 6)
    }

    @Test func initWithShapeType() {
        let t = Tensor<Float16>(shape: Shape([4, 5]))
        #expect(t.shape == Shape([4, 5]))
        #expect(t.strides.values == [5, 1])
    }

    @Test func isContiguous() {
        let t = Tensor<Float>(shape: [2, 3, 4])
        #expect(t.isContiguous)
    }

    @Test func scalarTensor() {
        let t = Tensor<Float>(shape: [])
        #expect(t.rank == 0)
        #expect(t.elementCount == 1)
    }

    @Test func differentScalarTypes() {
        let f32 = Tensor<Float>(shape: [2])
        let f16 = Tensor<Float16>(shape: [2])
        let i8 = Tensor<Int8>(shape: [2])
        let u8 = Tensor<UInt8>(shape: [2])
        #expect(f32.elementCount == 2)
        #expect(f16.elementCount == 2)
        #expect(i8.elementCount == 2)
        #expect(u8.elementCount == 2)
    }
}
```

#### 3.6 Run ALL tests — confirm they pass

```bash
swift test 2>&1
# Expected: ALL tests pass — TensorScalarTests, TensorTests, ShapeTests, MetalBackendTests
```

#### 3.7 Commit

```bash
git add Sources/EdgeRunnerCore/TensorScalar.swift \
       Sources/EdgeRunnerCore/Tensor.swift \
       Tests/EdgeRunnerCoreTests/TensorScalarTests.swift \
       Tests/EdgeRunnerCoreTests/TensorPlaceholderTests.swift
git commit -m "feat: add TensorScalar protocol and upgrade Tensor<T>

TensorScalar protocol with BitwiseCopyable + Sendable constraints.
Conformances for Float, Float16, Int8, UInt8.
Tensor<T> now uses Shape/Strides and TensorScalar constraint."
```

---

## Output Contract

Return ONLY:
1. **Files created/modified** — full path and one-line description of each
2. **Test results** — copy of `swift test` output after each task showing pass/fail counts
3. **Commit hashes** — the SHA for each of the 3 commits
4. **Final verification** — output of `swift build`, `swift test`, and `swift package describe --type json | head -5` after all tasks complete

Do not include: narration of what you plan to do, intermediate reasoning, proposed alternatives, or explanations of why code works.

---

## Verification

Before declaring this phase complete, verify ALL of the following:

- [ ] `swift build` succeeds with zero errors
- [ ] `swift test` passes ALL tests (minimum 27 tests across 5 suites)
- [ ] `swift package describe --type json` outputs valid JSON with no dependency cycles
- [ ] 3 git commits exist on `main` with the messages specified above
- [ ] No files exist beyond those specified (no README, no .gitignore, no CLAUDE.md, no extra directories)
- [ ] `Sources/EdgeRunnerMetal/Shaders/` does NOT exist yet (Metal shaders come in Phase 1)
- [ ] `Sources/EdgeRunnerCore/Graph/` does NOT exist yet (graph types come in Phase 1)
- [ ] Zero external dependencies in `Package.swift`
- [ ] All public types conform to `Sendable`
- [ ] `MetalBackend` is an `actor`, not a `class`
- [ ] `Tensor<T>` is constrained to `TensorScalar`, not raw `Sendable`
- [ ] Tests use Swift Testing (`import Testing`, `@Test`, `@Suite`, `#expect`), NOT XCTest

---

## Constraints

- **Do NOT create any files beyond those listed in the 3 tasks.** No README, no .gitignore, no CLAUDE.md, no Benchmarks/, no Plugins/ — those come in later phases.
- **Do NOT add dependencies to Package.swift.** EdgeRunner is zero-dependency at this stage.
- **Do NOT use XCTest.** Use Swift Testing exclusively.
- **Do NOT add docstrings or comments beyond what is shown in the code blocks.** Minimal code — elaboration comes in Phase 1.
- **Do NOT create `Sources/EdgeRunnerMetal/Shaders/`** — Metal shaders come in Phase 1.
- **Do NOT create `Sources/EdgeRunnerCore/Graph/`** — graph types come in Phase 1.
- **Do NOT create `Sources/EdgeRunnerCore/TensorStorage.swift`** — storage comes in Phase 1.
- **TDD is mandatory.** Write tests first, confirm they fail, then implement. Do not write implementation before tests.
- **If `swift build` fails**, read the error carefully, fix it, and rebuild. Common pitfalls:
  - Missing `import` statements
  - SwiftPM C target needs at least one `.c` file (that's why `ShaderTypes.c` exists)
  - Actor properties accessed from non-async context require `await`
  - `@_exported import` requires the target to actually depend on the imported module
- **If `swift test` fails**, diagnose and fix. Do not skip tests, comment them out, or mark them as expected failures.
- **Commit after each task completes.** 3 tasks = 3 commits. Do not squash into one.

---

## Success Criteria

This phase is complete when:
1. All files from Tasks 1-3 exist at their exact specified paths
2. `swift build` succeeds with zero errors
3. `swift test` passes all tests (5 suites, 27+ tests)
4. 3 sequential git commits exist on `main`
5. No extra files, no missing files
6. The codebase is ready for Phase 1 (Metal kernels, BufferCache, TensorStorage, lazy graph, fusion engine)
