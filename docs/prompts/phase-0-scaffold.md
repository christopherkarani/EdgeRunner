# Phase 0: EdgeRunner Package Scaffold — Agent Prompt

## Instructions for Use
Copy everything below the `---` line and paste it as a prompt to a coding agent (Claude Code, Cursor, etc.) in a new session opened at `/Users/chriskarani/CodingProjects/EdgeRunner`.

---

## Role

You are a senior Swift engineer implementing the foundational scaffold for **EdgeRunner**, a Metal 4-native inference engine for running LLMs on Apple Silicon. You are executing Task 1 of an approved implementation plan.

## Context

### What EdgeRunner Is
EdgeRunner is a Swift 6.2 framework for neural-network inference on Apple Silicon using Metal 4 compute shaders. It targets iOS 26+ / macOS 26+ exclusively (no Metal 3 fallback). The key differentiators are:
- Native GGUF weight format support
- Full SwiftPM CLI compatibility (no Xcode project required)
- Swift 6 strict concurrency (`Sendable` everywhere, actor-isolated GPU resources)
- Hybrid MTLBuffer storage + MTLTensor views for GPU dispatch
- 3-tier kernel fusion: function constants (hot) → function stitching (warm) → JIT (cold)

### Repository State
- **Location**: `/Users/chriskarani/CodingProjects/EdgeRunner`
- **Current state**: Greenfield — only `docs/plans/` and `Plan review summary.docx` exist. No `Package.swift`, no `Sources/`, no `Tests/`.
- **Git**: Initialized on `main` branch with 2 commits (design doc + implementation plan).
- **Design doc**: `docs/plans/2026-02-28-edgerunner-m1-design.md` — read this for full architectural context.
- **Implementation plan**: `docs/plans/2026-02-28-edgerunner-m1-implementation.md` — you are executing Task 1 from this plan.

### Architecture Decisions Already Made
These are non-negotiable. Do not deviate from them:

| Decision | Choice |
|----------|--------|
| Swift tools version | 6.2 |
| Platforms | `.iOS(.v26)`, `.macOS(.v26)` |
| Package structure | Multi-target: `EdgeRunnerSharedTypes` (C) → `EdgeRunnerMetal` (Swift+Metal) → `EdgeRunnerCore` (Swift) → `EdgeRunner` (facade) |
| Shared Metal/Swift types | Dedicated C target with `publicHeadersPath: "include"`, NOT inline in `.metal` files |
| Concurrency model | `MetalBackend` is an `actor`, caches use `Mutex<T>` from `Synchronization` |
| Buffer allocation | `MTLResourceOptions: [.storageModeShared, .hazardTrackingModeUntracked]` |
| Test framework | Swift Testing (`import Testing`, `@Test`, `@Suite`, `#expect`) — NOT XCTest |
| `Tensor<T>` | Generic struct constrained to `TensorScalar` protocol (implemented in later task — use `Sendable` placeholder for now) |

### Target Dependency Graph
```
EdgeRunnerSharedTypes (C)    ← shared structs/enums for Metal ↔ Swift
        │
        ▼
EdgeRunnerMetal (Swift)      ← Metal device, command queue, shader loading, buffer cache
        │
        ▼
EdgeRunnerCore (Swift)       ← Tensor<T>, Shape, lazy graph, fusion engine
        │
        ▼
EdgeRunner (Swift)           ← thin public facade, re-exports Core + Metal
```

## Task

Create the complete project scaffold: `Package.swift`, all source directories, minimal compilable source files for each target, and placeholder tests that verify the build works end-to-end.

## Deliverables

You must create exactly these 8 files:

### 1. `Package.swift`
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
        .target(
            name: "EdgeRunnerSharedTypes",
            path: "Sources/EdgeRunnerSharedTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "EdgeRunnerMetal",
            dependencies: ["EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerMetal"
        ),
        .target(
            name: "EdgeRunnerCore",
            dependencies: ["EdgeRunnerMetal"],
            path: "Sources/EdgeRunnerCore"
        ),
        .target(
            name: "EdgeRunner",
            dependencies: ["EdgeRunnerCore"],
            path: "Sources/EdgeRunner"
        ),
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

### 2. `Sources/EdgeRunnerSharedTypes/include/ShaderTypes.h`
C header defining types shared between Metal shaders and Swift. Must include:
- `ERDType` enum: `ERDTypeFloat32 = 0`, `ERDTypeFloat16 = 1`, `ERDTypeInt8 = 2`, `ERDTypeUInt8 = 3`
- `ERElementwiseParams` struct: `uint32_t elementCount`
- `ERReductionParams` struct: `uint32_t elementCount`, `uint32_t reductionSize`, `uint32_t outerSize`
- `ERTransposeParams` struct: `uint32_t rows`, `uint32_t cols`
- Use `__attribute__((enum_extensibility(closed)))` on the enum for Swift bridging
- Include guard: `SHADER_TYPES_H`
- Include `<stdint.h>` for `uint32_t`

### 3. `Sources/EdgeRunnerSharedTypes/ShaderTypes.c`
Empty stub file. SwiftPM requires at least one `.c` source file for C targets. Add a comment explaining why it exists.

### 4. `Sources/EdgeRunnerMetal/MetalBackend.swift`
Minimal `MetalBackend` actor with:
- `import Metal`
- `public actor MetalBackend`
- `public static let shared = MetalBackend()`
- `public let device: MTLDevice` — from `MTLCreateSystemDefaultDevice()`
- `public let commandQueue: MTLCommandQueue` — from `device.makeCommandQueue()`
- `private init()` — fatalError if Metal is unavailable
- No other methods yet (those come in Task 12)

### 5. `Sources/EdgeRunnerCore/Tensor.swift`
Minimal placeholder `Tensor<T>`:
- `import EdgeRunnerMetal`
- `public struct Tensor<T: Sendable>: Sendable` (the real `TensorScalar` constraint comes in Task 3)
- `public let shape: [Int]`
- `public init(shape: [Int])`
- Nothing else — this is a stub that will be replaced in Task 5

### 6. `Sources/EdgeRunner/EdgeRunner.swift`
Public API facade:
- `@_exported import EdgeRunnerCore`
- `@_exported import EdgeRunnerMetal`
- Nothing else

### 7. `Tests/EdgeRunnerCoreTests/PlaceholderTests.swift`
```swift
import Testing
@testable import EdgeRunnerCore

@Test func tensorInitialization() {
    let t = Tensor<Float>(shape: [2, 3])
    #expect(t.shape == [2, 3])
}
```

### 8. `Tests/EdgeRunnerMetalTests/PlaceholderTests.swift`
```swift
import Testing
@testable import EdgeRunnerMetal

@Test func metalBackendExists() async {
    let backend = MetalBackend.shared
    let device = await backend.device
    #expect(device.name.isEmpty == false)
}
```

## Verification Steps

After creating all files, run these commands **in order** and verify the expected output:

1. **Build the package:**
   ```bash
   cd /Users/chriskarani/CodingProjects/EdgeRunner && swift build 2>&1
   ```
   **Expected:** `Build complete!` with no errors. Warnings about unused imports are acceptable.

2. **Run all tests:**
   ```bash
   swift test 2>&1
   ```
   **Expected:** Both tests pass. `tensorInitialization` and `metalBackendExists` should show as passed.

3. **Verify the target dependency graph resolves correctly:**
   ```bash
   swift package describe --type json 2>&1 | head -5
   ```
   **Expected:** Valid JSON output, no dependency cycle errors.

## Commit

After all verification passes, create a single git commit:

```bash
git add Package.swift Sources/ Tests/
git commit -m "feat: scaffold EdgeRunner package with multi-target structure

Targets: EdgeRunnerSharedTypes (C), EdgeRunnerMetal, EdgeRunnerCore, EdgeRunner
Shared C header for Metal/Swift type bridging with ERDType, dispatch param structs.
Minimal MetalBackend actor with MTLDevice/MTLCommandQueue initialization.
Placeholder Tensor<T> and tests using Swift Testing framework.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Constraints

- **Do NOT add any files beyond the 8 listed above.** No README, no .gitignore, no CLAUDE.md, no extra directories.
- **Do NOT add dependencies to Package.swift.** EdgeRunner is a zero-dependency library at this stage.
- **Do NOT use XCTest.** Use Swift Testing (`import Testing`, `@Test`, `#expect`).
- **Do NOT add docstrings, comments, or type annotations beyond what is specified.** Keep files minimal — elaboration comes in later tasks.
- **Do NOT create the `Sources/EdgeRunnerMetal/Shaders/` directory yet.** Metal shaders come in Task 6.
- **Do NOT create `Sources/EdgeRunnerCore/Graph/` yet.** Graph types come in Task 9.
- **If `swift build` fails**, read the error carefully, fix it, and re-run. Common issues:
  - Missing `import` statements
  - SwiftPM C target needs at least one `.c` file (that's why `ShaderTypes.c` exists)
  - Actor properties accessed from non-async context require `await`
- **If `swift test` fails**, ensure the Metal backend test is marked `async` since `MetalBackend` is an actor.

## Success Criteria

The task is complete when:
1. All 8 files exist at their exact paths
2. `swift build` succeeds with no errors
3. `swift test` passes both tests
4. A git commit has been created with the staged changes
5. No extra files, no missing files
