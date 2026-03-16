# EdgeRunner Master Plan

**Date:** 2026-03-16
**Authors:** Christopher Karani + Claude
**Status:** Active

---

## Vision

EdgeRunner is a Metal-native Swift 6.2 inference engine purpose-built for running large language models on Apple Silicon. It targets iOS 26+ and macOS 26+ exclusively, leveraging Metal 4 compute shaders, hybrid MTLBuffer/MTLTensor architecture, and 3-tier kernel fusion to deliver best-in-class on-device inference performance. The project aims to make LLM inference a first-class citizen on Apple hardware -- no Python, no ONNX conversion, no compromise -- with native GGUF/SafeTensor support, strict Swift concurrency, and an iPhone-first memory design that enables 3B parameter models on 8 GB devices.

---

## Milestone Dependency Graph

```
 M1: Core Tensor & Metal Infrastructure
  |
  v
 M2: Transformer Primitives
  |
  v
 M3: Weight Loading & Quantisation
  |
  v
 M4: High-Level API & Developer Experience
  |
  v
 M5: Memory Optimization & Device Adaptation
```

Each milestone strictly depends on its predecessor. No milestone may begin implementation until the prior milestone passes all verification gates and a code review checkpoint is completed.

---

## Milestone Summary Table

| Milestone | Goal | Tasks | Est. Tests | Key Deliverables | Depends On |
|-----------|------|------:|------------|------------------|------------|
| **M1** | Core Tensor & Metal Infrastructure | 15 | ~120 | `Tensor<T>` with COW, Shape/Strides, TensorScalar, BufferCache (LRU), TensorStorage, element-wise/reduction/transpose Metal kernels, stitchable ops, TensorOp DAG, ComputeGraph, FusionEngine (3-tier), ResidencyManager, CommandBatcher, AutoTuner | None |
| **M2** | Transformer Primitives | 14 | ~100 | Tiled GEMM, GEMV, softmax, Flash Attention, GQA, ring-buffer KV cache, RoPE, LayerNorm/RMSNorm, SiLU/GELU activations, embedding lookup, linear projection, TransformerBlock, integration tests | M1 |
| **M3** | Weight Loading & Quantisation | 12 | ~80 | GGUF parser, SafeTensor loader, Q4_0/Q4_K_M/Q8_0 dequant kernels, mixed-precision matmul, memory-mapped weight storage, model config parser, weight layout transformer, quantisation-aware graph optimizer | M2 |
| **M4** | High-Level API & Developer Experience | 10 | ~60 | `EdgeRunner.generate()` streaming API, tokenizer integration (BPE), sampling strategies (top-k/top-p/temperature), speculative decoding, Foundation Models framework backend, EdgeRunnerChat demo app (iOS), documentation, error diagnostics | M3 |
| **M5** | Memory Optimization & Device Adaptation | 7 | ~50 | KV cache compression (MiniCache, Squeezed Attention, DuoAttention), context-aware memory budgeting, context summarisation/recycling, Metal 4/M5 device optimizations, long-context benchmarks | M4 |

---

## Cross-Cutting Concerns

### Concurrency Model

- `MetalBackend` is an `actor` -- all GPU resource access is actor-isolated
- Caches (BufferCache, KernelRegistry) use `Mutex<T>` from the `Synchronization` framework
- All public types conform to `Sendable` — `@unchecked Sendable` permitted only for Metal protocol wrappers (`MTLBuffer`, `MTLCommandQueue`) with justifying comments
- `MetalBackend.shared` is the only global — it is actor-isolated, so all access is serialized. Tests may create dedicated instances via `init(device:)`. No `DispatchQueue`-based synchronization.
- Swift 6.2 strict concurrency mode enforced project-wide

### Memory Management Strategy

- **Unified memory:** `MTLResourceOptions: [.storageModeShared, .hazardTrackingModeUntracked]` for CPU-GPU shared access
- **Buffer recycling:** LRU cache keyed by size class, validated by MLX architecture patterns
- **Memory-mapped weights:** `mmap`-based weight loading for GGUF files -- avoids doubling memory on load
- **Tiered quantisation fallback:** If device memory is constrained, fall back from FP16 to Q8_0 to Q4_K_M automatically
- **Residency management:** Single `MTLResidencySet` per queue, max 32 sets as recommended by Apple
- **Context window budgeting:** 4-8K context on 8 GB devices; larger on M-series Macs with more memory headroom

### Testing Philosophy

- **TDD mandatory:** Every task begins with failing tests, then implementation, then green tests, then commit
- **CPU reference implementations:** Every GPU kernel has a corresponding CPU reference for correctness validation
- **Tolerance thresholds:** Float32 kernels: 1e-5 absolute error; Float16 kernels: 1e-3 absolute error; quantised: 5e-2 relative error
- **Property-based tests:** Norm preservation (RoPE), associativity, broadcast correctness
- **Framework:** Swift Testing exclusively (`import Testing`, `@Test`, `@Suite`, `#expect`) -- no XCTest
- **Minimum test counts enforced per milestone** before the verification gate passes

### Performance Baselines and Regression Tracking

- Benchmarks target directory: `Benchmarks/EdgeRunnerBenchmarks/`
- Key metrics tracked: tok/s (prefill and decode), GPU kernel time, memory high-water mark, buffer cache hit rate
- Regression threshold: any commit that degresses throughput by >5% blocks merge
- AutoTuner framework (M1 Task 13) provides per-device threadgroup/tile configuration
- Baseline numbers captured at each milestone gate

### CI/CD Requirements

- **Self-hosted Mac runner required:** Metal GPU tests cannot run on Linux CI or standard GitHub-hosted runners
- Runner must have Apple Silicon (M1 or later) with macOS 26+ beta
- CI pipeline stages: `swift build` -> `swift test` -> benchmarks (optional gate)
- GPU tests tagged for conditional execution when no Metal device is available
- Code signing not required for test targets (command-line SwiftPM)

---

## Package Structure Evolution

### M1: Foundation

```
Targets:
  EdgeRunnerSharedTypes  (C)     -- ShaderTypes.h, dispatch param structs
  EdgeRunnerMetal        (Swift) -- MetalBackend, BufferCache, KernelRegistry,
                                    CommandBatcher, ResidencyManager, BarrierTracker,
                                    Shaders/{Elementwise,Reduction,Transpose,
                                    StitchableOps,FusedPatterns}.metal
  EdgeRunnerCore         (Swift) -- Tensor<T>, TensorScalar, Shape, Strides,
                                    TensorStorage, Graph/{TensorOp,ComputeGraph,
                                    FusionEngine}, AutoTuner
  EdgeRunner             (Swift) -- Public facade, re-exports Core + Metal

Test targets:
  EdgeRunnerCoreTests
  EdgeRunnerMetalTests
```

### M2: + Transformer Primitives

```
New in EdgeRunnerSharedTypes:
  include/GEMMParams.h, include/AttentionParams.h, include/RoPEParams.h,
  include/NormParams.h

New in EdgeRunnerMetal:
  Shaders/{GEMM,GEMV,Softmax,FlashAttention,RoPE,LayerNorm,Activations}.metal
  {GEMM,GEMV,Softmax,FlashAttention,GQA,KVCache,RoPE,Norm,Activation,
   Embedding,Linear}Kernel.swift
  TransformerBlock.swift

New test target:
  EdgeRunnerTransformerTests
```

### M3: + Weight Loading & Quantisation

```
New target:
  EdgeRunnerIO           (Swift) -- GGUF parser, SafeTensor loader,
                                    model config, weight layout transformer

New in EdgeRunnerMetal:
  Shaders/{Dequant_Q4_0,Dequant_Q4_K_M,Dequant_Q8_0,MixedPrecisionGEMM}.metal
  DequantKernel.swift, MixedPrecisionGEMMKernel.swift

New in EdgeRunnerCore:
  QuantisationConfig.swift, GraphOptimizer.swift

New test targets:
  EdgeRunnerIOTests
```

### M4: + High-Level API

```
New target:
  EdgeRunnerGeneration   (Swift) -- Streaming generate API, sampling,
                                    tokenizer integration

New in EdgeRunner (facade):
  FoundationModelsBackend.swift  -- Foundation Models framework adapter

New test target:
  EdgeRunnerGenerationTests
```

### M5: + Memory Optimization & Device Adaptation

```
New in EdgeRunnerCore:
  KVCacheCompressor.swift          -- MiniCache, Squeezed Attention, DuoAttention strategies
  MemoryBudgetPlanner.swift        -- Context-aware memory budgeting per device class
  ContextRecycler.swift            -- Context summarisation and recycling

New in EdgeRunnerMetal:
  Shaders/{MiniCache,SqueezedAttention,DuoAttention}.metal
  KVCompressionKernel.swift

New:
  Benchmarks/EdgeRunnerBenchmarks/  -- Long-context benchmark suite
```

---

## Model Support Roadmap

| Milestone | Models Supported | Format | Notes |
|-----------|-----------------|--------|-------|
| **M2** | GPT-2 124M (random weights) | CPU reference | Architecture validation only; verifies transformer pipeline correctness with random weights. Real weight loading comes in M3. |
| **M3** | Llama 3.2 1B, Llama 3.2 3B, any SafeTensor model | GGUF (Q4_0, Q4_K_M, Q8_0), SafeTensor (FP16/FP32) | First real-world model support; memory-mapped GGUF loading |
| **M4** | All M3 models + Foundation Models backend | Same as M3 + Foundation Models | Developer-facing API; streaming generation; sampling strategies |
| **M5** | All M4 models with extended context | Same as M4 | KV cache compression enables 8K context on 8GB devices; Metal 4 optimizations for M5 chips |

---

## Risk Registry

### M1: Core Tensor & Metal Infrastructure

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Metal 4 API changes in beta | Medium | High | Pin to specific Xcode beta; abstract Metal calls behind internal protocols |
| Function stitching performance worse than expected | Low | Medium | 3-tier fusion design allows fallback to function constants for hot paths |
| Buffer cache eviction policy suboptimal | Medium | Medium | Start with LRU, instrument hit rates, switch to size-class bucketing if needed |
| SwiftPM Metal shader compilation issues | Medium | High | Custom BuildToolPlugin with explicit `-I` flag handling; tested in Phase 0 |

### M2: Transformer Primitives

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flash Attention numerical instability on Metal | Medium | High | Online softmax with running max; extensive CPU reference comparison |
| GEMM performance far below theoretical | Medium | Medium | Tiled implementation with simdgroup intrinsics; AutoTuner for tile sizes |
| KV cache memory pressure at large context | High | High | Ring-buffer design with fixed max size; tiered context fallback |

### M3: Weight Loading & Quantisation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GGUF format versioning breaks parser | Medium | Medium | Support GGUF v2/v3; version detection with graceful error messages |
| Dequantisation kernel accuracy loss | Low | High | Validate against llama.cpp reference outputs; per-layer error tracking |
| 3B model exceeds 8 GB device memory | Medium | High | Memory-mapped weights + lazy loading; Q4_K_M reduces to ~2 GB weights |

### M4: High-Level API & Developer Experience

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Foundation Models API unavailable or restricted | Medium | High | Design as optional adapter; core API works standalone |
| Tokenizer edge cases (BPE merge order) | Medium | Medium | Port battle-tested tokenizer logic; fuzz test with known model outputs |
| Streaming API backpressure issues | Low | Medium | AsyncStream with bounded buffer; configurable batch size |

### M5: Memory Optimization & Device Adaptation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KV cache compression degrades generation quality | Medium | High | Validate perplexity loss < 1% on standard benchmarks; allow per-layer compression ratio tuning |
| Device compatibility across A-series and M-series chips | Medium | Medium | Abstract device capability detection; fallback to uncompressed KV cache on unsupported hardware |
| DuoAttention head classification accuracy | Medium | Medium | Use profiling-based head importance scoring; validate against full-attention baseline |

---

## Success Metrics

### M1: Core Tensor & Metal Infrastructure
- [ ] `swift build` succeeds with zero errors and zero warnings
- [ ] All tests pass (minimum 120 tests across all suites)
- [ ] Element-wise kernel throughput within 80% of theoretical memory bandwidth
- [ ] Buffer cache hit rate > 90% on repeated graph evaluations
- [ ] 3-tier fusion selects correct tier for known patterns (verified by unit tests)
- [ ] All `@unchecked Sendable` uses are justified (Metal protocol wrappers only) and commented

### M2: Transformer Primitives
- [ ] GEMM achieves >60% of peak TFLOPS on M-series (FP16)
- [ ] Flash Attention passes numerical validation against naive CPU implementation (tolerance 1e-4)
- [ ] KV cache supports context lengths up to 8192 on 8 GB devices
- [ ] Full TransformerBlock forward pass produces correct logits for GPT-2 124M
- [ ] All tests pass (minimum 100 tests)

### M3: Weight Loading & Quantisation
- [ ] GGUF parser loads Llama 3.2 1B and 3B models correctly
- [ ] Q4_K_M 3B model fits within 3.6 GB total memory at 8K context
- [ ] Dequantisation + matmul error < 5e-2 relative to FP32 reference
- [ ] Memory-mapped weight loading uses < 50 MB transient allocation
- [ ] SafeTensor loader handles models with architecture metadata or weight-name inference; errors clearly on ambiguous files

### M4: High-Level API & Developer Experience
- [ ] `EdgeRunner.generate()` produces coherent text from Llama 3.2 3B
- [ ] Streaming API delivers first token within 500ms on M1
- [ ] Token throughput: 30-40 tok/s on iPhone, 200+ tok/s on M-series desktop
- [ ] Foundation Models backend passes Apple's conformance tests (if available)
- [ ] Public API surface is fully documented with DocC

### M5: Memory Optimization & Device Adaptation
- [ ] MiniCache achieves >4x KV cache compression with <1% perplexity loss
- [ ] 8K context runs within 4GB memory budget on iPhone 16 Pro
- [ ] DuoAttention reduces memory by >2x for long sequences
- [ ] Metal 4 tensor_ops achieve >1.3x speedup on M5 vs custom kernels on M4

---

## Research-Validated Constraints

These numbers come from the plan review process and are validated against Metal documentation, MLX source code, and hardware specifications. They are non-negotiable design constraints.

| Constraint | Value | Source |
|-----------|-------|--------|
| Max Metal buffer arguments per kernel | 31 | Metal Best Practices Guide |
| Max fusion depth (stitched functions) | 11 | Empirical testing of function stitching |
| Q4_K_M 3B model weight size | ~2 GB | GGUF spec: 3B params x 4.5 bits/param |
| Q4_K_M 3B total memory at 8K context | ~3.6 GB | Weights + KV cache + activations + scratch |
| Target throughput (iPhone) | 30-40 tok/s | Comparable to MLX on similar hardware class |
| Target throughput (M-series desktop) | 200+ tok/s | MLX benchmarks on M2/M3 Ultra |
| Metal dispatch overhead (empty kernel) | ~120 us | Measured on M1; justifies command batching |
| Command batch size | 20-50 dispatches per buffer | MLX-validated pattern; amortizes dispatch overhead |
| Max MTLResidencySets per queue | 32 | Apple recommendation; we use 1 |
| Flash Attention memory complexity | O(N) | Tiled online softmax; no materialized N x N matrix |
| Optimal GEMM tile size | 32x32 or 64x64 | Hardware-dependent; AutoTuner selects per device |
| Hazard tracking | Manual (untracked) | 5-15% GPU perf gain; validated by MLX |

---

## Execution Strategy

### Subagent-Driven Development

Each milestone is executed by dedicated Claude subagents, with the main agent maintaining context continuity and architectural oversight.

**Per-Milestone Workflow:**

1. **Planning subagent** -- Reviews the milestone implementation plan, validates assumptions, identifies gaps
2. **Implementation subagents** -- One per task (or small task group), executing strict TDD: failing test -> implementation -> green test -> commit
3. **Review subagent** -- After all tasks complete, performs full milestone verification: builds, tests, benchmarks, code quality

**Subagent Boundaries:**

- Each subagent receives: the milestone implementation plan, relevant source files, and the design doc
- Each subagent returns: committed code, test results, and a brief summary
- Main agent handles: cross-milestone decisions, architecture questions, lesson capture

### Code Review Checkpoints

Between each milestone, a mandatory checkpoint occurs:

1. **Full test suite runs green** -- all tests from current and all prior milestones
2. **No regressions** -- benchmark numbers stable or improved
3. **API review** -- public API surface is minimal, consistent, well-named
4. **Concurrency audit** — all `@unchecked Sendable` justified (Metal wrappers only), no data races under `-strict-concurrency=complete`
5. **Memory audit** -- no leaks under Instruments; buffer cache behaving correctly
6. **Lessons captured** -- `tasks/lessons.md` updated with patterns learned during the milestone

### Milestone Execution Order

```
M1 -> Checkpoint -> M2 -> Checkpoint -> M3 -> Checkpoint -> M4 -> Checkpoint -> M5 -> Final Release
```

No milestone may be started until the prior checkpoint passes. If a checkpoint reveals architectural issues, the plan is revised before proceeding.

### Commit Discipline

- One commit per task (or logical sub-task)
- Commit messages follow conventional commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`
- Each commit must leave the project in a buildable, testable state
- No WIP commits on main; use feature branches if exploratory work is needed
