# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

EdgeRunner is a from-scratch LLM inference engine for Apple Silicon, written in Swift 6.2 with custom Metal compute kernels. It targets fast on-device decode of quantized GGUF models (primarily Qwen3-0.6B Q8_0). See `AGENTS.md` for the autoresearch optimization loop — that file is the source of truth for performance-tuning work and takes precedence over this file when they disagree on benchmarking.

## Repository Layout

```
Package.swift              # SwiftPM manifest — do NOT modify dependencies
README.md                  # Public-facing overview and quick start
AGENTS.md                  # Autoresearch / optimization loop rules (PRECEDENCE for perf work)
TROUBLESHOOTING.md         # Common build / load / runtime issues
CLAUDE.md                  # This file
docs/
  ROADMAP.md               # Phased roadmap and perf context
  EdgeRunner-Framework-Deep-Documentation.md
  arch/                    # Architecture references (pipeline, kernels, API, memory)
Sources/
  EdgeRunnerSharedTypes/   # C header shims shared with Metal shaders
  EdgeRunnerMetal/         # Metal kernels + wrappers (GPU hot path)
    Shaders/               # *.metal source files (compiled as package resource)
  EdgeRunnerIO/            # Model loading: GGUF, SafeTensors, NPY/NPZ, dequant kernels
    GGUF/                  # GGUF parser + memory-mapped file
    Protocols/             # LoadableModel protocol
  EdgeRunnerCore/          # Tensors, sampling, tokenization, graph, structured generation
    Sampling/              # Greedy / temperature / top-k / top-p / min-p / rep-penalty
    Tokenizer/             # BPE, SentencePiece, chat templates, pre-tokenizer
    Graph/                 # ComputeGraph + FusionEngine + TensorOp
    Generation/            # SpeculativeDecoder
    StructuredGeneration/  # JSON schema / grammar-constrained decoding
  EdgeRunner/              # Public façade + high-level API
    Models/                # LlamaLanguageModel (primary), GPT2*
    Transformer/           # Generic transformer block scaffolding
    Module/                # nn-module-style wrappers (Linear, Sequential, TensorBox)
    Backends/              # Backend factory, local backend, Foundation Models backend
    Streaming/             # TokenStream + GenerationSession
    ToolCalling/           # Tool protocol, parser, executor, tool choice
    Chat/                  # ChatMessage, ChatViewModelState, ModelInfo
    Metrics/               # Perplexity
    Documentation.docc/    # DocC catalog
  ANEInteropIO/            # C code for ANE / IOSurface interop
  EspressoEdgeRunner/      # Experimental ANE/Espresso backend (weight conversion, RoPE bridge)
Tests/
  EdgeRunnerMetalTests/    # Kernel-level tests + KV cache / memory benchmarks
  EdgeRunnerIOTests/       # Loader / dequant tests
  EdgeRunnerCoreTests/     # Sampling, tokenizer, graph, structured generation
  EdgeRunnerTests/         # Integration, parity, publishable/framework benchmarks
  EspressoEdgeRunnerTests/
Examples/
  EdgeRunnerChat/          # SwiftUI sample chat app
benchmarks/
  pinned_qwen3_0.6b_q8_0.json  # Canonical contract — source of truth
  experiment_log.md             # Append-only experiment history
  baseline.json
  run_long_prompt_framework_benchmark.py
```

## Module Dependency Graph

Layered from bottom up (each depends only on layers below):

```
EdgeRunnerSharedTypes  (C headers — scalar type defs shared with Metal)
        │
EdgeRunnerMetal        (Metal kernels, buffer cache, residency, KV cache)
        │
EdgeRunnerIO           (GGUF/SafeTensors loaders, dequantization kernels)
        │
EdgeRunnerCore         (Tensors, sampling, tokenizers, graph, structured gen)
        │
EdgeRunner             (Public façade: EdgeRunner actor, LlamaLanguageModel, streaming)
```

`EspressoEdgeRunner` is a parallel experimental target layered on `EdgeRunnerIO + EdgeRunnerMetal + ANEInteropIO`. `ANEInteropIO` is C-only with `IOSurface` linkage.

## Platform & Toolchain

- Swift tools 6.2, platforms: `.iOS(.v26)`, `.macOS(.v26)`
- Requires Apple Silicon (M1+), Xcode 26 beta or newer
- `@_exported import` surface lives in `Sources/EdgeRunner/EdgeRunner.swift` (re-exports Core, IO, Metal, SharedTypes)

## Key Entry Points

| Type | File | Purpose |
|---|---|---|
| `EdgeRunner` (actor) | `Sources/EdgeRunner/EdgeRunnerFacade.swift` | High-level public API — `init(modelPath:)`, `stream(_:)`, `generate(_:)` |
| `EdgeRunnerLanguageModel` (protocol) | `Sources/EdgeRunner/EdgeRunnerLanguageModel.swift` | Contract for model implementations |
| `LlamaLanguageModel` | `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` | **Primary inference engine** — hot path for all optimization work |
| `ModelLoader` | `Sources/EdgeRunner/ModelLoader.swift` | Dispatches GGUF/SafeTensors loading |
| `GGUFLoader` / `GGUFParser` | `Sources/EdgeRunnerIO/GGUF/` | Memory-mapped GGUF parsing |
| `MetalBackend` | `Sources/EdgeRunnerMetal/MetalBackend.swift` | Device / command queue / shader library |
| `KVCache` | `Sources/EdgeRunnerMetal/KVCache.swift` | GPU-resident K/V cache for incremental decode |
| `SamplingPipeline` | `Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift` | Composable sampling (temp → top-k → top-p → min-p → rep penalty) |
| `BPETokenizer` | `Sources/EdgeRunnerCore/Tokenizer/BPETokenizer.swift` | BPE; Qwen-compatible via `ChatTemplateEngine` |

## Metal Kernel Surface

Each kernel in `Sources/EdgeRunnerMetal/` has a Swift wrapper + a `.metal` shader in `Shaders/`. Notable kernels:

- `GEMMKernel`, `GEMVKernel` — dense matmul / matvec
- `GQAKernel`, `FlashAttentionKernel` — attention (grouped-query + flash)
- `RMSNormKernel`, `LayerNormKernel`, `ActivationKernels`, `SoftmaxKernel`, `RoPEKernel`
- `TurboQuantKernel` + `TurboQuant.swift` — EdgeRunner's custom quant format
- `FusedPatterns.metal` / `StitchableOps.metal` — fused RMSNorm+GEMV, dequant+GEMV+residual, etc.
- `BufferCache`, `ResidencyManager`, `CommandBatcher`, `BarrierTracker` — GPU resource/scheduling infrastructure

Dequantization kernels for `Q2_K / Q3_K / Q4_0 / Q4_K_M / Q5_0 / Q5_1 / Q5_K / Q6_K / Q8_0` live in `Sources/EdgeRunnerIO/Dequant*.swift` with matching shaders in `Sources/EdgeRunnerMetal/Shaders/Dequant_*.metal`.

## Building and Testing

```bash
# Build (debug)
swift build

# Unit tests (all targets)
swift test

# Run a specific suite
swift test --filter "KernelBenchmarks"
swift test --filter "SamplingPipelineTests"

# The canonical publishable benchmark (release, 128-token greedy decode)
swift test -c release --filter "PublishableBenchmark/fullBenchmark"

# Smoke/regression (4-token — NOT apples-to-apples with publishable)
swift test -c release --filter "QwenBenchmark/decodeBenchmark"
```

### Benchmark Ground Rules (from AGENTS.md)

- **Canonical metric:** `PublishableBenchmark/fullBenchmark` — 128-token greedy decode, TTFT separated, release build. Use p50 decode tok/s.
- **Pinned model:** Qwen3-0.6B Q8_0 at `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`, expected size `639,446,688` bytes.
- **Contract source of truth:** `benchmarks/pinned_qwen3_0.6b_q8_0.json` — the pinned hash/prefix contract.
- **Correctness guard:** Greedy prefix must start with `[1, 1479, 35]`; full 128-token hash must match the pinned value. Any divergence = correctness regression, roll back.
- **Mega fused GQA kernel is disabled** on the safe decode path until it regains determinism on the pinned artifact — do not re-enable it without a correctness plan.
- **Metal 4** is available on macOS 26+, but the Metal 3 decode path remains default. Set `EDGERUNNER_DECODE_PREFER_METAL4=1` to compare.
- Cached JSON artifacts under `benchmarks/` (other than the pinned contract and `experiment_log.md`) are record-keeping — always rerun for truth.
- Benchmark JSON outputs are gitignored (see `.gitignore`).

### Editable vs. Off-Limits Files

Editable for performance work:
- `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` — **primary target**
- `Sources/EdgeRunnerMetal/*.swift` and the matching `Shaders/*.metal`
- `Sources/EdgeRunnerIO/Dequant*.swift`
- `Sources/EdgeRunnerCore/Sampling/*.swift`

**Do not modify** without explicit intent and documentation:
- `Tests/EdgeRunnerTests/PublishableBenchmark.swift` — benchmark harness semantics
- `Tests/EdgeRunnerTests/QwenBenchmark.swift` — smoke/regression harness semantics
- `Package.swift` — no dependency changes allowed
- `benchmarks/pinned_qwen3_0.6b_q8_0.json` — only update when the pinned artifact legitimately changes

## Coding Conventions

- **Swift 6 concurrency**: the public façade is an `actor`; protocol conformers are `Sendable`. Keep new public types `Sendable` and use actors/`async` for anything touching GPU state.
- **Metal resource discipline**: prefer reusing `MTLBuffer` via `BufferCache` / `ResidencyManager`. Do not allocate per-token. Prefer single-command-buffer encodes over per-op command buffers (see AGENTS.md Pattern 3).
- **KV cache is authoritative** for decode — don't recompute full-sequence K/V unless the call is prefill (`tokenIDs.count != cachedPosition + 1`).
- **Fused kernels** are preferred over separate dispatches on the hot path (see `FusedPatterns.metal`, `StitchableOps.metal`).
- **Public API** lives in `Sources/EdgeRunner/` — keep doc comments (`///`) with at least one usage snippet when adding or modifying public types. The DocC catalog is at `Sources/EdgeRunner/Documentation.docc/`.
- **Tests go next to the target they exercise** (`Tests/EdgeRunner<Target>Tests/`). Parity / integration tests that load the pinned model live in `Tests/EdgeRunnerTests/`.
- **No new dependencies.** Add stdlib/Foundation/Metal-only code.

## Model Config Reference (Qwen3-0.6B)

```
embeddingDim: 1024      layerCount: 28        headCount: 16
kvHeadCount: 8          headDim: 128          intermediateDim: 3072
vocabSize: 151936       ropeFreqBase: 1e6     rmsNormEpsilon: 1e-6
```

## Workflow Expectations

### For any code change
1. **Read before editing.** Open the file(s) you intend to modify in full; understand surrounding invariants.
2. **Build quickly:** `swift build 2>&1 | tail -20`. Fix or roll back on failure — do not paper over warnings.
3. **Run the narrowest relevant tests** first (e.g. a single kernel test), then the broader suite.
4. **For any decode/prefill hot-path change**, run the publishable benchmark (release). Correctness guard must pass.
5. **Commit small, focused changes** with messages that describe intent. For perf work follow the `AGENTS.md` commit protocol: `perf: <what changed> — <old> → <new> tok/s (+<pct>%)`.
6. **Log perf experiments** in `benchmarks/experiment_log.md` using the format in `AGENTS.md` (Hypothesis / Change / Result / Status).

### For this branch
- Development branch: `claude/add-claude-documentation-Ov2iK`
- Develop, commit, and push only to this branch unless the user explicitly says otherwise.
- Do not create a PR unless explicitly asked.

### Git discipline
- New commits only — do not amend unless asked.
- Never push to `main`/`master`. Never force-push without explicit approval.
- Never skip hooks (`--no-verify`) or bypass signing.
- Stage files explicitly rather than `git add -A` to avoid pulling in model artifacts or cached benchmark JSON.

## Where to Look for Answers

| Question | Start here |
|---|---|
| "How does decode work?" | `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` + `docs/arch/inference_pipeline.md` |
| "How is KV cache laid out?" | `Sources/EdgeRunnerMetal/KVCache.swift` + `docs/arch/memory_compute.md` |
| "Which kernel runs for X?" | `docs/arch/metal_shaders.md` + `Sources/EdgeRunnerMetal/KernelRegistry.swift` |
| "How do I load a new GGUF?" | `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift`, `Sources/EdgeRunner/ModelLoader.swift` |
| "How is sampling wired up?" | `Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift` |
| "How do I add a tokenizer feature?" | `Sources/EdgeRunnerCore/Tokenizer/` — start at `TokenizerProtocol.swift` / `TokenizerFactory.swift` |
| "What's the public API shape?" | `docs/arch/public_api.md` + `Sources/EdgeRunner/EdgeRunnerFacade.swift` |
| "What perf work has been tried?" | `benchmarks/experiment_log.md` |
| "What's next on the roadmap?" | `docs/ROADMAP.md` |
| "Why does my build/load fail?" | `TROUBLESHOOTING.md` |

## Non-Negotiables

- `AGENTS.md` governs performance work and benchmarking protocol — defer to it.
- The publishable benchmark is the canonical perf metric. Rerun it; never trust stale JSON.
- Correctness guard (`[1, 1479, 35]` prefix + pinned 128-token hash) must pass after any hot-path change.
- No dependency changes. No benchmark-harness semantic changes without explicit intent.
- No emojis, no speculative abstractions, no unrequested refactors.
