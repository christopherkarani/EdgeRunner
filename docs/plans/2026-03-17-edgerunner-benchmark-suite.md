# EdgeRunner Benchmark Suite — Implementation Prompt

> **For Claude:** This prompt defines a comprehensive, publishable benchmark suite for EdgeRunner.
> Use TDD. Run tests after each file. Commit after each major section.

**Goal:** Build a benchmark harness that produces numbers we can publish in a README, blog post,
or academic comparison table. The output should be a Markdown report with reproducible results
that any developer can run on their own Apple Silicon Mac.

**Non-Goal:** We are NOT benchmarking against real GGUF model files (those require multi-GB downloads).
Instead we benchmark EdgeRunner's **kernel throughput**, **pipeline latency**, and **generation loop overhead**
using synthetic weights that exercise the real Metal code paths.

---

## Architecture Context

EdgeRunner's inference stack:

```
Token IDs → Embedding → [TransformerBlock × N] → LM Head → Logits → Sampling → Token ID
                              │
                    ┌─────────┴──────────┐
                    │  RMSNorm           │
                    │  MultiHeadAttention │ ← KVCache (ring buffer)
                    │  RoPE              │
                    │  FeedForward (SiLU) │
                    │  RMSNorm           │
                    └────────────────────┘
```

Key types:
- `MetalBackend` (actor, singleton) — GPU command dispatch, command batching (30-50 ops/buffer)
- `KVCache` — ring buffer per layer, fp32/fp16/fp8 precision
- `BufferCache` — LRU buffer reuse by size class
- `GPT2Model.forward(_ input: [Int]) async throws -> [Float]` — full forward pass (CPU LM head)
- `GenerationSession<Model>.stream(prompt:)` — autoregressive generation loop
- `SamplingPipeline` — composable logit transforms + token selection
- All Metal kernels: GEMM, GEMV, FlashAttention, GQA, RMSNorm, LayerNorm, RoPE, Softmax, Activations, Dequant

---

## Benchmark Categories

### Category 1: Metal Kernel Microbenchmarks

Measure individual kernel throughput at production-relevant sizes.
These are the atoms — everything else is composed from them.

**File:** `Tests/EdgeRunnerMetalTests/KernelBenchmarks.swift`

```
@Suite("Kernel Benchmarks")
```

| Benchmark | Sizes | Metric | What It Proves |
|-----------|-------|--------|----------------|
| GEMM (fp32) | 1024×1024, 2048×2048, 4096×4096 | GFLOPS | Raw matmul throughput |
| GEMM (fp16) | Same | GFLOPS | Half-precision advantage |
| GEMV (fp32) | M=4096, N=4096 (single token decode) | GB/s | Memory-bound decode speed |
| FlashAttention | seqLen=512/1024/2048, heads=32, dim=128 | ms/query | Attention at context lengths |
| GQA | seqLen=1024, heads=32, kvHeads=8, dim=128 | ms/query | Grouped query efficiency |
| RMSNorm | dim=4096, batch=1/32/512 | GB/s | Norm throughput |
| RoPE | seqLen=2048, dim=128, heads=32 | ms | Positional encoding cost |
| Softmax | vocab=128256 (Llama 3 vocab) | ms | Final softmax cost |
| SiLU activation | dim=14336 (Llama 3 intermediate) | ms | Activation throughput |
| Dequant Q4_0 | 1M elements | GB/s | Quantized weight decode |
| Dequant Q8_0 | 1M elements | GB/s | Higher-precision dequant |
| Dequant Q4_K_M | 1M elements | GB/s | K-quant decode speed |

**Implementation pattern for each kernel benchmark:**
```swift
@Test func gemmF32_4096x4096() async throws {
    let backend = MetalBackend.shared
    let M = 4096, N = 4096, K = 4096

    // Create random input buffers at production sizes
    let a = (0..<M*K).map { _ in Float.random(in: -1...1) }
    let b = (0..<K*N).map { _ in Float.random(in: -1...1) }

    // Warmup (3 runs, discard)
    for _ in 0..<3 {
        _ = try await backend.gemm(a: a, b: b, M: M, N: N, K: K)
    }
    await backend.synchronize()

    // Timed runs (10 iterations)
    let iterations = 10
    let clock = ContinuousClock()
    let start = clock.now
    for _ in 0..<iterations {
        _ = try await backend.gemm(a: a, b: b, M: M, N: N, K: K)
    }
    await backend.synchronize()
    let elapsed = start.duration(to: clock.now)

    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
    let flops = Double(2 * M * N * K * iterations)
    let gflops = flops / seconds / 1e9

    print("GEMM fp32 \(M)×\(N)×\(K): \(String(format: "%.1f", gflops)) GFLOPS (\(String(format: "%.2f", seconds/Double(iterations)*1000)) ms/op)")
    #expect(gflops > 10) // Sanity: at least 10 GFLOPS on any Apple Silicon
}
```

**CRITICAL:** Each benchmark must:
1. Use `ContinuousClock` (not `CFAbsoluteTimeGetCurrent`)
2. Call `await backend.synchronize()` before AND after timing to flush GPU work
3. Warmup 3 iterations (JIT, pipeline compilation)
4. Time 10+ iterations and average
5. Print result in parseable format: `BENCHMARK: <name> <value> <unit>`
6. Have a sanity `#expect` (loose lower bound — not perf regression, just "GPU is working")

### Category 2: KV Cache Throughput

**File:** `Tests/EdgeRunnerMetalTests/KVCacheBenchmarks.swift`

| Benchmark | Config | Metric |
|-----------|--------|--------|
| Append throughput (fp32) | 32 layers, dim=4096, 2048 tokens | tokens/sec |
| Append throughput (fp16) | Same | tokens/sec |
| Retrieve after full fill | 32 layers, 2048 tokens | ms |
| Ring buffer wraparound | Write 4096 tokens into 2048-slot cache | correctness + ms |

### Category 3: Weight Loading

**File:** `Tests/EdgeRunnerIOTests/LoadingBenchmarks.swift`

| Benchmark | Config | Metric |
|-----------|--------|--------|
| GGUF parse (header only) | Synthetic 1000-tensor file | ms |
| SafeTensor parse + mmap | Synthetic 1000-tensor file | ms |
| NPZ parse | Synthetic 100-tensor file | ms |
| Metal buffer creation | 100 tensors × 1MB each | ms + GB/s |

### Category 4: Sampling Pipeline

**File:** `Tests/EdgeRunnerCoreTests/SamplingBenchmarks.swift`

| Benchmark | Config | Metric |
|-----------|--------|--------|
| Greedy (argmax) | vocab=128256 | µs/sample |
| Top-K (k=40) | vocab=128256 | µs/sample |
| Top-P (p=0.9) | vocab=128256 | µs/sample |
| Full pipeline (temp+topK+topP+rep) | vocab=128256, 100 prev tokens | µs/sample |
| 1000 sequential samples | Full pipeline | ms total |

### Category 5: End-to-End Generation (Synthetic Model)

**File:** `Tests/EdgeRunnerTests/GenerationBenchmarks.swift`

This is the headline number. Use `GPT2Model` with a small but non-trivial config
that exercises real Metal kernels (not the tiny 32-dim test config).

```swift
// "Benchmark config" — small enough to run fast, large enough to be meaningful
static let benchConfig = GPT2Config(
    vocabSize: 1024,      // Small vocab for speed
    maxSeqLen: 256,       // Meaningful context
    numLayers: 6,         // Enough layers to show pipeline effects
    numHeads: 8,          // Multi-head
    hiddenDim: 512,       // Production-like ratio
    layerNormEps: 1e-5
)
```

| Benchmark | What It Measures | Metric |
|-----------|-----------------|--------|
| Prefill latency | Time to process prompt (64 tokens) | ms |
| Prefill throughput | Prompt tokens / prefill time | tokens/sec |
| Decode throughput | Autoregressive generation of 128 tokens | tokens/sec |
| Time to first token (TTFT) | Prefill + first decode step | ms |
| Generation session overhead | GenerationSession vs raw forward loop | % overhead |
| Speculative decoding speedup | SpeculativeDecoder vs baseline | speedup ratio |
| Memory high-water mark | Peak Metal buffer usage during generation | MB |

**Prefill benchmark pattern:**
```swift
@Test func prefillThroughput() async throws {
    let model = try GPT2Model(config: Self.benchConfig)
    // ... init with random weights ...

    let promptTokens = Array(0..<64) // 64-token prompt

    // Warmup
    _ = try await model.forward(promptTokens)

    // Timed
    let clock = ContinuousClock()
    let start = clock.now
    let iterations = 10
    for _ in 0..<iterations {
        _ = try await model.forward(promptTokens)
    }
    let elapsed = start.duration(to: clock.now)
    let seconds = /* extract */
    let tokensPerSec = Double(promptTokens.count * iterations) / seconds

    print("BENCHMARK: prefill_throughput \(String(format: "%.0f", tokensPerSec)) tokens/sec")
    print("BENCHMARK: prefill_latency \(String(format: "%.1f", seconds/Double(iterations)*1000)) ms")
}
```

**Decode benchmark pattern:**
```swift
@Test func decodeThroughput() async throws {
    let model = try GPT2Model(config: Self.benchConfig)
    // ... init with random weights ...

    let generateCount = 128
    var tokenIDs = [0] // Start token

    // Warmup
    _ = try await model.forward(tokenIDs)

    let clock = ContinuousClock()
    let start = clock.now

    for _ in 0..<generateCount {
        let logits = try await model.forward(tokenIDs)
        // Greedy decode from last position
        let vocabSlice = Array(logits.suffix(Self.benchConfig.vocabSize))
        let nextToken = vocabSlice.enumerated().max(by: { $0.element < $1.element })!.offset
        tokenIDs.append(nextToken)
    }

    let elapsed = start.duration(to: clock.now)
    let seconds = /* extract */
    let tokensPerSec = Double(generateCount) / seconds

    print("BENCHMARK: decode_throughput \(String(format: "%.1f", tokensPerSec)) tokens/sec")
    print("BENCHMARK: decode_latency_per_token \(String(format: "%.1f", seconds/Double(generateCount)*1000)) ms/token")
}
```

### Category 6: Memory Efficiency

**File:** `Tests/EdgeRunnerTests/MemoryBenchmarks.swift`

| Benchmark | What It Measures | Metric |
|-----------|-----------------|--------|
| Buffer cache hit rate | Run 1000 ops, measure reuse | % hit rate |
| Buffer cache memory | Peak cache size during generation | MB |
| KV cache memory | Per-layer × precision × context length | MB |
| Quantization memory savings | fp32 vs fp16 vs q8_0 vs q4_0 estimate | compression ratio |

---

## Output Format: The Report

**File:** `Sources/EdgeRunner/Benchmarks/BenchmarkReportGenerator.swift` (utility, not in test target)

After running, the benchmark suite should print a parseable report. Create a simple
`BenchmarkReport` struct that collects results and renders Markdown:

```
# EdgeRunner Benchmark Report

**Device:** Apple M3 Max (40-core GPU, 48 GB unified memory)
**OS:** macOS 26.0
**Swift:** 6.2
**Date:** 2026-03-17

## Kernel Throughput

| Kernel | Size | Metric | Result |
|--------|------|--------|--------|
| GEMM fp32 | 4096×4096 | GFLOPS | 1,247.3 |
| GEMM fp16 | 4096×4096 | GFLOPS | 2,891.0 |
| GEMV fp32 | 4096×4096 | GB/s | 312.5 |
| FlashAttention | seq=2048 | ms | 4.2 |
| ... | ... | ... | ... |

## Generation Performance

| Metric | Value |
|--------|-------|
| Prefill (64 tokens) | 2,450 tokens/sec |
| Decode (autoregressive) | 187 tokens/sec |
| Time to First Token | 28.3 ms |
| Speculative Decoding Speedup | 2.1x |

## Memory Efficiency

| Metric | Value |
|--------|-------|
| Buffer Cache Hit Rate | 94.2% |
| KV Cache (32L, 2048 ctx, fp16) | 512 MB |
| Peak Working Set | 1.2 GB |

## System Info

| Property | Value |
|----------|-------|
| GPU Cores | 40 |
| Memory Bandwidth | 400 GB/s |
| Unified Memory | 48 GB |
```

---

## Implementation Order

1. **BenchmarkHelpers.swift** — Shared utilities: `BenchmarkResult`, timing helpers, device info, report formatter
2. **KernelBenchmarks.swift** — Metal kernel microbenchmarks (Category 1)
3. **KVCacheBenchmarks.swift** — KV cache throughput (Category 2)
4. **LoadingBenchmarks.swift** — Weight loading (Category 3)
5. **SamplingBenchmarks.swift** — Sampling pipeline (Category 4)
6. **GenerationBenchmarks.swift** — End-to-end generation (Category 5)
7. **MemoryBenchmarks.swift** — Memory efficiency (Category 6)
8. **BenchmarkReportGenerator.swift** — Report renderer

---

## Critical Rules

1. **All benchmarks use `import Testing`** (NOT XCTest)
2. **All timing uses `ContinuousClock`** (wall-clock, not affected by system sleep)
3. **All GPU benchmarks call `await MetalBackend.shared.synchronize()`** before timing
4. **All benchmarks warmup 3 iterations** before timing
5. **All benchmarks run 10+ timed iterations** and report average
6. **Print format:** `BENCHMARK: <snake_case_name> <value> <unit>` (one per line, parseable)
7. **Loose sanity expects** — don't hardcode perf thresholds that break on different hardware
8. **Include device info** in report (GPU core count, memory, chip family)
9. **No real model files required** — all benchmarks use synthetic weights or small configs
10. **Separate test file per category** — don't put 50 benchmarks in one file

## Accessing Metal Device Info

```swift
let device = MTLCreateSystemDefaultDevice()!
let name = device.name                           // "Apple M3 Max"
let unified = device.hasUnifiedMemory            // true
let maxMem = device.recommendedMaxWorkingSetSize // bytes
```

For GPU core count (not directly available via Metal API), use:
```swift
import IOKit
// Or simply report device.name which implies core count
```

## What Makes This Publishable

1. **Reproducible** — Anyone with Apple Silicon can run `swift test --filter Benchmark`
2. **Contextualized** — Results include device info so readers understand the hardware
3. **Granular** — Kernel-level numbers let people compare specific ops against llama.cpp/MLX
4. **Honest** — We benchmark what we have (GPT-2 architecture), not what we claim
5. **Automated** — Report generated programmatically, no manual number copying
