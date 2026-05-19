# EdgeRunner Roadmap

> **⚠️ Status Update (May 2026):** This roadmap was written in March 2026. Since then, the project has advanced significantly faster than planned. Many items marked "TODO" below are actually **already shipped** (BPE tokenizer, chat templates, streaming, sampling, multi-model support, tool calling, Q4_K_M, memory mapping, and more). See the [README](../README.md) for the current ground truth. This document is kept for historical context and long-term vision.

---

## Where We Are

EdgeRunner is a from-scratch LLM inference engine in Swift/Metal for Apple Silicon. Built in ~13,000 lines of code across 26 optimization experiments, it produces correct, coherent text at competitive speed.

### Current Performance (May 2026)

```
Model: Qwen3-0.6B Q8_0 | Device: Apple M3 Max | 128-token decode

  MLX (Python):        277.8 tok/s    ████████████████████████████  target
  EdgeRunner (Swift):  234.8 tok/s    ███████████████████████▋      84.5%
  llama.cpp:           200.3 tok/s    ████████████████████▏         72.1%

  Time to First Token:  EdgeRunner 3.4ms  vs  MLX 77.2ms  (22x faster)
  Peak Memory:          EdgeRunner ~710MB vs MLX 638MB     (comparable)
```

### What Works Today ✅
- Full Llama/Qwen transformer inference on Metal GPU
- **BPE tokenizer** loaded from GGUF metadata (validated against HuggingFace)
- **Chat template engine** (Jinja2 subset) for multi-turn conversations
- **Streaming** via `AsyncThrowingStream<String, Error>`
- **Sampling** — temperature, top-p, top-k, repetition penalty, min-p
- **Tool calling** — `EdgeRunnerTool` protocol + JSON parser + executor
- **Multi-model support** — auto-detects Llama, Qwen, Mistral, Phi-3, Gemma, DeepSeek, Yi, and more
- **9 quantization types** — Q2_K through Q8_0, plus F16/F32
- **Memory-mapped loading** — instant startup
- **KV cache** with incremental decode and prefix reuse
- **Fused Metal kernels** — RMSNorm+GEMV, RoPE+GQA, SwiGLU, dequant+GEMV+residual
- **Single-simdgroup GQA mega-kernel** (zero threadgroup barriers)
- **Metal 4 argument table dispatch** (macOS 26+) for zero-dispatch-overhead decode
- Correct, coherent output: "The capital of France is Paris", "2 + 2 = 4"
- 22x faster time-to-first-token than MLX

### What We're Working On Next
- Lower OS requirements (macOS 15+ fallback path)
- Even faster decode via format-level optimizations
- Vision-language model support (Gemma 4 multimodal)
- ANE (Apple Neural Engine) offload for select layers
- Graceful memory-pressure degradation

---

## Phase 1: Beat MLX (2-3 weeks) — MOSTLY COMPLETE

**Goal: 280+ tok/s decode, <1GB memory**

### 1.1 Eliminate Float32 Weight Cache ✅
- **Status:** Done. Redundant float32 caches removed; decode path uses raw Q8_0 buffers.
- **Impact:** Peak memory dropped from ~4.9 GB to ~710 MB for Qwen3-0.6B.

### 1.2 Fused RMSNorm + LM Head GEMV ✅
- **Status:** Done. Final RMSNorm is fused into the LM head GEMV dispatch.

### 1.3 Reduce Q8_0 Bandwidth via Threadgroup Scale Caching 🔄
- **Status:** In progress. Experiments show ~5-10% GEMV improvement possible.
- **Risk:** Medium. Requires careful register pressure management.

### 1.4 Extended Decode Warmup ✅
- **Status:** Done. GPU pipeline warmed at diverse `kvSeqLen` values.

### Phase 1 Success Criteria
- [x] Peak memory < 2,500 MB (achieved: ~710 MB)
- [ ] 128-token decode throughput > 278 tok/s (still ~235 tok/s)
- [x] Correctness: greedy tokens unchanged
- [x] Coherence: "The capital of France is Paris" still works

---

## Phase 2: Real Tokenizer (1-2 weeks) — COMPLETE

**Goal: Accept text input, produce text output. No Python dependency.**

### 2.1 BPE Tokenizer from GGUF ✅
- **Status:** Shipped. Full byte-level BPE with merge priorities, special-token handling, and byte fallback.

### 2.2 Chat Template Engine ✅
- **Status:** Shipped. Jinja2 subset supporting `for`, `if/elif/else`, `set`, filters, `tojson`, `namespace()`, and more.

### 2.3 Sampling Pipeline ✅
- **Status:** Shipped. `SamplingConfiguration` wired into the decode path.

### Phase 2 Success Criteria
- [x] `runner.generate("What is the meaning of life?")` returns coherent text
- [x] Chat mode with proper turn-taking
- [x] No Python dependency for tokenization
- [x] Temperature and top-p sampling produce varied output

---

## Phase 3: Multi-Model Support (2-4 weeks) — COMPLETE

**Goal: Run Llama 3, Mistral, Phi-3, Gemma on the same engine**

### 3.1 Architecture Abstraction ✅
- **Status:** Shipped. `ModelLoader` auto-detects architecture from GGUF metadata.

### 3.2 Q4_K_M Quantization ✅
- **Status:** Shipped. Fused dequant+GEMV kernel for Q4_K_M added.

### 3.3 Llama 3 / Mistral / Phi-3 / Gemma Support ✅
- **Status:** Shipped and tested. Plus DeepSeek, Yi, InternLM2, StarCoder, Falcon, Command-R.

### Phase 3 Success Criteria
- [x] `ModelLoader.load(from: anyGGUF)` works for 10+ model families
- [x] Q4_K_M decode supported
- [x] All tested models produce coherent output

---

## Phase 4: Production API (2-3 weeks) — COMPLETE

**Goal: An API that app developers can ship**

### 4.1 Swift Package with Public API ✅
- **Status:** Shipped. `EdgeRunner` actor provides a clean one-liner API.

### 4.2 Streaming Token Generation ✅
- **Status:** Shipped. `AsyncThrowingStream` with backpressure and cancellation.

### 4.3 Memory-Mapped Model Loading ✅
- **Status:** Shipped. GGUF files are `mmap`'d; load time < 200 ms.

### 4.4 Graceful OOM Handling 🔄
- **Status:** Partial. `ModelConfiguration(contextWindowSize:)` lets users cap memory. Full pressure monitoring is planned.

### Phase 4 Success Criteria
- [x] Clean Swift Package Manager integration
- [x] Streaming output with < 5ms time-to-first-token
- [x] Model loading < 200ms
- [ ] No crashes under memory pressure (partial)

---

## Phase 5: Ship It (2-4 weeks)

**Goal: A macOS/iOS app or framework that regular people use**

### 5.1 macOS Menu Bar App
- **What:** Lightweight always-available LLM chat in the menu bar
- **Status:** Not started. Looking for contributors!

### 5.2 iOS On-Device Inference
- **What:** Same engine running on iPhone/iPad
- **Status:** Framework works on iOS 26+. Need a demo app and memory profiling.

### 5.3 Foundation Models Framework Integration
- **What:** Implement Apple's Foundation Models protocol for system-level integration
- **Status:** Stub exists at `FoundationModelsBackend.swift`. Needs fleshing out.

### Phase 5 Success Criteria
- [ ] App Store submission
- [ ] Real users running local LLMs on their devices
- [ ] The 10/10

---

## Performance Targets

| Metric | March 2026 | Today | Target |
|--------|-----------|-------|--------|
| Decode (0.6B, 128tok) | 235 tok/s | 235 tok/s | 280+ tok/s |
| Decode (7B, 128tok) | N/A | 30+ tok/s (Q4_K_M) | 40+ tok/s |
| Memory (0.6B) | 4,980 MB | ~710 MB | <700 MB |
| Memory (7B) | N/A | <6 GB (Q4_K_M) | <5 GB |
| TTFT | 3.4 ms | 3.4 ms | <3 ms |
| Model load | ~2s | <200ms | <100ms |
| Models supported | 1 | 10+ | 15+ |
| Quantization formats | Q8_0 | 9 types | 10+ types |

---

## Technical Debt & Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Prefill seqLen>1 uses separate (slower) code path | Medium | Metal 4 prefill path planned |
| LlamaLanguageModel.swift is ~6,000 lines | Medium | Refactoring into smaller types planned |
| No error recovery on GPU OOM | Medium | Graceful degradation in progress |
| Decode warmup adds ~18ms to first generation | Low | Acceptable trade-off |
| Non-deterministic output across process restarts | Low | Driver JIT variance; expected |
| GPT-2 module (Module/, Transformer/) is unused dead code | Low | Cleanup planned |
| macOS 26 requirement limits adoption | High | macOS 15 fallback path planned |

---

## What We Learned (26 Experiments)

### What Worked
1. **Fused kernels** — merging RMSNorm into GEMV, RoPE into GQA: +67% (Exp 10)
2. **Single command buffer** — 1 encoder for entire forward pass: +191% (Exp 4)
3. **Q8_0 fused dequant+GEMV** — read quantized weights directly: +41% (Exp 6)
4. **Single-simdgroup GQA** — zero barriers, 4 dims/thread: +13% (Exp 21)
5. **Decode warmup** — pre-warm GPU pipeline states: +16% (Exp 16)

### What Didn't Work
1. **f16 accumulation** — NaN from precision loss (Exp 22)
2. **NR=4 GEMV** — register pressure kills occupancy (Exp 15a)
3. **2-simdgroup GEMV** — occupancy loss exceeds dispatch savings (Exp 19)
4. **Flash-decode (extra dispatches)** — dispatch overhead > parallelism gain at 128 tokens (Exp 26)
5. **Swift COW reusable arrays** — COW semantics fight in-place mutation (Exp 25)

### Key Insight
> Apple Silicon GEMV is **occupancy-limited**: more small threadgroups with 1 simdgroup (32 threads) each outperform fewer large threadgroups. Bandwidth utilization = f(outstanding memory requests) = f(active threadgroups). Any optimization that reduces threadgroup count—even if it reduces total work—hurts effective bandwidth.

> The remaining gap to MLX is **structural**, not algorithmic: Q8_0 reads 20% more data per weight than MLX's 8-bit format, and 142 dispatches create 0.31ms of GPU scheduling overhead that MLX's graph-level lazy evaluation avoids. Closing this gap requires format-level and dispatch-level changes, not kernel-level tuning.

---

## The Journey So Far

```
May 2026

  0.058 tok/s  ▏                                          Experiment 0: baseline (broken)
  3.57  tok/s  █                                          Exp 1: GPU LM head + buffer cache
  16.1  tok/s  ████                                       Exp 4: single command buffer
  24.0  tok/s  ██████                                     Exp 6: fused Q8_0 dequant+GEMV
  64.0  tok/s  ████████████████                            Exp 9: KV cache decode
  120   tok/s  ██████████████████████████████               Exp 10: kernel fusion
  150   tok/s  █████████████████████████████████████        Exp 12: fused prefill
  210   tok/s  ████████████████████████████████████████████████████  Exp 13: mega-kernel
  235   tok/s  ███████████████████████████████████████████████████████████  Exp 21: 1-SG GQA

  Total improvement: 4,052x from baseline
  vs llama.cpp: +17%
  vs MLX: -15% (closing)
```
