# EdgeRunner Roadmap

## Where We Are

EdgeRunner is a from-scratch LLM inference engine in Swift/Metal for Apple Silicon. Built in ~13,000 lines of code across 26 optimization experiments, it produces correct, coherent text at competitive speed.

### Current Performance (March 2026)

```
Model: Qwen3-0.6B Q8_0 | Device: Apple M3 Max | 128-token decode

  MLX (Python):        277.8 tok/s    ████████████████████████████  target
  EdgeRunner (Swift):  234.8 tok/s    ███████████████████████▋      84.5%
  llama.cpp:           200.3 tok/s    ████████████████████▏         72.1%

  Time to First Token:  EdgeRunner 3.4ms  vs  MLX 77.2ms  (22x faster)
  Peak Memory:          EdgeRunner 4,980MB vs MLX 638MB    (7.8x worse)
```

### What Works Today
- Full Llama/Qwen transformer inference on Metal GPU
- Q8_0 quantized weight support (GGUF format)
- KV cache with incremental decode (only processes new token)
- Fused Metal kernels: RMSNorm+GEMV, RoPE+GQA, SwiGLU, dequant+GEMV+residual
- Single-simdgroup GQA mega-kernel (zero threadgroup barriers)
- Correct, coherent output: "The capital of France is Paris", "2 + 2 = 4"
- 22x faster time-to-first-token than MLX

### What Doesn't Work
- No BPE tokenizer (can't accept text input, only pre-tokenized IDs)
- Only 1 model supported (Qwen3-0.6B)
- Only Q8_0 quantization
- 8x memory bloat (stores redundant float32 weight copies)
- No streaming output, no sampling (temperature/top-p), no chat interface

---

## Phase 1: Beat MLX (2-3 weeks)

**Goal: 280+ tok/s decode, <1GB memory**

Three kernel-level changes close the gap. No new features, no architecture redesign. Pure autoresearch: modify, benchmark, keep or rollback.

### 1.1 Eliminate Float32 Weight Cache
- **What:** Remove redundant dequantized float32 weight buffers (2.4GB wasted)
- **Why:** Decode path uses only raw Q8_0 buffers. Float32 copies exist for a prefill fallback that can use Q8_0 fused GEMV instead
- **Impact:** 4,980MB -> ~2,500MB memory. Better DRAM bandwidth (less page table pressure)
- **Risk:** Low. Dummy buffer approach avoids changing any call sites
- **Effort:** 1 day

### 1.2 Fused RMSNorm + LM Head GEMV
- **What:** Merge the final RMSNorm dispatch into the LM Head GEMV kernel
- **Why:** Eliminates 1 GPU dispatch per decode (saves ~3us scheduling gap x 28 layers equivalent). Each GEMV threadgroup computes norm cooperatively, same pattern already used in fused QKV and Gate+Up+SiLU kernels
- **Impact:** 141 -> 140 dispatches. ~3us saved. Marginal but architecturally clean
- **Risk:** Low. Same cooperative RMSNorm pattern proven in Experiments 10-12
- **Effort:** 1 day

### 1.3 Reduce Q8_0 Bandwidth via Threadgroup Scale Caching
- **What:** Cache per-block scale factors in registers instead of re-reading from DRAM
- **Why:** Q8_0 reads 2-byte scale per 32-weight block. Scale accounts for 6% of weight bandwidth. Caching scale across the simdgroup reduces DRAM reads
- **Impact:** ~5-10% GEMV kernel improvement. 234.8 -> ~250 tok/s
- **Risk:** Medium. Requires careful register pressure management
- **Effort:** 2-3 days

### 1.4 Extended Decode Warmup
- **What:** Warm GPU pipeline at diverse kvSeqLen values (1, 4, 16, 32, 64, 128) instead of just 5 consecutive positions
- **Why:** GPU profiling showed first-call JIT penalty at new kvSeqLen values
- **Impact:** Eliminates residual cold-start penalties across the 128-token benchmark
- **Risk:** Very low. One-time cost at first prefill
- **Effort:** 0.5 day

### Phase 1 Success Criteria
- [ ] 128-token decode throughput > 278 tok/s (beat MLX)
- [ ] Peak memory < 2,500 MB
- [ ] Correctness: greedy tokens [1, 1479, 35, 5371, 1] unchanged
- [ ] Coherence: "The capital of France is Paris" still works

---

## Phase 2: Real Tokenizer (1-2 weeks)

**Goal: Accept text input, produce text output. No Python dependency.**

### 2.1 BPE Tokenizer from GGUF
- **What:** Load vocabulary + merge rules from GGUF metadata, implement BPE encode/decode in Swift
- **Why:** Currently uses raw UTF-8 bytes as "tokens" which is wrong. GGUF already contains the full Qwen3 BPE vocabulary (151,936 tokens) and merge table (151,387 rules)
- **Implementation:** Port the GPT-2 BPE algorithm (byte-level BPE with merge priorities)
- **Effort:** 3-5 days

### 2.2 Chat Template Engine
- **What:** Parse and apply Jinja2-style chat templates from GGUF metadata
- **Why:** Qwen3 requires `<|im_start|>user\n...<|im_end|>` formatting. The template is already in the GGUF file
- **Implementation:** Minimal Jinja2 subset (variable substitution, for loops, conditionals). No full Jinja2 needed
- **Effort:** 2-3 days

### 2.3 Sampling Pipeline
- **What:** Temperature, top-p, top-k, repetition penalty, min-p
- **Why:** Greedy decode produces repetitive output. Real generation needs sampling
- **Implementation:** `SamplingConfiguration` struct already exists. `GreedySampler` and `MinPSampler` are in EdgeRunnerCore. Wire them into the Llama decode path
- **Effort:** 1-2 days

### Phase 2 Success Criteria
- [ ] `model.generate(prompt: "What is the meaning of life?")` returns coherent multi-sentence response
- [ ] Chat mode with proper turn-taking
- [ ] No Python dependency for tokenization
- [ ] Temperature and top-p sampling produce varied, high-quality output

---

## Phase 3: Multi-Model Support (2-4 weeks)

**Goal: Run Llama 3, Mistral, Phi-3, Gemma on the same engine**

### 3.1 Architecture Abstraction
- **What:** Factor model-specific config (RoPE type, norm type, activation, head arrangement) out of LlamaLanguageModel into a config-driven architecture
- **Why:** Currently hardcoded for Qwen3's NeoX RoPE + per-head Q/K norm. Llama 3 uses standard RoPE without Q/K norm. Mistral uses sliding window attention
- **Effort:** 1 week

### 3.2 Q4_K_M Quantization
- **What:** Add fused dequant+GEMV kernel for Q4_K_M (the most popular GGUF quantization)
- **Why:** Q4_K_M uses ~4.5 bits/weight vs Q8_0's 8.5. Halves memory and bandwidth. Most models are distributed in Q4_K_M
- **Impact:** ~2x faster decode (half the weight data). 7B models fit in 4GB RAM
- **Effort:** 1 week (kernel + Swift integration)

### 3.3 Llama 3 / Mistral / Phi-3 / Gemma Support
- **What:** Test and validate each architecture variant
- **Why:** Coverage. These are the most widely used open models
- **Effort:** 2-3 days per model family (config + integration tests)

### Phase 3 Success Criteria
- [ ] `LlamaLanguageModel.load(from: anyGGUF)` works for 5+ model families
- [ ] Q4_K_M decode throughput > 400 tok/s for 0.6B models
- [ ] Llama-3.1-8B-Q4_K_M runs at > 30 tok/s in < 6GB memory
- [ ] All models produce coherent output verified by coherence tests

---

## Phase 4: Production API (2-3 weeks)

**Goal: An API that app developers can ship**

### 4.1 Swift Package with Public API
```swift
let model = try await EdgeRunner.load("Qwen3-0.6B-Q8_0.gguf")
for try await token in model.stream("Tell me a story") {
    print(token, terminator: "")
}
```

### 4.2 Streaming Token Generation
- **What:** `AsyncSequence`-based token streaming with backpressure
- **Why:** Users expect to see tokens appear one at a time
- **Implementation:** `GenerationSession` and `TokenStream` already exist in the codebase. Wire them to LlamaLanguageModel

### 4.3 Memory-Mapped Model Loading
- **What:** mmap the GGUF file instead of reading into memory
- **Why:** Load time drops from ~2 seconds to < 100ms. OS manages page faults
- **Impact:** Instant model loading, shared memory across processes

### 4.4 Graceful OOM Handling
- **What:** Memory pressure monitoring, context length reduction, quantization fallback
- **Why:** Currently crashes on OOM. Must degrade gracefully for production use

### Phase 4 Success Criteria
- [ ] Clean Swift Package Manager integration
- [ ] Streaming output with < 5ms time-to-first-token
- [ ] Model loading < 200ms
- [ ] No crashes under memory pressure

---

## Phase 5: Ship It (2-4 weeks)

**Goal: A macOS/iOS app or framework that regular people use**

### 5.1 macOS Menu Bar App
- **What:** Lightweight always-available LLM chat in the menu bar
- **Why:** Prove the technology works in a real product
- **Implementation:** SwiftUI, local model management, conversation persistence

### 5.2 iOS On-Device Inference
- **What:** Same engine running on iPhone/iPad
- **Why:** Apple Silicon is the same architecture. Metal kernels work unchanged
- **Challenges:** Memory constraints (iPhone has 6-8GB shared). Need Q4_K_M for 7B models

### 5.3 Foundation Models Framework Integration
- **What:** Implement Apple's Foundation Models protocol for system-level integration
- **Why:** Enables EdgeRunner models to be used by any app via the system ML framework
- **Implementation:** `FoundationModelsBackend.swift` already exists as a stub

### Phase 5 Success Criteria
- [ ] App Store submission
- [ ] Real users running local LLMs on their devices
- [ ] The 10/10

---

## Performance Targets by Phase

| Metric | Today | Phase 1 | Phase 3 | Phase 5 |
|--------|-------|---------|---------|---------|
| Decode (0.6B, 128tok) | 235 tok/s | 280+ tok/s | 400+ tok/s (Q4) | 400+ tok/s |
| Decode (7B, 128tok) | N/A | N/A | 30+ tok/s | 40+ tok/s |
| Memory (0.6B) | 4,980 MB | <2,500 MB | <800 MB | <700 MB |
| Memory (7B) | N/A | N/A | <6 GB | <5 GB |
| TTFT | 3.4 ms | 3.4 ms | <10 ms | <5 ms |
| Model load | ~2s | ~2s | <200ms | <100ms |
| Models supported | 1 | 1 | 5+ | 10+ |
| Quantization formats | Q8_0 | Q8_0 | Q4_K_M, Q8_0 | Q4_0-Q8_0, F16 |

---

## Technical Debt & Known Issues

| Issue | Severity | Phase to Fix |
|-------|----------|-------------|
| 4.9GB peak memory (redundant float32 caches) | High | Phase 1 |
| No BPE tokenizer (UTF-8 byte hack) | High | Phase 2 |
| Only Qwen3 architecture supported | High | Phase 3 |
| Prefill seqLen>1 uses separate (slower) code path | Medium | Phase 1 |
| Metal 4 path not tested on real macOS 26 hardware | Medium | Phase 3 |
| LlamaLanguageModel.swift is 2,337 lines | Medium | Phase 3 |
| No error recovery on GPU OOM | Medium | Phase 4 |
| Decode warmup adds ~18ms to first generation | Low | Acceptable |
| Non-deterministic output across benchmark runs | Low | Phase 3 |
| GPT-2 module (Module/, Transformer/) is unused dead code | Low | Phase 3 |

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
March 2026

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
