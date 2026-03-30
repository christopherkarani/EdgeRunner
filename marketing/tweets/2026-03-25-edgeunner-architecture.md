# Standalone Tweets — 2026-03-25

## Tweet 1 — Architecture showcase

We just fully documented EdgeRunner — a Metal LLM inference engine for Apple Silicon.

10 quantization formats. Mega-kernels that collapse 5 GPU dispatches into 1. Zero-allocation inference. Three auto-detected inference modes.

📎 Image: architecture diagram

🔗 https://github.com/YourHandle/EdgeRunner

---

## Tweet 2 — Metric / technical depth

160 tok/s decode throughput. 310MB memory footprint. Qwen3 0.6B Q8_0 on Apple Silicon.

That's with ~3ms TTFT and a KV cache that handles overflow via circular buffer rotation.

📎 Image: benchmark metrics

🔗 Repo in bio

---

## Tweet 3 — Code flex

Three lines to load a model and stream tokens:

```swift
let model = try await ModelLoader.load(from: modelURL)
let session = GenerationSession(model: model, maxTokens: 1024)
for try await text in session.stream(prompt: "Write a story") {
    print(text, terminator: "")
}
```

That's the entire public API. Everything else is implementation detail.

📎 Image: `../assets/code-images/2026-03-25-generation-api.png`

🔗 https://github.com/YourHandle/EdgeRunner

---

## Tweet 4 — Technical insight

The most interesting kernel in EdgeRunner: `fused_qk_norm_rope_gqa`

It computes Q/K RMSNorm + RoPE + GQA attention — across a 128-dim head — with ZERO threadgroup barriers.

Pure SIMD reductions. 32 threads. No synchronization overhead.

---

## Tweet 5 — Architecture insight

Most LLM runtimes have one decode path. EdgeRunner has three:

1. **Full Prefill** — reset KV cache, process entire sequence
2. **Decode** — single new token, reuse KV cache
3. **Prefix Reuse** — extend cached prefix, attend over full KV

Auto-detected per call. No configuration needed.

The model just gets faster on subsequent turns.

---

## Tweet 6 — TIL

TIL Metal's `powr(base, exp)` is faster than `pow(base, exp)` for computing RoPE frequencies.

`powr` uses hardware reciprocal units. `pow` goes through the generic path.

Swapped one for the other. Saw the gain in profiles.

---

## Tweet 7 — Hot take

Stop writing separate dequantization passes.

EdgeRunner's Q8_0 path passes raw quantized bytes directly to the GEMV kernel. The scale lives in a 2-byte header. The math happens inline.

~3.8× less bandwidth than dequantizing first.

Your quantized model doesn't need to be dequantized to run. It needs to be dequantized inside the matmul.

---

## Tweet 8 — Mega-kernel insight

The FFN mega-kernel (`dequant_q8_0_fused_ffn_block`) uses 1024 threads per threadgroup — 32 SIMD groups × 32 threads.

Phase 1: Wo GEMV + residual add (1024 threads)
Phase 2: Cooperative RMSNorm across 1024 elements (cross-SIMDgroup barrier)
Phase 3: Gate + Up + SwiGLU GEMVs
Phase 4: Down GEMV + residual add

One dispatch. Five operations.

---

## Tweet 9 — Framework breadth

EdgeRunner supports: Llama, Qwen2, Qwen3, Gemma, Gemma2, Gemma3, Mistral, Phi3, StarCoder, DeepSeek, InternLM, Falcon.

All loaded via GGUF with automatic architecture detection.

```swift
let model = try await ModelLoader.load(from: modelURL)
```

The framework handles the rest.

🔗 https://github.com/YourHandle/EdgeRunner
