# We fully documented EdgeRunner — a Metal LLM inference engine for Apple Silicon. Here's what we found that was actually interesting.

## TL;DR

~160 tok/s on Qwen3 0.6B Q8_0. 310MB memory. Three auto-detected inference modes. Mega-kernels that collapse 5 GPU dispatches into 1. Full architecture doc: `docs/arch/EdgeRunner_Complete_Reference.md`

## What this is

EdgeRunner is an open-source Metal-accelerated inference engine for GGUF models (Llama, Qwen, Mistral, Gemma, Phi3, DeepSeek, etc.). I spent a few hours reading the actual source code — all of it — and documenting what makes it interesting architecturally.

## The quantization strategy isn't the feature list — it's the tiering

Everyone lists supported formats. What's more interesting is how EdgeRunner handles dequantization:

- Raw Q8_0: The quantized buffer goes straight to the GEMV kernel, no float32 materialization. The scale lives in a 2-byte header. The math happens inline, ~3.8× less memory bandwidth than dequantizing first.

- Other quants (Q4_0, Q5_0, K-quants): Dedicated Metal kernels dequantize on-GPU during matmul. Not a separate pass.

- Embeddings: CPU SIMD fallback.

Most runtimes dequantize everything upfront. EdgeRunner dequantizes only what's needed, where it's needed.

## The mega-kernels are the real optimization

`fused_qk_norm_rope_gqa` — This one kernel does per-head Q/K RMSNorm + RoPE + GQA attention in one dispatch. Zero threadgroup barriers.

How it works: 32 threads per head (one full SIMDgroup). Each thread handles 4 elements of a 128-dim vector: `[i, i+32, i+64, i+96]`. SIMD sum across 32 threads covers the full 128-dim head. No barrier between RMSNorm+RoPE and GQA — the KV threads just `return` when done writing to the KV cache.

The FFN mega-kernel (`dequant_q8_0_fused_ffn_block`) uses 1024 threads per threadgroup and does Wo GEMV + RMSNorm + Gate/Up/SwiGLU + Down in one dispatch.

## The KV cache is a circular buffer with overflow handling

When `totalWritten > maxSeqLen`, the cache "rotates" — retrieval reads in two chunks: `[writePos...maxSeqLen]` then `[0...writePos]`. This correctly reconstructs the causal view for attention. No recomputation.

## Three inference modes, auto-detected

- Decode: single new token, full KV cache. The common case.
- Prefix Reuse: sequence extends cached prefix. Only suffix tokens embedded. Full KV cache used. This is what makes multi-turn fast.
- Full Prefill: no useful prefix. Entire sequence processed.

The mode is detected by comparing token sequences on every call, no configuration needed.

## What didn't surprise me

The memory management is solid but conventional. LRU BufferCache, BarrierTracker for RAW hazards, CommandBatcher with per-chip thresholds. All necessary, none novel.

## What did surprise me

The scratch buffer pre-allocation. 19 persistent buffers covering every intermediate tensor. Forward passes never allocate GPU memory, ~17 MTLBuffer allocations eliminated per inference step.

Also: `powr` vs `pow` in the RoPE kernel. Free performance gain by using hardware reciprocal units instead of generic power.

## Benchmarks

Qwen3 0.6B Q8_0 on Apple Silicon:
- Decode throughput: ~160 tok/s
- TTFT: ~3ms
- Peak RSS: 310MB
- Model file: 804MB (Q8_0)

## Limitations

- Metal 4 only (macOS 26+/iOS 26+). Falls back to Metal 3 but optimized path requires newer OS.
- GGUF only. No Safetensors, no HuggingFace format.
- No speculative decoding yet.
- Streaming is async-only (AsyncThrowingStream), no sync callback option.

## Full doc

2,500 lines covering every quantization format, every kernel, every inference mode:

`docs/arch/EdgeRunner_Complete_Reference.md`

## Try it

```swift
import EdgeRunner

let model = try await ModelLoader.load(from: modelURL)
let session = GenerationSession(model: model, maxTokens: 1024)

for try await text in session.stream(prompt: "Write a story") {
    print(text, terminator: "")
}
```

Repo link: https://github.com/YourHandle/EdgeRunner

AMA about the architecture.
