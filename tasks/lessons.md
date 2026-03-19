# Lessons

## 2026-03-16
- Verify the actual milestone baseline from the repo before starting execution. This workspace is mid-M2, so M3/M4 implementation prompts must be deferred until M2 is complete and verified locally.
- Validate buffer shapes and requested element types at public API boundaries before encoding Metal work. GPU kernels should never be the first place mismatched dimensions or precisions are discovered.
- If a model-layer API carries positional metadata like `startPos` or configuration like `ropeTheta`, wire it through the reference implementation immediately or keep it out of the public surface until supported.
- For autoregressive models, prefer prefix-equivalence tests over naive “offset output must differ” assertions. Causal correctness is best proven by comparing full-sequence logits with growing-prefix logits, not by assuming positional metadata always changes local outputs.

## 2026-03-19
- Verify the benchmark target model from the harness before downloading assets or quoting results. For EdgeRunner, the pinned benchmark target is `Qwen3-0.6B-Q8_0`, not a presumed 1B variant.
- Any decode path that writes to `hazardTrackingModeUntracked` KV buffers and then reads them in a later dispatch needs an explicit buffer barrier. If the Metal 4 path has barriers and the Metal 3 path does not, benchmark instability is a correctness warning, not just noise.
- When a model uses tied embeddings, every fallback path must read from the same tied weight tensor as the fast path. Swapping `lmHead.weight` for `embedding.weight` in a low-memory fallback can keep a short prefix intact while silently breaking long-run determinism.
- Treat single publishable wins as low-quality evidence. Keep a performance change only after at least two repeated 128-token publishable runs stay deterministic and remain in the same throughput band.
- When the user explicitly asks for subagents on the EdgeRunner optimization workflow, use `gpt-5.4-mini` with `xhigh` reasoning unless they say otherwise.
- Tiny per-dispatch kernel parameter blocks are not automatically better in a shared params buffer. If a Metal kernel consumes a small constant struct, `setBytes` can beat buffer indirection even in an otherwise params-buffer-heavy decode path.
- Generation-side optimizations must be benchmarked through `nextToken` or streaming paths, not `logits + argmax` in the benchmark harness, or the measurement will hide the real win.
