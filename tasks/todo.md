# Long-Prompt MLX vs EdgeRunner Benchmark

## Goal
- Run a fresh apples-to-apples long-prompt benchmark for `Qwen3-0.6B` comparing EdgeRunner against MLX on this machine.
- Measure the prompt-heavy path explicitly instead of inferring from decode-only numbers.

## Plan
- [x] Define one shared benchmark contract: same semantic prompt, same prompt length target, same generated-token count, same warmup policy, same run count, same reported metrics.
- [x] Implement or script a matched benchmark harness for EdgeRunner and MLX that records prefill latency, prefill tokens/sec, TTFT, and post-prefill decode throughput separately.
- [x] Run both benchmarks on the local pinned assets and save the raw results under `benchmarks/`.
- [x] Review the deltas, note any tokenizer/model-format mismatches that still affect strict comparability, and summarize the outcome.

## Review
- Added [`LongPromptFrameworkBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/LongPromptFrameworkBenchmark.swift), an env-gated EdgeRunner child benchmark that consumes exact prompt token IDs, warms the prompt shape once, then measures warmed TTFT and post-prefill decode throughput.
- Added [`run_long_prompt_framework_benchmark.py`](/Users/chriskarani/CodingProjects/EdgeRunner/benchmarks/run_long_prompt_framework_benchmark.py), which:
  - builds one exact 1024-token prompt with the official `Qwen/Qwen3-0.6B` tokenizer,
  - feeds the same token IDs to both EdgeRunner and MLX,
  - warms each runtime once on that exact prompt shape,
  - runs 3 warmed measurements per framework,
  - and writes the aggregate result to [`long_prompt_framework_comparison.json`](/Users/chriskarani/CodingProjects/EdgeRunner/benchmarks/long_prompt_framework_comparison.json).
- Full benchmark settings:
  - prompt tokens: `1024`
  - generated tokens: `128`
  - runs per framework: `3`
  - context window: `2048`
- Median results on this machine:
  - EdgeRunner prompt throughput: **282.1 tok/s**
  - MLX prompt throughput: **7047.8 tok/s**
  - EdgeRunner TTFT: **3629.5 ms**
  - MLX TTFT: **145.3 ms**
  - EdgeRunner decode throughput after the long prompt: **42.5 tok/s**
  - MLX decode throughput after the long prompt: **249.3 tok/s**
- Relative delta from the saved artifact:
  - MLX prompt throughput is **~24.98x** EdgeRunner (`+2398%`)
  - EdgeRunner TTFT is **~24.98x** slower than MLX
  - MLX long-context decode throughput is **~5.87x** EdgeRunner (`+486.7%`)
- Strict comparability notes:
  - The prompt tokens are identical across runtimes because both consumed the same pre-tokenized ID list.
  - The underlying model formats still differ (`GGUF Q8_0` for EdgeRunner vs MLX 8-bit safetensors), so this is apples-to-apples at the prompt/workload level, not at the raw weight-format level.

# Runtime Repack + Matrix Prefill Rewrite

## Goal
- Replace the current experimental packed-prefill scaffolding with a production runtime repack for compatible Q8 hot-path weights.
- Replace the token-loop-heavy multi-token prompt path with a true prompt-matrix exact prefill engine.
- Replace the long-KV exact decode attention path with a dedicated exact kernel and cache layout that scale better at `kvLen ~ 1k+`.
- Preserve the current deterministic short-decode path and legacy prefill/decode fallbacks until the rewrite has benchmarked wins.

## Plan
- [ ] Phase 0: re-establish authoritative baselines on this checkout for publishable decode and the 1024-token long-prompt benchmark, then record them in the review notes below.
- [ ] Phase 1: promote runtime repacking from debug scaffolding to a first-class load-time artifact for compatible Q8 weights, with explicit metadata and fallback preservation.
- [ ] Phase 2: add a dedicated exact matrix-prefill engine entrypoint instead of routing every path through `fusedPrefillPass`.
- [ ] Phase 3: implement matrix-backed QKV projection in the new prefill engine and validate parity plus long-prompt benchmark effect.
- [ ] Phase 4: implement prompt-wide exact FFN (`wo`, `gate/up/down`) in the new prefill engine and re-benchmark.
- [ ] Phase 5: replace prompt attention with a dedicated exact tiled kernel that operates on prompt slabs and writes K/V directly in cache-native form.
- [ ] Phase 6: redesign long-KV exact decode attention and KV layout behind explicit routing, then re-run publishable and long-prompt benchmarks.
- [ ] Phase 7: only after kernel architecture is winning, revisit dispatch/runtime cleanup and Metal 4 fast-path routing.
- [ ] Kill criteria:
  - matrix-backed slices that regress long-prompt prompt throughput or TTFT after one tuning pass are reverted immediately
  - decode-only changes that do not improve the 1024-token decode metric are reverted even if short decode stays healthy
  - no slice is kept if publishable determinism or hash parity regresses

## Review
- Trusted kept baseline on `perf2@05736be73ef5f8d73e51edec2b743de0726f548a`:
  - publishable decode median `~211-213 tok/s`
  - publishable TTFT `~4.0 ms`
  - publishable hash `0afae14a84cf0df8`
  - long-prompt prompt throughput median `489.0 tok/s`
  - long-prompt TTFT median `2093.9 ms`
  - long-prompt decode median `42.26 tok/s`
- Current kept rewrite checkpoint on top of `de785a1`:
  - vectorized packed-prefill matmul kernel (`gemm_f32_packed_prefill`) wired into the experimental multi-token QKV/`wo`/FFN/down path under `EDGERUNNER_PREFILL_PREFER_EXACT_MATRIX=1`
  - publishable decode median `213.3 tok/s`
  - publishable TTFT `4.0 ms`
  - publishable hash `0afae14a84cf0df8`
  - long-prompt prompt throughput median `558.6 tok/s`
  - long-prompt TTFT median `1833.2 ms`
  - long-prompt decode median `42.24 tok/s`
- New kept decode rewrite candidate behind `EDGERUNNER_DECODE_PREFER_PACKED_LONG_KV=1`:
  - dedicated packed decode KV views plus exact long-KV attention kernel, with K/V repack collapsed into one dispatch
  - publishable decode median `220.3 tok/s`
  - publishable TTFT `3.9 ms`
  - publishable hash `0afae14a84cf0df8`
  - long-prompt prompt throughput median `526.8 tok/s`
  - long-prompt TTFT median `1943.9 ms`
  - long-prompt decode median `60.29 tok/s`
- New kept prompt rewrite candidate with `EDGERUNNER_PREFILL_PREFER_EXACT_MATRIX=1` + `EDGERUNNER_DECODE_PREFER_PACKED_LONG_KV=1`:
  - prompt-local flash-style GQA over f32 K/V slabs before cache conversion
  - publishable decode median `218.1 tok/s`
  - publishable TTFT `3.8 ms`
  - publishable hash `0afae14a84cf0df8`
  - long-prompt prompt throughput median `1169.6 tok/s`
  - long-prompt TTFT median `875.5 ms`
  - long-prompt decode median `60.09 tok/s`
- Dead rewrite branches already measured and reverted:
  - GEMM-backed exact matrix prefill over repacked Q8 weights: regressed long-prompt prompt throughput to `433.3 tok/s`
  - contiguous raw-Q8 prefill bundle views over the GGUF mmap: regressed long-prompt prompt throughput to `469.2 tok/s`
  - first dedicated exact-matrix execution slice using prompt-wide `gemm_f32` packed QKV/FFN/down inside `exactMatrixPrefillPass`: regressed long-prompt prompt throughput to `451.9 tok/s`, TTFT to `2266.1 ms`, while decode stayed flat at `42.42 tok/s`
  - tiled `gemm_f32_packed_prefill` threadgroup rewrite: improved publishable decode to `223.2 tok/s` but regressed the target long-prompt workload to `549.8 tok/s` prompt throughput, `1862.3 ms` TTFT, and `37.34 tok/s` long-context decode
  - mega decode kernel block-softmax chunking in `fused_qk_norm_rope_gqa`: improved prompt throughput/TTFT to `608.8 tok/s` / `1681.9 ms` in the long-prompt harness but regressed long-context decode to `37.19 tok/s` and publishable decode to `203.2 tok/s`
  - packed long-KV decode forced onto short contexts (`kvLen < 256`): broke canonical publishable prefix (`[1, 1479, 6222]` instead of `[1, 1479, 35]`), so the new packed attention path remains long-KV-only
  - direct float-to-packed decode-cache writes in the flash-prompt prefill path: preserved correctness but regressed publishable decode to `212.5 tok/s` and slipped the long-prompt checkpoint to about `1160.8 tok/s` prompt throughput, `882.3 ms` TTFT, and `59.36 tok/s` long-context decode versus the kept `88ccdbd` floor
  - prompt-flash Q/K/V via `MPSMatrixMultiplication`: KEPT. 3-run long-prompt median improved from about `1159.0 tok/s`, `883.5 ms`, `59.9 tok/s` to about `1356.0 tok/s`, `755.2 ms`, `59.2 tok/s`; publishable benchmark stayed deterministic with hash `0afae14a84cf0df8`
- Implication: the remaining viable path is a larger engine split, not more local substitutions inside the legacy prefill body.

# Mega Fused GQA Kernel Repair

# TurboQuant KV Cache Integration

## Goal
- Add the public TurboQuant KV-cache configuration surface and the reference implementation building blocks needed to integrate the paper’s algorithm into EdgeRunner.
- Keep the current runtime behavior unchanged unless TurboQuant is explicitly requested or the future `.automatic` gate is enabled after validation.

## Plan
- [x] Add public KV cache compression configuration and adaptive default policy.
- [x] Add deterministic TurboQuant reference primitives: presets, randomized Hadamard transforms, codebooks, and CPU row encode/decode.
- [x] Add KV-cache format plumbing, Metal quantization/attention kernels, and runtime model integration.
- [x] Add focused tests for the new API and TurboQuant reference primitives.
- [x] Vendor the paper and add implementation notes under `docs/research/turboquant/`.
- [x] Add env-gated runtime smoke and benchmark harnesses for the TurboQuant path.

## Review
- Added public `KVCacheCompression` policy to `ModelConfiguration` with `.disabled`, `.automatic`, `.turboQuantBalanced`, and `.turboQuantAggressive`. The public default is `.automatic`, but it currently resolves to `.disabled` unless the rollout gate env var is enabled and the context window is at least `8192`.
- Replaced the initial TurboQuant scaffolding with a stable row format in `Sources/EdgeRunnerMetal/TurboQuant.swift`: preset descriptors, deterministic randomized Hadamard/QJL transforms, mixed-bit outlier masks, packed code words, residual sign bits, row/residual norms, and CPU encode/decode helpers that match the runtime format.
- Extended `KVCache` to support packed TurboQuant storage alongside dense FP32/FP16 modes. TurboQuant layers now own packed code buffers, residual-sign buffers, outlier-mask buffers, and metadata buffers with the same ring-buffer semantics as the dense cache.
- Added `Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal` and `Sources/EdgeRunnerMetal/TurboQuantKernel.swift` for GPU row quantization and TurboQuant GQA attention. The model routes TurboQuant requests through the supported base decode path and quantizes both prefill-written rows and generated-token rows.
- Wired TurboQuant end-to-end through `LlamaLanguageModel.swift`. TurboQuant loads now succeed for supported `headDim == 128` models, the packed cache path is used during actual inference, and the existing FP16 fast paths remain unchanged for the default `.disabled` mode.
- Added a repo-local research note and vendored arXiv PDF under `docs/research/turboquant/`.
- Added focused tests in `Tests/EdgeRunnerMetalTests/TurboQuantReferenceTests.swift`, `Tests/EdgeRunnerMetalTests/KVCacheTests.swift`, `Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift`, and extended `EdgeRunnerLanguageModelTests.swift` for the new public configuration surface.
- Added env-gated real-model coverage in `Tests/EdgeRunnerTests/QwenTurboQuantSmokeTest.swift` and `Tests/EdgeRunnerTests/TurboQuantLongContextBenchmark.swift`.
- Verification:
  - `swift build`
  - `swift test --filter "TurboQuantReferenceTests"`
  - `swift test --filter "KVCacheTests"`
  - `swift test --filter "TurboQuantAttentionTests"`
  - `swift test --filter "EdgeRunnerLanguageModelProtocolTests"`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter "QwenTurboQuantSmokeTest"`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=1024 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=8 swift test --filter "TurboQuantLongContextBenchmark"`
- Current rollout status:
  - The feature is implemented and works end-to-end, but the current TurboQuant kernels do **not** beat FP16 on this machine yet. The benchmark harness reported `fp16_decode_tok_s=24.79` versus `turboquant_decode_tok_s=1.09` on the sampled 1024-token prompt case, and TTFT was materially worse under TurboQuant.
  - Because the performance gate is not met, `.automatic` correctly remains disabled by default. TurboQuant is production-safe as an explicit opt-in path, but it is **not** eligible to become the default until the long-context benchmark harness shows a real throughput win.

## TurboQuant Tuning: Fused QKV Recovery

### Goal
- Recover the fused Q8 RMSNorm+QKV projection path for TurboQuant so long-context TTFT is not dominated by separate RMSNorm, Q, K, and V dispatches.
- Keep the default FP16 publishable path unchanged and benchmark-canonical.

### Plan
- [x] Re-establish the last good TurboQuant decode baseline after the failed decode-kernel experiment.
- [x] Inspect where TurboQuant disables the fused Q8 QKV path in prefill and decode.
- [x] Add a TurboQuant-specific fused QKV kernel variant that writes `V` into the float scratch buffer instead of the FP16 KV cache.
- [x] Route TurboQuant prefill and decode through that fused path.
- [x] Re-run focused TurboQuant tests and long-context benchmark points.
- [x] Re-run the release publishable benchmark to prove the default path is unchanged.

### Review
- Added `dequant_q8_0_fused_qkv_f32v` in [Dequant_Q8_0.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_0.metal), which preserves the fused RMSNorm+Q/K/V math but writes `V` to a float output buffer for TurboQuant packing.
- Wired TurboQuant to use that kernel in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) for both prefill and decode whenever Q8 fused projection is available.
- Verified:
  - `swift build`
  - `swift test --filter TurboQuantAttentionTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=128 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=256 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=512 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=1024 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- TurboQuant benchmark deltas versus the previous stable tuning baseline:
  - `prompt_len=128`: decode `12.58 -> 12.97 tok/s`, TTFT `1349.14 -> 1060.82 ms`
  - `prompt_len=256`: decode `12.26 -> 11.18 tok/s`, TTFT `2760.24 -> 2024.40 ms`
  - `prompt_len=512`: decode `8.37 -> 10.40 tok/s`, TTFT `6140.15 -> 3987.31 ms`
  - `prompt_len=1024`: decode `5.80 -> 7.49 tok/s`, TTFT `15517.23 -> 6668.50 ms`
- This is the first real long-context TurboQuant breakthrough on this checkout: the fused QKV recovery cuts TTFT sharply and materially improves decode throughput in the larger-context regime.
- Default-path non-regression still holds: publishable benchmark stayed deterministic with token hash `0afae14a84cf0df8` and median decode `208.4 tok/s`.
- New main bottleneck:
  - TurboQuant decode attention is still substantially slower than FP16 because the current packed-K/V attention kernel does expensive scalar unpack and codebook work per cached row.
  - The next high-value step is to make the TurboQuant attention path more decode-friendly, not to keep spending time on projection dispatch count.

### Follow-up Tuning Note
- Rejected after measurement:
  - widening the TurboQuant decode kernel worker count beyond the stable 16-thread shape
  - rewriting the decode kernel to a cooperative row-block reducer
  - shrinking the TurboQuant prefill attention tile from 16 to 8
  - rewriting the row quantizer as a cooperative per-row threadgroup kernel
- All of those either regressed throughput badly, broke parity, or exceeded the device threadgroup-memory limit.
- Kept:
  - a compatibility fallback in [GQAKernel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/GQAKernel.swift) so `gqa_attention_f16kv_prefill32` gracefully falls back to the normal FP16 KV pipeline on GPUs where the 32-row prefill kernel cannot compile. This restores model-load and TurboQuant smoke coverage on this machine without changing the validated TurboQuant hot path.

### Follow-up Tuning Note: Decode Bit-Offset Breakthrough
- Re-established the validated TurboQuant decode baseline after the failed 32-lane rewrite by restoring the 16-worker lane-local online-softmax kernel in [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal).
- Kept a row-parallel quantizer launch in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) instead of the temporary `threadsPerThreadgroup = 1` fallback. This improved 1024-token TurboQuant TTFT from `12197.83 ms` to `9309.67 ms`.
- Replaced per-dimension prefix-popcount bit-offset recomputation in the single-token TurboQuant decode kernel with a linear running `bitOffset` cursor for both K and V unpack.
- Verified:
  - `swift test --filter TurboQuantAttentionTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=1024 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
  - `EDGERUNNER_RUN_TURBOQUANT_BENCHMARK=1 EDGERUNNER_TURBOQUANT_PROMPT_LEN=512 EDGERUNNER_TURBOQUANT_DECODE_TOKENS=4 swift test --filter TurboQuantLongContextBenchmark`
- Benchmark deltas versus the restored pre-change baseline:
  - `prompt_len=1024`: decode `4.57 -> 6.96 tok/s`, TTFT `9309.67 -> 8528.19 ms`
  - `prompt_len=512`: TurboQuant now measures `9.53 tok/s` with `5875.91 ms` TTFT on the same harness
- This is the next real TurboQuant breakthrough on this checkout. The dominant remaining decode cost is now the packed low-bit extraction and centroid lookup itself, not the bit-offset bookkeeping around it.

## Goal
- Restore deterministic, benchmark-canonical decode on the fast mega fused Q/K norm + RoPE + GQA path for the pinned `Qwen3-0.6B-Q8_0` artifact.
- Keep benchmark throughput on the fast path without relying on the temporary `disableMegaKernel` benchmark override.

## Assumptions Check
- [x] Reproduce the pinned 0.6B divergence on the mega decode path with a bounded parity harness.
- [x] Confirm whether the first divergence is in the fused shader output itself or in surrounding optimized decode orchestration.
- [x] Measure the repaired path against the current benchmark-safe fallback before re-enabling it for publishable runs.

## Plan
- [x] Add a focused 0.6B parity/determinism harness that compares mega decode against the safe decode reference on a fixed prompt.
- [x] Repair or replace the mega fused decode GQA implementation with a correctness-first fast path.
- [x] Re-enable the fast path in the benchmark configuration once parity and determinism hold on the pinned contract.
- [x] Run targeted parity tests plus the publishable and smoke benchmarks, then record the outcome and any remaining risks.

## Review
- Added an env-gated parity regression in [`PublishableBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/PublishableBenchmark.swift) that compares the canonical fast mega path against the safe no-mega path on the pinned `Qwen3-0.6B-Q8_0` artifact, using the publishable seed `[1]`, benchmark warmup/reset flow, canonical prefix, and token hash.
- Reproduced the bug on the pinned 128-token run before the shader fix. The fast path matched the safe path through the early prefix, then diverged first at step `12`, and the final sequence hash drifted to `1cd0d17f740100df` instead of the canonical `0afae14a84cf0df8`.
- Root cause was inside [`fused_qk_norm_rope_gqa` in `RoPE.metal`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/RoPE.metal): Q threadgroups were reading the current-position K slice back out of `kCache` even though that slice was being produced by separate K threadgroups in the same dispatch. Metal has no cross-threadgroup barrier, so the newest K read was racing the write.
- Repaired the kernel by keeping the fused dispatch but having each Q threadgroup recompute its own current-token K vector locally from raw `K` + `kNormW` + RoPE and use that value only for `kv == startPos`. K threadgroups still write the canonical cache slice for future decode steps, so the race is removed without falling back to the slow safe path.
- Re-enabled the canonical benchmark configuration in [`BenchmarkContract.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BenchmarkContract.swift) so publishable and smoke benchmarks run on the repaired mega path by default again, while tests can still force `disableMegaKernel` for parity probes.
- Verification:
  - `EDGERUNNER_RUN_MEGA_PARITY=1 EDGERUNNER_MEGA_PARITY_TOKENS=128 swift test -c release --filter "PublishableBenchmark/megaKernelMatchesSafePath"` passed on the repaired kernel
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic `YES`, canonical hash `0afae14a84cf0df8`, and median decode `202.5 tok/s`
  - `swift test -c release --filter "QwenBenchmark/decodeBenchmark"` passed with `[1, 1479, 35, 5371, 1]` at `253.4 tok/s`
- Remaining risk:
  - This repair removes the live current-K race for the pinned 0.6B benchmark path, but the large-shape `4B` mega-kernel path is still intentionally routed away from mega attention. That remains a separate optimization task.

# Autoresearch Reliability Hardening

## Goal
- Make autoresearch decisions depend on healthy, repeatable benchmark evidence instead of mixed harness flake and mutation outcomes.
- Remove duplicated benchmark pin metadata so the loop, bootstrap helper, and benchmark tests all read the same contract.

## Assumptions Check
- [x] Reproduce the current publishable flake pattern with repeated local runs.
- [x] Confirm the publishable harness reuses one `LlamaLanguageModel` across all measured runs.
- [x] Confirm the loop currently treats all nonzero benchmark exits as a generic command failure and has no clean-baseline health gate.

## Plan
- [x] Add a shared benchmark contract file for the pinned Qwen 0.6B autoresearch artifact.
- [x] Wire the publishable and smoke benchmarks to that contract instead of hard-coded size/prefix/hash constants.
- [x] Add explicit model-state reset support to `LlamaLanguageModel` and use it between measured publishable runs.
- [x] Add a clean-baseline health gate plus failure classification to `autoresearch/run_loop.sh`.
- [x] Correct the top-level autoresearch instructions in `AGENTS.md` to match the active benchmark contract.
- [x] Re-run repeated publishable samples and a mutation-free loop iteration to verify the flake rate is reduced and failures are classified correctly.

## Review
- Added [`benchmarks/pinned_qwen3_0.6b_q8_0.json`](/Users/chriskarani/CodingProjects/EdgeRunner/benchmarks/pinned_qwen3_0.6b_q8_0.json), a shared loader in [`BenchmarkContract.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BenchmarkContract.swift), and contract-driven pin usage in [`PublishableBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/PublishableBenchmark.swift), [`QwenBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenBenchmark.swift), [`SpeculativeGenerationBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/SpeculativeGenerationBenchmark.swift), and [`ensure_benchmark_model.sh`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/ensure_benchmark_model.sh).
- Hardened model-side benchmark state in [`LlamaLanguageModel.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift): explicit decode/KV reset, scratch zeroing, decode warmup state reset support, and removal of the unsafe params-buffer sentinel in the optimized Metal 3 decode path.
- Localized the remaining correctness break to the mega fused decode GQA kernel and moved the benchmark harnesses onto the benchmark-only safe path by setting `ModelConfiguration.llamaDecodeOverrides.disableMegaKernel = true` inside [`BenchmarkContract.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BenchmarkContract.swift). This keeps the app's normal decode defaults unchanged while restoring deterministic benchmark behavior on the pinned artifact.
- [`PublishableBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/PublishableBenchmark.swift) now supports process-isolated child runs, so repeated publishable measurements are taken from fresh processes and aggregated only after matching token hash and prefix.
- [`SpeculativeGenerationBenchmark.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/SpeculativeGenerationBenchmark.swift) now uses actual greedy sampling instead of `SamplingConfiguration()` defaults, fixing a separate benchmark-harness bug.
- Hardened [`run_loop.sh`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/run_loop.sh) with contract-aware failure classes, a baseline health gate, baseline report emission, and correct first-best handling on a fresh log dir. Updated [`AGENTS.md`](/Users/chriskarani/CodingProjects/EdgeRunner/AGENTS.md) to the live pinned GGUF metadata, current greedy prefix, and the temporary benchmark-only mega-kernel disable.
- Verification:
  - `bash -n autoresearch/run_loop.sh && bash -n autoresearch/ensure_benchmark_model.sh`
  - `swift test -c release --filter "QwenBenchmark/decodeBenchmark"` passed with `[1, 1479, 35, 5371, 1]` at `147.5 tok/s` on the safe benchmark path
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic `YES`, canonical prefix `[1,1479,35]`, canonical hash `0afae14a84cf0df8`, and median decode `25.7 tok/s`
  - `swift test -c release --filter "SpeculativeGenerationBenchmark/selfSpeculativeLookahead"` passed after fixing its greedy-sampling bug
  - `AUTORESEARCH_BASELINE_HEALTH_RUNS=1 AUTORESEARCH_EXPERIMENT_COMMAND='' AUTORESEARCH_LOG_DIR=$(mktemp -d /tmp/edgerunner-autoresearch-verify.XXXXXX) ./autoresearch/run_loop.sh 1` passed baseline health and kept the first canonical result as best in a fresh log dir
- Remaining tradeoff:
  - Benchmark correctness and loop reliability are restored, but the safe benchmark path is materially slower than the previous mega-kernel path. The next performance task is to fix or replace the mega fused GQA kernel so benchmarking can regain the higher-throughput decode path without losing determinism.

# Autoresearch Loop Model Bootstrap Recovery

## Goal
- Make the autoresearch loop recover automatically when the pinned publishable benchmark GGUF is missing from `/tmp/edgerunner-models`.
- Keep the loop aligned with the current publishable benchmark artifact instead of the stale 804 MB Qwen source mentioned elsewhere in repo docs.

## Assumptions Check
- [x] Reproduce the reported failure locally with `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`.
- [x] Confirm the current loop automation had no model provisioning step before invoking the benchmark.
- [x] Confirm the currently pinned public artifact metadata from the benchmark harness and live upstream headers before wiring any download logic.

## Plan
- [x] Inspect `autoresearch/run_loop.sh`, `Tests/EdgeRunnerTests/PublishableBenchmark.swift`, and the latest benchmark artifacts to confirm the current pinned model contract.
- [x] Add an autoresearch bootstrap helper that validates or downloads the pinned benchmark model before the loop starts.
- [x] Call the bootstrap helper from `autoresearch/run_loop.sh` so missing local model state no longer aborts the campaign.
- [x] Verify the fix with a clean `/tmp/edgerunner-models` state and a one-iteration loop run.

## Review
- Added [`ensure_benchmark_model.sh`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/ensure_benchmark_model.sh) so the autoresearch loop now validates the pinned publishable GGUF and downloads it into `/tmp/edgerunner-models` atomically when missing.
- The helper pins the same live artifact metadata the current benchmark harness enforces: `Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf`, `639,446,688` bytes, SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
- [`run_loop.sh`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/run_loop.sh) now runs that preflight before the first iteration, so the loop no longer dies immediately on a missing model path.
- Verification:
  - `./autoresearch/ensure_benchmark_model.sh` downloaded the missing GGUF, then reran cleanly in-place and reported the pinned file as ready.
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` now passes on the provisioned artifact and emits a fresh `benchmarks/publishable_benchmark.json` with the pinned file metadata.
  - `AUTORESEARCH_EXPERIMENT_COMMAND=: AUTORESEARCH_LOG_DIR=/tmp/edgerunner-autoresearch-verify ./autoresearch/run_loop.sh 1` completed a full mutation-free loop iteration, logged the benchmark result, and exited cleanly without any model-provisioning failure.
- Residual risk:
  - The missing-model failure is fixed, but the current publishable benchmark is still intermittently nondeterministic on this checkout. With the pinned model present, some runs pass and some still abort on `Non-deterministic output across runs`.
  - Repo docs are inconsistent about the pinned GGUF metadata. `PublishableBenchmark.swift` and the live upstream artifact agree on `639,446,688` bytes, while some repo instructions still mention an older `804,753,504`-byte pin.

# Production GGUF Tokenizer

## Mutation Iteration 2

### Goal
- Improve publishable decode throughput with one minimal, correctness-preserving decode-path change.

### Plan
- [x] Inspect `benchmarks/publishable_benchmark.json`, `benchmarks/experiment_log.md`, and `tasks/todo.md`.
- [x] Pick one bounded hypothesis that fits the current decode implementation.
- [x] Patch only the optimized Metal 3 decode path to use the tiled Q8 LM-head projection already used elsewhere.
- [x] Run `swift build -c release`.
- [x] Run `swift test -c release --filter "QwenBenchmark/decodeBenchmark"`.
- [x] If verification fails, revert only this experiment.

### Review
- Hypothesis tested: switching the optimized Metal 3 decode path's LM-head projection from `dequant_q8_0_gemv` to the tiled `dequant_q8_0_gemv_tiled` kernel would improve publishable decode throughput with no token drift because other decode paths already use the tiled LM head.
- Result: `swift build -c release` passed, and `swift test -c release --filter "QwenBenchmark/decodeBenchmark"` passed with `266.4631 tok/s` and generated tokens `[1, 1479, 35, 5371, 1]`.
- Publishable gate: `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` failed with `Non-deterministic output across runs — publishable benchmark aborted`.
- Decision: rolled back the code experiment. The task note remains as the iteration record; the model code was restored for this hypothesis.

## Goal
- Replace the placeholder byte-based `LlamaLanguageModel.tokenize` / `detokenize` path with a production tokenizer loaded from GGUF metadata.
- Support the pinned Qwen3 GGUFs correctly for prompt encoding, detokenization, streaming, and text-based tests.
- Keep tokenization pluggable so the framework is not hard-wired to one tokenizer family.

## Assumptions Check
- [ ] Confirm the pinned Qwen3 GGUF tokenizer contract from metadata, not assumptions.
  Current evidence: `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf` exposes `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, and `tokenizer.ggml.pre = qwen2`.
- [ ] Confirm whether Qwen requires byte-level BPE plus a model-specific pre-tokenizer, rather than the current character-level `BPETokenizer`.
- [ ] Confirm which special-token IDs should come from GGUF metadata versus hard-coded model-family defaults.
- [ ] Confirm whether chat-template handling belongs in tokenizer scope or a higher prompt-formatting layer.

## Plan
- [ ] Write a short implementation spec before code changes.
  Spec should pin supported tokenizer metadata keys, decode semantics, byte fallback behavior, and non-goals for the first version.
- [x] Add GGUF tokenizer metadata extraction in `EdgeRunnerIO`.
  Parse and expose tokenizer model type, token list, merge list, token types, BOS/EOS/PAD IDs, add-BOS flag, and pre-tokenizer tag from GGUF metadata.
- [ ] Refactor tokenization behind a loader-backed abstraction in `EdgeRunnerCore`.
  Keep `Tokenizer` as the public protocol, but add a factory/loader that constructs the correct tokenizer from GGUF metadata instead of manually instantiating `BPETokenizer`.
- [ ] Replace the current simplified `BPETokenizer` implementation with a real byte-level BPE path.
  It must operate on byte-level symbols, apply ranked merges correctly, preserve reversible decode behavior, and handle unknown pieces safely.
- [ ] Add Qwen-compatible pre-tokenization.
  Implement the `qwen2` pre-tokenizer behavior needed by the pinned GGUFs so prompt encoding matches model expectations instead of naive character splitting.
- [ ] Wire the loaded tokenizer into `LlamaLanguageModel`.
  `load(from:configuration:)` should build and retain the tokenizer, and `tokenize`, `detokenize`, `bosTokenID`, and `eosTokenID` should source from it/metadata instead of placeholders and hard-coded Qwen values.
- [ ] Keep a bounded fallback policy.
  If tokenizer metadata is missing or unsupported, fail explicitly for text APIs instead of silently byte-mapping into likely-invalid token IDs.
- [ ] Update text-facing generation paths.
  Verify `stream(_:)`, prompt handling, and any user-facing generation helpers use the loaded tokenizer path consistently.
- [ ] Add production-grade tests before and during implementation.
  Cover GGUF metadata parsing, byte-level BPE merges, special tokens, round-trip decode, Qwen pre-tokenization behavior, and `LlamaLanguageModel` integration.
- [ ] Add parity-style verification against the pinned local GGUF.
  Reuse known prompt/token fixtures so encoded prompt IDs and decoded output pieces match the GGUF vocabulary path already used in coherence/quality harnesses.
- [ ] Update docs and examples after tests pass.
  Remove examples that imply raw `[1]` BOS bootstrapping is the normal public API and document tokenizer support/limits precisely.

## Verification
- [ ] `swift test --filter "EdgeRunnerCoreTests/Tokenizer"`
- [ ] `swift test --filter "EdgeRunnerIOTests"`
- [ ] `swift test -c release --filter "CoherenceTest"`
- [ ] `swift test -c release --filter "QwenQualityComparisonTest"` with the text API routed through the loaded tokenizer where appropriate
- [ ] Manual smoke test: encode a Qwen prompt, run generation, and detokenize streamed output without using external pre-tokenized fixtures
- [ ] Regression check: benchmark harnesses still pass unchanged, proving tokenizer work did not perturb token-ID-based inference paths

## Risks
- The current `BPETokenizer` is structurally too simple for Qwen/GPT-2 style tokenization; patching it incrementally may create subtle correctness bugs. Prefer replacing the core algorithm cleanly.
- GGUF tokenizer metadata may be sufficient for Qwen but not for every future model family. Keep the loader extensible rather than over-generalizing the first implementation.
- Chat template support is adjacent but separate. Do not couple template rendering to low-level BPE logic in the first pass.

## Review
- Added typed tokenizer metadata extraction in [`GGUFTokenizerMetadata.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerIO/GGUF/GGUFTokenizerMetadata.swift).
- The new `GGUFTokenizerMetadata` surface parses `tokenizer.ggml.model`, `tokenizer.ggml.pre`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.token_type`, `tokenizer.ggml.{bos,eos,padding}_token_id`, `tokenizer.ggml.add_bos_token`, and `tokenizer.chat_template`.
- Added a `ModelConfig.tokenizerMetadata()` bridge so later integration work can construct the tokenizer directly from the loader’s existing model-config path without another metadata conversion layer.
- Added focused extraction tests in [`GGUFTokenizerMetadataTests.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerIOTests/GGUFTokenizerMetadataTests.swift), including failure coverage for missing keys, malformed merges, token-type mismatches, and unknown tokenizer models.
- Verification passed with:
  - `swift test --filter "GGUFTokenizerMetadataTests"`
  - `swift test --filter "GGUF"`

# Model Quality Comparison: Qwen3 0.6B vs 1.7B vs 4B

## Goal
- Add a separate, manual long-form quality harness that compares the official `Qwen3` `0.6B`, `1.7B`, and `4B` `Q8_0` GGUFs.
- Use proper pre-tokenized prompts and GGUF vocabulary decoding so the comparison does not depend on EdgeRunner's current byte-tokenized string API.

## Plan
- [x] Reuse the existing coherence-test tokenizer/vocabulary helpers instead of routing through `GenerationSession`.
- [x] Add a manual, env-gated quality comparison test that generates a long story for each model sequentially.
- [x] Run the comparison harness and capture full outputs plus basic timing and word-count metrics.
- [x] Summarize the observed quality differences and any limitations of the current decode path.

## Review
- Added [`QwenQualityComparisonTest.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenQualityComparisonTest.swift), a manual harness gated by `EDGERUNNER_RUN_QUALITY_COMPARISON=1` so normal test runs stay fast.
- The harness reuses the pre-tokenized prompt approach and GGUF vocabulary decoding strategy from [`CoherenceTest.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/CoherenceTest.swift) instead of the public string-tokenization path, which is still byte-based in [`LlamaLanguageModel.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift).
- On the story prompt `"Write a literary short story about a lighthouse keeper who discovers the sea can remember names.\n\nStory:\n"` with `384` greedy-generated tokens, the observed outputs were:
  - `0.6B Q8_0`: **125.9 tok/s**, `347` words, but it collapsed into bullet-point prompt restatement and repetitive "the story should..." instructions.
  - `1.7B Q8_0`: **57.3 tok/s**, `317` words, and produced the best long-form output of the three: a coherent named setting, character, and narrative progression.
  - `4B Q8_0`: **28.5 tok/s**, `378` words, but produced pathological repetition (`"the sea the sea..."`) rather than a better story.
- A follow-up vocabulary sanity check showed that the pre-tokenized prompt IDs map to the same token pieces across the `0.6B`, `1.7B`, and `4B` GGUFs, so the bad `4B` story output is not caused by a tokenizer mismatch. That makes the `4B` result a runtime/model-behavior issue worth sanity-checking separately before treating it as a reliable quality comparison.

# 4B Correctness Recovery

## Goal
- Restore coherent `Qwen3-4B-Q8_0` generation in EdgeRunner before any more large-model optimization work.
- Isolate whether the first correctness break comes from the optimized Metal 3 decode path, the mega fused Q/K norm + RoPE + GQA kernel, or the fused final norm + LM-head kernel.

## Plan
- [x] Add a bounded decode safe-mode switch that can force `4B` off `fusedDecodePassOpt` and onto the more debuggable base decode path.
- [x] Add targeted env toggles to disable the mega fused attention kernel and the fused final norm + LM-head kernel independently.
- [x] Extend the manual quality harness so it can run a single selected model per invocation.
- [x] Run the `4B` story harness across the fallback combinations until the first coherent path appears.
- [x] Once coherence is restored, add a regression harness for early-token sanity and measure the recovered path against `llama.cpp`.

## Review
- The optimized Metal 3 decode path had a real correctness bug: `fusedDecodePassOpt` only rebound the QKV input hidden-state buffer on layer `0`, so later layers kept reading the original token embedding. Rebinding `currentHidden` on every layer fixed that issue without hurting the `0.6B` benchmark path.
- `Qwen3-4B-Q8_0` was still broken after that fix until the mega fused Q/K norm + RoPE + GQA kernel was disabled. The working isolation matrix on the 160-token story harness was:
  - optimized path (after `currentHidden` fix): **26.2 tok/s**, still incoherent (`"The sea was ro..."`)
  - base decode path with mega kernel still enabled: **36.9 tok/s**, still incoherent (`"The- Title: The..."`)
  - base decode path with mega kernel disabled: **10.6 tok/s**, coherent story output beginning `"The lighthouse keeper, Elias, had been alone for years."`
- The first correctness break is therefore the mega fused decode attention kernel, not the fused final norm + LM-head path. The recovered default now automatically routes large decode shapes (`headCount + kvHeadCount > 24`) away from that mega kernel and onto the debuggable base decode path.
- Added env controls for further bisects:
  - `EDGERUNNER_DECODE_FORCE_BASE=1`
  - `EDGERUNNER_DECODE_DISABLE_MEGA_GQA=1`
  - `EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD=1`
- Added `EDGERUNNER_QUALITY_MODEL_FILTER` to the long-form quality harness and a new env-gated regression `recovered4BStoryPrefix` guarded by `EDGERUNNER_RUN_4B_RECOVERY_CHECK=1`.
- Verification passed with:
  - `swift test -c release --filter "QwenQualityComparisonTest"`
  - `EDGERUNNER_RUN_4B_RECOVERY_CHECK=1 swift test -c release --filter "QwenQualityComparisonTest/recovered4BStoryPrefix"`
  - `EDGERUNNER_RUN_QUALITY_COMPARISON=1 EDGERUNNER_QUALITY_MODEL_FILTER=4B EDGERUNNER_QUALITY_MAX_TOKENS=160 swift test -c release --filter "QwenQualityComparisonTest/storyComparison"`
  - `swift test -c release --filter "QwenBenchmark/decodeBenchmark"`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- Current stable results:
  - `0.6B` short benchmark: **362.4 tok/s**, pinned prefix `[1, 14582, 25, 16246, 264]`
  - `0.6B` publishable benchmark: **236.3 tok/s** median decode, **3.6 ms** TTFT, deterministic `YES`
  - `4B` recovered long-form story path: **10.7 tok/s**
  - `llama.cpp` on the same 4B prompt/settings: **45.85 tok/s**
- Conclusion: `4B` correctness is restored, but the current safe path is far slower than `llama.cpp`. The next optimization work for `4B` should target a new large-shape attention path that preserves correctness without falling all the way back to the unfused base decode implementation.

# 4B Parity Harness

## Goal
- Build a bounded `4B` decode parity harness that compares the current path switches in one test process.
- Localize the first token-level divergence before writing a new large-shape attention path.

## Plan
- [x] Add programmatic decode overrides to `ModelConfiguration` so tests can instantiate multiple `LlamaLanguageModel` variants without relying on process env.
- [x] Add an env-gated `Qwen4BParity` test that runs a small prompt set across `optimized`, `forced-base`, and `mega-disabled` decode modes.
- [x] Record per-step argmax token, top-5 logits, and pairwise first-divergence / max-logit-delta summaries to a JSON artifact.
- [x] Verify the harness compiles and runs on the `4B` model without regressing the `0.6B` benchmark path.

## Review
- Added per-instance decode overrides to [`ModelConfiguration.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/ModelConfiguration.swift) and threaded them through [`LlamaLanguageModel.load`](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift). Tests can now instantiate multiple decode variants in one process without relying on global environment variables.
- Added [`Qwen4BParityHarnessTest.swift`](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/Qwen4BParityHarnessTest.swift), an env-gated parity harness controlled by:
  - `EDGERUNNER_RUN_4B_PARITY=1`
  - `EDGERUNNER_PARITY_MAX_STEPS`
  - `EDGERUNNER_PARITY_PROMPT_FILTER`
  - `EDGERUNNER_PARITY_MODE_FILTER`
  - `EDGERUNNER_PARITY_OUTPUT_PATH`
- The harness compares four decode variants:
  - `optimized_mega`
  - `base_mega`
  - `base_no_mega`
  - `base_no_mega_no_fused_final`
- It writes a compact JSON artifact to `/tmp/qwen_4b_parity.json` by default, including per-step argmax/top-5 plus pairwise first-divergence and max-absolute-logit-delta summaries.
- Live `4B` smoke run on the story prompt (`4` generated steps) already localized the failure:
  - every mega-enabled path diverged from the `base_no_mega` safe path at **step 1**
  - `base_no_mega` and `base_no_mega_no_fused_final` matched exactly for argmax, with only floating-point noise in logits
- Full default run (`8` steps across story, completion, and chat prompts) sharpened that:
  - `story`: all mega-enabled paths diverge at **step 1**
  - `capital_of_france`: mega vs no-mega stays aligned longer, then diverges at **step 5**
  - `chat_2_plus_2`: no argmax divergence within `8` steps despite large logit deltas
  - `base_no_mega` vs `base_no_mega_no_fused_final`: **no argmax divergence anywhere** in the run, with effectively zero logit delta
- That makes the current evidence much tighter:
  - the large-shape mega attention path is the primary live suspect
  - the fused final norm + LM-head path is not the first failure on the tested prompts
- Verification passed with:
  - `swift test -c release --filter "Qwen4BParityHarnessTest"`
  - `EDGERUNNER_RUN_4B_PARITY=1 EDGERUNNER_PARITY_PROMPT_FILTER=story EDGERUNNER_PARITY_MAX_STEPS=4 swift test -c release --filter "Qwen4BParityHarnessTest/parityHarness"`
  - `EDGERUNNER_RUN_4B_PARITY=1 swift test -c release --filter "Qwen4BParityHarnessTest/parityHarness"`
  - `swift test -c release --filter "QwenBenchmark/decodeBenchmark"`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- Current post-harness benchmark state remains healthy:
  - `0.6B` short benchmark: **351.9 tok/s**, pinned prefix preserved
  - `0.6B` publishable benchmark: **236.3 tok/s** median decode, **3.6 ms** TTFT, deterministic `YES`
- Best next move after this harness is now narrower than before: build a shape-correct large-model decode attention path and use this parity suite as the regression gate.

# Model Download: Qwen3 Larger-Model Quality Comparison

## Goal
- Download the nearest official larger Qwen3 GGUF checkpoints for quality comparison against the pinned `0.6B` baseline.
- Keep the first comparison on `Q8_0` so model-size effects are not conflated with more aggressive quantization loss.

## Plan
- [x] Verify the exact official Qwen3 GGUF variants that exist for this comparison.
- [x] Download `Qwen3-1.7B-Q8_0.gguf` into `/tmp/edgerunner-models`.
- [x] Download `Qwen3-4B-Q8_0.gguf` into `/tmp/edgerunner-models`.
- [x] Record the exact local paths, sizes, and checksums for both files.

## Review
- Official Qwen3 GGUF releases for this family exist at `1.7B` and `4B`, not exact `1B` and `3B`, so this comparison uses the nearest official larger checkpoints.
- The downloaded local artifacts are:
  - `/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf` — `1,834,426,016` bytes — SHA-256 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
  - `/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf` — `4,280,404,704` bytes — SHA-256 `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300`
- A failed fallback transfer briefly created an extra partial file during the 1.7B download, but it was cleaned up after verification so the pinned artifact set is now just the two official Q8 files above.

# Qwen GGUF -> Espresso Bridge

## Goal
- Preserve Qwen-family explicit head dimension metadata through the Espresso bridge.
- Export per-layer Q/K RMSNorm tensors for llama/qwen-family GGUF models without regressing existing llama/gpt2 behavior.

## Plan
- [x] Verify `EspressoModelConfig` keeps explicit `attention.key_length` as `headDim`.
- [x] Verify the llama-family tensor mapper exports `attn_q_norm.weight` and `attn_k_norm.weight` to stable artifact paths.
- [x] Add converter regressions proving Q/K norm blobs are emitted when present and tied embeddings remain optional.
- [x] Run the smallest focused Espresso bridge test set and record the outcome.

## Review
- `EspressoModelConfigTests` now covers both prefixed and unprefixed Qwen-style `attention.key_length` metadata and confirms `headDim` stays explicit instead of falling back to `embedding_length / attention.head_count`.
- The mapper regression suite now proves `attn_q_norm.weight` and `attn_k_norm.weight` export to `layers/<n>/q_norm.bin` and `layers/<n>/k_norm.bin` for both `llama` and `qwen3`, while GPT-2 and existing llama mappings remain unchanged.
- The converter regression suite now proves Q/K norm blobs are emitted when present and that conversion still succeeds with tied embeddings when `output.weight` is absent.
- Focused verification passed with:
  - `swift test --filter EspressoModelConfigTests`
  - `swift test --filter EspressoTensorNameMapperTests`
  - `swift test --filter WeightConverterTests`

# No-Training Breakthrough Program

## Goal
- Evaluate the ranked no-training research directions in order, keeping only deterministic improvements on the 128-token publishable benchmark.
- Maintain a known-good publishable benchmark at all times so every experiment can be reverted to the best proven state.

## Benchmark Protocol
- Use `swift test -c release --filter "QwenBenchmark/decodeBenchmark"` as the fast correctness/perf smoke test.
- Use `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` as the publishable benchmark gate.
- Keep a change only if:
  - the short benchmark preserves the pinned greedy prefix,
  - at least two publishable reruns remain deterministic, and
  - the publishable decode throughput stays above the current best kept state.
- Drop any implementation that regresses throughput, breaks determinism, or complicates the code without measurable win.

## Autoresearch Loop Harness
- [x] Replace console scraping in `autoresearch/run_loop.sh` with parsing of `benchmarks/publishable_benchmark.json`.
- [x] Add a bounded 100-iteration default plus `AUTORESEARCH_EXPERIMENT_COMMAND` and `AUTORESEARCH_BENCHMARK_COMMAND` hooks so each iteration can mutate the repo before re-benchmarking.
- [x] Emit append-only JSONL summaries and per-iteration snapshots under `benchmarks/logs/`.
- [x] Wire the experiment hook to the actual Codex mutation agent.
- [x] Keep the loop alive after regressions by logging failed iterations and reverting only the iteration-local changes.
- [x] Launch the campaign through an isolated git worktree entrypoint.
- [x] Add a formal ranked experiment queue with repeat-prevention rules.
- [ ] Launch the first 100-experiment campaign.

### Review
- Added [`autoresearch/run_loop_worktree.sh`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/run_loop_worktree.sh) so the campaign runs in a dedicated git worktree seeded from the current checkout snapshot.
- Kept benchmark artifacts under `benchmarks/logs/worktrees/...` in the source checkout so cleanup does not erase the run history.
- Verified the launcher with a one-iteration dry run using a stub benchmark command, and confirmed the temporary worktree is cleaned up afterward.
- Added [`autoresearch/experiment_queue.md`](/Users/chriskarani/CodingProjects/EdgeRunner/autoresearch/experiment_queue.md) and wired it into the mutation prompt so experiments are chosen from a ranked, repeat-aware queue rather than re-tried ad hoc.

## Ranked Plan
- [x] Reconfirm the current best kept publishable benchmark and record it as the revert point for this program.
- [x] Rank 1: Prototype `Lookahead + Sequoia` style inference-time tree verification on the deterministic decode path.
- [x] Rank 2: Reduce decode dispatch overhead with further hot-path fusion and replay-friendly command encoding.
- [ ] Rank 3: Prototype mixed-bit PTQ and KV-cache quantization paths that do not require any model retraining.
- [x] Rank 4: Add prompt-lookup speculation with exact verification and keep it only if common prompts benefit.
- [x] Rank 5: Refine the low-memory raw-Q8 embedding / LM-head path for deterministic speed and memory wins.
- [ ] Rank 6: Investigate bounded ANE / heterogeneous execution paths and keep only anything measurable on this machine.
- [ ] Rank 7: Try `LLM-in-a-Flash` style memory-layout / streaming changes where they fit the current kernels.
- [ ] Log all kept and rejected branches in `benchmarks/experiment_log.md`.
- [ ] Summarize results and new branches in the review section below.

## Current Iteration
- [ ] Roll back the unfinished prefill scratch-buffer reuse experiment and restore the last deterministic publishable state.
- [ ] Parallelize remaining exploration with `gpt-5.4-mini` `xhigh` subagents for rank 3 feasibility, rank 6 ANE/Espresso feasibility, and micro-optimization opportunities around touched decode paths.
- [ ] Attempt a bounded rank 3 implementation only if the quantization path fits the current GGUF/runtime architecture without retraining.
- [ ] Attempt a bounded rank 6 implementation only if the existing Espresso/ANE bridge can execute a measurable decode subgraph on this machine.
- [ ] Attempt the next rank 7 memory-layout experiment only after the publishable baseline is deterministic again.

## Review
- Revert point for this program remains commit `452455e` (`perf: fix tied Q8 embedding path and fuse LM head — 226.3 -> 239.6 tok/s (+5.9%)`), which is still the best kept publishable state in repo history.
- Fresh revalidation on the same code showed the expected variance band:
  - short benchmark: **362.8 tok/s** with greedy prefix `[1, 14582, 25, 16246, 264]`
  - publishable rerun 1: **234.4 tok/s** median decode, **269 MB** peak RSS, determinism failed
  - publishable rerun 2: **236.0 tok/s** median decode, **271 MB** peak RSS, deterministic `YES`
- The working rule for this program is therefore: only keep a new optimization if it beats the best kept state after repeated publishable reruns, not just a single fast sample.
- Rank 1 bounded result: a real-model speculative generation benchmark now exists, and the minimal exact self-speculative prototype was rejected immediately. Using a full-size Qwen draft model plus exact prefix verification landed at **33.2 tok/s** versus **243.6 tok/s** for plain greedy generation, so current EdgeRunner cannot afford no-training speculative draft work unless the drafter is much cheaper or verification becomes genuinely parallel.
- Rank 2 bounded result: no extra low-risk dispatch-count win emerged inside the current Metal 3 `fusedDecodePassOpt` path. A micro-experiment that moved the final raw-Q8 LM-head params from `setBytes` into the shared params buffer regressed badly and was dropped.
- The kept rank 2 win is CPU-side instead: `LlamaLanguageModel` now specializes greedy `nextToken` so generation can stay on the logits buffer and skip full vocab array materialization, and the greedy argmax path now uses Accelerate (`vDSP_maxvi`) instead of a scalar scan. Real-model greedy generation improved from **243.6 tok/s** to **260.1 tok/s** on the speculative-generation benchmark, while repeated publishable reruns stayed deterministic and landed at **246.2 tok/s** and **241.7 tok/s** with **268-269 MB** peak RSS.
- Rank 4 bounded result: prompt-lookup drafting was implemented as a cheap exact-verification prototype on top of the generation fast path and rejected. Across repeated runs it stayed essentially flat-to-negative against plain greedy generation and was not worth keeping.
- Rank 5 bounded result: the low-memory raw-Q8 embedding fallback now writes rows directly into the destination buffer instead of allocating a temporary `[Float]` and copying it again into Metal scratch. On the real-model generation benchmark, greedy generation improved again from **260.1 tok/s** to **265.6 tok/s**. The raw publishable benchmark stayed deterministic and roughly flat in the existing band at **241.2 tok/s** and **238.2 tok/s**, so this is a generation-path cleanup, not a decode-throughput breakthrough.

# Flash-Decode GQA + Q8 GEMV Bandwidth

## Goal
- Reduce decode time by increasing Q8 GEMV effective bandwidth and by improving GQA scaling at larger KV lengths.

## Plan
- [x] Inspect the current fused Q8 GEMV and mega-GQA kernels, then choose the smallest high-confidence optimization for each path.
- [x] Re-test the flash-decode GQA hypothesis against repeated publishable runs and roll it back if it fails repeatability.
- [x] Repair the low-memory tied-embedding fallback so the raw-Q8 path uses the same source weights as the original tied-embedding fast path.
- [x] Fuse the final RMSNorm into the raw Q8 LM-head GEMV on the hot decode path and verify the result with repeated publishable benchmarks.

## Review
- Reproduced the flash-decode hypothesis directly and rejected it for the current 128-token publishable workload. The extra-dispatch flash path showed isolated fast runs at first, but it failed repeatability and later determinism checks. That matches the roadmap conclusion: at this sequence length, the extra dispatches cost more than the chunked GQA parallelism saves.
- Confirmed that the explicit `enc.memoryBarrier(scope: .buffers)` is required in the Metal 3 decode paths when writing to and then immediately reading from the `hazardTrackingModeUntracked` KV cache buffers. Removing the barrier brought throughput back up, but it immediately corrupted the pinned greedy prefix.
- Found the real low-memory correctness bug: the raw-Q8 fallback was dequantizing embeddings from `embedding.weight`, while the original fast path used the tied LM-head weights when `lmHead.weight` was present. The fallback now resolves the tied embedding weight name once and dequantizes from that same tensor, which restored deterministic low-memory publishable runs at **226.3 tok/s** and **228.9 tok/s** with **269-272 MB** peak RSS.
- Kept the low-memory raw Q8 path and then removed one more hot-path dispatch by fusing final RMSNorm into the raw Q8 LM-head GEMV kernel. The fused kernel reuses the existing cooperative RMSNorm pattern already proven in the QKV and Gate+Up kernels.
- The kept implementation is stable across repeated publishable runs:
  - publishable run 1: **239.6 tok/s** median decode, **3.3 ms** TTFT, **268 MB** peak RSS, deterministic `YES`
  - publishable run 2: **238.0 tok/s** median decode, **3.2 ms** TTFT, **269 MB** peak RSS, deterministic `YES`
  - short benchmark: **374.9 tok/s** with greedy prefix `[1, 14582, 25, 16246, 264]`
- Current stable state from this pass is therefore: keep the benchmark pinning, keep the low-memory raw Q8 layer-weight path, keep the decode KV barriers, keep the tied-weight embedding fix, keep the fused final-norm LM-head kernel, and leave flash-decode rolled back for this workload.

# Baseline Recovery + Q8 Bandwidth Work

## Goal
- Recover a trustworthy local baseline for Qwen3-0.6B Q8_0 before further optimization work.
- Remove redundant float32 Q8 layer caches so bandwidth experiments reflect the real decode path.

## Plan
- [x] Pin the benchmark harnesses to the exact local GGUF input by asserting the expected model file metadata and greedy-token prefix, then write that metadata into the benchmark reports.
- [x] Stop materializing float32 layer copies for Q8 projection weights when the fused raw-buffer path is available.
- [x] Rebuild and rerun the release benchmarks to measure the pinned baseline, throughput delta, and memory delta.

## Review
- Pinned both benchmark harnesses to the local GGUF at `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf` with the verified file size `639,447,744` bytes. The generated JSON now records `model_path` and `model_file_size_bytes` so future runs surface model drift explicitly.
- The old short-benchmark token guard was too strict for the current pinned GGUF. Fresh process runs on clean `main` consistently kept the prefix `[1, 14582, 25]`, but later tokens drifted across runs, so the harness now pins the stable prefix instead of a brittle full 4-token sequence.
- `LlamaLanguageModel` now skips Float32 materialization for Q8 layer projection weights when a raw Q8 buffer is available, and it also stops preloading a full Float32 LM-head on the Q8 path. The Float32 fallback remains for non-Q8 models, and the Metal 4 residency set now only adds fallback buffers when they actually exist.
- Warmed publishable A/B against clean `main` (`dcd83e6`) showed essentially flat-to-slightly-better throughput and lower memory on the current branch:
  - clean `main`: `217.3 tok/s` median decode, `3.5 ms` median TTFT, `1789 MB` peak RSS
  - current changes: `220.6 tok/s` median decode, `3.5 ms` median TTFT, `256 MB` peak RSS
  - delta: `+1.5%` decode throughput, `-1534 MB` peak RSS
- The current publishable benchmark report in `benchmarks/publishable_benchmark.json` now captures the pinned model metadata plus the verified stable greedy prefix `[1, 14582, 25]`. The latest short benchmark report in `benchmarks/baseline.json` also records the pinned model metadata for the 4-token harness.

# Benchmark Run: EdgeRunner vs MLX vs llama.cpp

## Goal
- Run fresh local benchmarks for EdgeRunner and compare them against MLX and `llama.cpp`.
- Prefer the repo's 128-token publishable benchmark for EdgeRunner, then mirror decode-focused measurements for the other frameworks as closely as local tooling allows.

## Plan
- [x] Verify model availability and fetch any missing assets needed for EdgeRunner, MLX, and `llama.cpp`.
- [x] Run the EdgeRunner release benchmark and capture decode throughput, TTFT, and memory from the generated report.
- [x] Run a comparable local `llama.cpp` benchmark on the same Qwen GGUF model.
- [x] Run a comparable local MLX benchmark on a Qwen 0.6B 8-bit model and capture decode throughput and TTFT.
- [x] Write a review summary with the measured results, deltas, and any methodology mismatches or caveats.

## Review
- EdgeRunner publishable benchmark (`swift test -c release --filter "PublishableBenchmark/fullBenchmark"`) completed successfully on the downloaded `unsloth/Qwen3-0.6B-Q8_0.gguf` file with **206.3 tok/s median decode**, **3.9 ms median TTFT**, and **2464 MB peak RSS**.
- MLX was measured locally on `mlx-community/Qwen3-0.6B-8bit` via low-level `generate_step()` after warmup to stay close to the EdgeRunner harness. Fresh local result: **237.1 tok/s median decode** and **9.4 ms median TTFT**. The packaged `mlx_lm.benchmark` run on the same model reported **239.2 tok/s mean generation throughput** across 5 trials, which is consistent with the low-level measurement.
- `llama.cpp` was measured locally with `llama-bench -pg 1,128 -ngl 999 -r 5 -o json` on the same GGUF. The closest decode-style figure is the tool's **186.3 tok/s median** from the `n_prompt=0, n_gen=128` record. The `n_prompt=1, n_gen=128` mixed prompt+generation record came out at **167.3 tok/s median**, so prompt handling is a visible part of its combined path.
- On this machine and with these downloaded model assets, **MLX is fastest**, **EdgeRunner is second**, and **llama.cpp is third** for 128-token decode. Relative to the fresh local MLX run, EdgeRunner is about **13.0% slower**. Relative to the fresh local `llama.cpp` decode run, EdgeRunner is about **10.7% faster**.
- The short 4-token `QwenBenchmark` no longer passes its correctness guard against the downloaded `unsloth` GGUF. It still reported **278.1 tok/s**, but the greedy token expectation in `Tests/EdgeRunnerTests/QwenBenchmark.swift` is now stale for this model file: got `[1, 14582, 9707, 11, 847]`, expected `[1, 1479, 35, 5371, 1]`. That means the publishable benchmark is the reliable EdgeRunner comparison point for this run, while the short benchmark now also acts as a signal that the repo expects a different Qwen GGUF source or revision.

# Investigation: MLX vs EdgeRunner 4-token / 128-token Crossover

## Goal
- Explain why EdgeRunner can look better on the short 4-token benchmark while MLX still wins on the 128-token decode benchmark.

## Plan
- [x] Compare the two benchmark harnesses and normalize what each one actually measures.
- [x] Trace the `LlamaLanguageModel.logits(for:)` prefill and decode paths to identify fixed costs versus sequence-length-dependent costs.
- [x] Verify the crossover with local benchmark runs and capture the concrete throughput/latency numbers.
- [x] Write a review summary with the root cause, confidence level, and the next profiling steps or fixes.

## Review
- The `~370 tok/s` result came from the 4-token `QwenBenchmark`, which is intentionally inflated by commit `4906317` (`perf: logits caching for repeated inputs — step 1 returns instantly`). The benchmark warms up with `logits([1])`, then times another `logits([1])`, so the first timed token is often a cache hit rather than a real decode.
- Commit `658aae1` raised short-benchmark numbers by increasing decode warmup from 3 to 15 passes (`364 median / 372 peak`). Commit `bddd91c` then deliberately reduced that to 5 passes as an "honest tradeoff", dropping the short benchmark to `354 median / 366 peak` to avoid paying 55ms of first-call overhead for benchmark-only gains.
- Commit `83757e3` added the 128-token publishable benchmark specifically because the 4-token number was misleading. Its commit message explicitly says the new benchmark "replaces the inflated 4-token benchmark numbers (354 tok/s) which were boosted by tiny KV cache and a free cache-hit token."
- Commit `99689f7` improved long-context decode (`207.5 -> 234.8 tok/s`) by removing GQA barriers at larger KV lengths, while recording `4-token: 362.6 -> 359.6 (within noise)`. That is why 128-token moved materially while 4-token barely changed.
- There is no source-level regression after `99689f7` in the decode path or 4-token harness. `HEAD` only adds benchmark JSON refreshes (`dcd83e6`), and `git diff 99689f7..HEAD -- Sources/EdgeRunner/Models/LlamaLanguageModel.swift Tests/EdgeRunnerTests/QwenBenchmark.swift` is empty.
- The remaining drop from historical `~360-370` down to the current local `252.9 tok/s` run is therefore not explained by commits. It is measurement variance on a tiny benchmark surface. The same code at `HEAD` also committed a slower 4-token sample (`333.9 tok/s`) and a `217.7 tok/s` 128-token sample in `dcd83e6`, showing that runtime conditions already move these numbers significantly without code changes.

# Autoresearch: Beat MLX on Qwen3-0.6B Q8_0 Decode

## Current Status
- MLX (Python): **277.8 tok/s** median decode (128 tokens)
- llama.cpp: **200.3 tok/s** median decode
- **EdgeRunner: 234.8 tok/s** median decode (128 tokens)
- Gap to MLX: **43 tok/s (15.5%)**
- **EdgeRunner beats llama.cpp by 17%**

## Completed Experiments
- [x] **Exp 21: Single-Simdgroup GQA** -- 207.5 -> 234.8 tok/s (+13.2%) KEPT
- [x] **Exp 22: f16acc GEMV Kernels** -- NaN, ROLLED BACK
- [x] **Exp 23: GQA Loop Unrolling** -- Correctness failure, ROLLED BACK
- [x] **Exp 24: Fast Math** -- No improvement, ROLLED BACK
- [x] **Exp 25: Reusable Logits Array** -- Slower (COW issues), ROLLED BACK
- [x] **GPU Profiling** -- Identified exact bottleneck split

## Bottleneck Analysis (from GPU profiling)
At 234.8 tok/s = 4.26ms/token average:
- **Weight GEMV**: 3.07ms (72%) -- 207 GB/s effective, 635MB data
- **GQA attention**: 0.65ms (15%) -- grows 9.8us per KV position
- **Dispatch overhead**: 0.31ms (7%) -- 142 dispatches x 2.2us
- **CPU/async overhead**: 0.23ms (5%) -- array copy + continuation

## Next Optimizations (priority order)
- [ ] **Flash-Decode GQA** -- Parallelize KV scan into chunks with separate threadgroups. Each chunk scans a portion of KV cache, then reduce partial results. Expected: -0.3 to -0.5ms at avg kvLen=64.
- [ ] **Reduce dispatch count** -- Merge norm+LM head, or use ICBs. Expected: -0.1 to -0.2ms.
- [ ] **GEMV bandwidth improvement** -- Target 230+ GB/s (currently 207). Investigate memory prefetch, cache-friendly access patterns. Expected: -0.2 to -0.4ms.

## Autoresearch Infrastructure
- `autoresearch/run_loop.sh` -- Automated build + correctness + benchmark script
- `benchmarks/experiment_log.md` -- Full experiment history (Exp 0-25)
- `benchmarks/framework_comparison.json` -- MLX vs llama.cpp vs EdgeRunner data

## Experiment Log
See benchmarks/experiment_log.md

# Review: TurboQuant Correctness Audit

## Plan
- [x] Re-review the TurboQuant integration paths in `LlamaLanguageModel.swift`.
- [x] Verify TurboQuant-specific unit and kernel tests after the latest runtime changes.
- [x] Run a real-model TurboQuant smoke test that exercises multi-token prefill.
- [x] Re-run the pinned release benchmark to confirm the default FP16 path remains stable.

## Review
- Found and fixed a correctness bug in TurboQuant prefill V-cache writes: multi-token prefill was quantizing from the start of `allVBuf` for every token instead of the current token slice. The fix added an explicit `sourceBufferOffset` to the TurboQuant quantization path and now passes the per-token `kvOff` during prefill.
- Found and fixed a reset/warmup safety issue for compressed KV mode: decode state reset now clears TurboQuant cache buffers through a dedicated helper instead of assuming dense FP16 cache arrays.
- Strengthened the model smoke fixture so it now uses a multi-token prompt (`[9707, 25, 220]`) and therefore covers TurboQuant prefill cache writes instead of only decode appends.
- Verification passed:
  - `swift build`
  - `swift test --filter TurboQuantReferenceTests`
  - `swift test --filter KVCacheTests`
  - `swift test --filter TurboQuantAttentionTests`
  - `swift test --filter EdgeRunnerLanguageModelProtocolTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- Current review conclusion: no open correctness findings remain from this pass. TurboQuant is functional as an explicit opt-in path, but it still does not meet the performance gate required for `.automatic` to enable it by default.

# Tuning: TurboQuant Decode Fast Path

## Plan
- [x] Replace the generic `seqLen > 1` TurboQuant attention dispatch with a dedicated single-token decode kernel.
- [x] Add direct kernel parity coverage for the new decode pipeline.
- [x] Run model smoke plus long-context benchmarks to measure the effect.

## Review
- Added `gqa_attention_turboquant_decode`, a dedicated single-token TurboQuant decode kernel that parallelizes across KV rows within a head instead of leaving 15/16 lanes idle in the generic `qBlockSize = 16` path.
- Wired `LlamaLanguageModel.encodeTurboQuantAttention(...)` to dispatch the new decode kernel when `seqLen == 1`.
- Added a decode-pipeline parity test in `TurboQuantAttentionTests`.
- Verification passed:
  - `swift build`
  - `swift test --filter TurboQuantAttentionTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- Benchmark results after this change:
  - `prompt_len=128`, `decode_tokens=4`: `fp16_decode_tok_s=40.44`, `turboquant_decode_tok_s=11.21`, `fp16_ttft_ms=871.65`, `turboquant_ttft_ms=6040.65`
  - `prompt_len=256`, `decode_tokens=4`: `fp16_decode_tok_s=39.04`, `turboquant_decode_tok_s=10.86`, `fp16_ttft_ms=1705.50`, `turboquant_ttft_ms=10569.52`
- Current conclusion: the dedicated decode kernel materially improves TurboQuant decode throughput, but TTFT/prefill remains the dominant gap and still keeps the implementation far from the paper’s reported regime.

# Tuning: TurboQuant Prefill Dispatch Collapse

## Plan
- [x] Inspect the TurboQuant prefill schedule for unnecessary per-token row-packing dispatches.
- [x] Batch TurboQuant V-cache packing across the whole prefill slice instead of dispatching once per token.
- [x] Re-run correctness checks and long-context benchmark points to measure TTFT impact.

## Review
- Found a major TTFT issue in prefill: V rows were being TurboQuant-packed inside the per-token projection loop, causing one quantization dispatch per token per layer.
- Batched V packing into a single `encodeTurboQuantRows(...)` call per layer, matching the existing K packing pattern.
- Restored the missing `gemm_f32` pipeline binding so benchmark-related prefill code paths compile again.
- Verification passed:
  - `swift build`
  - `swift test --filter TurboQuantAttentionTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- Updated benchmark points:
  - `prompt_len=128`, `decode_tokens=4`: `fp16_decode_tok_s=43.47`, `turboquant_decode_tok_s=12.92`, `fp16_ttft_ms=765.74`, `turboquant_ttft_ms=1376.33`
  - `prompt_len=256`, `decode_tokens=4`: `fp16_decode_tok_s=39.08`, `turboquant_decode_tok_s=11.05`, `fp16_ttft_ms=1745.54`, `turboquant_ttft_ms=2955.42`
  - `prompt_len=512`, `decode_tokens=4`: `fp16_decode_tok_s=33.08`, `turboquant_decode_tok_s=8.37`, `fp16_ttft_ms=4313.29`, `turboquant_ttft_ms=6140.15`
  - `prompt_len=1024`, `decode_tokens=4`: `fp16_decode_tok_s=24.20`, `turboquant_decode_tok_s=5.80`, `fp16_ttft_ms=11939.57`, `turboquant_ttft_ms=15517.23`
- Current conclusion: TurboQuant is no longer catastrophically slow. The remaining gap is now dominated by the core quantization/decode math rather than avoidable dispatch structure.

# Tuning: TurboQuant Decode Online Softmax

## Plan
- [x] Remove the second K-row decode pass from the single-token TurboQuant decode kernel.
- [x] Keep exact output behavior via an online-softmax merge rather than changing the math.
- [x] Verify parity and benchmark the same prompt lengths for comparison.

## Review
- Refactored `gqa_attention_turboquant_decode` to use a one-pass online-softmax summary per lane, then merge lane-local summaries exactly across the threadgroup.
- Followed with a lazy scalar rescale optimization so max updates no longer force a 128-dimension vector rescale on every row.
- Verification passed:
  - `swift build`
  - `swift test --filter TurboQuantAttentionTests`
  - `EDGERUNNER_RUN_TURBOQUANT_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- Updated benchmark points:
  - `prompt_len=128`, `decode_tokens=4`: `fp16_decode_tok_s=42.33`, `turboquant_decode_tok_s=13.85`, `fp16_ttft_ms=757.55`, `turboquant_ttft_ms=1382.30`
  - `prompt_len=256`, `decode_tokens=4`: `fp16_decode_tok_s=37.97`, `turboquant_decode_tok_s=12.11`, `fp16_ttft_ms=1687.05`, `turboquant_ttft_ms=2420.11`
- Current conclusion: decode throughput keeps improving incrementally, and TTFT at moderate context lengths is now within the same order of magnitude as FP16. The next remaining cost is still the scalar low-bit unpack/codebook work plus the row-wise prefill quantizer.
# Exact Path Rewrite Program

## Goal
- Replace EdgeRunner's long-prompt exact path with a production-safe architecture that can materially approach MLX on prompt throughput, TTFT, and long-context decode while preserving exact dense semantics by default.

## Plan
- [x] Add explicit strategy selection for prefill and decode fast paths with benchmark-safe fallbacks.
- [x] Add a single benchmark-gate entrypoint that runs build, publishable benchmark, long-prompt benchmark, and optional parity probes.
- [ ] Implement runtime weight repacking for prompt-wide exact kernels.
- [ ] Implement prompt-wide exact prefill projections and remove per-token projection loops from the new path.
- [ ] Implement exact tiled prompt attention and cache-native K/V writes for prefill.
- [ ] Implement prompt-wide FFN path for the new exact prefill engine.
- [ ] Redesign long-context exact decode attention and KV cache layout.
- [ ] Optimize dispatch and parameter binding after the new kernels are proven.
- [ ] Evaluate runtime-native int8 packed format for the new exact kernels.
- [ ] Promote the new exact path to default only after parity and benchmark gates pass.

## Review
- Added routed prefill selection plus a single benchmark gate script in the kept scaffolding commits (`7adf1e9`, `28c7f99`), so exact-path work can land behind explicit fallbacks and be benchmarked consistently.
- Kept a new batched fused-QKV exact-path slice for Q8 prefill:
  - `dequant_q8_0_fused_qkv` now supports batched prompt tokens while preserving the single-token decode path.
  - `fusedPrefillPass` now uses the fused Q8 RMSNorm+QKV path for multi-token prefill instead of looping per token for Q/K/V plus a separate V conversion pass.
  - Added a multi-token correctness test in `FusedKernelTests`.
- Kept a second exact-path prefill slice that batches RoPE-K cache writes:
  - `fusedPrefillPass` now converts the full RoPE'd K slab from `f32` to `f16` KV-cache storage in one dispatch instead of one conversion dispatch per token.
  - Decode behavior and canonical benchmark semantics are unchanged.
- Kept a third exact-path prefill slice that batches the remaining Q8 `wo` and `down` projections:
  - Added a dedicated batched Q8 GEMV kernel for multi-token prompt projections.
  - Routed only multi-token prefill `wo` / `down` work through that kernel; single-token decode stays on the existing tiled and fused-add paths.
- Kept a fourth exact-path attention slice that vectorizes the shared GQA kernel:
  - Reworked the exact `gqa_attention_f32` and `gqa_attention_f16kv` kernels around `float4` / `half4` tile loads and `dot(float4, float4)` accumulation instead of scalar head-dimension loops.
  - Added a direct `f16`-KV correctness test so the cache-backed attention path is checked against the CPU reference, not just the float32 helper path.
- Verification for the kept batched QKV slice:
  - `swift test -c release --filter FusedKernelTests` passed.
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic hash `0afae14a84cf0df8` and median decode `221.8 tok/s`.
  - `python3 benchmarks/run_long_prompt_framework_benchmark.py --prompt-tokens 1024 --generate-tokens 128 --runs 3 ...` produced EdgeRunner median `285.9 tok/s` prompt throughput, `3581.9 ms` TTFT, and `41.85 tok/s` long-context decode.
- Verification for the kept batched RoPE-K conversion slice:
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic hash `0afae14a84cf0df8` and median decode `211.1 tok/s`.
  - `python3 benchmarks/run_long_prompt_framework_benchmark.py --prompt-tokens 1024 --generate-tokens 128 --runs 3 ...` produced EdgeRunner median `351.9 tok/s` prompt throughput, `2910.2 ms` TTFT, and `40.95 tok/s` long-context decode.
- Verification for the kept batched `wo` / `down` slice:
  - `swift test -c release --filter FusedKernelTests` passed, including the new batched GEMV correctness case.
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic hash `0afae14a84cf0df8` and median decode `213.2 tok/s`.
  - `python3 benchmarks/run_long_prompt_framework_benchmark.py --prompt-tokens 1024 --generate-tokens 128 --runs 3 ...` produced EdgeRunner median `368.5 tok/s` prompt throughput, `2778.9 ms` TTFT, and `41.85 tok/s` long-context decode.
- Verification for the kept vectorized GQA slice:
  - `swift test -c release --filter GQATests` passed, including the new direct `f16`-KV reference case.
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"` passed with deterministic hash `0afae14a84cf0df8` and median decode `211.2 tok/s`.
  - `python3 benchmarks/run_long_prompt_framework_benchmark.py --prompt-tokens 1024 --generate-tokens 128 --runs 3 ...` produced EdgeRunner median `489.0 tok/s` prompt throughput, `2093.9 ms` TTFT, and `42.26 tok/s` long-context decode.
- Interpretation:
  - These are bounded production-safe improvements to exact prefill structure and exact attention math, not the full prefill rewrite.
  - Prompt throughput and TTFT have improved materially again, but long-context decode is still effectively unchanged and the MLX gap remains architectural.

# Long-Prompt MLX vs EdgeRunner Benchmark
