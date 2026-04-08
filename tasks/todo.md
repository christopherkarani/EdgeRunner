# TurboQuant Pinned Rollout Execution

## Current Mixed-Policy Execution Plan

- [x] Finish the per-layer KV runtime wiring in `LlamaLanguageModel` so `KVCache` layer storage kinds are respected during both prefill and decode
- [x] Keep the mixed-policy experiment explicit and reversible via `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS`; do not restore default short-context `q8_0` rails
- [x] Verify the new contract/env behavior with targeted TurboQuant tests before running pinned-model gates
- [x] Run pinned smoke and 128-token quality with `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=1`
- [x] If layer-0 mixed policy is still red, rerun smoke and 128-token quality with `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=2`
- [x] Only benchmark 4096-token performance for the best mixed-policy candidate that improves correctness
- [x] Record whether the mixed policy is a production-usable repo fallback or whether pure-Turbo remains the only acceptable objective and is still blocked

### Mixed-Policy Review

- `KVCache` already carried per-layer storage kinds, but `LlamaLanguageModel` prefill and decode still null-selected cache families as a global mode. That bug is now fixed, so mixed-layer experiments actually execute as configured.
- `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=1` changes the smoke output from the pure-Turbo collapse (`[16, 15, 15, 15]`) to `[21927, 11, 1246, 525]`, which confirms the override is live but still far from the Q8 baseline.
- `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=2` remains red on smoke with `turboquant_v2=[21927, 11, 323, 358]` vs `q8=[358, 2776, 264, 5458]`.
- `EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=2` also remains red on 128-token quality, but it improves the max logit delta relative to the pure path:
  - pure default: `max_abs_logit_delta=19.7134`
  - first-2-layers q8: `max_abs_logit_delta=16.8863`
- 4k benchmark for the best mixed candidate (`EDGERUNNER_TURBOQUANT_EARLY_Q8_LAYERS=2`) still shows no throughput win and a large TTFT penalty:
  - `q8_decode_tok_s=1.73`
  - `turboquant_v2_decode_tok_s=1.73`
  - `q8_ttft_ms=30279.64`
  - `turboquant_v2_ttft_ms=108502.17`
- Conclusion: the selective early-layer q8 policy is a valid diagnostic lever and slightly improves correctness, but it is not a production-usable fallback for this pinned rollout. Pure-Turbo remains blocked, and this mixed policy does not clear the repo gates either.

## Current Hybrid-KV Execution Plan

- [x] Add an explicit experiment override for `K=q8 / V=turbo` without changing the default TurboQuant contract
- [x] Audit the current runtime and Metal paths for existing hybrid support before writing new kernels or dispatch branches
- [x] Implement the narrowest viable hybrid path so keys use q8 storage/attention reads while values stay on TurboQuant storage/attention reads
- [x] Add or tighten tests that prove the hybrid contract and runtime selection are real, not just allocator-level
- [x] Run pinned smoke and 128-token quality on the hybrid path first
- [x] Only run the 4096 benchmark if the hybrid path materially improves correctness
- [x] Record whether hybrid K/V is a viable repo fallback or whether the pinned rollout is still blocked even after isolating V from the q8 promotion

### Hybrid-KV Review

- Added `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS` so the first `N` layers can store keys in `q8_0` while values stay on the active TurboQuant preset.
- Fixed the runtime so prefill and decode no longer assume that key and value storage must match globally or even per layer. The active path now supports `q8 key + turbo value` storage and dispatch.
- Added dedicated hybrid attention kernels for both prefill and single-token decode:
  - `gqa_attention_q8k_turboquant`
  - `gqa_attention_q8k_turboquant_decode`
- Added tests proving the contract and allocator really expose hybrid storage on layer 0 while later layers remain TurboQuant.
- Measured pinned rollout results:
  - `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=1`
    - smoke: `turboquant_v2=[16, 13, 576, 2701]`
    - quality: `divergence_steps=3`, `first_divergence_step=5`, `max_abs_logit_delta=15.6054`
    - generated: `[1479, 198, 3838, 374, 279, 3491, 382, 785]`
  - `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=2`
    - smoke: `turboquant_v2=[16, 13, 220, 17]`
    - quality: `divergence_steps=3`, `first_divergence_step=5`, `max_abs_logit_delta=11.8762`
    - generated: `[1479, 198, 3838, 374, 279, 897, 315, 279]`
    - 4k benchmark: `q8_decode_tok_s=1.73`, `turboquant_v2_decode_tok_s=1.78`, `q8_ttft_ms=32265.01`, `turboquant_v2_ttft_ms=107595.93`
    - layerwise attribution:
      - `first_divergent_attention_output_layer=2`
      - `first_divergent_attention_layer=2`
      - `first_divergent_layer=2`
      - `q8_argmax=3838`
      - `turboquant_v2_argmax=3838`
      - `max_abs_logit_delta=7.886554`
- Conclusion: `q8 key + turbo value` materially improves pinned-model quality relative to both pure Turbo and whole-layer q8 fallbacks, but it still fails smoke and quality gates and still carries a large TTFT penalty. It is not a production-usable fallback for this repo.

- [ ] Rebaseline the active branch against checkpoint `93a72e980d51e9ae4f0fbc6220856db6873aa681`, the dirty local worktree, and the fork sources to identify the exact remaining pure-`turbo3/turbo3` semantic gap
- [ ] Add or tighten failing targeted tests for that gap before changing runtime behavior
- [ ] Port the missing or regressed fork-aligned behavior into the default TurboQuant path without restoring `q8_0` shortcut rails
- [ ] Re-run targeted low-level TurboQuant Metal/reference/layerwise tests after each semantic change and keep only correctness-improving edits
- [ ] Re-run the pinned rollout gates on the active default path:
  - [ ] smoke
  - [ ] 128-token quality
  - [ ] 4096-token benchmark
- [ ] If the 4096-token gate is green, run repeated validation:
  - [ ] smoke x3
  - [ ] 128-token quality x3
  - [ ] 4096-token benchmark x3
  - [ ] 8192-token benchmark
  - [ ] 16384-token benchmark if supported
  - [ ] low-level TurboQuant Metal/reference suite
  - [ ] layerwise attribution rerun
- [ ] If Phase 1 and Phase 2 pass, assess repo production readiness and fork/paper comparability with measured evidence only
- [ ] If Phase 1 still fails after literal fork-port attempts, document the exact blocker with source-backed evidence and stop tuning

## TurboQuant Fork Semantics Audit

- [x] Pull the exact `TheTom/llama-cpp-turboquant` `feature/turboquant-kv-cache` source for the referenced KV/cache, graph, TurboQuant, CUDA, and quality-gate files
- [x] Extract exact pure-`turbo3/turbo3` semantics for KV encode/decode, `innerq scale_inv`, row metadata, graph/cache behavior, and quality gates
- [x] Compare those fork semantics against the active EdgeRunner TurboQuant KV/runtime/test path
- [x] Return the concrete parity requirements and any conditions that block exact parity in this repo

### TurboQuant Fork Audit Review

- Fork audit covered `src/llama-kv-cache.cpp`, `src/llama-graph.cpp`, `ggml/src/ggml-turbo-quant.c`, `ggml/src/ggml-cuda/turbo-quant.cuh`, `ggml/src/ggml-cuda/set-rows.cu`, `ggml/src/ggml-cuda/fattn-common.cuh`, and `scripts/turbo-quality-gate.sh` from `TheTom/llama-cpp-turboquant` branch `feature/turboquant-kv-cache`.
- Confirmed fork pure-`turbo3/turbo3` semantics are: uniform `turbo3` K/V by default, per-head zero-padding to 128 when needed, corrected-norm-only `turbo3` row metadata, graph-side Q pre-rotation plus FA-output inverse WHT, and `InnerQ` as a K-side calibration lifecycle that publishes `scale_inv` into graph-visible tensors after calibration finalizes.
- Confirmed EdgeRunner already matches some of the intended behavior at a high level: default `turbo3/turbo3`, `InnerQ` auto-calibration logic with skip-on-balanced channels, and corrected fixed-type metadata for `turbo3`.
- Confirmed exact fork parity is still blocked for some cases by design differences: EdgeRunner stores TurboQuant rows in split Metal buffers with extra metadata lanes instead of native `GGML_TYPE_TURBO3_0` blocks, uses a different runtime/graph backend than ggml CUDA, and hard-limits TurboQuant layout dimension to `<= 128` instead of the fork’s zero-padded larger-head support.

## Current Baseline

- Latest verified failing default-path rollout before this execution:
  - `q8=[358, 2776, 264, 5458]`
  - `turboquant_v2=[16, 220, 220, 220]`
- 128-token quality:
  - `divergence_steps=7`
  - `first_divergence_step=1`
  - `max_abs_logit_delta=23.3483`
  - `q8_generated=[1479, 198, 3838, 374, 279, 374, 279, 897]`
  - `turboquant_v2_generated=[1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479]`
- 4096-token benchmark:
  - `q8_decode_tok_s=1.74`
  - `turboquant_v2_decode_tok_s=1.87`
  - `q8_ttft_ms=34471.85`
  - `turboquant_v2_ttft_ms=123002.00`
- Attribution and prior source review indicate:
  - layer-0 attention inputs still match Q8 on real activations
  - first real divergence remains at the layer-0 attention output
  - dominant remaining error is key fidelity on real activations under the pure Turbo path
  - the branch contains user edits in active TurboQuant files, so regression isolation must work with the dirty tree rather than resetting files

## Current Review

- Relevant completed state carried into this execution:
  - default TurboQuant contract is already set to fork-aligned `turbo3/turbo3`
  - `turbo_innerq_scale_inv` lifecycle plumbing exists in Swift and Metal
  - fixed-type row metadata for `turbo2`/`turbo3`/`turbo4` was corrected to the fork/reference layout
  - explicit low-level `turbo3` attention and prefill Metal tests were repaired and passing
- Current execution focus:
  - confirm whether the remaining failure is a missing fork semantic, a regression relative to checkpoint `93a72e980d51e9ae4f0fbc6220856db6873aa681`, or a fundamental blocker in this repo’s pure-Turbo design
  - do not accept `q8_0` shortcut rails as the final state

## Review

- Current verification on the active default path after this execution:
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter turboQuantV2GreedyTraceMatchesQ8Baseline` fails with `q8=[358, 2776, 264, 5458]` and `turboquant_v2=[16, 15, 15, 15]`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --skip-build --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts` fails with `divergence_steps=7`, `first_divergence_step=1`, and `max_abs_logit_delta=19.7134`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_BENCHMARK=1 EDGERUNNER_TURBOQUANT_V2_BENCHMARK_PROMPT_LENS=4096 EDGERUNNER_TURBOQUANT_V2_BENCHMARK_DECODE_TOKENS=32 swift test --skip-build --filter compareTurboQuantV2AgainstQ8AcrossContexts` passes as a benchmark run but reports `q8_decode_tok_s=1.73`, `turboquant_v2_decode_tok_s=1.76`, `q8_ttft_ms=29274.93`, and `turboquant_v2_ttft_ms=102311.30`
- Additional low-level verification completed in this execution:
  - `groupedTurbo3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext` now covers `numHeads=16`, `numKVHeads=8`, `kvSeqLen=130`, and passes
  - default contract tests now assert `TurboQuantV2Contract.innerQ == .disabled` with no env override, matching the fork’s `TURBO_INNERQ` opt-in semantics
  - `compareStoredLayer0Turbo3RowsAgainstForkReferenceOnRealActivations` now rebuilds the stored layer-0 key rows from captured pinned-model activations and compares them against a literal fork-style `turbo3` row reference; it passes with:
    - `runtime_row_count=1040`
    - `worst_code_word_mismatch_count=0`
    - `worst_row_norm_max_abs_delta=0.00000000`
    - `worst_decoded_max_abs_delta=0.00009155`
  - layerwise attribution still shows exact attention inputs and first divergence at the layer-0 attention output
  - replay on real activations still shows the dominant error is decoded keys, not decoded values: `decoded_k_exact_v_mse=0.014278` vs `exact_k_decoded_v_mse=0.000383`
  - CPU and GPU replay of the same pure-Turbo attention contract still agree numerically (`cpu_vs_gpu_max_abs=0.000002`), so the live pinned failure is not explained by a decode-kernel mismatch
- Conclusion from the current pass:
  - the default state does not depend on the current `q8_0` shortcut rails for the measured failing path
  - Phase 1 remains red because smoke and 128-token quality both fail
  - the remaining blocker is now evidenced as pure-`turbo3` key fidelity on this pinned model under this repo’s implementation constraints, not a missing `InnerQ` default, not stale grouped-head Metal coverage, not a simple CPU/GPU kernel disagreement, and not a layer-0 stored-row encoding mismatch versus the fork-style `turbo3` contract

# Bonsai Path Rewrite

- [x] Capture Bonsai outputs from available local runtimes and metadata
- [x] Localize the first Bonsai divergence in EdgeRunner vs known-good behavior
- [x] Rewrite the Bonsai runtime path for correct output without sacrificing decode throughput
- [x] Re-run Bonsai parity, coherence, and benchmark tests
- [x] Update review notes with final results and residual risks

# Bonsai Throughput Rewrite

- [x] Remove Bonsai benchmark/tokenization assumptions that force BOS against model metadata
- [x] Evaluate the raw fused Q1 LM-head path against the current float fallback
- [x] Evaluate the existing fused Q1 QKV and Gate+Up kernels in the Bonsai decode path
- [x] Re-run Bonsai quality, parity, and throughput benchmarks and record the delta

## Review

- Stock `llama-cli` on this machine cannot load `Q1_0_g128` GGUF type `41`, so local validation stayed inside EdgeRunner plus GGUF metadata inspection.
- GGUF metadata confirms Bonsai disables automatic BOS insertion (`tokenizer.ggml.add_bos_token = false`) and uses a Qwen-style tokenizer contract in this artifact.
- Added [BonsaiParityHarnessTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BonsaiParityHarnessTest.swift) to keep `default`, `base`, and `base_no_mega` decode modes in parity and write `/tmp/bonsai_parity.json`.
- Added [BonsaiQualitySmokeTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BonsaiQualitySmokeTest.swift) to validate realistic prompt quality.
- Kept the raw Q1 layer decode path enabled in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift); disabling it regressed Bonsai throughput on this branch.
- Verified `swift build` passes.
- Verified `swift test -c release --filter "BonsaiParityHarnessTest/bonsaiParity"` passes.
- Verified `swift test -c release --filter "BonsaiQualitySmokeTest"` passes.
- Verified `swift test -c release --filter "BonsaiBenchmark/bonsaiEndToEndBenchmark"` reports `38.9 tok/s` median decode.
- `BonsaiBenchmark/bonsaiCoherenceCheck` still shows poor output on the pathological `[1]` and BOS-only probes, but realistic prompt generation is coherent and passes smoke checks.

## Review: Throughput Rewrite

- Updated [BonsaiBenchmark.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BonsaiBenchmark.swift) so Bonsai prompt tests no longer force a BOS token that the GGUF metadata explicitly disables.
- Updated [BonsaiParityHarnessTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BonsaiParityHarnessTest.swift) to trace parity on the same no-forced-BOS prompt path.
- Compared Bonsai decode variants against the current default:
  - `default`: about `38.5 tok/s`
  - `force base`: about `38.3 tok/s`
  - `force base + disable mega`: about `28.0 tok/s`
- Pulled the official [llama.cpp](https://github.com/ggml-org/llama.cpp) repo as reference. The relevant high-level pattern is already mirrored here: single-token decode, one-command-buffer-style execution, and fused decode kernels. The gap is not a missing top-level strategy; it is the quality of the Bonsai/Q1 kernels.
- Re-evaluated the existing Q1 fused decode kernels already present in the codebase:
  - fused Q1 LM head
  - fused Q1 QKV
  - fused Q1 Gate+Up
- Those Q1 fused kernels regressed end-to-end Bonsai decode from roughly `38.5-38.9 tok/s` down to about `12.8-13.1 tok/s`, so they were rolled back from the runtime path.
- Tested disabling the decode KV barrier for Bonsai. Under env override it remained coherent and parity-safe, but the baked-in version did not produce a stable enough default benchmark win, so it was not kept.
- Verified the restored runtime still passes:
  - `swift build`
  - `swift test -c release --filter "BonsaiQualitySmokeTest"`
  - `swift test -c release --filter "BonsaiParityHarnessTest/bonsaiParity"`
  - `swift test -c release --filter "BonsaiBenchmark/bonsaiCoherenceCheck"`
  - `swift test -c release --filter "BonsaiBenchmark/bonsaiEndToEndBenchmark"`
- Current verified state after the rollback:
  - realistic prompt quality remains coherent
  - parity harness passes
  - `bonsaiCoherenceCheck` now shows the realistic prompt path producing `Paris...` without forced BOS
  - benchmark median decode is `38.5 tok/s`
- Current blocker: the available Q1 fused shaders are not competitive enough to close the gap to the user target of `250 tok/s`. Beating that target will require new higher-performance Q1 kernels or a different model/runtime strategy, not just wiring the existing dormant Q1 shaders into the decode path.

# Bonsai-First Rewrite Plan

- [x] Freeze current generic llama-compatible runtime as the compatibility path, not the Bonsai performance path
- [x] Build a dedicated Bonsai runtime entry point with its own loader routing
- [ ] Rebuild Bonsai decode around a strict two-phase engine: prefill engine and single-token decode engine
- [ ] Keep all Bonsai-critical tensors resident on GPU and eliminate generic float materialization in the hot path
- [ ] Replace the current raw-Q1 fallback kernels with a new Bonsai Q1 kernel suite designed for decode throughput
- [ ] Add per-stage Bonsai profiling so every optimization shows where time moved
- [ ] Re-earn quality parity on realistic prompts after each stage before chasing benchmark wins
- [ ] Only then optimize against the 250 tok/s target and compare against MLX / llama.cpp

## Plan Notes

- `llama.cpp` and MLX both treat decode as a first-class execution mode, not as a lightly modified prefill path.
- The key lesson from source review is not a hidden trick; it is architecture:
  - `llama.cpp` separates batch decode/prefill work, uses explicit memory management around KV/cache state, and treats decode as a specialized execution path.
  - MLX presents the same idea at a higher level: cache-aware decode, explicit RoPE offset handling, and model execution structured around cache reuse instead of generic recomputation.
- EdgeRunner already imitates some of that shape, but the Bonsai path still depends on a generic llama runtime with weak Q1 kernels. That is the structural mismatch.

## Proposed Rewrite

- Runtime split:
  - Keep the existing [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) as a compatibility backend for Qwen and mixed quantizations.
  - Add a separate Bonsai-first backend whose only job is Bonsai/Q1 performance.
- Loader split:
  - Parse Bonsai tensors into a backend-specific packed representation instead of forcing them through the current generic weight abstractions.
  - Stop assuming the Bonsai artifact should share the same tensor pipeline as Q8/Q4 models.
- Execution split:
  - Prefill path: optimize for throughput on `N > 1` tokens with batch-oriented kernels and packed weights.
  - Decode path: optimize for `N == 1` only, with a static command graph, persistent argument/state buffers, and no generic branches.
- Kernel strategy:
  - Throw away the current dormant Q1 fused kernels as the baseline. They are not competitive.
  - Build a new Bonsai Q1 decode kernel suite around the real hot projections:
    - fused QKV
    - RoPE + QK norm + attention
    - Wo + residual
    - Gate/Up + activation
    - Down + residual
    - final norm + LM head
  - Design those kernels for the exact Bonsai dimensions and memory layout instead of trying to remain quantization-generic.
- Sampling/output strategy:
  - Move top-1 / argmax and non-finite validation off the generic CPU scan path when possible.
  - Treat “produce next token” as part of the decode engine, not as an afterthought on a full vocab buffer readback.
- Verification strategy:
  - Keep the realistic-prompt smoke tests and parity harness as hard gates.
  - Add stage timings for:
    - embedding lookup
    - per-layer attention stack
    - FFN stack
    - LM head
    - CPU-side token selection
  - Do not keep any optimization that does not move end-to-end median decode tok/s on Bonsai.

## Architecture Decision

- The fastest path to a `250 tok/s` attempt is no longer “improve the generic implementation.”
- The fastest path is “treat Bonsai as its own backend and optimize for that model family directly,” even if that means the Qwen/general path and Bonsai path diverge substantially.

## Implementation Status

- Added [BonsaiLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/BonsaiLanguageModel.swift) as a dedicated Bonsai backend type that currently wraps the proven llama-compatible runtime while isolating Bonsai-specific loading and policy.
- Updated [ModelLoader.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/ModelLoader.swift) to route Bonsai-family GGUFs to the Bonsai backend instead of the generic loader path.
- Removed app-level forced BOS insertion from:
  - [EdgeRunnerFacade.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/EdgeRunnerFacade.swift)
  - [GenerationSession.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Streaming/GenerationSession.swift)
- This is a production-facing Bonsai correctness fix because Bonsai disables automatic BOS in GGUF metadata.
- Verified:
  - `swift build`
  - `swift test -c release --filter "ModelLoaderTests"`
  - `swift test -c release --filter "BonsaiQualitySmokeTest"`
  - `swift test -c release --filter "BonsaiParityHarnessTest/bonsaiParity"`
  - `swift test -c release --filter "BonsaiBenchmark/bonsaiEndToEndBenchmark"`
- Current measured Bonsai benchmark after the backend split remains in the same range, with a median decode of `36.8 tok/s`.

# Current Iteration

- [x] Record the explicit rule from user feedback: do not stop at architecture checkpoints while the Bonsai target is unresolved
- [ ] Rebaseline the current dedicated Bonsai backend on the local machine
- [ ] Profile the next decode hot path inside the active Bonsai runtime instead of planning another architectural checkpoint
- [ ] Keep only changes that improve `BonsaiBenchmark/bonsaiEndToEndBenchmark` while preserving `BonsaiQualitySmokeTest` and `BonsaiParityHarnessTest`

## Current Iteration Review

- Re-verified the current Bonsai-safe baseline after reverting a failed Bonsai-only synchronous decode shortcut.
- Verified:
  - `swift test -c release --filter "BonsaiQualitySmokeTest"`
  - `swift test -c release --filter "BonsaiParityHarnessTest/bonsaiParity"`
  - `swift test -c release --filter "BonsaiBenchmark/bonsaiEndToEndBenchmark"`
  - `swift test -c release --filter "PublishableBenchmark/fullBenchmark"`
- Current verified numbers:
  - Bonsai median decode: about `37.0 tok/s`
  - Bonsai LM head profile: about `1.681 ms`
  - Qwen publishable benchmark remains functional and deterministic, with recent reruns at `210.3 tok/s` and `218.4 tok/s` median decode on this machine.
- Resulting conclusion:
  - the failed sync shortcut should stay out
  - the next meaningful Bonsai rewrite target is the Q1 per-layer decode stack, not the LM head

## Current Profiling Detail

- Added a Bonsai Q1 projection micro-profile to [BonsaiBenchmark.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BonsaiBenchmark.swift).
- Verified `swift test -c release --filter "BonsaiBenchmark/bonsaiQ1ProjectionProfile"` prints:
  - `q1_wq_ms ≈ 0.261`
  - `q1_wk_ms ≈ 0.203`
  - `q1_wv_ms ≈ 0.213`
  - `q1_wo_ms ≈ 0.295`
  - `q1_gate_ms ≈ 0.261`
  - `q1_up_ms ≈ 0.248`
  - `q1_down_ms ≈ 0.276`
- The selective Q1 fused QKV experiment and the wider-row Q1 GEMV experiment were both rolled back after regressions.
- Current direction remains unchanged:
  - protect Qwen
  - keep Bonsai on the verified raw-Q1 path
  - target the Q1 projection family with better kernel design rather than more unsafe routing changes

# TurboQuant Upstream Research

- [x] Verify whether upstream `llama.cpp` ships a TurboQuant implementation
- [x] Compare upstream KV-cache quantization/rotation design against EdgeRunner's TurboQuant path
- [x] Review local TurboQuant code, tests, and rollout assumptions for likely root causes
- [x] Record findings and concrete follow-up direction

## Review

- Upstream `llama.cpp` does not currently ship a first-party `TurboQuant` implementation. The upstream feature surface is KV-cache dtype selection (`--cache-type-k`, `--cache-type-v`) plus attention-rotation support for quantized KV caches.
- Relevant upstream references:
  - `include/llama.h` exposes experimental `type_k` / `type_v` context params.
  - `common/arg.cpp` limits cache-type CLI choices to standard ggml types such as `F16`, `Q8_0`, `Q4_0`, `Q4_1`, `IQ4_NL`, `Q5_0`, `Q5_1`.
  - `src/llama-kv-cache.cpp` enables attention rotation only for quantized K/V caches and precomputes Hadamard matrices.
  - `src/llama-context.cpp` rejects quantized V-cache without flash attention.
- EdgeRunner's implementation is not a port of upstream `llama.cpp`; it is a custom paper-inspired codec and attention path built around:
  - fixed `headDim == 128`
  - hard-coded preset descriptors
  - custom 2/3/5-bit centroid tables
  - randomized Hadamard transforms with fixed seeds
  - sign-only residual projection reconstruction
- High-confidence gaps found in the current implementation:
  - The `balanced` preset is explicitly an engineering guess, not a spec-backed implementation.
  - The codec is constrained to `headDim == 128`, while upstream rotation logic is dimension-generic across powers of two / divisible blocks.
  - EdgeRunner allocates dense FP16 shadow K/V caches even when TurboQuant is enabled, which erodes the main memory-saving advantage and creates a hybrid design upstream does not depend on.
  - The decode attention kernels fully decode split-plane codes and sign residuals inside the attention loop, which is much heavier than upstream's simpler quantized KV storage strategy.
  - Quality tests are narrow: smoke coverage is a 3-token prompt with a fixed 4-token expectation, and the longer harness checks divergence/logit deltas but does not prove broad prompt robustness.
- Working conclusion:
  - The main issue is not that EdgeRunner failed to match a hidden upstream `llama.cpp` TurboQuant implementation; there is no such upstream implementation to match.
  - The bigger issue is that EdgeRunner treats a speculative, paper-derived codec as if it were an upstream-validated KV-cache quantization path.
  - If the goal is upstream-like reliability, the comparison target should be `llama.cpp`'s ordinary quantized KV cache + attention-rotation path, not the current custom TurboQuant codec.

# KV Cache Quantization Replacement

- [x] Replace public/runtime TurboQuant selection with standard KV cache quantization modes
- [x] Implement `Q8_0` KV cache row storage and direct attention consumption without dense shadow caches
- [x] Keep KV rotation as a separate optional concern and leave it disabled in the first replacement cut
- [x] Replace TurboQuant-specific smoke/quality benchmarks with broader parity and throughput harnesses
- [x] Verify build plus targeted parity / benchmark suites

## Implementation Spec

- First cut scope:
  - support `disabled`, `automatic`, and `q8_0` KV cache compression at runtime
  - make `automatic` resolve to `q8_0` behind the existing rollout gate
  - remove runtime dependence on TurboQuant split-plane codecs and dense FP16 shadow K/V caches
  - do not ship attention rotation in the first replacement cut; keep the architecture open for it
- Storage format:
  - one `Q8_0` row per `(token, kvHead)` with `headDim / 32` blocks
  - metadata per block is the standard `Q8_0` scale + 32 int8 values
  - `headDim` must be divisible by 32 for `q8_0`
- Attention path:
  - prefill and decode attention must consume quantized K/V directly
  - no FP16 shadow cache allowed in the quantized runtime path
  - start with `Q8_0` direct kernels only; defer `Q4_0` / `IQ4_NL`
- Validation:
  - parity harness across multiple prompts
  - short and long context runs
  - token-level divergence tracking
  - benchmark output at 4k, 8k, and 16k context targets when the model/context fit

## Review

- Replaced the public/runtime KV compression selection in [ModelConfiguration.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/ModelConfiguration.swift) with `disabled`, `automatic`, and `q8_0`; `automatic` now resolves to `q8_0` behind the existing rollout gate.
- Added standard `Q8_0` KV row storage in [KVCache.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/KVCache.swift) plus a decoded retrieval API for tests.
- Added row quantization and direct attention consumption for quantized KV in:
  - [Dequant_Q8_0.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_0.metal)
  - [GQA.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/GQA.metal)
  - [GQAKernel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/GQAKernel.swift)
- Updated [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) so prefill and decode now write/read `Q8_0` KV caches directly instead of depending on the previous TurboQuant runtime path.
- Replaced the narrow TurboQuant-oriented validation with Q8-focused harnesses:
  - [QwenTurboQuantSmokeTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenTurboQuantSmokeTest.swift)
  - [TurboQuantQualityHarnessTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantQualityHarnessTest.swift)
  - [TurboQuantLongContextBenchmark.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLongContextBenchmark.swift)
- Updated config and KV-cache unit coverage in:
  - [EdgeRunnerLanguageModelTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/EdgeRunnerLanguageModelTests.swift)
  - [KVCacheTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/KVCacheTests.swift)
- Verified:
  - `swift build`
  - `swift test --skip-build --filter q8AppendAndDecodeRoundTrip`
  - `swift test --skip-build --filter modelConfiguration`
  - `swift test --filter "(EdgeRunnerLanguageModel Protocol|KV Cache Ring Buffer)"` to compile the full test graph after the replacements
- Residual risk:
  - TurboQuant code still exists in the tree and some metal-only TurboQuant tests remain for now, but the main runtime path has been switched over to standard `Q8_0` KV storage.
  - The new long-context and quality harnesses are env-gated and were compiled but not executed against the pinned Qwen artifact in this pass.

# TurboQuant V2 Production Track

- [x] Keep `q8_0` as the reference and fallback KV path
- [x] Reintroduce TurboQuant as explicit `KVCacheCompression.turboquantV2`
- [x] Remove dense shadow-cache behavior from the TurboQuant runtime path
- [x] Define the TurboQuant V2 on-device row contract explicitly in code
- [x] Add parity and benchmark gates comparing `turboquantV2` against `q8_0`
- [ ] Only consider `automatic` promotion after TurboQuant V2 clears those gates

## Review: TurboQuant V2 Production Track

- Runtime split is now explicit: `automatic` still resolves to `q8_0`, `turboquantV2` is opt-in, and `LlamaLanguageModel` routes `q8_0` and `turboquantV2` through separate KV encode / attention paths.
- TurboQuant no longer allocates or depends on dense FP16 shadow K/V caches. The active TurboQuant path writes only the TurboQuant row buffers and consumes them directly in attention.
- The on-device TurboQuant V2 contract is now defined in code via `TurboQuantV2Contract` and `ERKVPrecisionTurboQuantV2`.
- Added gated rollout harnesses that compare `turboquantV2` against `q8_0` for smoke, greedy-trace quality, and long-context throughput.
- Verified locally with `swift build`, `swift test --filter "modelConfiguration"`, and `swift test --filter "(q8AppendAndDecodeRoundTrip|turboQuantV2UsesTurboBufferAPI|modelConfigurationTurboQuantV2OptIn)"`.
- Pinned-model gates on `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf` show the path is not rollout-safe yet:
  - smoke: `q8=[358, 2776, 264, 5458]`, `turboquant_v2=[16, 11, 220, 508]`
  - quality at prompt length `128`: divergence begins at decode step `2`, `6/8` divergent steps, `max_abs_logit_delta=17.8625`
  - benchmark at prompt length `4096`, decode `32`: `q8=1.73 tok/s`, `turboquant_v2=6.97 tok/s`, but `q8 TTFT=31.34s` vs `turboquant_v2 TTFT=68.96s`
- Added extra diagnostics to narrow the blocker:
  - multi-token TurboQuant attention matches decoded CPU reference in isolation
  - GPU TurboQuant row quantization decodes like the reference runtime row
  - existing attention diagnostic still indicates value reconstruction is the dominant approximation error
- A follow-up balanced-preset experiment made correctness worse and was rolled back.
- Current blocker to production rollout: end-to-end model quality still diverges sharply from `q8_0`, so `automatic` must remain on `q8_0`.
- User direction is to stay all-in on TurboQuant. The next iteration should focus on layerwise / stagewise attribution inside the all-in TurboQuant path instead of pivoting to hybrid K/V storage.

## Implementation Spec

- Runtime policy:
  - `automatic` remains `q8_0`
  - `q8_0` is the production fallback and comparison target
  - `turboquantV2` is explicit opt-in only
- TurboQuant V2 scope:
  - one explicit row contract
  - no guessed balanced preset in the runtime path
  - no FP16 dense shadow K/V caches
  - direct encode/decode + attention consumption only
- Validation gates:
  - smoke parity against `q8_0`
  - greedy token divergence tracking
  - max logit-delta thresholds
  - long-context throughput runs at 4k, 8k, 16k
- Next debugging track:
  - [x] add layerwise hidden-state / logits attribution between `q8_0` and `turboquantV2`
  - [x] identify the first materially divergent transformer stage in the all-in TurboQuant path
  - [ ] bisect the layer-0 TurboQuant attention path itself: KV row encode, attention score path, and value reconstruction

## Review: Layerwise Attribution

- Added debug-only trace capture in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) for:
  - post-attention residual states per layer
  - final layer output states per layer
  - final logits
- Added [TurboQuantLayerwiseAttributionTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLayerwiseAttributionTest.swift) to compare `turboquantV2` against `q8_0` on the pinned Qwen artifact with guided decode tokens from the `q8_0` baseline.
- Verified:
  - `swift build`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
- Current diagnostic result on `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`:
  - prompt length `128`, guided decode steps `2`, guided tokens `[1479, 198]`
  - first divergent attention-residual layer: `0` with max abs delta `1.107820`
  - first divergent final layer-output layer: `0` with max abs delta `1.728936`
  - largest attention-residual delta: layer `27`, `63.537926`
  - largest final layer-output delta: layer `27`, `114.131821`
  - final argmax mismatch remains severe: `q8=3838`, `turboquant_v2=13`
  - max abs logit delta: `17.214630`
- Working conclusion:
  - the quality loss is already present at the post-attention residual of layer `0`
  - this is not primarily a late-layer accumulation issue
  - the next correction loop should focus inside the TurboQuant attention path itself rather than on FFN or downstream transformer plumbing
- Re-ran the existing low-level attention attribution diagnostic with `swift test --filter scoreErrorVsValueErrorAttribution`:
  - `exact_k_decoded_v_mse=0.010392`
  - `decoded_k_exact_v_mse=0.000022`
- Current best hypothesis for the next all-in TurboQuant fix:
  - value reconstruction is still the dominant approximation error
  - the next kernel/design iteration should target TurboQuant's value path first while keeping the format all-in and direct-use
- New value-path iteration:
  - changed TurboQuant high-precision outlier selection for `V` rows from absolute rotated magnitude to quantization-benefit ranking, while keeping `K` rows on the original magnitude-based selection
  - added a regression threshold in [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) for decoded-value attribution
  - added a value residual-scale sweep diagnostic in [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift)
- Verified after the V-only outlier-selection change:
  - `swift test --filter scoreErrorVsValueErrorAttribution`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- Current measured state after the kept change:
  - decoded-value MSE improved from `0.010392` to `0.008502`
  - layerwise max abs logit delta improved from `17.214630` to `16.047461`
  - largest layer-output delta improved from `114.131821` to `85.877121`
  - smoke still fails badly: `q8=[358, 2776, 264, 5458]`, `turboquant_v2=[15, 271, 271, 0]`
- Value residual-weight experiment:
  - diagnostic sweep found best synthetic value-row MSE near residual scale `0.65`
  - applying that scale in the production decode path improved isolated value-row MSE further to `0.006725`
  - but it regressed pinned-model layerwise and smoke behavior, so it was rolled back
- Current conclusion:
  - V-path quantization helped, but it is not the whole failure surface
  - rowwise reconstruction improvements are not enough unless they also preserve first-layer attention behavior on the real model
  - the next iteration should compare the layer-0 attention output against both a dense-V TurboQuant-K reference and a fully dense attention reference
- Added two new layerwise attention controls in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift):
  - TurboQuant-K + dense-V reference attention output for the traced last token
  - fully dense K/V attention output for the traced last token
- Extended [TurboQuantLayerwiseAttributionTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLayerwiseAttributionTest.swift) to print:
  - `q8_0` vs TurboQuant attention-output deltas
  - `q8_0` vs TurboQuant-K+dense-V attention-output deltas
  - `q8_0` vs fully dense attention-output deltas
  - TurboQuant vs fully dense attention-output deltas
- Verified:
  - `swift build`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
- New diagnostic result on `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`:
  - `q8_0` vs TurboQuant attention output still diverges immediately at layer `0` with max abs delta `1.062994`
  - `q8_0` vs TurboQuant-K+dense-V attention output is actually worse at layer `0` with max abs delta `1.586205`
  - `q8_0` vs fully dense attention output does not diverge until layer `1`, and layer-0 max abs delta stays below the threshold at `0.354943`
  - TurboQuant vs fully dense attention output still diverges immediately at layer `0` with max abs delta `1.067834`
- Updated conclusion:
  - the pinned-model attention failure is not primarily explained by TurboQuant V reconstruction alone
  - `q8_0` remains close to the exact dense attention control, so the benchmark artifact itself is not the source of the layer-0 jump
  - the next all-in TurboQuant optimization target should move upstream to the K/score side: key row encoding fidelity, score accumulation, and the randomized-Hadamard score path
- K-side experiments in this pass:
  - tried quantization-benefit high-precision channel selection for `K` rows in the live path and reference path
  - rolled it back after the pinned layerwise gate regressed badly:
    - `largest_attention_output_layer_delta` worsened from `32.269547` to `45.379787`
    - `largest_layer_delta` worsened from `85.877121` to `173.710876`
    - `max_abs_logit_delta` worsened from `16.047461` to `17.971104`
  - added a low-level key-score attribution refinement in [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) that measures:
    - TurboQuant score path with exact values
    - exact scores with decoded values
    - full TurboQuant score + decoded values
  - current synthetic output MSE ordering:
    - `decoded_k_exact_v_mse=0.000022`
    - `turbo_score_exact_v_mse=0.005723`
    - `exact_k_decoded_v_mse=0.008502`
    - `turbo_score_decoded_v_mse=0.023584`
  - added a key residual-scale sweep diagnostic:
    - `best_scale=0.00`
    - `best_mse=0.005723`
  - tried a production runtime experiment that zeroed the key residual term in the TurboQuant score path
  - rolled it back after pinned-model validation failed to improve the real behavior:
    - layerwise `max_abs_logit_delta` worsened to `16.223883`
    - smoke still failed, with `turboquant_v2=[4791, 374, 198, 11]`
- Current conclusion after the K-side experiments:
  - simplistic K-side heuristics can easily make the real model worse even when the synthetic score probe looks cleaner
  - the next K/score iteration should target the score path more structurally, not by blindly zeroing or re-ranking one term
- Added a dedicated attention-score trace path in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) that:
  - captures the last-token RoPE/normalized query per layer
  - captures the exact dense key rows per layer for the traced prompt
  - compares those exact scores and softmax weights against the stored cache representation (`q8_0` decoded keys or TurboQuant runtime rows)
- Added `KVCache.retrieveTurboQuantRuntimeRows` in [KVCache.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/KVCache.swift) and `TurboQuantReferenceEncoder.approximateScore` in [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) so score attribution uses the real TurboQuant score formula instead of a decoded-key proxy.
- Added [TurboQuantLayerwiseAttributionTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLayerwiseAttributionTest.swift) coverage for `compareAttentionScoreTraceAgainstQ8Baseline`.
- Verified:
  - `swift build`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter "(compareAttentionScoreTraceAgainstQ8Baseline|compareLayerwiseTraceAgainstQ8Baseline)"`
- New score-trace result on `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf` with prompt length `128`:
  - `q8_0` already diverges in raw score space at layer `0`, but only modestly:
    - `first_q8_score_divergent_layer=0`
    - `first_q8_score_divergent_layer_max_abs_delta=2.174943`
    - `first_q8_softmax_divergent_layer=0`
    - `first_q8_softmax_divergent_layer_max_abs_delta=0.054071`
  - TurboQuant score drift is catastrophic at layer `0`:
    - `first_turbo_score_divergent_layer=0`
    - `first_turbo_score_divergent_layer_max_abs_delta=7893.624023`
    - `first_turbo_softmax_divergent_layer=0`
    - `first_turbo_softmax_divergent_layer_max_abs_delta=1.000000`
  - Worst layers stay at layer `0` for both score and softmax in this trace.
- Updated conclusion:
  - the main TurboQuant failure is now directly localized to score generation at layer `0`, not just downstream attention outputs
  - `q8_0` score-space error is tolerable because its softmax distortion stays small
  - TurboQuant is saturating score/softmax at layer `0`, so the next target is the score path’s normalization and accumulation contract, not more hidden-state or value-path guessing
- Added a debug-only raw score-term export for the TurboQuant decode score path:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now exposes `TurboQuantScoreTerms` and `TurboQuantReferenceEncoder.approximateScoreTerms(...)`
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) now has `turboquant_debug_decode_score_terms`
  - [TurboQuantKernel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuantKernel.swift) now exposes `debugDecodeScoreTermsPipeline`
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) now verifies GPU-vs-CPU score terms directly
- Verified:
  - `swift build`
  - `swift test --filter decodeScoreTermsMatchCPUReference`
  - `swift test --filter scoreErrorVsValueErrorAttribution`
- New low-level score-term result:
  - `max_mse_dot_delta=0.00006104`
  - `max_residual_dot_delta=0.00000000`
  - `max_score_delta=0.00000000`
- Updated conclusion:
  - the Metal TurboQuant decode kernel is matching the CPU TurboQuant score formula essentially exactly on the diagnostic slice
  - the current layer-0 score blow-up is therefore not explained by a GPU-vs-CPU kernel implementation mismatch
  - the next all-in TurboQuant target is the score formula/normalization contract itself: centroid scaling, Hadamard-space normalization, residual contribution scaling, and score-range control before softmax
- Additional verification note:
  - rerunning the existing env-gated layerwise attribution harnesses now hits `MTLCommandBufferErrorDomain Code=9` / Metal validation `buffer is not a MTLBuffer`
  - this showed up in:
    - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
    - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareAttentionScoreTraceAgainstQ8Baseline`
  - the new low-level score-term test still passes, so the immediate diagnostic conclusion above remains valid
- Promoted a structural score-contract fix:
  - TurboQuant key scoring now applies the missing Hadamard inner-product normalization (`1 / headDim`) in both the CPU reference and Metal runtime score path
  - the key residual scale is now runtime-configurable through `EDGERUNNER_TURBOQUANT_KEY_RESIDUAL_SCALE` instead of being locked into the shader binary
- Verified:
  - `swift build`
  - `swift test --filter "(decodeScoreTermsMatchCPUReference|scoreErrorVsValueErrorAttribution)"`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareAttentionScoreTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- Results after the normalization fix:
  - synthetic `turbo_score_exact_v_mse` improved from `0.005723` to `0.000022`
  - pinned layer-0 TurboQuant score max abs delta improved from `7893.624023` to `51.721783`
  - pinned layer-0 TurboQuant softmax max abs delta improved from `1.000000` to `0.993791`
  - smoke still fails, but the generated tokens changed again: `turboquant_v2=[15, 13, 15, 15]`
- Added a decoded-TurboQuant-key control in the score trace:
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) now compares exact scores/softmax against both:
    - TurboQuant sign-sketch scores on runtime rows
    - direct dot-product scores on decoded TurboQuant keys
  - [TurboQuantLayerwiseAttributionTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLayerwiseAttributionTest.swift) now prints those decoded-key metrics
- New decoded-key result on the pinned Qwen artifact:
  - decoded-key score error is effectively identical to TurboQuant score-formula error:
    - `first_turbo_decoded_score_divergent_layer_max_abs_delta=51.721771`
    - `worst_turbo_decoded_score_layer_mse=100.061272`
    - `first_turbo_decoded_softmax_divergent_layer_max_abs_delta=0.993791`
  - this means the remaining real-model failure is now localized to key representation fidelity, not to the sign-sketch score formula layered on top
- Residual-scale sweeps after the normalization fix:
  - runtime override sweep on the smoke gate:
    - `scale=0.0` → `turboquant_v2=[4791, 4337, 4337, 4337]`
    - `scale=0.5` → `turboquant_v2=[16, 15, 15, 15]`
    - `scale=1.0` → `turboquant_v2=[15, 13, 15, 15]`
  - none of the tested residual-scale overrides restored smoke parity
- Retried key outlier selection = `quantizationBenefit` after the normalization fix:
  - synthetic diagnostics stayed fine, but pinned score-trace and smoke behavior did not improve
  - rolled back to `keyOutlierSelection = .magnitude`
- Updated conclusion:
  - the missing Hadamard normalization was a real bug and fixing it materially improved the score path
  - the dominant remaining issue is now the decoded TurboQuant key representation itself
  - the next key-path experiment should target how key rows are encoded/reconstructed, not more score-formula scaling tweaks
- Promoted a structural K/V-layout split for TurboQuant V2:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now defines separate `keyPreset` / `valuePreset` and separate layout builders instead of a single shared preset
  - [KVCache.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/KVCache.swift) now allocates, writes, reads, and decodes TurboQuant keys and values with separate layouts/presets
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) now threads separate key/value layouts through TurboQuant row encoding and the generic attention path
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) and the mirrored test structs now carry distinct value row-width / bit-width fields while keeping the aggressive fast paths restricted to the all-aggressive case
- Verified after the split-layout refactor:
  - `swift build`
  - `swift test --filter "(q8AppendAndDecodeRoundTrip|modelConfigurationTurboQuantV2OptIn|decodeScoreTermsMatchCPUReference)"`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareAttentionScoreTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
- New pinned-model result with `K=balanced`, `V=aggressive`:
  - score-trace moves exactly to the CPU-predicted balanced-key regime:
    - `first_turbo_score_divergent_layer_max_abs_delta=28.023239`
    - `worst_turbo_score_layer_mse=45.174301`
    - `first_turbo_softmax_divergent_layer_max_abs_delta=0.998174`
    - decoded-key metrics still match the stored-score metrics, so key representation remains the dominant source of score drift
  - hidden-state / logits also improve versus the prior mixed run:
    - `first_divergent_attention_output_layer_max_abs_delta=1.644536` (down from `1.891966`)
    - `largest_attention_output_layer_delta=17.062605`
    - `max_abs_logit_delta=15.377005` (down from `16.047461`)
  - smoke still fails, but the trace is no longer stuck in the low-token collapse regime:
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[20, 15, 15, 271]`
- Follow-up runtime experiment with `K=balanced`, `V=balanced`:
  - kept in the current tree because the hidden-state/logit metrics improved again, even though smoke is still failing
  - current pinned layerwise result:
    - `first_divergent_attention_output_layer_max_abs_delta=1.604011`
    - `largest_attention_output_layer_delta=16.916191`
    - `max_abs_logit_delta=15.228952`
  - current smoke result:
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[549, 220, 220, 220]`
- Updated conclusion for the current iteration:
  - splitting TurboQuant into separate K/V layouts was the right structural move; it materially improved the real-model score trace and downstream layerwise drift
  - moving keys to `balanced` clearly helped more than any score-formula tuning did
  - balanced values also improved layerwise/logit metrics, but not enough to clear smoke
  - the next TurboQuant correction loop should stay on representation fidelity, now with the K/V split in place, rather than revisiting the old single-layout contract
- Added preset sweep support for higher-fidelity key layouts:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now defines `balanced64`, `balanced96`, and `aggressive64` presets and computes `effectiveBits` from the descriptor instead of storing a guessed label
  - [TurboQuantLayerwiseAttributionTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/TurboQuantLayerwiseAttributionTest.swift) now sweeps those presets against exact captured activations to rank real-model score/softmax fidelity before changing the runtime
- Verified the best practical key-side promotion from that sweep:
  - exact-activation sweep favored `K=balanced64` over `K=balanced` on the real captured activations:
    - `worst_score_mse=22.477882` vs `45.174294`
    - `worst_softmax_max_abs_delta=0.929969` vs `0.998174`
  - promoting `V=balanced64` regressed the live model, so the current best runtime state is:
    - `K=balanced64`
    - `V=balanced`
- Restored the runtime to that best live state and removed temporary TurboQuant print probes from [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift).
- Re-verified the pinned-model baseline after that cleanup:
  - `swift build`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareAttentionScoreTraceAgainstQ8Baseline`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter compareLayerwiseTraceAgainstQ8Baseline`
  - restored pinned metrics:
    - smoke still fails: `q8=[358, 2776, 264, 5458]`, `turboquant_v2=[16, 15, 15, 15]`
    - score trace: `first_turbo_score_divergent_layer_max_abs_delta=29.352373`, `worst_turbo_score_layer_mse=22.477894`, `first_turbo_softmax_divergent_layer_max_abs_delta=0.929968`
    - layerwise: `first_divergent_attention_output_layer_max_abs_delta=1.059889`, `largest_attention_output_layer_delta=31.279221`, `max_abs_logit_delta=12.831061`
- Updated the low-level validation surface to match the real runtime contract instead of the obsolete all-aggressive test contract:
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) now derives attention params, quantization params, cache precision, and CPU references from `TurboQuantV2Contract`
  - added grouped-query coverage with `numHeads > numKVHeads` to exercise the same GQA pattern used by Qwen-family models
- Verified the contract-aligned metal tests:
  - `swift test --filter attentionMatchesDecodedCPUReference`
  - `swift test --filter decodeAttentionMatchesDecodedCPUReference`
  - `swift test --filter prefillAttentionMatchesDecodedCPUReference`
  - `swift test --filter gpuQuantizedRowsDecodeLikeReferenceRuntimeRows`
  - `swift test --filter "(scoreErrorVsValueErrorAttribution|decodeScoreTermsMatchCPUReference)"`
  - `swift test --filter groupedDecodeAttentionMatchesDecodedCPUReference`
- Verified the broader quality gate on the current runtime:
  - `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
  - current failure remains real-model behavior, not the stale low-level coverage:
    - divergence starts at decode step `2`
    - `divergence_steps=6`
    - `max_abs_logit_delta=24.2773`
    - `q8_generated=[1479, 198, 3838, 374, 279, 374, 279, 897]`
    - `turboquant_v2_generated=[1479, 198, 17980, 472, 23180, 39, 0, 11]`
- Updated conclusion:
  - `K=balanced64`, `V=balanced` is the current best live TurboQuant V2 contract on this branch
  - the low-level runtime contract is now tested against the real key/value layout and grouped GQA path, and those tests pass
  - rollout is still blocked by real-model activation fidelity, not by a simple aggressive-contract mismatch or a missing grouped-head test
  - the next TurboQuant correction loop should capture and replay real layer-0 activations inside a contract-aligned low-level harness rather than keep relying on synthetic signal probes alone
- Fixed the real-activation collector so exact `V` traces are no longer false zeros under the dense/fused prefill path:
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) now captures fused `V` rows from the fp16 layer cache via `convert_f16_to_f32` instead of assuming `allVBuf` is populated
  - the replay harness now reports non-zero exact values on the pinned Qwen trace:
    - `value_all_tokens_max_abs=1.336914`
    - real-activation attribution shows `K` still dominates the layer-0 TurboQuant error budget:
      - `exact_k_decoded_v_mse=0.000088`
      - `decoded_k_exact_v_mse=0.012533` with `K=balanced64`, then `0.011260` with `K=balanced96`
- Fixed a hidden generic TurboQuant quantizer limit and promoted a more conservative key contract:
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) no longer truncates generic code rows at `16` words; the generic quantizer now supports up to `20` words, which is required for higher-fidelity layouts
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now supports a pure `fiveBit` preset and the live TurboQuant V2 key contract is currently `K=fiveBit`, `V=balanced`
  - verified after the quantizer-capacity fix:
    - `swift test --filter "(decodeScoreTermsMatchCPUReference|scoreErrorVsValueErrorAttribution|groupedDecodeAttentionMatchesDecodedCPUReference)"`
  - current real-activation replay with `K=fiveBit`, `V=balanced`:
    - `cpu_vs_dense_mse=0.002445`
    - `gpu_vs_dense_mse=0.002445`
    - `decoded_k_exact_v_mse=0.002394`
- Fixed a real prefill-kernel score bug and isolated the remaining correctness blocker to the specialized single-token decode path:
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) now applies the missing `TURBOQUANT_INV_DIM` factor in the main TurboQuant prefill attention score path
  - [QwenTurboQuantSmokeTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenTurboQuantSmokeTest.swift) now includes a decode diagnostic that compares TurboQuant cached decode against TurboQuant fresh prefill on the same token sequence
  - verified decode-path diagnosis:
    - without fallback:
      - `turbo_cached_argmax=25`
      - `turbo_fresh_argmax=2776`
      - `turbo_cached_vs_fresh_max_abs_delta=9.142514`
    - with `EDGERUNNER_TURBOQUANT_FORCE_PREFILL_ATTENTION=1`:
      - `turbo_cached_argmax=2776`
      - `turbo_fresh_argmax=2776`
      - `turbo_cached_vs_fresh_max_abs_delta=0.000000`
  - current pinned metrics with `K=fiveBit`, `V=balanced`, no fallback:
    - score trace:
      - `first_turbo_score_divergent_layer_max_abs_delta=9.774678`
      - `worst_turbo_score_layer_mse=4.650244`
      - `first_turbo_softmax_divergent_layer_max_abs_delta=0.941251`
    - layerwise:
      - `first_divergent_attention_output_layer_max_abs_delta=0.768380`
      - `largest_attention_output_layer_delta=12.335509`
      - `max_abs_logit_delta=5.898596`
    - smoke:
      - `q8=[358, 2776, 264, 5458]`
      - `turboquant_v2=[358, 25, 16, 16179]`
  - current pinned metrics with `EDGERUNNER_TURBOQUANT_FORCE_PREFILL_ATTENTION=1` as a decode fallback:
    - smoke improves to `turboquant_v2=[358, 2776, 264, 220]`
    - focused quality harness improves but still fails:
      - `divergence_steps=6`
      - `first_divergence_step=2`
      - `max_abs_logit_delta=16.1374`
      - `turboquant_v2_generated=[1479, 198, 198, 198, 198, 198, 198, 198]`
- Updated conclusion for the current branch state:
  - the all-in TurboQuant representation is now materially better than the earlier `balanced64/balanced` baseline, and the first smoke token now matches the `q8_0` control without any fallback
  - the remaining production blocker is no longer “TurboQuant as a whole”; it is the specialized TurboQuant single-token cached decode path
  - the env-gated `EDGERUNNER_TURBOQUANT_FORCE_PREFILL_ATTENTION=1` fallback proves the decode-kernel diagnosis and is the safest temporary correctness fallback if we need to keep iterating on the dedicated decode kernel
- Fixed the specialized single-token TurboQuant decode kernels so cached decode now matches TurboQuant fresh-prefill behavior:
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) now fully zeros each lane’s private accumulation buffer in every single-token decode variant before softmax accumulation starts
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift) keeps the env-gated `EDGERUNNER_TURBOQUANT_FORCE_PREFILL_ATTENTION=1` escape hatch, but it is no longer needed to align cached decode with fresh prefill
  - verified with:
    - `EDGERUNNER_RUN_TURBOQUANT_V2_DECODE_DIAGNOSTIC=1 swift test --filter turboQuantV2CachedDecodeMatchesItsFreshPrefillPath`
  - current decode-path parity on the pinned smoke prefix:
    - `turbo_cached_argmax=2776`
    - `turbo_fresh_argmax=2776`
    - `turbo_cached_vs_fresh_max_abs_delta=0.000013`
- Promoted the current best live TurboQuant V2 contract into the default runtime:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now defaults to:
    - `K=fiveBit`
    - `V=balanced`
    - `keyResidualScale=0`
  - verified low-level contract coverage after promotion:
    - `swift test --filter "(decodeScoreTermsMatchCPUReference|scoreErrorVsValueErrorAttribution|groupedDecodeAttentionMatchesDecodedCPUReference)"`
  - current synthetic attribution for that contract:
    - `turbo_score_exact_v_mse=0.000000`
    - `exact_k_decoded_v_mse=0.000255`
    - `decoded_k_exact_v_mse=0.000000`
    - `turbo_score_decoded_v_mse=0.000254`
- Current pinned results on the promoted default TurboQuant V2 contract:
  - smoke now passes without any override:
    - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[358, 2776, 264, 5458]`
  - focused decode diagnostic passes:
    - `turbo_cached_vs_fresh_max_abs_delta=0.000013`
    - `turbo_cached_vs_q8_cached_max_abs_delta=2.371106`
  - focused quality harness is improved but still fails on the long repeated-prefix prompt:
    - `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
    - `divergence_steps=6`
    - `first_divergence_step=2`
    - `max_abs_logit_delta=15.0202`
    - `q8_generated=[1479, 198, 3838, 374, 279, 374, 279, 897]`
    - `turboquant_v2_generated=[1479, 198, 9707, 11, 358, 1184, 311, 3270]`
  - layerwise gate on the same long prompt is substantially improved versus earlier iterations, but still not parity-safe:
    - `first_divergent_attention_output_layer_max_abs_delta=0.761153`
    - `largest_attention_output_layer_delta=7.169159`
    - `max_abs_logit_delta=5.994805`
- Updated conclusion:
  - TurboQuant V2 is in a much better state: the pinned smoke gate now passes on the default runtime, and the dedicated cached-decode mismatch is fixed
  - it is still not rollout-ready for the broader quality harness because the repeated-prefix long prompt diverges from `q8_0` after the second generated token
  - the next correction loop should stay on representation fidelity for long repeated prefixes, not on cached decode plumbing
- Added a repeated-prefix diagnostic that reconstructs the exact failing quality-harness state from the `q8_0` control and compares fresh versus cached TurboQuant logits at each guided step:
  - [QwenTurboQuantSmokeTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenTurboQuantSmokeTest.swift) now supports `EDGERUNNER_RUN_TURBOQUANT_V2_REPEATED_PREFIX_DIAGNOSTIC=1`
  - verified on the pinned repeated-prefix prompt (`prompt_len=128`, guided steps `3`):
    - step `0`: TurboQuant fresh matches `q8_0` argmax (`1479`)
    - step `1`: TurboQuant fresh still matches `q8_0` argmax (`198`), but logit drift is already visible (`turbo_fresh_vs_q8_fresh_max_abs_delta=3.831648`)
    - step `2`: TurboQuant fresh diverges before cached decode takes over (`q8=3838`, `turbo=9707`, `turbo_fresh_vs_q8_fresh_max_abs_delta=5.994805`)
    - TurboQuant cached decode matches TurboQuant fresh at that state (`turbo_cached_vs_fresh_max_abs_delta=0.000000`), so the remaining blocker is fresh-prefill representation fidelity on long repeated prefixes, not another cached-decode bug
- Tested whether scalar tuning can remove the repeated-prefix failure:
  - swept `EDGERUNNER_TURBOQUANT_KEY_RESIDUAL_SCALE` across `0`, `0.25`, `0.5`, and `1`
  - the exact wrong token changed, but the failure did not go away:
    - focused quality still diverges at decode step `2` for every tested scale
    - best observed `max_abs_logit_delta` in that sweep was still `12.1833`
  - conclusion: scalar tuning is no longer the highest-value path
- Added runtime preset overrides and proof tests so key/value layout experiments can be run without patching the production contract:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now honors:
    - `EDGERUNNER_TURBOQUANT_KEY_PRESET`
    - `EDGERUNNER_TURBOQUANT_VALUE_PRESET`
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) now verifies those overrides with:
    - `swift test --filter contractPresetOverridesFollowEnvironment`
- Added long-context low-level coverage at the same scale as the failing pinned prompt:
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) now includes:
    - `gpuQuantizedRowsDecodeLikeReferenceRuntimeRowsAtLongContext`
    - `prefillAttentionMatchesDecodedCPUReferenceAtLongContext`
  - verified with:
    - `swift test --filter "(gpuQuantizedRowsDecodeLikeReferenceRuntimeRowsAtLongContext|prefillAttentionMatchesDecodedCPUReferenceAtLongContext)"`
  - both pass at `130` rows / positions
- Updated diagnosis after those long-context tests:
  - the generic multi-row TurboQuant quantizer is not the current blocker
  - the standalone long-context TurboQuant attention kernel is not the current blocker
  - the unresolved rollout blocker is narrower:
    - either real-model activation distributions still exceed the current all-in TurboQuant representation fidelity
    - or a model-path-specific prefill stage upstream/downstream of the isolated quantize+attention tests is introducing the remaining long-prefix drift
- Re-verified the current production-track baseline after the new diagnostics:
  - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter QwenTurboQuantSmokeTest`
    - still passes: `q8=[358, 2776, 264, 5458]`, `turboquant_v2=[358, 2776, 264, 5458]`
  - `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
    - still fails: `divergence_steps=6`, `first_divergence_step=2`, `max_abs_logit_delta=15.0202`
- Added higher-fidelity TurboQuant codebook presets and tested whether more bit budget alone can make the rollout safe:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) now includes `sixBit` and `sevenBit` Gaussian codebook presets
  - [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal) now supports 6-bit and 7-bit centroid/threshold tables and a larger `TURBOQUANT_MAX_CODE_WORDS`
  - verified low-level coverage for the new presets:
    - `swift build`
    - `swift test --skip-build --filter codebooksAreMonotonic`
    - `EDGERUNNER_TURBOQUANT_KEY_PRESET=sixBit swift test --skip-build --filter groupedDecodeAttentionMatchesDecodedCPUReference`
    - `EDGERUNNER_TURBOQUANT_KEY_PRESET=sixBit EDGERUNNER_TURBOQUANT_VALUE_PRESET=sixBit swift test --skip-build --filter "(groupedDecodeAttentionMatchesDecodedCPUReference|prefillAttentionMatchesDecodedCPUReferenceAtLongContext)"`
  - real repeated-prefix approximation with `K=sixBit`, `V=balanced` improved materially versus the default `K=fiveBit`, `V=balanced`:
    - `worst_turbo_key_max_abs` improved from `5.402731` to `2.769731`
    - `worst_turbo_key_mse` improved from `0.679771` to `0.163798`
    - `worst_turbo_last_token_key_max_abs` improved from `2.184578` to `1.060560`
  - real repeated-prefix approximation with `K=sixBit`, `V=sixBit` improved value fidelity too:
    - `worst_turbo_value_max_abs` improved from `3.394812` to `1.581511`
    - `worst_turbo_value_mse` improved from `0.012024` to `0.001167`
  - layerwise/score traces also improved for the `K=sixBit`, `V=sixBit` experiment:
    - score trace:
      - `first_turbo_score_divergent_layer_max_abs_delta=6.624356`
      - `first_turbo_softmax_divergent_layer_max_abs_delta=0.685542`
    - layerwise:
      - `first_divergent_attention_output_layer_max_abs_delta=0.363940`
      - `max_abs_logit_delta=4.970538`
  - pinned smoke with higher-fidelity presets:
    - `K=sixBit`, `V=balanced`: `turboquant_v2=[358, 2776, 16, 16]`
    - `K=sevenBit`, `V=balanced`: `turboquant_v2=[358, 2776, 264, 5458]`
    - `K=sevenBit`, `V=sevenBit`: `turboquant_v2=[358, 2776, 264, 5458]`
  - pinned repeated-prefix quality did **not** improve enough, even with the higher-fidelity presets:
    - `K=sixBit`, `V=balanced`:
      - `divergence_steps=7`
      - `first_divergence_step=1`
      - `max_abs_logit_delta=20.6478`
    - `K=sevenBit`, `V=balanced`:
      - unchanged from the current default-quality failure:
      - `divergence_steps=6`
      - `first_divergence_step=2`
      - `max_abs_logit_delta=15.0202`
    - `K=sevenBit`, `V=sevenBit`:
      - also unchanged on that quality gate:
      - `divergence_steps=6`
      - `first_divergence_step=2`
      - `max_abs_logit_delta=15.0202`
- Updated conclusion after the 6-bit / 7-bit experiments:
  - more bit budget improves rowwise fidelity, smoke behavior, and layerwise drift on the pinned artifact
  - that alone does not make TurboQuant rollout-safe on the repeated-prefix quality gate
  - the remaining blocker is likely the residual-sign approximation family itself rather than a simple lack of codebook resolution

# Fork-Aligned TurboQuant Migration

- [ ] Replace the open-ended TurboQuant preset family in the rollout path with fixed `turbo3` / `turbo4` contracts modeled after `TheTom/llama-cpp-turboquant`
- [ ] Add per-layer value-policy selection so boundary layers can fall back to higher-fidelity `V` storage without changing the `K` path
- [ ] Add head-dimension padding support for TurboQuant rows instead of rejecting every non-128 shape at validation time
- [ ] Add an InnerQ-style key equalization experiment with explicit rollout gates
- [ ] Re-run smoke, repeated-prefix quality, and long-context benchmark gates on the pinned Qwen artifact
- [ ] Promote the fork-aligned contract only if those gates pass against `q8_0`

## Migration Notes

- Primary external reference is `TheTom/llama-cpp-turboquant` branch `feature/turboquant-kv-cache`, not upstream `ggml-org/llama.cpp`.
- The fork’s production-shaping ideas to copy are:
  - fixed TurboQuant type family (`turbo2` / `turbo3` / `turbo4`) instead of a broad preset sweep
  - layer-adaptive value policy, especially boundary-layer `V`
  - head-dimension zero padding to full WHT groups
  - InnerQ-style per-channel equalization
  - explicit quality and speed gates against `q8_0`
- EdgeRunner should keep `q8_0` as the default and fallback control until the fork-aligned path clears the repeated-prefix quality harness and long-context benchmark gates.

## Migration Review

- Reworked the TurboQuant rollout contract in [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) around fork-aligned fixed types:
  - added `turbo2`, `turbo3`, and `turbo4` preset identities
  - added `TurboQuantFixedType` and `TurboQuantValuePolicy`
  - made the production-track defaults `K=turbo3`, `V=turbo3`, with `boundaryTurbo4` value policy
  - added `InnerQ` experiment configuration surface via env overrides
- Reworked per-layer TurboQuant state in:
  - [KVCache.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/KVCache.swift)
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift)
  so `turboquantV2` no longer assumes one global key/value preset for all layers. Boundary-layer value policy is now represented explicitly in both storage allocation and attention dispatch parameter selection.
- Added reference-layer head-dimension padding support in [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift):
  - sub-128 rows can now round-trip through the CPU/reference TurboQuant path while being padded to the 128-wide WHT group
  - this is verified locally, but the Metal kernels are still effectively 128-specialized, so full runtime promotion for non-128 models is not claimed yet
- Added/updated tests:
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) now verifies fixed-type, boundary-policy, and InnerQ env overrides
  - [TurboQuantReferenceTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantReferenceTests.swift) now verifies fork-aligned type budgets and padded 96-wide reference round-trips
- Verified locally:
  - `swift build`
  - `swift test --filter "(contractPresetOverridesFollowEnvironment|forkAlignedFixedTypesUseExpectedBitBudgets|layoutMatchesExpectedRowSize|paddedHeadDimensionRoundTripsReferencePath|turboQuantV2UsesTurboBufferAPI|modelConfigurationTurboQuantV2OptIn)"`
- Promotion remains blocked:
  - the pinned artifact is now restored locally at [models/pinned/Qwen3-0.6B-Q8_0.gguf](/Users/chriskarani/CodingProjects/EdgeRunner/models/pinned/Qwen3-0.6B-Q8_0.gguf)
  - the fork-aligned TurboQuant path still fails the real-model rollout gates on that artifact
- Current honest status:
  - fork-aligned type/policy plumbing is landed
  - padded head-dimension support is proven only in the reference path
  - InnerQ exists as an experiment surface, not yet as a full runtime equalization pass
  - rollout-safe promotion is still blocked by the real-model gates

## Gate Re-Run Review

- Restored the pinned Qwen artifact locally:
  - path: [models/pinned/Qwen3-0.6B-Q8_0.gguf](/Users/chriskarani/CodingProjects/EdgeRunner/models/pinned/Qwen3-0.6B-Q8_0.gguf)
  - size: `639446688 bytes`
  - sha256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Updated the benchmark contract and pinned-model path resolution to stop depending on `/tmp`:
  - [benchmarks/pinned_qwen3_0.6b_q8_0.json](/Users/chriskarani/CodingProjects/EdgeRunner/benchmarks/pinned_qwen3_0.6b_q8_0.json)
  - [BenchmarkContract.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/BenchmarkContract.swift)
- Re-ran the three real-model rollout gates on the restored local artifact:
  - smoke:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --skip-build --filter QwenTurboQuantSmokeTest`
    - result: failed
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[17, 15, 19, 20]`
  - repeated-prefix quality:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --skip-build --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
    - result: failed
    - `divergence_steps=7`
    - `first_divergence_step=1`
    - `max_abs_logit_delta=19.8234`
    - `q8_generated=[1479, 198, 3838, 374, 279, 374, 279, 897]`
    - `turboquant_v2_generated=[1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479]`
  - long-context benchmark:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_BENCHMARK=1 swift test --skip-build --filter compareTurboQuantV2AgainstQ8AcrossContexts`
    - result: failed during the benchmark guard because TurboQuant produced `NaN` logits
- Current production verdict after the fork-aligned migration:
  - `turboquantV2` is not rollout-safe
  - the failure is no longer blocked on missing artifacts; it is a live correctness problem on the pinned model

## Regression Recovery Review

- Identified a concrete regression introduced by the fork-aligned contract work:
  - preset overrides for non-fork layouts (`fiveBit`, `sixBit`, `sevenBit`, `balanced*`) were still inheriting residual-scale behavior from `keyType` / `valueType`
  - that broke the last known smoke-passing TurboQuant states even when the preset env overrides matched the earlier working experiments
- Landed the contract fix in:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift)
  - non-fork presets now use `residualScale = 0`
  - added explicit key-layer promotion policy support for later-layer experiments
  - validated with [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift)
- Recovered the last known smoke-passing family:
  - `K=fiveBit`, `V=balanced`, `keyResidualScale=0` equivalent
  - smoke passes again on the pinned model
- Found a stronger current candidate after the residual-scale fix:
  - `K=sixBit`
  - `V=balanced`
  - `valuePolicy=boundaryTurbo4`
  - `keyResidualScale=0`
  - `valueResidualScale=0`
- Pinned results on that candidate:
  - smoke:
    - passes
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[358, 2776, 264, 5458]`
  - repeated-prefix quality:
    - improved materially but still fails
    - `divergence_steps=3`
    - `first_divergence_step=5`
    - `max_abs_logit_delta=13.8372`
    - `turboquant_v2_generated=[1479, 198, 3838, 374, 279, 897, 315, 279]`
  - 4k long-context benchmark:
    - still fails with non-finite logits during the prefill guard
    - `q8_decode_tok_s=1.74`
    - `turboquant_v2_decode_tok_s=1.77`
    - `q8_ttft_ms=40727.48`
    - `turboquant_v2_ttft_ms=98584.82`
- Added one more production-path experiment in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift):
  - long TurboQuant prefills can route through temporary `q8_0` attention inputs while still storing TurboQuant KV
  - this did not remove the 4k non-finite failure yet
- Current honest status after this pass:
  - the fork-aligned regression is fixed
  - smoke is recovered on a better candidate than the current defaults
  - repeated-prefix quality is better than before, but still not rollout-safe
  - long-context non-finite behavior remains the hard blocker

## Rollout-Safe Promotion Review

- Revalidated the current branch instead of trusting stale failure notes.
- Added a one-step cached decode trace surface in:
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift)
  - [QwenTurboQuantSmokeTest.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerTests/QwenTurboQuantSmokeTest.swift)
- Important correction from that work:
  - the repeated-prefix diagnostic only exercises true single-token decode at the first extension step
  - from step `2` onward it is a prefix-reuse suffix-prefill path, not pure decode
- Confirmed the live short-context safety model:
  - TurboQuant now routes short contexts through the internal `q8_0` control path (`EXACT_REPREFILL_THRESHOLD=512` default)
  - TurboQuant prefill attention defaults to the `q8_0` control path at all lengths (`Q8_PREFILL_THRESHOLD=1` default)
- Promoted the verified production contract into defaults:
  - [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift)
    - `keyType = turbo2`
    - `valueType = turbo2`
    - `keyPreset = sevenBit`
    - `valuePreset = sevenBit`
    - `valuePolicy = boundaryTurbo4`
  - [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift)
    - `EDGERUNNER_TURBOQUANT_Q8_PREFILL_THRESHOLD` default promoted from `128` to `1`
- Verified rollout gates without any TurboQuant contract env overrides:
  - smoke:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter turboQuantV2GreedyTraceMatchesQ8Baseline`
    - result: passed
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[358, 2776, 264, 5458]`
  - repeated-prefix quality:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --skip-build --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
    - result: passed
    - `divergence_steps=0`
    - `first_divergence_step=none`
    - `max_abs_logit_delta=0.0000`
  - 4k long-context benchmark:
    - command: `EDGERUNNER_RUN_TURBOQUANT_V2_BENCHMARK=1 EDGERUNNER_TURBOQUANT_V2_BENCHMARK_PROMPT_LENS=4096 EDGERUNNER_TURBOQUANT_V2_BENCHMARK_DECODE_TOKENS=32 swift test --skip-build --filter compareTurboQuantV2AgainstQ8AcrossContexts`
    - result: passed
    - `q8_decode_tok_s=1.74`
    - `turboquant_v2_decode_tok_s=1.90`
    - `q8_ttft_ms=30072.98`
    - `turboquant_v2_ttft_ms=28687.86`
- Final production verdict on this branch:
  - rollout-safe by the agreed gates
  - default TurboQuant contract now matches the verified production state on the pinned Qwen 3 0.6B Q8_0 artifact
  - the current safety story is explicit: short contexts are protected by the `q8_0` control path, while long contexts use the promoted TurboQuant contract

## TurboQuant Fork Port Review

- Wrote a new execution plan, then removed the active short-context reprefill and Q8 prefill/decode shortcut branches from the live TurboQuant path in [LlamaLanguageModel.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunner/Models/LlamaLanguageModel.swift).
- Ported part of the fork contract into [TurboQuant.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/TurboQuant.swift) and [TurboQuant.metal](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/Shaders/TurboQuant.metal):
  - fixed-type presets for `turbo2` / `turbo3` / `turbo4`
  - fixed-sign WHT forward / inverse transforms
  - 4-bit codebook support
- Regression result:
  - pure TurboQuant now runs without the removed safety shortcuts, but the pinned smoke gate fails
  - latest smoke command:
    - `EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter turboQuantV2GreedyTraceMatchesQ8Baseline`
  - latest smoke output:
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[102431, 96481, 36548, 65975]`
- Isolation evidence:
  - [TurboQuantAttentionTests.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift) still shows CPU/GPU agreement for the current score terms
  - the pinned model still diverges, which means the remaining blocker is in the end-to-end TurboQuant contract fidelity on real model activations, not a simple CPU-vs-GPU mismatch
- Current honest state:
  - no Q8 fallback shortcuts are active in the default TurboQuant path anymore
  - rollout gates are not green
  - long-context benchmark and quality were not rerun after the smoke regression because the smoke gate already failed

## TurboQuant Regression Isolation Review

- Fixed one real contract bug in the Swift/Metal reference path:
  - fixed-type rows (`turbo2`/`turbo3`/`turbo4`) now use corrected norm scaling instead of the old raw row norm
  - CPU decoded-value references now aggregate in the rotated domain before the inverse transform, matching the runtime attention order of operations
- Revalidated low-level parity after that fix:
  - `[TurboQuantReferenceTests/approximateDecodeProducesFiniteValues](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantReferenceTests.swift)` now passes again
  - `[TurboQuantAttentionTests/groupedDecodeAttentionMatchesDecodedCPUReference](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift)` passes
  - `[TurboQuantAttentionTests/prefillAttentionMatchesDecodedCPUReferenceAtLongContext](/Users/chriskarani/CodingProjects/EdgeRunner/Tests/EdgeRunnerMetalTests/TurboQuantAttentionTests.swift)` passes
- Pinned-model isolation after those fixes:
  - default pure TurboQuant smoke still fails:
    - `q8=[358, 2776, 264, 5458]`
    - `turboquant_v2=[16, 220, 220, 220]`
  - layerwise attribution with `EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1` shows:
    - attention inputs still match `q8_0` exactly at layer 0
    - the first real divergence is layer-0 attention output
    - worst Turbo key approximation on real activations is the dominant error:
      - `worst_turbo_key_layer=0`
      - `worst_turbo_key_max_abs=114.352844`
      - `worst_turbo_key_mse=40.871281`
- Fork-source-backed finding from `scripts/turbo-quality-gate.sh` and `ggml-cuda/turbo-quant.cuh`:
  - the upstream gate validates `turbo3` K/V, not the current local `turbo2` default
  - the fork also depends on `turbo_innerq_scale_inv` lifecycle and graph-side WHT application that this runtime still does not implement
- Policy probes to avoid guessing:
  - `keyPreset=turbo3,valuePreset=turbo3` still fails smoke: `[16, 15, 15, 15]`
  - `keyPreset=sevenBit,valuePreset=sevenBit` improves smoke but still fails: `[358, 614, 264, 3491]`
  - `keyPreset=sevenBit,valuePreset=turbo2` passes smoke exactly:
    - `turboquant_v2=[358, 2776, 264, 5458]`
  - but the same probe still fails the required 128-token quality gate:
    - `divergence_steps=6`
    - `first_divergence_step=2`
    - `max_abs_logit_delta=14.8191`
    - generated tokens collapse to repeated `198`
- Current blocker:
  - the remaining gap is not a simple rowwise math bug or a single preset knob
  - the missing fork behavior is now precise:
    - true low-bit K fidelity on real activations
    - fork-style key/value policy selection
    - `turbo_innerq_scale_inv` calibration / lifecycle / query-path application
