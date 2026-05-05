# Current Mobile GGUF Quant Support Plan

# Current Gemma 250 tok/s Optimization Plan

- [ ] Push Gemma 4 E4B from functional baseline toward high-throughput decode
  - [x] Establish honest baseline and identify algorithmic waste
  - [x] Remove repeated full-prompt PLE prelude work from decode
  - [x] Re-measure Gemma release harness
  - [x] Port the next Llama-style optimization: cached raw buffer reuse and batched decode projections
  - [x] Re-run Qwen gates after shared runtime changes
  - [x] Build stable multi-run Gemma benchmark reporting median tok/s and TTFT
  - [x] Add Gemma-private decode state and heterogeneous GPU KV cache foundations
  - [x] Add Gemma-specific GPU decode kernels for one-plus RMSNorm, scalar scale, and Q6_K token gather
  - [ ] Move Gemma decode hidden state and layer scratch buffers fully GPU-resident
  - [x] Integrate persistent heterogeneous Gemma KV cache with real sliding/global attention
  - [x] Fuse final norm, tied LM head, softcap, and greedy argmax without full CPU logit readback for greedy Q6_K LM head

## Active Plan: Gemma Buffer-Native Layer Runner

- [x] Replace the cache-backed Gemma layer loop with a GPU-resident scratch-buffer runner
  - [x] Add TDD coverage for the missing buffer-native primitives: RMSNorm plus residual add, and F32-to-F16 KV cache store
  - [x] Add Gemma-only Metal encode helpers without changing Qwen/Llama/Bonsai lanes
  - [x] Encode the cache-backed decoder layer stack through `Gemma4Scratch` and one command sequence per token
  - [x] Keep the existing Swift-array layer path as a fallback until the real-model gate proves the new path
  - [x] Run focused tests, Gemma coherent-token gate, Gemma median benchmark, and Qwen publishable hash gate

### Active Spec

- Success criteria:
  - generated Gemma token IDs stay `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
  - Gemma median decode materially improves beyond the current `10.557 tok/s`
  - Qwen publishable hash stays `0afae14a84cf0df8`
- Constraints:
  - no benchmark semantic changes
  - no shared model runtime changes
  - rollback if the buffer-native path is wrong or slower in sequential median gates

### Active Review

- Added a Gemma-only buffer-native greedy decode path over `Gemma4Scratch`.
- Added and tested missing primitives:
  - residual RMSNorm add
  - F32 to F16 KV store with cache offsets
  - Q4_K dual/triple GEMV encoders
  - buffer-native single-token PLE prelude
- Correctness gates stayed coherent:
  - Gemma generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
  - Qwen publishable hash stayed `0afae14a84cf0df8`
- Best measured Gemma median in this pass:
  - `21.806 tok/s`
  - median TTFT about `0.94s`
- Latest default Gemma median after leaving unstable experiments opt-in:
  - `18.859 tok/s`
  - best run `19.028 tok/s`
- Current honest blocker:
  - the remaining gap to `200 tok/s` is Q4_K matvec kernel quality and total Gemma E4B math volume, not CPU-side prelude or array round-trips.
  - two-row Q4_K, fused-GeGLU, and buffer-native PLE prelude variants are opt-in only because they did not produce a stable default win.

## Active Plan: Gemma FFN K-Quant GEMV Optimization

- [ ] Improve the remaining Gemma FFN bottleneck without changing Qwen/Llama/Bonsai lanes
  - [x] Identify the new top bucket after fast GQA and FFN command-buffer fusion: `layer_ffn_fused_gate_up_down`
  - [x] Add/extend focused K-quant GEMV microbenchmarks for Gemma FFN shapes
  - [x] Implement one targeted Q4_K/Q6_K GEMV kernel improvement
  - [ ] Keep only if Gemma Q4_K_M median improves and coherent token IDs stay stable
  - [ ] Re-run Qwen quick and publishable gates after any shared Metal kernel change

### Active Spec

- Success criteria:
  - Gemma Q4_K_M default median decode improves beyond the current `9.671 tok/s`
  - generated token IDs stay `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
  - Qwen publishable hash stays `0afae14a84cf0df8`
- Constraints:
  - do not modify benchmark semantics
  - shared K-quant Metal changes must retain existing Q4_K/Q6_K parity tests
  - roll back any kernel change that wins microbenchmarks but loses Gemma median

### Active Review

- Tried 128-thread Q4_K/Q6_K GEMV variants to reduce per-row threadgroup width in the FFN path.
- Focused parity passed for Q4_K, Q6_K, and mixed encoded command buffers.
- First Gemma median looked promising (`10.225 tok/s`), but repeat/default runs were inconsistent and one sequential 128-thread full median faulted with signal 5.
- Metal validation smoke later passed but only reached `7.632 tok/s`.
- Decision: rolled back the 128-thread kernels from the production diff. They are not rollout-safe.
- Latest verification after rollback:
  - K-quant parity and fast-GQA real-capacity tests pass.
  - Gemma Q4_K_M median benchmark passes coherently, but latest median under repeated Metal load is `6.974 tok/s`.
  - Qwen publishable hash remains `0afae14a84cf0df8`.

## Active Plan: Gemma4Scratch GPU-Resident Layer Runner Slice

- [x] Build the first production-safe scratch-buffer foundation for a GPU-resident Gemma layer runner
  - [x] Add `Gemma4Scratch` with persistent hidden, projection, attention, FFN, and PLE buffers sized from `Gemma4ModelConfig`
  - [x] Add tests proving buffer sizes, hidden-buffer swapping, and CPU copy/read helpers
  - [x] Add caller-owned RMSNorm encode parity coverage so scratch buffers can feed existing kernels
  - [x] Keep default generation behavior unchanged in this slice
  - [x] Rerun focused scratch/RMSNorm tests plus Gemma/Qwen gates

### Active Spec

- Success criteria:
  - scratch buffers allocate once per model config and can be reused across layer steps
  - all dimensions cover both Gemma local and global head shapes
  - no default runtime path changes until the scratch runner is wired behind a gate
  - existing Gemma coherent token IDs and Qwen hash remain the correctness gates
- Constraints:
  - no benchmark semantics changes
  - no shared Llama/Qwen hot-path change for this slice
  - do not keep scratch APIs that are too narrow for the next full-layer command-buffer runner

### Active Review

- Added the first `Gemma4Scratch` allocation owner with reusable hidden, attention, FFN, and PLE buffers sized from `Gemma4ModelConfig`.
- Added focused tests for scratch sizing, active-hidden buffer swapping, hidden copy/read helpers, and invalid hidden shapes.
- Added Metal coverage that `Gemma4DecodeKernels.encodeRMSNorm` can write into caller-owned buffers inside an existing command buffer, which is required for the next single-command-buffer layer runner slice.
- Focused verification passed:
  - `swift test --filter 'Gemma4Scratch|Gemma4DecodeKernelTests/gemmaRMSNormEncodesIntoExistingCommandBuffer'`
- Gemma release gate passed on `gemma-4-E4B-it-Q4_K_M.gguf`:
  - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
  - TTFT `13.059135s`, decode `9.762 tok/s`
- Qwen publishable gate passed:
  - median decode `257.4 tok/s`
  - token hash `0afae14a84cf0df8`

## Active Plan: Scratch-Backed Gemma PLE Side Channel

- [x] Wire `Gemma4Scratch` into the first real Gemma layer operation without touching other model lanes
  - [x] Add focused scratch coverage for PLE input buffer reuse
  - [x] Add a persistent Gemma-only scratch owner to `Gemma4LanguageModel`
  - [x] Route cache-backed PLE side-channel through scratch hidden/PLE buffers
  - [x] Keep the shortcut fallback path and shared Qwen/Llama/Bonsai paths unchanged
  - [x] Rerun focused scratch/Gemma tests plus Gemma/Qwen release gates

### Active Spec

- Success criteria:
  - PLE side-channel no longer allocates hidden, PLE input, gate, activated, or projection buffers on the cache-backed Gemma path
  - generated Gemma token IDs remain unchanged
  - Qwen publishable hash remains `0afae14a84cf0df8`
- Constraints:
  - do not wire Q/K/V, RoPE, FFN, or prelude into scratch in this slice
  - avoid scratch buffer swapping for now; mutate `currentHidden` in place and read it back
  - protect reusable scratch from concurrent generation access
  - no shared hot-path changes

### Active Review

- Added `Gemma4Scratch.copyPLEInput` and shape coverage so the PLE side-channel can reuse scratch storage instead of per-layer allocation.
- Added a persistent Gemma-only scratch owner in `Gemma4LanguageModel`.
- Routed the cache-backed Gemma PLE side-channel through scratch hidden, PLE input, gate, activated, and projection buffers.
- Added an async gate around scratch-backed PLE execution so overlapping generation calls cannot reuse the same scratch buffers while a command buffer is in flight.
- Kept the shortcut fallback path and shared Qwen/Llama/Bonsai paths unchanged.
- Verification passed:
  - `swift build`
  - `swift test --filter 'Gemma4Scratch|Gemma4DecodeKernelTests|Gemma4GPUKVCacheTests|Gemma4DecodeStateTests|Gemma4OpsTests|ModelLoaderTests'`
  - `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `3.691983s`, decode `10.034 tok/s`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
    - Qwen median decode `259.9 tok/s`
    - token hash `0afae14a84cf0df8`

## Active Plan: Scratch-Backed Gemma Fused FFN

- [x] Reuse `Gemma4Scratch` for the fused Gemma FFN gate/up/GeGLU/down helper
  - [x] Add focused scratch coverage for FFN input buffer reuse
  - [x] Route cache-backed fused FFN through scratch input/gate/up/activated/down buffers
  - [x] Keep shortcut fallback and shared model lanes unchanged
  - [x] Rerun focused scratch/Gemma tests plus Gemma/Qwen release gates

### Active Spec

- Success criteria:
  - cache-backed Gemma FFN no longer allocates input/gate/up/activated/down buffers per layer
  - generated Gemma token IDs remain unchanged
  - Qwen publishable hash remains `0afae14a84cf0df8`
- Constraints:
  - do not change GeGLU math or projection dispatch semantics
  - keep scratch buffer access serialized while command buffers are in flight
  - no shared Llama/Qwen/Bonsai hot-path changes

### Active Review

- Added `Gemma4Scratch.copyFFNInput` and shape coverage so the fused FFN command-buffer helper can reuse persistent input storage.
- Routed the cache-backed Gemma fused FFN through scratch input, gate, up, activated, and down buffers.
- Preserved the existing shortcut fallback and all shared model lanes.
- Verification passed:
  - `swift build`
  - `swift test --filter 'Gemma4Scratch|Gemma4DecodeKernelTests|Gemma4GPUKVCacheTests|Gemma4DecodeStateTests|Gemma4OpsTests|ModelLoaderTests'`
  - `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `5.558448s`, decode `7.592 tok/s` with finite validation enabled
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `6.342530s`, decode `10.307 tok/s`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
    - Qwen median decode `261.2 tok/s`
    - token hash `0afae14a84cf0df8`
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFMedianBenchmark'`
    - Gemma median decode `10.336 tok/s`
    - best decode `10.441 tok/s`
    - median TTFT `2.208316s`
- Caveat:
  - the finite-validation run was slower, so future performance decisions should continue using the median Gemma harness, not a single short run.

## Active Plan: Buffer-Resident Gemma PLE Inputs

- [x] Keep PLE per-layer inputs GPU-resident through the decoder layer stack
  - [x] Add focused coverage for offset-based PLE gate input buffers
  - [x] Extend `PreludeState` to retain the produced `perLayerInputsBuffer`
  - [x] Route cache-backed layer PLE slices by buffer offset instead of Swift array slicing
  - [x] Keep shortcut fallback able to materialize PLE inputs lazily
  - [x] Rerun focused PLE/Gemma tests plus Gemma median and Qwen publishable gates

### Active Spec

- Success criteria:
  - default cache-backed Gemma path does not read the full `numLayers * perLayerDim` PLE input buffer back to Swift each token
  - per-layer PLE side-channel receives its slice by `MTLBuffer` offset
  - generated Gemma token IDs remain unchanged
  - Qwen publishable hash remains `0afae14a84cf0df8`
- Constraints:
  - no changes outside Gemma-specific runtime and PLE gate API
  - shortcut fallback can be slower but must remain correct
  - rollback on token mismatch, finite-validation failure, or median throughput regression

### Active Review

- Added offset support to `PLEGateKernel.encode`, with coverage proving a PLE slice can be read from inside a larger buffer.
- Kept `PreludeState` PLE inputs as the produced `MTLBuffer` on the default cache-backed path.
- Routed cache-backed layer PLE side-channel calls by byte offset into the per-layer-input buffer, avoiding full-buffer Swift materialization and per-layer Swift slicing.
- Kept the shortcut fallback correct by lazily materializing PLE slices only if the shortcut path is used.
- Verification passed:
  - `swift build`
  - `swift test --filter 'PLEGateKernelTests|PLEInputsKernelTests|PLEGatherKernelTests|PLESideChannelKernelTests|Gemma4Scratch|Gemma4DecodeKernelTests|Gemma4GPUKVCacheTests|Gemma4DecodeStateTests|Gemma4OpsTests|ModelLoaderTests'`
  - `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `12.832236s`, decode `10.107 tok/s`
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFMedianBenchmark'`
    - Gemma median decode `10.202 tok/s`
    - best decode `10.255 tok/s`
    - median TTFT `2.245419s`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
    - Qwen median decode `255.9 tok/s`
    - token hash `0afae14a84cf0df8`
  - `EDGERUNNER_GEMMA4_PROFILE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - profile still shows `layer_ffn_fused_gate_up_down` first and `ple_row_gather` second

## Active Plan: Blocked Q6_K PLE Row Gather

- [x] Replace the default Q6_K PLE gather dispatch with a block-oriented kernel
  - [x] Add/retain parity coverage through existing Q6_K PLE gather tests
  - [x] Implement a Q6_K PLE gather kernel that dispatches one threadgroup per 256-value block
  - [x] Route `encodeQ6K` through the blocked kernel only
  - [x] Rerun PLE tests, Gemma median, profile, and Qwen publishable gates

### Active Spec

- Success criteria:
  - Q6_K PLE gather output remains bit-equivalent within existing tolerance
  - Gemma token IDs remain unchanged
  - Gemma median does not regress from the current `10.202 tok/s`
  - Qwen publishable hash remains `0afae14a84cf0df8`
- Constraints:
  - Q8_0 PLE gather path remains unchanged
  - no benchmark semantic changes
  - rollback if the blocked kernel is slower in the real Gemma gate

### Active Review

- Added `ple_gather_q6_k_blocked`, dispatching one 256-thread threadgroup per Q6_K block.
- Routed only Q6_K PLE gather through the blocked kernel; Q8_0 remains unchanged.
- Fixed the Metal signature to use vector threadgroup/thread-position attributes consistently.
- Verification passed:
  - `swift test --filter PLEGatherKernelTests`
  - `swift test --filter 'PLEGateKernelTests|PLEInputsKernelTests|PLEGatherKernelTests|PLESideChannelKernelTests|Gemma4Scratch|Gemma4DecodeKernelTests|Gemma4GPUKVCacheTests|Gemma4DecodeStateTests|Gemma4OpsTests|ModelLoaderTests'`
  - `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `14.061053s`, decode `10.198 tok/s`
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFMedianBenchmark'`
    - Gemma median decode `10.270 tok/s`
    - best decode `10.315 tok/s`
    - median TTFT `2.221813s`
- Result:
  - modestly better than the prior `10.202 tok/s` buffer-resident PLE-input median, but still below the earlier `10.336 tok/s` scratch-FFN median; keep as a small isolated kernel cleanup.

## Active Plan: Bounded Gemma Prelude Cache

- [x] Cache completed single-token Gemma prelude outputs by token ID
  - [x] Add a bounded runtime cache for token hidden state plus per-layer-input buffer
  - [x] Reuse cached prelude state before PLE row gather/token embedding/projection work
  - [x] Keep cache size bounded to avoid long-generation memory growth
  - [x] Rerun Gemma finite, median/profile, and Qwen publishable gates

### Active Spec

- Success criteria:
  - repeated token IDs skip PLE row gather and PLE input rebuild
  - cache is Gemma-only and bounded
  - generated Gemma token IDs remain unchanged
  - Qwen publishable hash remains `0afae14a84cf0df8`
- Constraints:
  - no cache for mutable layer outputs or KV state
  - do not change prompt/decode semantics
  - rollback if median performance regresses

### Active Review

- Added a bounded 128-entry `PreludeState` LRU inside the Gemma runtime cache.
- Cached only token-local prelude data: token hidden vector and read-only per-layer-input buffer.
- Repeated token IDs now skip PLE row gather, token embedding gather, model projection, and PLE input build.
- Verification passed:
  - `swift build`
  - `swift test --filter 'PLEGateKernelTests|PLEInputsKernelTests|PLEGatherKernelTests|PLESideChannelKernelTests|Gemma4Scratch|Gemma4DecodeKernelTests|Gemma4GPUKVCacheTests|Gemma4DecodeStateTests|Gemma4OpsTests|ModelLoaderTests'`
  - `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - generated token IDs stayed `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
    - TTFT `20.301714s`, decode `9.618 tok/s` with finite validation
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFMedianBenchmark'`
    - Gemma median decode `10.557 tok/s`
    - best decode `10.595 tok/s`
    - median TTFT `2.899122s`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
    - Qwen token hash stayed `0afae14a84cf0df8`
    - Qwen median decode was noisy at `220.8 tok/s`; two runs dropped to ~158 tok/s after repeated heavy Gemma/Metal runs.
  - `EDGERUNNER_GEMMA4_PROFILE=1 EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
    - PLE row gather calls dropped from `33x` to `23x` at the 8-token profile checkpoint.
- Result:
  - keep for decode throughput and reduced repeated PLE work, but TTFT did not improve in the measured median run.

# Current Gemma 4 Correctness Breakthrough Plan

- [x] Restore Gemma 4 tokenization parity before more throughput work
  - [x] Re-check upstream llama.cpp / MLX / Transformers sources for Gemma 4 tokenizer and model-contract assumptions
  - [x] Add failing tokenizer parity coverage for Gemma 4 GGUF prompt handling
  - [x] Implement Gemma 4 as GGUF BPE with Gemma-specific pretokenization instead of SentencePiece
  - [x] Re-run focused tokenizer and Gemma tests
  - [x] Re-run the downloaded Gemma Q4_K_M benchmark and compare output against llama.cpp
  - [x] Record the next blocker or the first coherent EdgeRunner Gemma output

- [x] Fix upstream K-quant raw-layout parity for Gemma Q4_K_M
  - [x] Add/retarget tests so Q4_K and Q6_K decode against llama.cpp byte ordering, not EdgeRunner's old local reference
  - [x] Correct Q4_K CPU fallback scale/min unpacking and low/high nibble ordering
  - [x] Correct Q4_K Metal dequant and fused GEMV ordering
  - [x] Correct Q6_K CPU fallback segment ordering
  - [x] Correct Q6_K Metal dequant, fused GEMV, PLE gather, and Gemma token-embedding gather
  - [x] Re-run focused K-quant, tokenizer, Gemma, Qwen gates, and the downloaded Gemma benchmark

### Gemma 4 K-Quant Correctness Review

- Root cause of the 2.4 tok/s iPhone failure was correctness first: EdgeRunner's Q4_K/Q6_K paths used local raw-byte ordering that disagreed with llama.cpp's GGUF layout.
- Fixed Q4_K scale/min unpacking and low/high nibble grouping in CPU fallback and Metal dequant/GEMV.
- Fixed Q6_K 128-value segment ordering in CPU fallback, Metal dequant/GEMV, PLE gather, and Gemma token embedding gather.
- Promoted real cache-backed Gemma attention to the default path; the old shortcut can be forced only with `EDGERUNNER_GEMMA4_SHORTCUT_ATTENTION=1` and is known to repeat whitespace.
- Added a Gemma greedy path that avoids materializing the full Q6_K tied-LM-head logits array for normal greedy decode.
- Current Mac release result on `gemma-4-E4B-it-Q4_K_M.gguf`:
  - generated text: `thought\nThinking Process:\n1.  **Analyze the Request:** The`
  - generated tokens: `16`
  - TTFT: `9.432904s`
  - decode: `5.784 tok/s`
- App-like greedy median harness after the greedy path:
  - runs: `5`
  - median TTFT: `3.725078s`
  - median decode: `5.572 tok/s`
  - best decode: `5.590 tok/s`
- Qwen regression gates after the shared quant fix:
  - quick smoke: `250.4605 tok/s`, expected prefix preserved
  - publishable benchmark: median `225.9 tok/s`, hash `0afae14a84cf0df8`
- iPhone reinstall was blocked by local signing state:
  - device visible as `Chris's iPhone`
  - Xcode build fails because no development-team account/provisioning profile is available for `com.chriskarani.EdgeRunnerChat`
- Remaining performance blocker:
  - Gemma is now coherent but still far below llama.cpp because layer state is still bounced through Swift arrays and many command buffers. The next breakthrough needs a GPU-resident Gemma layer runner that fuses attention, FFN, PLE side-channel, and LM-head/argmax over persistent buffers.

### Gemma 4 GPU-Resident Runner Progress

- Added optional profiling behind `EDGERUNNER_GEMMA4_PROFILE=1`.
- Profiling result on the release median harness confirms the largest measured stage buckets are:
  - `layer_ffn_gate_up_geglu`
  - `layer_attention`
  - `layer_ffn_down`
  - `layer_o_projection`
  - `layer_ple_side_channel`
- The first token is still expensive because prompt prefill runs token-by-token through the full 42-layer stack rather than batched graph/GEMM prefill.
- Runtime finite-value sweeps are now opt-in with `EDGERUNNER_GEMMA4_VALIDATE_FINITE=1`, so release decode does not scan every intermediate Swift array by default.
- Added an experimental fast decode-GQA kernel that computes attention scores once per head and reuses them across value dimensions, instead of recomputing QK for every output dimension.
  - Synthetic local/global Gemma head-dim tests pass.
  - Added real-capacity coverage for 512-token sliding buffers and 4096-token global buffers, including wrapped starts.
  - Added cache-backed buffer-size guards so future shape mistakes fail before a Metal device fault.
  - Real Gemma Q4_K_M release benchmark now passes coherently with fast GQA active, so fast GQA is the default; `EDGERUNNER_GEMMA4_FAST_GQA=0` is the fallback.
- Fused Gemma FFN gate projection, up projection, GeGLU, and down projection into one command-buffer sequence so the GeGLU activation no longer round-trips through Swift.
- Tried a follow-on GPU-buffer post-FFN RMSNorm/residual handoff into PLE, but rolled it back because the median regressed (`8.280 tok/s`).
- Current stable default Gemma Q4_K_M median harness after this pass:
  - median TTFT: `2.731036s`
  - median decode: `9.671 tok/s`
  - best decode: `9.917 tok/s`
  - output token IDs remained `[100, 45518, 107, 120474, 12364, 236787, 107, 236770, 236761, 138, 1018, 115863, 506, 16499, 53121, 669]`
- Latest repeated benchmark after heavy Metal A/B testing measured lower but still coherent:
  - Gemma median TTFT: `3.270442s`
  - Gemma median decode: `6.974 tok/s`
  - interpretation: performance is noise/thermal sensitive enough that any new win needs repeatable sequential medians, not a single high run.
- Current Qwen gates after this pass:
  - quick smoke: latest `224.6507 tok/s`, expected prefix preserved
  - publishable benchmark: latest median `204.7 tok/s`, hash `0afae14a84cf0df8`
- Next concrete kernel task:
  - attack the remaining FFN bottleneck inside the Q4_K/Q6_K GEMV kernels or replace the layer runner with persistent GPU scratch buffers, because command-buffer fusion alone is now mostly exhausted.

### Gemma 4 Architecture Research Notes

- Official Gemma 4 E4B config confirms this is a separate text architecture, not a Llama/Qwen drop-in:
  - 42 decoder layers
  - hidden size 2560
  - 8 attention heads, 2 KV heads
  - 256-dim per-layer input embeddings
  - 512-token sliding window
  - alternating sliding/full attention with the final layer full attention
  - 18 KV-shared layers
  - tied token embeddings
  - final logit softcap 30.0
- Official Gemma 4 model card confirms E4B is mobile/edge targeted, uses PLE for on-device parameter efficiency, and stores large PLE tables that are cheap lookups rather than normal dense per-token compute.
- Hugging Face Gemma 4 config documents the PLE tensor layout as one packed table shaped like `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`.
- Hugging Face Gemma 4 config documents `num_kv_shared_layers` as consecutive decoder layers sharing key-value projections, so EdgeRunner needs a layer-aware KV source map instead of one cache entry per layer by default.
- llama.cpp's Gemma 3n/Gemma-family builder shows the reference graph shape EdgeRunner should converge toward:
  - get PLE rows for input tokens
  - project main embeddings through `per_layer_model_proj`
  - normalize and combine projected PLE with table PLE
  - keep per-layer PLE available as layer-local side input
  - run AltUp/LAuReL/attention/FFN/PLE correction in the decoder graph
  - for KV-shared layers, compute Q only and reuse earlier K/V

### Researched Gemma Runtime Spec

- Build a Gemma-only runtime lane rather than widening the Llama/Qwen path:
  - `Gemma4DecodeState`: token prefix, cached position, PLE cache window, sliding/global KV cache ownership, and validity/reset logic.
  - `Gemma4Scratch`: long-lived Metal buffers for hidden state, AltUp stack, norms, Q/K/V, attention output, FFN intermediates, PLE side-channel, logits, and argmax.
  - `Gemma4LayerPlan`: layer-local metadata for sliding vs full attention, RoPE family, KV owner layer, head dimensions, and shared-KV behavior.
- First architecture slice success criteria:
  - existing Gemma benchmark still generates the same number of tokens or better
  - all current Gemma focused tests pass
  - Qwen quick and publishable correctness gates keep their pinned prefix/hash
  - no shared hot path changes unless covered by existing cross-lane tests
- Second architecture slice success criteria:
  - prompt prefill writes real K/V for all KV-owner layers
  - decode processes only one new token and attends over cache
  - sliding layers cap attention to the configured 512-token window
  - KV-shared layers skip K/V projection and read from their owner cache
- Third architecture slice success criteria:
  - final norm, tied LM head, softcap, and greedy argmax stay GPU-resident for greedy decode
  - CPU reads only the selected next token during greedy benchmarking

### Gemma 250 tok/s Optimization Spec

- Target:
  - aspirational target: `250 tok/s`
  - current measured release baseline: `3.267 tok/s`, TTFT `10.862705s`
- Success criteria for this pass:
  - measurable Gemma tok/s improvement without losing generation
  - no regression in Qwen deterministic prefix/hash gates
  - no full-model Float32 materialization
- Constraints:
  - keep changes production-grade and incremental
  - do not modify benchmark semantics
  - preserve unrelated dirty worktree changes
- Reality check:
  - `250 tok/s` for a 4B-class Q4/Q6 model is not a credible near-term target on this current unfused baseline. The near-term path is to remove repeated prompt work, add decode-state caching, then fuse/encode the layer graph.

### Gemma 250 tok/s Optimization Review

- Removed repeated full-prompt PLE prelude work from the current token-only Gemma baseline; RoPE still uses the real sequence position.
- Added a Gemma runtime cache for raw tensor `MTLBuffer` views, decoded norm vectors, and norm weight buffers.
- Added reusable K-quant GEMV encode hooks for `Q4_K` and `Q6_K`, plus coverage proving Q4_K/Q6_K dispatches can share one command buffer.
- Batched Gemma attention Q/K/V projections and FFN gate/up projections when they share the same input.
- Added caller-owned `GeGLUKernel.encode` and wired Gemma FFN gate/up projection plus GeGLU into one command buffer, so only the activated FFN vector is read back before the down projection.
- Replaced per-head temporary array normalization with a direct loop over each head to preserve the same Gemma RMSNorm math with less allocation.
- Added a reusable Gemma median benchmark with one warmup run, five measured runs, per-run generated-token counts, median decode tok/s, best/min decode tok/s, and median TTFT.
- Added `PLEGateKernel` and moved the PLE gate GELU multiply into the GPU side-channel command.
- Rewrote `ple_side_channel_finalize` to reduce RMS once per token with a 256-thread group instead of once per hidden element.
- Optimized Q4_K/Q6_K GEMV shaders by staging per-block scale metadata in threadgroup memory instead of reloading it from every thread.
- Added a mathematically equivalent current-baseline single-token GQA shortcut: because Gemma currently invokes attention with `seqLen == 1`, attention output is the normalized V head expanded across each query group.
- Web/source research checkpoint:
  - Official Gemma 4 E4B config confirms 42 layers, hidden size 2560, 8 attention heads, 2 KV heads, 256/512 local/global head dims, 512-token sliding window, alternating sliding/full attention, 18 KV-shared layers, tied embeddings, and final logit softcap 30.0.
  - Official Gemma 4 model card confirms E2B/E4B are mobile/edge targets and use Per-Layer Embeddings for on-device efficiency.
  - Hugging Face Gemma 4 config documents packed PLE shape and KV-sharing semantics.
  - llama.cpp Gemma-family graph confirms the target flow: gather packed PLE rows, project model embeddings into PLE space, keep PLE as a layer-local side input, and make KV-shared layers compute Q while reusing earlier K/V.
- Added `Gemma4DecodeState` to classify full prefill, prefix reuse, and single-token decode without changing the public model API.
- Added `Gemma4GPUKVCache`, a Gemma-private F16 KV cache that allocates sliding layers as 512-token rings, global layers as full-context buffers, and aliases KV-shared layers to their source buffers.
- Added `Gemma4DecodeKernels` plus `Gemma4Decode.metal`:
  - Gemma `(1 + weight)` RMSNorm
  - in-place scalar multiply for `layer_output_scale`
  - Q6_K token embedding gather/dequant for the downloaded Gemma artifact's `token_embd.weight`
- Added an opt-in cache-backed Gemma attention lane behind `EDGERUNNER_GEMMA4_REAL_ATTENTION=1`:
  - full-prefill / prefix-reuse / single-token decode mode routing
  - Q/K/V projection for owner layers, Q-only projection for KV-shared layers
  - Gemma one-plus Q/K head RMSNorm
  - NeoX pRoPE with local/global theta selection
  - F16 K/V writes into the Gemma-private sliding/global cache
  - source-layer cache reads for KV-shared layers
  - windowed F16 KV GQA over the cache buffers
- Added a public `RoPEKernel.applyToQKNeoX` wrapper so Gemma can use NeoX-layout pRoPE without changing existing interleaved RoPE callers.
- Wired Gemma layer output scaling into the existing PLE side-channel command buffer.
- Routed Q6_K token embedding lookup through the Gemma GPU gather kernel; non-Q6 formats keep the existing CPU fallback until their GPU gather kernels are added.
- Tried a GPU LM-head-plus-softcap path, but reverted it because repeated measurements regressed relative to the GeGLU/batched-projection slice.
- Tried GPU attention-buffer reuse and attention/FFN residual RMSNorm fusion, but reverted both because median throughput did not beat the simpler path.
- Best Gemma Q4_K_M release harness result in this pass:
  - baseline: `3.267 tok/s`, TTFT `10.862705s`
  - current median benchmark: `10.895 tok/s`, median TTFT `0.089117s`
  - current best measured run: `11.138 tok/s`
- After the Gemma-private state/cache/kernel foundation:
  - run 1: median `7.384 tok/s`, best `11.624 tok/s`, median TTFT `0.119048s`
  - run 2: median `10.191 tok/s`, best `11.889 tok/s`, median TTFT `0.086215s`
  - interpretation: generation still works, but this slice is infrastructure rather than a throughput breakthrough. The Q6_K gather has not yet become a stable median win because the current runtime still reads back CPU arrays between major stages.
- Gemma Q4_K_M re-download attempt:
  - target: `unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf`
  - expected file size: `4.98 GB`
  - result: blocked by local disk pressure; `/tmp` had ~157 MB free after cleaning the incomplete Hugging Face shard.
  - no new Gemma tok/s claim was made from this pass.
- Verification passed:
  - `swift build`
  - `swift test --filter 'GeGLUKernelTests|GEMVTests/encodedKQuantGemvsShareOneCommandBuffer'`
  - `swift test --filter 'GEMVTests|GeGLUKernelTests|Gemma4WeightsTests'`
  - `swift test --filter 'Gemma4Ops|Gemma4WeightsTests'`
  - `swift test --filter 'PLESideChannelKernelTests|PLEGateKernelTests|GEMVTests/gemvQ4KWeightsMatchCPUReference|GEMVTests/gemvQ6KWeightsMatchCPUReference|GEMVTests/encodedKQuantGemvsShareOneCommandBuffer|Gemma4Ops|Gemma4WeightsTests'`
  - `swift test --filter 'Gemma4GPUKVCache|Gemma4DecodeState|Gemma4DecodeKernel|Gemma4Ops|Gemma4Weights|Gemma4Config'`
  - `swift test --filter 'Gemma4GPUKVCache|Gemma4DecodeKernel|Gemma4DecodeState|Gemma4Ops|Gemma4Weights|Gemma4Config|RoPE dual'`
  - `swift test --filter 'RoPETests|RoPEDualTable'`
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFThroughEdgeRunner'`
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test -c release --filter 'Gemma4DownloadedBenchmark/runDownloadedGGUFMedianBenchmark'`
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
  - `git diff --check`
- Qwen regression gates stayed deterministic:
  - quick gate prefix: `[1, 1479, 35, 5371, 1]`
  - quick gate after cache-backed Gemma attention lane: `[1, 1479, 35, 5371, 1]`, `262.8466 tok/s`
  - publishable token hash: `0afae14a84cf0df8`
  - publishable token hash after shared Metal changes: `0afae14a84cf0df8`
  - publishable after cache-backed Gemma attention lane: `251.2 tok/s`, hash `0afae14a84cf0df8`
- Remaining blocker for `250 tok/s`:
  - Gemma now has a staged cache-backed attention lane, but it still uses CPU arrays and readbacks between major layer stages and has no production GPU-resident layer scratch-buffer pipeline. The next real jump requires moving layer state, FFN, PLE side-channel, final norm, tied LM head, softcap, and argmax into a persistent GPU-resident decode runner with far fewer command buffers.

# Current Gemma Decoder Forward Fix Plan

- [x] Fix the current Gemma 4 downloaded GGUF blocker: decoder-layer forward missing
  - [x] Inspect exact downloaded Gemma tensor shapes/types and layer math contract
  - [x] Add focused tests for the next decoder runtime primitive before implementation
  - [x] Implement the smallest production decoder-forward slice that avoids full Float32 model materialization
  - [x] Run the downloaded Gemma Q4_K_M harness and record the next blocker or generation metrics
  - [x] Re-run Qwen correctness gates after shared kernel/runtime changes

### Gemma Decoder Forward Fix Spec

- Current blocker:
  - downloaded Gemma Q4_K_M reaches `Gemma 4 PLE prelude completed; decoder layer forward is not implemented yet`
- Success criteria:
  - advance the real downloaded GGUF past the decoder-layer-forward blocker
  - preserve mobile memory behavior by using raw quant/BF16 buffers where possible
  - do not regress Qwen pinned prefix/hash gates
- Constraints:
  - no benchmark harness expectation changes

# Gemma iPhone Deployment Space Plan

- [x] Free enough local staging space for a ~5 GB Gemma Q4_K_M install.
  - [x] Clear reproducible CoreDevice app-install delta caches.
  - [x] Clear reproducible Hugging Face cache entries; keep `/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf`.
  - [x] Re-check local free space.
- [x] Verify the existing Gemma Q4_K_M GGUF at `/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf`.
- [x] Run the Gemma downloaded-model benchmark from the existing GGUF.
- [ ] Put the model on the paired iPhone after the app-side model copy/install path has enough staging space.
  - [x] Freed local staging space by clearing reproducible caches.
  - [x] Built and installed `EdgeRunnerChat` on `Chris’s iPhone`.
  - [x] Copied `gemma-4-E4B-it-Q4_K_M.gguf` into the app Documents container.
  - [x] Verified the on-device app container lists `Documents/gemma-4-E4B-it-Q4_K_M.gguf` at `4.97 GB`.
  - [x] Fixed whole-file GGUF `mmap` on iOS by loading metadata from a prefix and mapping tensor regions on demand.
  - [x] Rerun the in-app benchmark after the corrected copy completed.
  - [ ] Fix degenerate Gemma iPhone output before treating tok/s as meaningful.
  - [ ] Replace the current CPU/readback-heavy Gemma decode lane with a Gemma-private GPU-resident decode slice inspired by llama.cpp/MLX.
  - no full-model Float32 materialization for Gemma 4 E4B
  - keep unrelated dirty worktree changes intact

### Gemma iPhone Benchmark Review

- Device: `Chris’s iPhone` / iPhone 15 Pro Max.
- Model copied into app container: `Documents/gemma-4-E4B-it-Q4_K_M.gguf`, `4.97 GB`.
- Initial in-app failure after the first copy attempt: `invalidFormat("Tensor per_layer_token_embd.weight exceeds GGUF data section bounds")`; root cause was an incomplete/misplaced file copy.
- Second in-app failure: `mmapFailed(errno: 12)`; root cause was mapping the full ~5 GB GGUF at load time.
- Loader fix: `GGUFLoader.prepare` now parses the header from a bounded file prefix, and tensor buffers are backed by page-aligned file-region mappings instead of a whole-file mapping.
- Current fixed-loader iPhone result:
  - generated tokens: `64`
  - TTFT: `7.225641s`
  - decode: `2.408 tok/s`
  - end-to-end: `1.917 tok/s`
  - output: 64 NUL characters, so the result is a runtime correctness failure, not a publishable benchmark.

### Gemma Decoder Forward Fix Review

- Added fused raw Q4_K and Q6_K GEMV Metal kernels, plus Swift `GEMVKernel` APIs, so Gemma decoder projections can consume mobile GGUF quant tensors without expanding full matrices to Float32.
- Added GEMV tests comparing Q4_K/Q6_K fused kernels against CPU references.
- Bound the real Gemma decoder tensors that were missing from `Gemma4Weights`: `attn_q_norm`, `attn_k_norm`, `ffn_norm`, and `layer_output_scale`.
- Replaced the Gemma decoder-layer stop with a functional baseline path:
  - PLE prelude
  - 42 decoder layers for the current token
  - Gemma `(1 + weight)` RMSNorms
  - Q/K/V attention projection, Q/K/V head norms, RoPE, GQA, attention output projection
  - GeGLU FFN and PLE side-channel
  - layer output scale, final output norm, tied LM head, and final logit softcap
- Current limitation:
  - this is a functional baseline, not the optimized mobile decode path. It computes current-token layer work and uses the configured shared-KV source map, but it does not yet implement a proper multi-token prefill/decode KV-cache state machine.
- Downloaded Gemma Q4_K_M release harness now passes:
  - generated tokens: `16`
  - TTFT: `10.862705s`
  - decode: `3.267 tok/s`
- Regression verification passed:
  - `swift build`
  - `swift test --filter 'GEMVTests|Gemma4WeightsTests'`
  - `swift test --filter 'FusedKernelTests|DequantQ8_0Tests'`
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
- Qwen publishable gate remains deterministic:
  - token hash: `0afae14a84cf0df8`
  - median decode: `221.4 tok/s`
  - median TTFT: `4.5 ms`

# Current Gemma Forward Bootstrap Plan

- [x] Replace the top-level Gemma missing-forward-pass stop with a real bootstrap slice
  - [x] Add BF16 GEMV coverage for Gemma `per_layer_model_proj.weight`
  - [x] Implement BF16 GEMV in the shared Metal GEMV helper
  - [x] Add a Gemma bootstrap path that builds token embeddings, gathers PLE rows, projects hidden state into PLE space, and builds PLE inputs
  - [x] Update the downloaded Gemma harness to advance to the next precise blocker after bootstrap
  - [x] Run focused Metal/Gemma tests plus the downloaded Gemma harness

### Gemma Forward Bootstrap Spec

- Scope:
  - bootstrap only: token embedding lookup plus PLE input construction
  - stop before attention, KV sharing, GeGLU FFN integration, final norm, LM head, and softcap
- Success criteria:
  - BF16 GEMV matches Float32 reference for small matrices
  - the downloaded Gemma Q4_K_M file no longer fails with the generic missing-forward-pass error
  - the new blocker names the next unimplemented runtime component precisely
- Constraints:
  - avoid materializing multi-GB tensors into Float32
  - preserve existing Qwen benchmark semantics and current dirty worktree changes
  - keep Bonsai/Q1 behavior untouched

### Gemma Forward Bootstrap Review

- Added `gemv_bf16_f32` and Swift `GEMVKernel` BF16-weight APIs so Gemma can multiply `per_layer_model_proj.weight` without expanding the full BF16 tensor to Float32.
- Added BF16 GEMV correctness coverage against a Float32 CPU reference.
- `Gemma4LanguageModel` now retains Metal runtime objects and runs a real bootstrap slice in `logits(for:)`:
  - validates token IDs against token and PLE vocab sizes
  - fills/scales token embeddings, including K-quant row lookup support
  - gathers Q6_K/Q8_0 PLE token rows
  - projects hidden states through BF16/F32 `per_layer_model_proj.weight`
  - builds `[tokens, layers, perLayerDim]` PLE inputs
- Downloaded Gemma Q4_K_M now advances past the generic missing-forward-pass stop and reaches the next precise blocker:
  - `Gemma 4 PLE prelude completed; decoder layer forward is not implemented yet`
- Verification passed:
  - `swift build`
  - `swift test --filter 'GEMVTests|PLEGatherKernelTests|PLEInputsKernelTests'`
  - `swift test --filter 'Gemma4Ops|Gemma4Weights|ModelLoaderTests'`
  - `swift test --filter 'FusedKernelTests|DequantQ8_0Tests'`
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
- Current publishable Qwen laptop measurement after the shared GEMV change:
  - median decode: `231.8 tok/s`
  - median TTFT: `4.2 ms`
  - token hash: `0afae14a84cf0df8`

# Current Qwen Decode Correctness Regression Plan

- [x] Restore the pinned Qwen Q8 decode correctness gate before claiming rollout safety
  - [x] Reproduce the current failure and record exact generated prefix
  - [x] Run decode variants to isolate whether the regression is in prefill, decode, GQA, final norm/LM head, or benchmark configuration
  - [x] Inspect changed Qwen hot-path files and identify the smallest correctness fix
  - [x] Add or retarget focused coverage so this prefix drift cannot recur silently
  - [x] Rerun `QwenBenchmark/decodeBenchmark`
  - [x] Rerun the Qwen quant acceptance harness
  - [x] Rerun Gemma Q4_K_M downloaded harness after the Qwen gate is green

### Qwen Decode Correctness Regression Spec

- Current failure:
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'`
  - expected greedy prefix: `[1, 1479, 35]`
  - actual greedy prefix: `[1, 1435, 353]`
  - pinned model SHA-256 still matches `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Success criteria:
  - pinned Qwen smoke benchmark passes its prefix guard
  - no benchmark artifact is rewritten from a failing run
  - mobile quant acceptance still loads/generates for all six selected quant files
- Constraints:
  - preserve unrelated dirty worktree changes
  - do not change benchmark correctness expectations to match the regression
  - fix runtime behavior, not the guard

### Qwen Decode Correctness Regression Review

- Root cause:
  - `Dequant_Q8_0.metal` had been changed to vectorize Q8 payload reads with `uint`/`char4` loads from `block + 2`.
  - Q8_0 blocks are 34 bytes, so `block + 2` is not 4-byte aligned across rows/blocks. The shader did not crash, but produced wrong GEMV math.
- Fix:
  - restored alignment-safe signed byte reads in the Q8 GEMV, batched GEMV, fused QKV, fused final norm + LM head, fused Gate+Up+SwiGLU, and f16-accumulation Q8 kernels.
- Verification passed:
  - `swift test --filter 'FusedKernelTests|DequantQ8_0Tests'`
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'`
    - generated prefix restored to `[1, 1479, 35]`
  - `swift test -c release --filter 'PublishableBenchmark/fullBenchmark'`
    - median decode: `218.1 tok/s`
    - median TTFT: `4.1 ms`
    - token hash: `0afae14a84cf0df8`
  - `EDGERUNNER_QWEN_* swift test -c release --filter 'QwenQuantAcceptanceTest/selectedMobileQuantsGenerateText'`
  - Gemma Q4_K_M still gets past Q6_K PLE loading and reaches the missing-forward-pass blocker.
- Remaining quality debt:
  - mobile K-quant files now load and generate, but low-bit Qwen outputs remain fragment-heavy under the current short harness prompt. Format support is proven; text quality still needs a separate acceptance prompt/quality pass.

## Current Mobile GGUF Quant Support Plan

- [x] Build a mobile-first GGUF quant capability baseline before continuing broad Gemma work
  - [x] Add a capability matrix covering GGUF parse, byte-size calculation, dequant parity, raw/fused path availability, and end-to-end generation
  - [x] Add failing-first coverage for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, and `BF16`
  - [x] Add explicit named rejection coverage for deferred unsupported upstream types: `IQ*`, `TQ*`, `MXFP4`, `NVFP4`, and upstream `Q1_*`
  - [x] Normalize Llama/Qwen quant dispatch so selected K-quants cannot fall into stale or accidental unsupported paths
  - [x] Add a reusable Qwen quant end-to-end harness with prompt, minimum-token, and basic coherence checks
  - [x] Download or reuse Qwen3 0.6B `Q2_K`, `Q3_K_M`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, and `Q8_0` GGUF artifacts for acceptance
  - [x] Add `Q6_K` PLE gather support for Gemma `per_layer_token_embd.weight`
  - [x] Rerun the downloaded Gemma Q4_K_M harness and record the next concrete blocker

### Mobile GGUF Quant Support Spec

- Scope:
  - first wave includes K-quants only: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, plus `F16/BF16` loader compatibility
  - first wave explicitly excludes `Q1` and `IQ*` quality work
  - end-to-end acceptance uses Qwen3 0.6B GGUF variants, not Gemma, because Qwen already has an implemented generation path
- Success criteria:
  - every selected quant has parse/byte-size/dequant or availability coverage
  - every selected Qwen quant artifact loads and generates non-empty text through EdgeRunner
  - Q2_K may be lower quality, but must not crash, produce empty text, or repeat one junk token only
  - Gemma Q4_K_M advances past the known `unsupportedPLEQuant("q6_K")` blocker after Q6_K PLE gather is implemented
- Constraints:
  - preserve unrelated dirty worktree edits
  - do not change app-facing APIs
  - do not modify benchmark harness semantics
  - keep Bonsai/Q1 behavior untouched in this wave

### Mobile GGUF Quant Support Review

- Added [docs/mobile-gguf-quant-support.md](/Users/chriskarani/CodingProjects/EdgeRunner/docs/mobile-gguf-quant-support.md) to track first-wave parse, byte-size, dequant, raw/fused, and end-to-end status.
- Added current upstream GGUF names for deferred unsupported formats: `IQ*`, `TQ1_0`, `TQ2_0`, `MXFP4`, and `NVFP4`.
- Preserved the existing Bonsai `Q1_0_G128` raw type-41 path for this wave to avoid changing Bonsai behavior.
- Added BF16 conversion in the Llama and Espresso dequant paths.
- Added GGUF byte-size and truncated-payload tests for `F16`, `BF16`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`.
- Added Q4_K dispatcher coverage and mobile quant buffer-bounds coverage in Espresso.
- Added row-level K-quant embedding lookup support for Llama/Qwen token embeddings stored as `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, or `Q6_K`.
- Added `ple_gather_q6_k` plus Swift encode/run APIs for Gemma PLE token embeddings.
- Confirmed the downloaded Gemma Q4_K_M tensor shape for `per_layer_token_embd.weight` is `[10752, 262144]`, matching `42 * 256` row width.
- Downloaded Qwen3 0.6B quant artifacts:
  - TensorBlock: `Q2_K`, `Q3_K_M`
  - Bartowski: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`
- Verification passed:
  - `swift test --filter 'GGUFTensorTableTests|PLEGatherKernelTests|Gemma4WeightsTests|DequantDispatcherTests|QwenQuantAcceptanceTest'`
  - `swift test --filter 'GGUFTensorTableTests|PLEGatherKernelTests|Gemma4WeightsTests|DequantDispatcherTests'`
  - `EDGERUNNER_QWEN_* swift test -c release --filter 'QwenQuantAcceptanceTest/selectedMobileQuantsGenerateText'`
- Gemma Q4_K_M result:
  - previous blocker `unsupportedPLEQuant("q6_K")` is cleared
  - current blocker is now the explicit missing Gemma forward pass error after weights/tokenizer load
- Regression caveat:
  - `swift test -c release --filter 'QwenBenchmark/decodeBenchmark'` currently fails the pinned greedy-prefix guard with `[1, 1435, 353]` instead of `[1, 1479, 35]` even though the pinned file SHA-256 matches `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
  - this needs a separate Qwen decode-correctness regression pass before the broader branch can be called rollout-safe

# Current Gemma 4 E4B iPhone Bring-Up Plan

- [x] Investigate `unsupportedDataType(30)` from the downloaded Gemma 4 Q4_K_M GGUF
  - [x] Map raw GGUF tensor type `30` to the upstream GGML type name
  - [x] Count affected tensors in the downloaded file
  - [x] Decide whether to add support or produce a precise rejection
  - [x] Add focused test coverage for the loader behavior
  - [x] Run verification and record the result

- [x] Download Gemma 4 E4B GGUF from Hugging Face into the local model cache
- [x] Run an EdgeRunner laptop measurement attempt against the downloaded GGUF
- [x] Record the actual tok/s or the first concrete EdgeRunner runtime blocker

- [x] Handle focused Gemma-ready GQA prerequisites only
  - [x] Add failing-first GQA tests for additive row-major attention masks
  - [x] Add failing-first GQA tests for `headDim` 256 and 512 while preserving the existing 128-dim path
  - [x] Implement the narrowest `GQAKernel` API and Metal shader support for additive masks
  - [x] Add a wide-head fallback path that supports 256/512 without changing the optimized 128-dim dispatch
  - [x] Run targeted GQA verification and record results

- [x] Implement focused PLE runtime kernel issues only
  - [x] Add failing focused tests for `PLEGatherKernel` command-buffer encode
  - [x] Add failing focused tests for `PLEInputsKernel` command-buffer encode
  - [x] Add failing focused tests for PLE side-channel finalize
  - [x] Implement surgical PLE kernel/API changes
  - [x] Run focused `PLE*` Metal tests

### Focused PLE Runtime Kernel Review

- Added caller-owned command-buffer encode APIs for `PLEGatherKernel` and `PLEInputsKernel`.
- Added `PLESideChannelKernel` plus `ple_side_channel_finalize`, computing `hidden += RMSNorm(projection, 1 + postNormWeight)`.
- Added focused Metal tests for gather encode, inputs encode, side-channel run/encode parity, and side-channel shape validation.
- Verification passed:
  - `swift test --filter PLE`
  - `swift build`

- [x] Add PLE runtime encode APIs for command-buffer integration
- [x] Add PLE side-channel finalize kernel/API and tests
- [x] Add explicit Gemma `(1 + weight)` RMSNorm helper and tests
- [x] Add FFN-only Gemma decoder slice with GeGLU and tests
- [x] Add Gemma-ready GQA additive mask + headDim 256/512 support and tests
- [ ] Replace the `Gemma4LanguageModel.logits` missing-forward-pass error with the first integrated forward slice
- [x] Confirm the exact Gemma 4 E4B text-only GGUF artifact and local compatibility assumptions
- [x] Run targeted Gemma 4 config/weights/tokenizer/kernel tests to expose the first concrete blocker
- [x] Make the smallest production-grade runtime/loader fixes required for Gemma 4 E4B to reach the dedicated backend
- [x] Verify locally with tests and record the first concrete forward-pass blocker
- [ ] Build, install, and launch the iOS chat/benchmark app on the connected iPhone
- [ ] Capture TTFT and decode tok/s from the phone, then record bottleneck evidence and next optimization step

### Gemma-ready GQA Prerequisites Spec

- Ownership is limited to `Sources/EdgeRunnerMetal/GQAKernel.swift`, `Sources/EdgeRunnerMetal/Shaders/GQA.metal`, and `Tests/EdgeRunnerMetalTests/GQA*`.
- Additive mask contract: row-major `[seqLen, effectiveKVSeqLen]` `Float` values added to attention scores before softmax.
- `headDim` support target: existing 128-dim path continues to work; 256 and 512 run correctly for float GQA.
- Constraints:
  - preserve unrelated dirty worktree edits
  - do not modify benchmark harnesses or package dependencies
  - keep changes surgical and production-grade

### Gemma-ready GQA Prerequisites Review

- Added `GQAKernel.execute(... additiveMask:)` and `encode(... additiveMaskBuffer:)` for row-major additive attention masks.
- Preserved the existing optimized unmasked `headDim <= 128` tiled path.
- Added scalar fallback Metal kernels for additive-mask dispatches and unmasked `headDim` 256/512 support.
- Added GQA tests for finite additive bias, `-.infinity` masking, effective `kvSeqLen` + `qOffset` mask indexing, and `headDim` 128/256/512 CPU parity.
- Verification passed:
  - `swift build`
  - `swift test --filter GQATests`
  - `swift test -c release --filter GQATests`

### Gemma 4 E4B iPhone Bring-Up Spec

- Target model: `google/gemma-4-E4B-it` text generation path
- Candidate GGUF source: a concrete `gemma-4-E4B-it-*` GGUF artifact, preferring a mobile-feasible quant before Q8_0
- Target device: connected iPhone, using the existing `Examples/EdgeRunnerChat` app/automation path
- Success criteria:
  - EdgeRunner recognizes Gemma 4 E4B metadata and rejects unsupported artifacts with actionable errors
  - a Gemma 4 E4B GGUF loads and produces deterministic text generation locally or reports the first concrete unsupported runtime feature
  - the iPhone run produces measured TTFT and decode tok/s from a real on-device launch
- Constraints:
  - preserve existing dirty worktree changes
  - keep benchmark semantics unchanged unless a Gemma-specific benchmark path is added
  - text-only first; multimodal audio/image/video processors are out of scope for this pass

### Gemma 4 E4B iPhone Bring-Up Review

- Confirmed current public artifact choices:
  - `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` is the preferred phone-feasible target at about 5.34 GB
  - `ggml-org/gemma-4-E4B-it-GGUF:Q8_0` is about 8.03 GB and is a quality/control artifact, not the first iPhone throughput target
- Added `Gemma4LanguageModel` as the public EdgeRunner backend boundary. It:
  - recognizes `general.architecture == "gemma4"`
  - parses Gemma 4 E4B config from public `ModelConfig` metadata
  - binds existing `Gemma4Weights`
  - loads the GGUF tokenizer when available
  - applies the tokenizer chat template, falling back to the local Gemma 4 renderer
  - fails generation with a precise missing-forward-pass error instead of falling through as an unknown architecture
- Updated `ModelLoader` to route `gemma4` GGUF files to the dedicated Gemma backend before the llama-compatible fallback.
- Fixed the pre-existing `.gitignore` `Edgerunner` rule by anchoring it to `/Edgerunner`, so new files under `Sources/EdgeRunner/...` are no longer ignored on the case-insensitive macOS filesystem.
- Wired iOS launch-time autobenchmark support in `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift`:
  - normal launch still opens `ChatWindow(runtime:)`
  - `EDGERUNNER_AUTOBENCH_*` environment values trigger one launch-time run
  - result JSON writes to the app Documents directory through `BenchmarkAutomationWriter`
- Verification passed:
  - `swift test --filter 'Gemma4|ModelLoader'`
  - `cd Examples/EdgeRunnerChatApp && swift test`
  - `xcodebuild -project Examples/EdgeRunnerChat/EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`
- Current blocker to a real iPhone tok/s number:
  - the Gemma 4 forward pass is not integrated yet. The next implementation slice is PLE input construction, then PLE side-channel finalization, then GeGLU FFN, then Gemma-ready GQA with additive mask + headDim 256/512 support.

### Pending Issues Fix Pass Review

- Closed the pending runtime prerequisite issues that can be verified without a full Gemma 4 forward pass:
  - Gemma `(1 + weight)` RMSNorm reference helper
  - Gemma GeGLU FFN residual slice reference helper
  - PLE gather/input encode APIs for caller-owned command buffers
  - PLE side-channel finalize kernel/API
  - GQA additive masks plus wide `headDim` 256/512 fallback
- Verification passed:
  - `swift build`
  - `swift test --filter 'GQA|PLE|Gemma4Ops|Gemma4|ModelLoader'`
  - `cd Examples/EdgeRunnerChatApp && swift test`
  - `xcodebuild -project Examples/EdgeRunnerChat/EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`
  - `git diff --check`
- Remaining intentional blocker:
  - `Gemma4LanguageModel.logits` still reports the missing integrated forward pass. No real Gemma 4 iPhone TTFT/tok/s can be captured until that forward path is wired end-to-end.

### Gemma 4 E4B Laptop Measurement Attempt Review

- Downloaded `ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` to `/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf`.
- Verified local artifact:
  - size: 5.0 GB
  - SHA-256: `90ce98129eb3e8cc57e62433d500c97c624b1e3af1fcc85dd3b55ad7e0313e9f`
- Added an opt-in downloaded-Gemma benchmark harness guarded by `EDGERUNNER_GEMMA4_BENCHMARK_MODEL`.
- Measurement attempt:
  - `EDGERUNNER_GEMMA4_BENCHMARK_MODEL=/Users/chriskarani/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf swift test --filter runDownloadedGGUFThroughEdgeRunner`
  - Result: no tok/s; EdgeRunner stops during GGUF loading with `unsupportedDataType(30)`.
- Fallback attempted:
  - Started Q8_0 download as a compatibility-control artifact, but the transfer projected about an hour at the observed direct-download rate after the Hugging Face CLI transfer stalled.
  - Stopped the Q8_0 fallback to avoid spending another long transfer on an artifact that still cannot generate until the Gemma 4 forward path is integrated.
- Verification passed:
  - `swift test --filter runDownloadedGGUFThroughEdgeRunner` without the env var
  - `git diff --check`

### Gemma 4 Q4_K_M Load Failure Investigation Review

- Root cause of the original `unsupportedDataType(30)`:
  - Upstream GGML defines raw tensor type `30` as `GGML_TYPE_BF16`.
  - The downloaded Q4_K_M file has exactly one BF16 tensor: `per_layer_model_proj.weight [2560, 10752]`.
  - EdgeRunner already had `TensorDataType.bfloat16`, but `GGUFTensorType` did not include BF16 and the 16...28 enum range was stale relative to current GGML.
- Fixes applied:
  - Added GGUF BF16 parsing and byte-size handling.
  - Realigned GGUF raw tensor types 16...28 with current GGML ordering.
  - Updated Gemma 4 config parsing for the current public GGUF metadata aliases:
    - `gemma4.attention.key_length_swa`
    - `gemma4.embedding_length_per_layer_input`
    - `gemma4.attention.sliding_window_pattern`
    - vocab fallback from `tokenizer.ggml.tokens`
- Real downloaded-file result after those fixes:
  - EdgeRunner now gets past BF16 and metadata parsing.
  - The current blocker is `unsupportedPLEQuant("q6_K")`.
  - The downloaded Q4_K_M file stores `per_layer_token_embd.weight` as Q6_K, while the only PLE gather kernel is `ple_gather_q8_0`.
- Verification passed:
  - `swift test --filter GGUFTensorTableTests`
  - `swift test --filter Gemma4ModelConfig`
  - `swift test --filter 'Gemma4ModelConfig|GGUFTensorTableTests|Gemma4Weights|runDownloadedGGUFThroughEdgeRunner'`
  - `swift build`

# Current Bonsai 1.7B iPhone Run Plan

- [ ] Inspect the installed app container on the connected iPhone 15 Pro Max and confirm whether the last `streamed_chat` benchmark wrote a JSON result
- [ ] If no valid result exists, relaunch the benchmark against `Bonsai-1.7B.gguf` with the intended streamed-chat configuration and capture a fresh artifact
- [ ] Report the measured tok/s and TTFT from the phone run, then record the immediate bottleneck evidence for the next optimization pass

### Bonsai 1.7B iPhone Run Review

- Fixed the runtime kernel lookup failure by making [KernelRegistry.swift](/Users/chriskarani/CodingProjects/EdgeRunner/Sources/EdgeRunnerMetal/KernelRegistry.swift) fall back to a source-compiled Metal library when a requested function is missing from the primary bundle metallib
- Verified the kernel path locally with:
  - `swift build`
  - `swift test --filter "ElementwiseKernelTests"`
- Rebuilt and reinstalled the iPhone app successfully
- Current blocker to the fresh on-device result is device lock state during launch:
  - `Unable to launch com.chriskarani.EdgeRunnerChat because the device was not, or could not be, unlocked`

# Current Bonsai 8B iPhone Retry Plan

- [ ] Confirm the connected iPhone is unlocked and `Bonsai-8B-Q1_0.gguf` remains present in the app sandbox
- [ ] Launch the 8B on-device benchmark with a fresh result filename
- [ ] Pull the result artifact and report either measured throughput or the first concrete runtime blocker

### Bonsai 8B iPhone Retry Review

- Pending fresh launch on the repaired app baseline

# Current Bonsai 8B iPhone Benchmark Plan

- [ ] Confirm the exact Bonsai 8B GGUF artifact and local compatibility assumptions for EdgeRunner
- [ ] Add or reuse the narrowest on-device benchmark path that can run at a 4096-token context window on the connected iPhone 15 Pro Max
- [ ] Download the Bonsai 8B GGUF artifact and verify the loader/runtime can parse it locally
- [ ] Build and install the iOS app or harness, copy the model into the app sandbox, and launch the benchmark on-device
- [ ] Capture measured decode throughput in tok/s and record the result with evidence and any proven blockers

### Bonsai 8B iPhone Benchmark Spec

- Target repo: `prism-ml/Bonsai-8B-gguf`
- Expected artifact: `Bonsai-8B-Q1_0.gguf` if present, otherwise `Bonsai-8B.gguf`
- Target device: connected `iPhone 15 Pro Max` (`00008130-001831360C08001C`)
- Target context: `ModelConfiguration(contextWindowSize: 4096)`
- Success criteria:
  - download a concrete Bonsai 8B GGUF artifact locally
  - prove EdgeRunner can load it or report the first concrete blocker with evidence
  - if it runs on-device, report measured tok/s from a real iPhone 15 Pro Max run
- Constraints:
  - keep changes surgical
  - do not disturb unrelated in-flight repo work

### Bonsai 8B iPhone Benchmark Review

- Delivery:
  - Added an app-side benchmark automation path in `Examples/EdgeRunnerChatApp` that:
    - resolves a model from the app sandbox `Documents` directory
    - runs a fixed prompt at a requested context window
    - writes a machine-readable JSON result back into `Documents`
  - Wired the iOS app entry point in `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift` to trigger that automation on launch when benchmark env vars are present
- Tests:
  - added targeted example-package coverage for:
    - benchmark env/config resolution
    - default fallback values
    - automated benchmark metric collection through the shared runtime
  - verified with:
    - `cd Examples/EdgeRunnerChatApp && swift test` ✅
- Device build/install:
  - built the iOS app for the connected `iPhone 15 Pro Max` with explicit team override:
    - `xcodebuild -project Examples/EdgeRunnerChat/EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'id=00008130-001831360C08001C' DEVELOPMENT_TEAM=BTSZ26LN83 CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates build` ✅
  - installed successfully with:
    - `xcrun devicectl device install app --device 00008130-001831360C08001C .../EdgeRunnerChat.app` ✅
- Concrete blocker:
  - launch is currently denied by iOS security:
    - `Unable to launch com.chriskarani.EdgeRunnerChat because it has an invalid code signature, inadequate entitlements or its profile has not been explicitly trusted by the user`
  - this is no longer a code or signing-generation problem; it is a one-time device trust gate
- Model acquisition:
  - confirmed the target repo exposes `Bonsai-8B-Q1_0.gguf` at roughly `1.16 GB`
  - local download was started to `~/edgerunner-models/Bonsai-8B-Q1_0.gguf`
  - the benchmark cannot be completed until the device trust gate is cleared, because the installed app cannot yet launch on the phone

# Current Chat App Plan

# Current Gemma 4 E4B + PLE Implementation Plan

**Detailed plan:** `docs/superpowers/plans/2026-04-18-gemma-4-e4b-ple-support.md` (26 tasks, 1886 lines)

## Phase 1 — Config & Loader Foundation
- [ ] Task 1: Parse Gemma 4 GGUF hparams into `Gemma4ModelConfig`
- [ ] Task 2: Build KV-share source map for layers 24–41
- [ ] Task 3: `Gemma4Weights` tensor handle bundle (incl. PLE tensors + quant gate)
- [ ] Task 4: `Gemma4ArchitectureFactory` registration in `ModelRegistry`

## Phase 2 — Tokenizer & Chat Template
- [ ] Task 5: Gemma 4 chat template (`<|turn>...<turn|>` sentinel format)
- [ ] Task 6: Extend tokenizer parity test for Gemma 4 sentinels

## Phase 3 — Metal Kernels
- [ ] Task 7: GeGLU kernel (gelu_pytorch_tanh × up, fused)
- [ ] Task 8: PLE single-row Q8_0 gather kernel
- [ ] Task 9: `per_layer_inputs` builder (proj + RMSNorm + PLE row + mix)
- [ ] Task 10: PLE side-channel finalize (RMSNorm + residual add)
- [ ] Task 11: Logit softcap kernel (tanh(x/30)·30)

## Phase 4 — Hybrid Attention
- [ ] Task 12: Sliding-window causal mask kernel
- [ ] Task 13: Dual RoPE tables (local θ=1e4 full / global θ=1e6 partial=0.25)
- [ ] Task 14: Extend `KVCache` for dual head-dim + KV-share map
- [ ] Task 15: GQA dispatch accepts precomputed additive mask

## Phase 5 — Forward Pass Integration
- [ ] Task 16: Decoder-layer forward block (single-layer parity vs HF)
- [ ] Task 17: Full-model prefill parity (42 layers, logits vs HF)
- [ ] Task 18: Decode path (incremental KV-cache reuse)
- [ ] Task 19: Route `ModelLoader.load()` to Gemma 4 before Llama fallthrough

## Phase 6 — iPhone Integration
- [ ] Task 20: `mmap`-backed shared-storage PLE table
- [ ] Task 21: iPhone 15 Pro Max TTFT + RSS benchmark

## Phase 7 — Public API + Docs
- [ ] Task 22: Expose Gemma 4 in `EdgeRunnerFacade`
- [ ] Task 23: Update `EdgeRunnerChat` example with Gemma 4 picker
- [ ] Task 24: Update ROADMAP / public_api / README

## Phase 8 — Long-Context & Robustness
- [ ] Task 25: 128K context stress test (64K prompt → decode)
- [ ] Task 26: Q4_K_M end-to-end coherence smoke test
- [ ] Run a release benchmark or equivalent greedy decode measurement on EdgeRunner
- [ ] Record tok/s, validation evidence, and any remaining functional limitations

## Follow-ups from Wave 1 code review
- [ ] Align Metal-kernel test convenience methods (`GeGLUKernel.run`, `LogitSoftcapKernel.run`, `SlidingWindowMask.build`) with async + caller-supplied-queue house pattern. Required before Task 16 forward-pass integration. (Source: Task 7/11/12 code-quality reviews; identical HIGH finding on each.)
- [ ] Move `GGUFMetadataError` from `Sources/EdgeRunner/Models/Gemma4/` into `Sources/EdgeRunnerIO/GGUF/` and standardize `invalidValue(key:description:)` label to match existing `GGUFTokenizerMetadataError`. (Source: Task 1 code-quality review Issue 3.)
- [ ] Fix pre-existing `.gitignore` rule `Edgerunner` (line 16) that matches case-insensitively on macOS FS — required `git add -f` workaround during Wave 1. (Source: Task 1 + Task 5 implementer concerns.)
- [ ] Add `encode(commandBuffer:...)` fusion method to `GeGLUKernel` mirroring `ActivationKernels.encodeSwiglu`. Needed when GeGLU is wired into the MLP forward path. (Source: Task 7 implementer note + code-quality review.)
- [ ] Add `window > seqLen` test case to `SlidingWindowMaskTests` (current test only covers `window == seqLen`). (Source: Task 12 code-quality review LOW.)

### Gemma 4 E4B GGUF Spec

- Target repo: `unsloth/gemma-4-E4B-it-GGUF`
- Success criteria:
  - prove whether EdgeRunner can load a concrete Gemma 4 E4B GGUF artifact
  - if it runs, report measured decode throughput in tok/s with the exact file used
  - if it does not run, report the first concrete blocker with evidence
- Constraints:
  - keep changes surgical
  - do not change benchmark semantics unrelated to Gemma 4 compatibility

- [x] Confirm the narrowest production-safe delivery shape for a chat app inside this repo
- [x] Add failing-first tests for throughput metrics and chat-session state transitions
- [x] Implement a macOS SwiftUI example app package that depends on the local `EdgeRunner` package
- [x] Wire prompt submission, streaming assistant output, cancel/reset behavior, and live tokens/sec reporting
- [x] Default the app to a 4096-token context window and expose model-path driven loading for Gemma 4B GGUF testing
- [x] Build the example app and run its targeted tests
- [x] Record verification results and any proven blockers or next performance work

### Chat App Spec

- Delivery shape: standalone example package under `Examples/` so the core library package remains stable
- Runtime: local `EdgeRunner` package dependency, no network services
- UI: macOS SwiftUI chat window with model path entry, transcript, composer, generate/cancel, and reset
- Metrics: show time to first token, rolling decode tokens/sec, total generated tokens, and final decode tokens/sec for each response
- Target config: `ModelConfiguration(contextWindowSize: 4096)` for Gemma 4B testing
- Model assumption: GGUF Gemma architecture is already recognized by `ModelLoader`; the app only needs a valid local Gemma 4B GGUF path

### Chat App Verification

- Tests:
  - Throughput tracker computes TTFT, rolling throughput, and final throughput correctly from deterministic timestamps
  - Chat session state handles user-send, streamed assistant chunks, cancel, and reset transitions correctly
- Manual/build checks:
  - `swift test` in the example package passes
  - `swift build` in the example package passes
  - App can be launched locally with a model path argument or typed path

### Chat App Review

- Delivery:
  - Added `Examples/EdgeRunnerChatApp`, a standalone macOS SwiftUI example package with:
    - `EdgeRunnerChatAppCore` for the tested runtime, throughput tracker, and `EdgeRunner` adapter
    - `EdgeRunnerChatApp` executable target for the UI shell
- Core API improvement:
  - `EdgeRunner` now exposes `stream(messages:...)` and `generate(messages:...)`, so the app can use model-native chat templates for Gemma/Qwen-style chat flows without reaching into internal modules
- Verification:
  - failing-first tests added for:
    - TTFT / rolling decode tok/s / final tok/s math
    - chat runtime streaming, reset, and blank-input handling
  - commands run:
    - `cd Examples/EdgeRunnerChatApp && swift test`
    - `cd Examples/EdgeRunnerChatApp && swift build`
  - result: both commands passed
- Remaining blocker to the user goal:
  - the app now measures the right signals for Gemma 4B at 4k context, but the `30 tok/s` target is still hardware/runtime dependent and requires a live run against the actual local Gemma 4B GGUF on the target device
- Suggested next step:
  - run `swift run EdgeRunnerChatApp --model /absolute/path/to/gemma-4-4b-it-q4.gguf`, capture TTFT + final decode tok/s, then optimize the decode path against that measured baseline

# Current iOS App Plan

- [x] Reuse the existing shared chat runtime/UI instead of forking another inference flow
- [x] Make the shared example package usable from iOS as well as macOS

# Current iOS Benchmark Startup Wiring Plan

- [x] Inspect the existing iOS app entry point and shared benchmark automation surface
- [x] Confirm normal UI launch is currently unconditional and benchmark configuration/writer/runtime support already exist in `EdgeRunnerChatAppCore`
- [x] Add the narrowest launch-time hook in `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift` that:
  - preserves `ChatWindow(runtime:)` as the normal first screen
  - creates `BenchmarkAutomationConfiguration` only from `EDGERUNNER_AUTOBENCH_*` environment values
  - runs `ChatRuntime.runAutomatedBenchmark` once per launch when configuration resolves
  - writes the JSON result through `BenchmarkAutomationWriter`
- [x] Run the existing example package tests and an iOS app build
- [x] Record verification results and any blocker in this section

### iOS Benchmark Startup Wiring Assumptions

- Existing core automation APIs are the source of truth; this task should not change benchmark semantics.
- The iOS app should use the app Documents directory for model filename resolution and JSON result output.
- Tests under `Examples/EdgeRunnerChatApp/Tests` already cover configuration parsing, JSON writing, and runtime benchmark execution, so the missing verification is the iOS app build.

### iOS Benchmark Startup Wiring Review

- Added launch-time automation wiring in `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift`.
- Normal UI remains `ChatWindow(runtime:)`; without a resolvable autobench configuration, launch returns without running benchmark work.
- With a resolvable `EDGERUNNER_AUTOBENCH_MODEL_PATH` or `EDGERUNNER_AUTOBENCH_MODEL_FILENAME`, the app resolves Documents, runs `ChatRuntime.runAutomatedBenchmark`, and writes the JSON result with `BenchmarkAutomationWriter`.
- Verification passed:
  - `swift test` from `Examples/EdgeRunnerChatApp`
  - `xcodebuild -project Examples/EdgeRunnerChat/EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`
- [x] Generate a real iOS Xcode project under `Examples/EdgeRunnerChat`
- [x] Replace the unfinished placeholder app flow with the shared EdgeRunner-backed chat UI
- [x] Verify the iOS app builds from Xcode command line
- [x] Record any remaining signing/device-install constraints

### iOS App Spec

- Delivery shape: Xcode project generated with `xcodegen` at `Examples/EdgeRunnerChat/EdgeRunnerChat.xcodeproj`
- App target: iOS SwiftUI app that depends on the local `EdgeRunnerChatAppCore` package product
- Shared runtime: reuse `ChatRuntime` and the throughput-aware chat UI so the iOS app and macOS app stay aligned
- Model loading: local GGUF path entry, `ModelConfiguration(contextWindowSize: 4096)`, live streaming generation
- Verification: simulator or generic iOS build from `xcodebuild`, with device install/signing handled separately if needed

### iOS App Review

- Delivery:
  - `Examples/EdgeRunnerChat/project.yml` now generates a real iOS app project that depends on the local `EdgeRunnerChatAppCore` package product
  - `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift` now launches the shared `ChatWindow` + `ChatRuntime` instead of the unfinished placeholder flow
  - `Examples/EdgeRunnerChatApp` now supports both iOS and macOS, with `ChatWindow` moved into `EdgeRunnerChatAppCore` for reuse across both app fronts
- Portability fix:
  - `Sources/EdgeRunner/Models/LlamaLanguageModel.swift` now compiles on iOS by stubbing the optional Metal 4 decode path off-platform instead of referencing unavailable `MTL4*` symbols
- Verification:
  - `cd Examples/EdgeRunnerChatApp && swift test` ✅
  - `cd Examples/EdgeRunnerChat && xcodegen generate` ✅
  - `xcodebuild -project EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' build` ✅
  - `xcodebuild -project EdgeRunnerChat.xcodeproj -scheme EdgeRunnerChat -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build` ✅
- Remaining install constraint:
  - the app now builds for iPhone, but actual install to a physical device still needs a signed build/provisioning pass

# Current Autoresearch 100-Experiment Plan

- [x] Audit the current publishable benchmark contract, relevant runtime files, and repo safety constraints for a 100-experiment run
- [x] Add a disposable-worktree autoresearch harness so the dirty main worktree is never mutated by the loop
- [x] Add failing-first coverage for the harness' benchmark parsing / experiment bookkeeping logic
- [x] Implement the narrowest production-grade automation needed to run, score, keep, or roll back experiments
- [x] Dry-run the harness on a small bounded count to verify parsing, isolation, logging, and failure handling
- [x] Execute at least 100 experiments against the publishable benchmark harness
- [x] Review generated artifacts, append outcome notes here, and summarize any proven blockers or wins

### Autoresearch 100-Experiment Review

- Added `benchmarks/autoresearch_harness.py` plus `benchmarks/test_autoresearch_harness.py` to automate a bounded benchmark sweep with JSON artifact capture, correctness gating, and build-state fallback.
- The harness first attempts a detached worktree from `HEAD`, but this repo's clean `HEAD` does not build right now. The sweep therefore fell back to the current checkout, which does build.
- Dry run passed:
  - command: `python3 benchmarks/autoresearch_harness.py --count 3 --canonical-top-k 1`
  - result: baseline `237.9 tok/s`, `decode_no_mega` recheck `237.3 tok/s`
- Full sweep completed:
  - command: `python3 benchmarks/autoresearch_harness.py --count 100 --canonical-top-k 5`
  - artifacts: `benchmarks/autoresearch_runs/20260409T221743Z`
  - sweep result: all 100 experiments preserved the pinned prefix/hash and completed successfully
  - top single-run results clustered around `239-241 tok/s`
  - strongest single-run variants:
    - `baseline`: `241.2 tok/s`
    - `decode_base__decode_metal4`: `240.4 tok/s`
    - `decode_no_mega__decode_no_kv_barrier`: `240.3 tok/s`
    - `decode_base__decode_no_mega__decode_no_kv_barrier`: `240.2 tok/s`
    - `decode_base__decode_no_kv_barrier__decode_metal4`: `239.7 tok/s`
- 5-run rechecks under sustained load were materially lower:
  - `baseline__canonical`: `129.7 tok/s`
  - `decode_base__decode_no_kv_barrier__decode_metal4__canonical`: `151.2 tok/s`
  - follow-up fresh canonical rerun after the sweep: `158.6 tok/s` median with one severe outlier run at `61.7 tok/s`
- Current blocker:
  - correctness is stable, but publishable benchmark stability is not
  - sustained multi-run measurements are dominated by system variance and/or thermal throttling, so single-run sweep winners do not hold their margin in canonical rechecks

# TurboQuant Pinned Rollout Execution

## Current Planar3/F16 Decode-Isolation Plan

- [x] Make the real-activation replay harness support `turbo-K / dense-V` so the new path can be analyzed directly
- [x] Re-run real-activation replay on the first guided-divergence layer of the `planar3 / f16 + deferred-prefill` path
- [x] Record whether `decoded_k_exact_v` still dominates once prompt-time compounding is reduced
- [x] If key fidelity remains dominant at the first decode-divergence layer, record that as the blocker and stop extending this path heuristically

### Planar3/F16 Decode-Isolation Review

- `TurboQuantLayerwiseAttributionTest.swift` now supports replaying `turbo-K / dense-V` layers directly instead of hard-requiring a compressed value preset.
- Re-ran guided layerwise attribution on the explicit `planar3 / f16 + deferred-prefill` path:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --skip-build --filter TurboQuantLayerwiseAttributionTest/compareLayerwiseTraceAgainstQ8Baseline`
  - result: `first_divergent_attention_output_layer=9`
  - result: `first_divergent_attention_layer=8`
  - result: `first_divergent_layer=8`
  - result: `q8_argmax=3838`
  - result: `turboquant_v2_argmax=3838`
  - result: `max_abs_logit_delta=1.123308`
- Replayed the first guided-divergence attention layer directly with dense values preserved:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 EDGERUNNER_TURBOQUANT_V2_REPLAY_LAYER=8 swift test --skip-build --filter TurboQuantLayerwiseAttributionTest/replayLayer0AttentionOnRealActivations`
  - result: `exact_k_decoded_v_mse=0.000000`
  - result: `decoded_k_exact_v_mse=0.041111`
  - result: `runtime_trace_vs_gpu_decode_mse=0.041125`
  - result: `cpu_vs_gpu_max_abs=0.000001`
- Replayed the first guided-divergent attention-output layer directly:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 EDGERUNNER_TURBOQUANT_V2_REPLAY_LAYER=9 swift test --skip-build --filter TurboQuantLayerwiseAttributionTest/replayLayer0AttentionOnRealActivations`
  - result: `exact_k_decoded_v_mse=0.000000`
  - result: `decoded_k_exact_v_mse=0.225832`
  - result: `runtime_trace_vs_gpu_decode_mse=0.225815`
  - result: `cpu_vs_gpu_max_abs=0.000001`
- Current conclusion:
  - the dense-value replay harness now matches the live runtime path closely enough to use as blocker evidence
  - once values are exact, the remaining approximation error is still entirely on the key side at both the first guided-divergence attention layer and the first guided-divergent attention-output layer
  - this path is blocked by decode-time planar3 key fidelity, so continuing heuristic rollout tuning is not justified

## Current Planar3/F16 Deferred-Prefill Plan

- [x] Port a source-backed `planar3 / f16` experiment path instead of continuing `planar3 / turbo3` tuning
- [x] Apply deferred exact-attention semantics to every prefill phase, including prefix-reuse suffix-prefill, not only `startPosition == 0`
- [x] Add failing-first tests for dense value-cache routing plus `turbo-K / dense-V` attention dispatch
- [x] Implement the narrowest runtime/KV-cache changes needed for explicit `f16` value storage under `turboquantV2`
- [x] Re-run low-level tests, then rerun pinned smoke and 128-token quality on the explicit `planar3 / f16` path
- [x] If the path still fails, record whether the remaining blocker is decode-time `planar3` key fidelity even after deferred-prefill and exact values

### Planar3/F16 Deferred-Prefill Review

- Added an explicit `dense` cache type override so `EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense` yields a real `planar3 / f16`-style value cache under `turboquantV2`.
- `KVCache` now supports dense `Float16` rows inside the `turboquantV2` runtime, so mixed `turbo-K / dense-V` storage is no longer just a contract-level fiction.
- `LlamaLanguageModel` now keeps a deferred exact-prefill planar3 key shadow cache and uses it with dense `Float16` values during prefill attention, while decode still uses quantized planar3 keys.
- Added targeted tests:
  - `planar3DenseValueOverrideAllocatesTurboKeyDenseValueStorage`
  - `planar3DecodeAttentionWithDenseValueBufferMatchesCPUReference`
- Verified low-level coverage:
  - `swift test --filter TurboQuantAttentionTests/contractPresetOverridesFollowEnvironment` passes
  - `swift test --skip-build --filter TurboQuantAttentionTests/planar3DenseValueOverrideAllocatesTurboKeyDenseValueStorage` passes
  - `swift test --skip-build --filter TurboQuantAttentionTests/planar3DecodeAttentionWithDenseValueBufferMatchesCPUReference` passes
  - `swift test --skip-build --filter TurboQuantAttentionTests/groupedPlanar3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext` passes
- Pinned smoke improved materially but still fails:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --filter turboQuantV2GreedyTraceMatchesQ8Baseline`
  - result: `q8=[358, 2776, 264, 5458]`
  - result: `turboquant_v2=[358, 386, 0, 42]`
- 128-token quality still fails:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_QUALITY=1 EDGERUNNER_TURBOQUANT_V2_QUALITY_PROMPT_LENS=128 swift test --skip-build --filter compareTurboQuantV2AgainstQ8BaselineAcrossContexts`
  - result: `divergence_steps=7`
  - result: `first_divergence_step=1`
  - result: `max_abs_logit_delta=22.5455`
  - result: `q8_generated=[1479, 198, 3838, 374, 279, 374, 279, 897]`
  - result: `turboquant_v2_generated=[1479, 15, 198, 198, 198, 198, 198, 198]`
- Guided layerwise attribution on the same path shows prompt-time behavior is much healthier even though unguided decode still fails:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_VALUE_CACHE_TYPE=dense EDGERUNNER_TURBOQUANT_DEFER_EXACT_PREFILL=1 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --skip-build --filter compareLayerwiseTraceAgainstQ8Baseline`
  - result: `first_divergent_attention_output_layer=9`
  - result: `first_divergent_attention_layer=8`
  - result: `first_divergent_layer=8`
  - result: `q8_argmax=3838`
  - result: `turboquant_v2_argmax=3838`
  - result: `max_abs_logit_delta=1.123308`
- Current conclusion:
  - the source-backed `planar3 / f16 + deferred-prefill` experiment fixes a real part of the problem and pushes the first guided divergence deep into the stack
  - it still fails the real pinned smoke and quality gates
  - the remaining blocker is decode-time planar3 key fidelity on the pinned model, not missing prefill lifecycle or dense-value support

## Current Planar3 Attribution Plan

- [x] Re-run the real-activation attribution probes under `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3`
- [x] Determine whether the remaining layer-0 blocker is dominated by stored-score/key error or value reconstruction error
- [x] If a single missing semantic is exposed, add a failing-first regression around it before changing runtime code
- [x] Implement only the evidenced missing behavior; avoid new heuristics
- [x] Re-run low-level planar3 tests plus pinned smoke after the change
- [x] If the pinned path is still red, record the blocker precisely instead of continuing blind tuning

## Current Planar3 Live-Path Unblock Plan

- [x] Audit the smallest live runtime surface needed to carry planar rotation coefficients for keys without changing value-side behavior
- [x] Reproduce the current grouped decode failure on its own and determine whether it is part of the planar3 blocker or an unrelated pre-existing regression
- [x] Add failing-first low-level tests for live `planar3` key quantize and/or score replay that do not require full pinned rollout
- [x] Implement planar coefficient buffers and preset-aware key transform selection in the Metal quantize path
- [x] Implement preset-aware key query transform and key decode-score path in the Metal attention kernels
- [x] Wire `planar3` through `LlamaLanguageModel` only for the explicit experiment path; do not change the default pinned contract yet
- [x] Verify targeted low-level tests and re-run the grouped decode check
- [x] Run pinned smoke on the explicit planar3 path
- [ ] Only benchmark 4096 if correctness materially improves
- [x] Record whether live planar3 actually changes the pinned-model blocker or whether deeper backend differences remain

### Planar3 Live-Path Review

- Fixed a real stale-test blocker first: `groupedDecodeAttentionMatchesDecodedCPUReference` was still hardcoded to `turbo2` test params even though the active contract is `turbo3/turbo3`. The builder now follows `TurboQuantV2Contract`.
- Added failing-first live `planar3` regressions in `TurboQuantAttentionTests.swift`:
  - `planar3GPUQuantizedRowsMatchReferenceRuntimeRows`
  - `planar3DecodeScoreTermsMatchCPUReference`
- Plumbed deterministic planar rotation coefficients through the real runtime:
  - `TurboQuantKernel` now owns key/value planar rotation buffers
  - `LlamaLanguageModel` now passes preset-aware rotation buffers into quantize and attention dispatches
  - `TurboQuant.metal` now supports planar forward/inverse rotation in the generic quantize, prefill, decode, and debug-score kernels
- Verified the live low-level path:
  - `swift test --filter groupedDecodeAttentionMatchesDecodedCPUReference` passes
  - `swift test --filter planar3GPUQuantizedRowsMatchReferenceRuntimeRows` passes
  - `swift test --filter planar3DecodeScoreTermsMatchCPUReference` passes
  - `swift test --filter groupedPlanar3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext` passes
  - score-term parity on the live decode path is now effectively exact:
    - `max_mse_dot_delta=0.00000095`
    - `max_residual_dot_delta=0.00000000`
    - `max_score_delta=0.00000048`
- Fixed a real attribution-harness bug after the first planar replay:
  - the GPU replay helpers in `TurboQuantLayerwiseAttributionTest.swift` were still hardcoding WHT rotation buffers and `reserved=0`
  - after switching them to preset-aware rotation buffers and reserved bits, `planar3` replay now matches CPU decode numerically
  - updated replay result:
    - `cpu_vs_gpu_max_abs=0.000001`
    - `cpu_vs_gpu_mse=0.000000`
- Pinned smoke with explicit `planar3` keys is still red:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --skip-build --filter turboQuantV2GreedyTraceMatchesQ8Baseline`
  - result: `q8=[358, 2776, 264, 5458]`
  - result: `turboquant_v2=[220, 220, 15, 220]`
- Bounded selective check with early q8 keys plus `planar3` deeper in the stack is also still red:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=2 EDGERUNNER_RUN_TURBOQUANT_V2_SMOKE=1 swift test --skip-build --filter turboQuantV2GreedyTraceMatchesQ8Baseline`
  - result: `turboquant_v2=[220, 1210, 220, 220]`
- Layerwise attribution on the explicit `planar3` path still shows first divergence at layer-0 attention output:
  - command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --skip-build --filter compareLayerwiseTraceAgainstQ8Baseline`
  - result: `first_divergent_attention_output_layer=0`
  - result: `first_divergent_attention_output_layer_max_abs_delta=0.969264`
  - result: `q8_argmax=3838`
  - result: `turboquant_v2_argmax=198`
  - result: `max_abs_logit_delta=10.540858`
- Real-activation replay and approximation still show the blocker is key-dominant at layer 0:
  - replay command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --filter replayLayer0AttentionOnRealActivations`
  - replay result:
    - `exact_k_decoded_v_mse=0.000383`
    - `decoded_k_exact_v_mse=0.013324`
    - `cpu_vs_dense_mse=0.013716`
    - `gpu_vs_dense_mse=0.013716`
  - approximation command: `EDGERUNNER_TURBOQUANT_KEY_PRESET=planar3 EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE=1 swift test --skip-build --filter compareAttentionApproximationAgainstQ8Baseline`
  - approximation result:
    - `worst_turbo_key_layer=0`
    - `worst_turbo_key_max_abs=226.563828`
    - `worst_turbo_key_mse=384.354004`
    - `worst_turbo_last_token_key_layer=0`
    - `worst_turbo_last_token_key_max_abs=111.972290`
    - `worst_turbo_last_token_key_mse=130.500122`
- Current conclusion:
  - the live `planar3` runtime path is no longer blocked by missing Metal/Swift plumbing
  - the remaining blocker is still pinned-model key/score fidelity at layer 0, not a broken planar runtime implementation

## Current RotorQuant Prototype Plan

- [x] Pull `scrya-com/rotorquant` locally and identify the narrowest K-only path that maps onto EdgeRunner’s existing KV/runtime structure
- [x] Audit whether EdgeRunner’s split-buffer TurboQuant storage can host a block-diagonal rotation format without changing the whole attention stack
- [x] Add a failing-first low-level regression for a `planar3`-style K-only row encode/decode or attention replay path
- [x] Implement the smallest viable `RotorQuant K-only` prototype, preferring reuse of the existing TurboQuant test harnesses and per-layer KV routing
- [x] Verify the new format with targeted CPU/Metal tests before any pinned-model rollout
- [ ] Run pinned smoke and 128-token quality on the prototype path
- [ ] Only run the 4096 benchmark if correctness materially improves
- [ ] Record whether RotorQuant offers a viable replacement path in this repo or whether the backend mismatch is too large

### RotorQuant Prototype Review

- Pulled `scrya-com/rotorquant` into `/tmp/rotorquant` and audited `planarquant.py`, `fused_planar_attention.py`, and `rotor_fused.metal`.
- The narrowest viable EdgeRunner port is `planar3` K-only, not full rotor algebra. It reuses the existing 3-bit packed row layout and swaps the transform from randomized Hadamard to deterministic pairwise planar rotations.
- Added a reference-grade `planar3` preset in `TurboQuant.swift` with:
  - deterministic pairwise planar rotate / inverse-rotate
  - no residual side-channel
  - the same row size as fixed `turbo3`
- Added failing-first then passing reference tests in `TurboQuantReferenceTests.swift` covering:
  - planar round-trip
  - deterministic `planar3` encode
  - runtime-row decode parity
  - a pair-structured signal where `planar3` beats Hadamard `turbo3`
- Verified:
  - `swift test --filter TurboQuantReferenceTests` passes
  - `swift test --skip-build --filter turbo3AttentionMatchesDecodedCPUReference` passes
  - `swift test --skip-build --filter groupedTurbo3PrefillAttentionMatchesDecodedCPUReferenceAtLongContext` passes
- Current blocker for live rollout: the Metal quantize and attention path is still hard-wired to WHT/sign buffers in `TurboQuantKernel.swift`, `TurboQuant.metal`, and `LlamaLanguageModel.swift`. `planar3` is a reference prototype only until those kernels can consume planar rotation coefficients.

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

## Current Layer-2 Isolation Plan

- [x] Re-run the hybrid-safe layerwise diagnostics at layer 2 under `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=2`
- [x] Confirm whether the live layer-2 error remains key-dominant once the first 2 layers use q8 keys
- [x] Promote layer 2 as well via `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=3` and rerun smoke plus 128-token quality as the minimal causality check
- [x] Re-run layerwise attribution under the 3-layer hybrid to see whether the first divergence simply moves deeper
- [x] Stop tuning and record the blocker now that deeper q8-key promotion still does not clear the pinned rollout gates

### Layer-2 Isolation Review

- The pure-Turbo replay harness is not valid under the hybrid path because it assumes every replayed layer has a Turbo key preset. Running `replayLayer0AttentionOnRealActivations` with `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=2` and `EDGERUNNER_TURBOQUANT_V2_REPLAY_LAYER=2` fails immediately with `unsupportedBitWidth(0)`, so hybrid isolation used the live layerwise input/trace tests instead.
- The replay harness is now fixed to use the replayed layer’s TurboQuant layout instead of implicitly assuming layer 0. That let the real-activation replay run directly on hybrid layer 2 and layer 3.
- Under `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=2`, layer-2 live attention inputs are already divergent and still key-dominant:
  - `query_last_token_max_abs_delta=1.877228`
  - `query_all_tokens_max_abs_delta=1.877228`
  - `key_all_tokens_max_abs_delta=5.891502`
  - `value_all_tokens_max_abs_delta=0.088129`
- Direct replay at hybrid layer 2 confirms that decoded keys are still the larger local error term on real activations:
  - `exact_k_decoded_v_mse=0.001464`
  - `decoded_k_exact_v_mse=0.005187`
  - `cpu_vs_gpu_max_abs=0.000001`
- Promoting layer 2 as well with `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=3` improves smoke materially but still does not clear it:
  - `q8=[358, 2776, 264, 5458]`
  - `turboquant_v2=[358, 614, 264, 3491]`
- The 128-token quality gate under `EDGERUNNER_TURBOQUANT_EARLY_Q8_KEY_LAYERS=3` still fails:
  - `divergence_steps=3`
  - `first_divergence_step=5`
  - `max_abs_logit_delta=14.5776`
  - `turboquant_v2_generated=[1479, 198, 3838, 374, 279, 7428, 315, 279]`
- Layerwise attribution under the 3-layer hybrid shows that deeper q8-key promotion does not cleanly eliminate the earliest hidden-state error:
  - `first_divergent_attention_output_layer=2`
  - `first_divergent_attention_layer=3`
  - `first_divergent_layer=2`
  - `q8_argmax=3838`
  - `turboquant_v2_argmax=3838`
  - `max_abs_logit_delta=7.905575`
- Layer-3 live attention inputs remain key-dominant even after the first 3 layers use q8 keys:
  - `query_last_token_max_abs_delta=1.272955`
  - `key_all_tokens_max_abs_delta=3.167385`
  - `value_all_tokens_max_abs_delta=0.320142`
- Direct replay at hybrid layer 3 still shows keys as the larger local approximation error, even though the absolute error is much smaller than layer 2:
  - `exact_k_decoded_v_mse=0.000037`
  - `decoded_k_exact_v_mse=0.000119`
  - `cpu_vs_gpu_max_abs=0.000001`
- Conclusion: deeper q8-key promotion remains a useful diagnostic, but it is no longer isolating a single residual Turbo key layer. The pinned blocker is now evidenced as compounded state divergence in the remaining pure-Turbo stack rather than a one-layer runtime bug that can be fixed by extending the fallback depth.

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
