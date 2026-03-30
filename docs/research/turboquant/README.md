# TurboQuant Research Notes

## Source Artifact

- Primary local paper copy: `turboquant-arxiv-2504.19874.pdf`
- Public source: <https://arxiv.org/abs/2504.19874>
- Cross-checked conference copy: <https://openreview.net/attachment?id=tO3ASKZlok&name=pdf>

## EdgeRunner-Relevant Takeaways

- TurboQuant is a **KV-cache compression** method, not a model-weight format.
- The paper's practical implementation uses **randomized Hadamard transforms** instead of dense QR rotations.
- The throughput story comes from **fused on-the-fly use of packed K/V rows**, not from explicitly reconstructing full-precision cache rows in HBM.
- The paper quantizes **generated tokens too**, so EdgeRunner should not keep decode-appended rows in an uncompressed side path.
- The paper gives an explicit `2.5`-bit `d=128` channel split. The exact `3.5`-bit split is not fully specified in the text copy, so the balanced preset must stay blocked until that detail is recovered from supplementary material or author code.

## Current Repo Status

- Public API scaffolding lives in `ModelConfiguration.kvCacheCompression`.
- `automatic` is the default API, but it currently resolves to `disabled`.
- CPU-side TurboQuant reference primitives are in `Sources/EdgeRunnerMetal/TurboQuant.swift`.
- The Metal decode/runtime path is not wired yet; explicit TurboQuant selections currently fail at model load with a descriptive error instead of silently changing behavior.
