# Lessons

## 2026-03-16
- Verify the actual milestone baseline from the repo before starting execution. This workspace is mid-M2, so M3/M4 implementation prompts must be deferred until M2 is complete and verified locally.
- Validate buffer shapes and requested element types at public API boundaries before encoding Metal work. GPU kernels should never be the first place mismatched dimensions or precisions are discovered.
- If a model-layer API carries positional metadata like `startPos` or configuration like `ropeTheta`, wire it through the reference implementation immediately or keep it out of the public surface until supported.
- For autoregressive models, prefer prefix-equivalence tests over naive “offset output must differ” assertions. Causal correctness is best proven by comparing full-sequence logits with growing-prefix logits, not by assuming positional metadata always changes local outputs.
