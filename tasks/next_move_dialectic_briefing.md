# Next Move Briefing: 4B Correctness vs Performance

## User Goal
- Beat `llama.cpp` on correctness and performance across `0.6B`, `1.7B`, and `4B`.
- Current blocking issue is `Qwen3-4B-Q8_0`: correctness is restored, but throughput is far behind `llama.cpp`.

## Current Evidence

### Strongly Supported
- `0.6B` publishable benchmark is healthy again:
  - `236.3 tok/s` median decode
  - `3.6 ms` TTFT
  - deterministic `YES`
- `4B` was incoherent on the previous optimized decode path:
  - optimized path after `currentHidden` fix: `26.2 tok/s`, still incoherent
  - base decode path with mega kernel still enabled: `36.9 tok/s`, still incoherent
  - base decode path with mega kernel disabled: `10.7 tok/s`, coherent
- `llama.cpp` on the same `4B` model and raw-completion story prompt is coherent at `45.85 tok/s`.
- The first real correctness break is therefore in EdgeRunner's large-shape mega fused decode attention path, not in the model checkpoint or tokenizer mapping.

### Moderately Supported
- The remaining `4B` performance gap is large:
  - EdgeRunner recovered `4B`: `10.7 tok/s`
  - `llama.cpp` `4B`: `45.85 tok/s`
- But the exact source of that gap is not isolated yet, because the recovered path changes more than just one kernel family:
  - it routes away from the optimized Metal 3 decode path
  - it also disables the mega fused attention kernel
- So the evidence shows where correctness breaks, but does not cleanly localize which retained simplifications dominate the speed loss.

## Candidate Next Moves

### Move A: Large-Shape Attention Kernel First
- Build a shape-correct replacement for the large-model decode attention path.
- Keep correctness guardrails, but prioritize restoring fast attention first because the current safe path is too slow to compete.

### Move B: Parity / Differential Harness First
- Build a token-parity and/or layerwise differential harness against `llama.cpp` before new kernel work.
- Treat the current evidence as insufficiently isolated; improve observability first, then optimize under a stricter correctness gate.

## Decision Criteria
- Maximize information gain per engineering day.
- Preserve the recovered `0.6B` benchmark.
- Reduce the chance of another “fast but wrong” branch.
- Produce a path that can realistically close a `~4.3x` `4B` throughput gap.
