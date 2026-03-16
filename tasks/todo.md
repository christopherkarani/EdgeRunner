# Remaining M2 Execution Tracker

## Status
- Current milestone: M2
- Current task: Task 12 - Transformer Block Composition (`in_progress`)
- Baseline before execution: `swift build` passes, `swift test` passes with 80 tests

## Checklist
- [x] Reconcile repo state against M2 plan
- [x] Add `EdgeRunnerTests` target scaffolding
- [x] Task 3: Softmax Kernel
- [x] Task 4: Flash Attention Forward Pass
- [x] Task 5: Grouped Query Attention (GQA)
- [x] Task 6: KV Cache (Ring Buffer)
- [x] Task 7: RoPE (Rotary Position Embeddings)
- [x] Task 8: RMSNorm Kernel
- [x] Task 9: LayerNorm Kernel
- [x] Task 10: Activation Kernels (SwiGLU, GELU, Sigmoid)
- [x] Task 11: EdgeRunnerModule Protocol
- [ ] Task 12: Transformer Block Composition
- [ ] Task 13: GPT-2 Reference Implementation
- [ ] Task 14: Integration Tests & Perplexity Verification
- [ ] Final verification: `swift build`
- [ ] Final verification: `swift test`

## Review
- Pending. Populate after task 14 with final verification notes, total test count, and git log summary.
