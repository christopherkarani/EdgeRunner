# EdgeRunner M3 Execution Tracker

## Status
- Current milestone: M3
- Current task: M3 complete
- Baseline before execution: `swift test` passes with 177 tests in 29 suites

## Checklist
- [x] Reconcile repo state against M3 plan and prompt constraints
- [x] Update task tracker for M3 execution
- [x] Task 1: EdgeRunnerWeightLoader protocol, WeightMap, ModelConfig
- [x] Task 2: GGUF header parser
- [x] Task 3: GGUF tensor table and memory mapping
- [x] Task 4: Q4_0 dequantisation kernel
- [x] Task 5: Q8_0 dequantisation kernel
- [x] Task 6: Q4_K_M dequantisation kernel
- [x] Task 7: SafeTensor loader
- [x] Task 8: NPZ loader
- [x] Task 9: Llama 3 architecture
- [x] Task 10: Convenience load API
- [x] Task 11: Memory pressure handler
- [x] Task 12: End-to-end integration and verification
- [x] Final verification: `swift build`
- [x] Final verification: `swift test`

## Review
- Added end-to-end IO verification suites under `Tests/EdgeRunnerIOTests/`.
- `swift test --filter "EndToEndLoadTests|PerformanceBenchmarkTests|PerplexityVerificationTests"` passed with 12 tests in 3 suites.
- `swift test` passed with 269 tests in 51 suites.
- `swift build` succeeded after Task 12 verification.

---

# Plan Document Review

## Checklist
- [x] Create review task tracker
- [x] Read all plan documents under `docs/plans/`
- [x] Check cross-document consistency, sequencing, and test strategy
- [x] Record review findings with severity and references

## Review
- High: M3 depends on `EdgeRunnerLanguageModel` before M4 introduces it.
- High: M3 convenience load API returns an uninitialized model and never loads weights.
- High: Master/M1 concurrency rules ban `@unchecked Sendable`, but multiple milestone plans require it.
- High: M4's logits-based protocol is incompatible with the planned Foundation Models backend.
- Medium: M2 claims GPT-2 124M validation, but the plan only validates randomly initialized reference models.
- Medium: M5 moves speculative decoding into M4 and omits master-plan deliverables for batched inference and model download/registry work.
- Medium: M5 LoRA training assumes gradient-bearing tensor APIs and optimizer kernels that are not introduced in prior milestones.

## M1 Reconciliation
- [x] Append reconciliation checklist and verify current baseline
- [x] Add fail-first tests for public API boundaries and `@unchecked Sendable` usage
- [x] Refactor Metal/Core internals so only Metal wrapper types use `@unchecked Sendable`
- [x] Re-run `swift build`, `swift test`, and `swift package describe --type json`
- [x] Record review notes and commit follow-up reconciliation changes

## M1 Reconciliation Review
- Narrowed `MetalBackend` to actor-safe public APIs only; raw Metal protocol types no longer appear on the public actor surface.
- Replaced non-wrapper `@unchecked Sendable` uses with package-level Metal wrapper handles for buffers, libraries, and pipeline states.
- Internalized subsystem support types (`BufferCache`, `KernelRegistry`, `CommandBatcher`, `ResidencyManager`, `BarrierTracker`, `GEMMKernel`) to keep the exported package surface focused on M1 user-facing APIs.
- Added source-backed architecture tests to prevent regressions in public API boundaries and `@unchecked Sendable` scope.
- Verification passed with `swift build`, `swift test` (74 tests), and `swift package describe --type json`.
- Wax CLI memory integration was attempted twice; one attempt returned `Invalid TOC: memory_binding not supported in v1` and one `remember` call hung until terminated.
