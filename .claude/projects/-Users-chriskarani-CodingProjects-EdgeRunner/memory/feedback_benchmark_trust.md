---
name: Never trust cached benchmark JSON
description: Always run fresh benchmarks instead of reading stale JSON files — cached results can be wildly wrong
type: feedback
---

Never cite benchmark numbers from `benchmarks/publishable_benchmark.json` or any cached JSON file as ground truth. The file showed 14 tok/s when the actual throughput was 224 tok/s — a 16x error.

**Why:** The JSON was from an early unoptimized build or a cold run. It was never updated after performance improvements. Agents that read the file and report those numbers give completely wrong answers.

**How to apply:** When asked about performance, always run a fresh benchmark yourself. Use a test like:
```swift
let model = try await LlamaLanguageModel.load(from: url, configuration: ModelConfiguration(contextWindowSize: 2048))
var tokenIDs = model.tokenize("Once upon a time")
// generate 128 tokens, measure wall time, compute tok/s
```
Run at least 3 iterations and take the median. Ignore run 0 (cold start). Report the numbers you measured, not what a file says.
