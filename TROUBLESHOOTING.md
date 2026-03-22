# EdgeRunner Troubleshooting Guide

## Installation Issues

### "No such module 'EdgeRunner'"

**Cause**: Swift Package Manager hasn't resolved dependencies.

**Solution**:
```bash
# In Xcode: File → Packages → Resolve Package Versions
# Or from terminal:
swift package resolve
```

### Build fails with Metal shader errors

**Cause**: Metal shaders failed to compile.

**Solution**:
1. Ensure you're on macOS 26.0+ or iOS 26.0+
2. Clean build folder: **Cmd+Shift+K** in Xcode
3. Delete derived data:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData
   ```

## Model Loading Issues

### "Model load failed: File not found"

**Cause**: GGUF file path is incorrect or file doesn't exist.

**Solution**:
```swift
let path = "/absolute/path/to/model.gguf"
let url = URL(fileURLWithPath: path)

// Verify file exists
assert(FileManager.default.fileExists(atPath: path))
```

### "Unsupported weight data type"

**Cause**: Model uses a quantization format not yet supported.

**Supported formats**:
- Q8_0 ✅ (recommended)
- Q4_0 ✅
- Q4_K_M ✅
- Q2_K ✅
- Q3_K ✅
- Q5_0 ✅
- Q5_1 ✅
- Q5_K ✅
- Q6_K ✅
- F16/F32 ✅

**Unsupported**: IQ-quants and other GGUF types without a corresponding dequant path

**Solution**: Convert model to supported format or download a different variant.

### "Failed to create MTLBuffer"

**Cause**: Out of GPU memory.

**Solutions**:
1. Use a smaller model
2. Reduce context window size:
   ```swift
   let config = ModelConfiguration(contextWindowSize: 1024)  // Default is 4096
   ```
3. Close other applications using GPU
4. Enable memory mapping (default):
   ```swift
   let config = ModelConfiguration(useMemoryMapping: true)
   ```

## Runtime Issues

### Slow first inference

**Cause**: Metal pipeline state compilation (one-time cost).

**Expected**: First `logits()` call takes ~50-100ms, subsequent calls are fast.

**Mitigation**: Warm up the model:
```swift
// Warm-up before real usage
_ = try await model.logits(for: [1])
```

### NaN or Inf in logits

**Cause**: Numerical instability (rare).

**Diagnosis**:
```swift
let logits = try await model.logits(for: tokens)
let hasNaN = logits.contains(where: { !$0.isFinite })
```

**Solutions**:
1. Check model file integrity (re-download)
2. Try fallback decode path:
   ```bash
   export EDGERUNNER_DECODE_FORCE_BASE=1
   ```
3. Disable mega kernel for large models:
   ```bash
   export EDGERUNNER_DECODE_DISABLE_MEGA_GQA=1
   ```

### Incorrect/garbage output

**Cause**: Model mismatch or tokenizer issues.

**Checks**:
1. Verify model file SHA-256 matches official release
2. Ensure you're using the correct tokenizer for the model
3. Check token IDs are in vocabulary range:
   ```swift
   print("Vocab size: \(model.vocabularySize)")
   ```

**For Qwen3 models**: the model BOS/EOS tokens are typically `151643` / `151645`.
The benchmark harnesses are different: the publishable and smoke Qwen benchmarks intentionally use a pinned seed token (`1`) for comparability with the checked-in benchmark history.

### Crash on large context

**Cause**: Exceeded available memory.

**Solutions**:
1. Reduce `contextWindowSize` in configuration
2. Use Q4_K_M quantized models (half the memory)
3. Clear KV cache periodically:
   ```swift
   // Model resets cache automatically on new sequence
   // For manual reset, start fresh tokens array
   ```

## Performance Issues

### Lower than expected tokens/second

**Benchmark**: Run the canonical publishable benchmark
```bash
swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```

**Expected** (Apple M3 Max):
- Qwen3-0.6B-Q8_0: ~230-245 tok/s median decode (128-token publishable benchmark)
- Qwen3-1.7B-Q8_0: 170+ tok/s
- Qwen3-4B-Q8_0: 50+ tok/s

**If slower, check**:
1. **Debug vs Release**: Always run benchmarks in release mode:
   ```bash
   swift test -c release --filter "PublishableBenchmark/fullBenchmark"
   ```
2. **Thermal throttling**: Check Activity Monitor for CPU/GPU pressure
3. **Other GPU apps**: Close browsers with video, ML training, etc.
4. **Power mode**: Ensure Mac is plugged in (not battery saving)

**Benchmark modes**:
- Canonical publishable run: the default benchmark tuple (`EDGERUNNER_BENCHMARK_TOKENS=128`, `EDGERUNNER_BENCHMARK_RUNS=5`, `EDGERUNNER_BENCHMARK_CONTEXT=2048`) with no decode/profiling overrides. Writes `benchmarks/publishable_benchmark.json`, emits `PUBLISH:`
- Profile run: any non-default benchmark override or any decode/profiling override writes `benchmarks/publishable_profile_benchmark.json`, emits `PROFILE:`
  - non-default `EDGERUNNER_BENCHMARK_TOKENS`
  - non-default `EDGERUNNER_BENCHMARK_RUNS`
  - non-default `EDGERUNNER_BENCHMARK_CONTEXT`
  - `EDGERUNNER_PROFILE_LMHEAD`
  - `EDGERUNNER_DECODE_FORCE_BASE`
  - `EDGERUNNER_DECODE_PREFER_METAL4`
  - `EDGERUNNER_DECODE_DISABLE_MEGA_GQA`
  - `EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD`
  - `EDGERUNNER_DECODE_DISABLE_KV_BARRIER`

**Canonical publishable validation**:
- pinned GGUF size must match the enforced benchmark artifact
- all runs must be deterministic in-process
- the output must keep the pinned greedy prefix and full token hash for the canonical 128-token harness

### High memory usage

**Expected memory** (with `useMemoryMapping: true`):
| Model | Weights | KV Cache (2K ctx) | Total |
|-------|---------|-------------------|-------|
| Qwen3-0.6B-Q8_0 | ~805 MB | ~70 MB | ~875 MB |
| Qwen3-1.7B-Q8_0 | ~1.8 GB | ~150 MB | ~2.0 GB |
| Qwen3-4B-Q8_0 | ~4.3 GB | ~280 MB | ~4.6 GB |

**If higher**:
1. Verify you are running a release build:
   ```bash
   swift test -c release --filter "PublishableBenchmark/fullBenchmark"
   ```
2. Reduce `contextWindowSize` to shrink KV cache usage.
3. Use a smaller model or more aggressive quantization.
4. Keep `useMemoryMapping: true` so weights are paged instead of eagerly loaded.
5. Close other GPU-heavy apps before benchmarking.

## Correctness Verification

### Verify installation

```bash
# Run coherence test (checks "Paris" for "capital of France")
swift test --filter "CoherenceTest"

# Run canonical publishable benchmark
swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```

### Compare with reference

```bash
# Install llama.cpp for comparison
brew install llama.cpp

# Run same prompt
llama-cli -m model.gguf -p "The capital of France is" -n 10
```

## Debugging Environment Variables

| Variable | Purpose |
|----------|---------|
| `EDGERUNNER_DECODE_FORCE_BASE=1` | Use base decode path (slower, more stable) |
| `EDGERUNNER_DECODE_PREFER_METAL4=1` | Prefer the Metal 4 decode path for profiling/comparison |
| `EDGERUNNER_DECODE_DISABLE_MEGA_GQA=1` | Disable fused attention kernel |
| `EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD=1` | Disable fused final layer |
| `EDGERUNNER_RUN_QUALITY_COMPARISON=1` | Run quality comparison tests |
| `EDGERUNNER_RUN_4B_RECOVERY_CHECK=1` | Run 4B model correctness check |

## Getting Help

1. **Check the logs**: Run with verbose output
   ```swift
   // Add to your code
   print("Tokens: \(tokenIDs)")
   print("Logits range: \(logits.min()!) to \(logits.max()!)")
   ```

2. **Test with known-good model**: Qwen3-0.6B-Q8_0 is the most tested

3. **File an issue** with:
   - macOS version
   - Mac model (Apple Silicon generation)
   - Model file name and size
   - Minimal reproduction code
   - Error message or unexpected output

## Common Error Messages

### "Context window exceeded"

Input sequence is longer than `contextWindowSize`. Either:
- Increase context window (uses more memory)
- Truncate input
- Use a model with larger native context

### "Invalid token ID"

Token ID is outside vocabulary range [0, vocabSize). Check:
- Tokenizer matches model
- Special tokens are correct for the model family

### "GPU buffer allocation failed"

Out of memory. See ["Failed to create MTLBuffer"](#failed-to-create-mtlbuffer) above.

### "Metal device not available"

Running on Intel Mac or in simulator. EdgeRunner requires Apple Silicon.

## Platform-Specific Notes

### macOS
- Requires macOS 26.0 (beta) or later
- Metal 4 features are available on supported OS versions, but the optimized Metal 3 decode path remains the default unless `EDGERUNNER_DECODE_PREFER_METAL4=1` is set

### iOS
- Requires iOS 26.0 (beta) or later
- Memory-constrained: use smaller models (<3B parameters)
- Thermal throttling more aggressive than macOS

### Simulator
- Not supported (no Metal GPU)
- Use "My Mac (Mac Catalyst)" or physical device
