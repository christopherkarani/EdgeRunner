# Implement Missing GGUF Dequantization Kernels for EdgeRunner

## Role

You are implementing Metal GPU dequantization kernels for the EdgeRunner Swift inference library. EdgeRunner currently supports Q4_0, Q8_0, and Q4_K_M quantization. You are adding support for: **Q5_0, Q5_1, Q2_K, Q3_K, Q5_K, Q6_K, Q8_K**.

## Context

EdgeRunner is a Swift 6.2 on-device LLM inference engine that runs GGUF models on Apple Silicon via Metal compute shaders. Each quantization type needs:

1. **A Metal compute shader** (`.metal` file) with a `dequant_*` kernel that unpacks quantized weights to float32
2. **A Swift kernel wrapper** (`.swift` file) that sets up Metal buffers, dispatches the kernel, and reads results
3. **Unit tests** that pack known float values into the quantized format, dequantize via GPU, and verify output within tolerance

### Existing Pattern to Follow

Every dequant kernel follows the same architecture. Study these three reference files as your template:

**Metal shader pattern** (`Sources/EdgeRunnerMetal/Shaders/Dequant_Q4_0.metal`):
- Struct for params (blockCount, outputOffset)
- Constants for block byte size and weights-per-block
- `kernel void dequant_*()` that reads packed bytes, extracts nibbles/bits, applies scale, writes float output
- One thread per block

**Swift wrapper pattern** (`Sources/EdgeRunnerIO/DequantQ4_0Kernel.swift`):
- `public final class Dequant*Kernel: Sendable`
- Init takes `MTLDevice`, creates pipeline via `KernelRegistry`
- `dequantise(blockData:blockCount:commandQueue:) async throws -> [Float]`
- Uses `DequantParams` or similar struct for GPU parameters
- Error handling via `DequantKernelError`

**Test pattern** (`Tests/EdgeRunnerIOTests/DequantQ4_0Tests.swift`):
- Helper function to pack float values into the quantized block format (CPU reference encoder)
- Helper function to dequantize on CPU (reference decoder)
- Test: pack known values → GPU dequant → compare against CPU reference within tolerance (1e-3)
- Test: multiple blocks
- Test: fused dequant+GEMV if applicable

**Shared types** (`Sources/EdgeRunnerSharedTypes/include/DequantParams.h`):
- C structs shared between Metal and Swift for kernel parameters
- Add new param structs here for K-quant types that need different parameters

**Error types and common params** (`Sources/EdgeRunnerIO/DequantKernel.swift`):
- `DequantParams`, `DequantGEMVParams`, `DequantQ4KParams` structs
- `DequantKernelError` enum
- Add new param structs here for the Swift side

## GGUF Quantization Block Formats

Each quantization type packs weights into fixed-size blocks. The formats below are from the GGUF specification (ggml). **These are the exact byte layouts you must implement.**

### Q5_0 (5-bit, zero-point quantization)
- **Block size:** 22 bytes, 32 weights
- **Layout:** `[half scale (2B)] [uint32 high_bits (4B)] [16 × uint8 packed_nibbles (16B)]`
- **Dequant:** Each weight uses 5 bits — 4 from the nibble + 1 from high_bits bitmap. `value = scale * (q - 16)` where q is the 5-bit value (0-31)
- **Nibble extraction:** Same as Q4_0 (low/high nibble from packed byte), plus bit from high_bits at the corresponding position

### Q5_1 (5-bit with min value)
- **Block size:** 24 bytes, 32 weights
- **Layout:** `[half scale (2B)] [half min (2B)] [uint32 high_bits (4B)] [16 × uint8 packed_nibbles (16B)]`
- **Dequant:** `value = scale * q + min` where q is the 5-bit value (0-31)

### Q2_K (2-bit K-quant)
- **Block size:** 256 weights per super-block
- **Super-block layout:** `[16 × uint8 scales (16B)] [16 × uint8 qs (64B)] [half d (2B)] [half dmin (2B)]` — total 84 bytes
- **Sub-blocks:** 16 sub-blocks of 16 weights each
- **Dequant per sub-block i:** `scale_i = d * (scales[i] & 0xF)`, `min_i = dmin * (scales[i] >> 4)`, each weight q is 2 bits from qs. `value = scale_i * q - min_i`

### Q3_K (3-bit K-quant)
- **Block size:** 256 weights per super-block
- **Super-block layout:** `[32 × uint8 hmask (32B)] [64 × uint8 qs (64B)] [12 × uint8 scales (12B)] [half d (2B)]` — total 110 bytes
- **Dequant:** 3-bit weights from qs (2 bits) + hmask (1 bit). Scales are 6-bit values packed into 12 bytes. `value = d * scale * (q - 4)`

### Q5_K (5-bit K-quant)
- **Block size:** 256 weights per super-block
- **Super-block layout:** `[half d (2B)] [half dmin (2B)] [12 × uint8 scales (12B)] [32 × uint8 qh (32B)] [128 × uint8 ql (128B)]` — total 176 bytes
- **Sub-blocks:** 8 sub-blocks of 32 weights each
- **Dequant:** 5-bit weights from ql (4 bits) + qh (1 bit). Scales/mins are 6-bit packed in 12 bytes (same as Q4_K_M). `value = d * scale * q - dmin * min`

### Q6_K (6-bit K-quant)
- **Block size:** 256 weights per super-block
- **Super-block layout:** `[128 × uint8 ql (128B)] [64 × uint8 qh (64B)] [16 × int8 scales (16B)] [half d (2B)]` — total 210 bytes
- **Sub-blocks:** 16 sub-blocks of 16 weights each
- **Dequant:** 6-bit weights from ql (4 bits) + qh (2 bits). Scales are signed int8. `value = d * scale * (q - 32)`

### Q8_K (8-bit K-quant)
- **Block size:** 256 weights per super-block
- **Super-block layout:** `[float d (4B)] [256 × int8 qs (256B)]` — total 260 bytes (plus 16B of bsums that can be ignored for dequant)
- **Dequant:** `value = d * qs[i]` — simplest K-quant, just scaled int8

## File Locations

Create files following the existing naming convention:

### Metal Shaders (one per quant type):
```
Sources/EdgeRunnerMetal/Shaders/Dequant_Q5_0.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q5_1.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q2_K.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q3_K.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q5_K.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q6_K.metal
Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_K.metal
```

### Swift Kernel Wrappers:
```
Sources/EdgeRunnerIO/DequantQ5_0Kernel.swift
Sources/EdgeRunnerIO/DequantQ5_1Kernel.swift
Sources/EdgeRunnerIO/DequantQ2_KKernel.swift
Sources/EdgeRunnerIO/DequantQ3_KKernel.swift
Sources/EdgeRunnerIO/DequantQ5_KKernel.swift
Sources/EdgeRunnerIO/DequantQ6_KKernel.swift
Sources/EdgeRunnerIO/DequantQ8_KKernel.swift
```

### Tests:
```
Tests/EdgeRunnerIOTests/DequantQ5_0Tests.swift
Tests/EdgeRunnerIOTests/DequantQ5_1Tests.swift
Tests/EdgeRunnerIOTests/DequantQ2_KTests.swift
Tests/EdgeRunnerIOTests/DequantQ3_KTests.swift
Tests/EdgeRunnerIOTests/DequantQ5_KTests.swift
Tests/EdgeRunnerIOTests/DequantQ6_KTests.swift
Tests/EdgeRunnerIOTests/DequantQ8_KTests.swift
```

### Modify:
```
Sources/EdgeRunnerSharedTypes/include/DequantParams.h  — add param structs for new types
Sources/EdgeRunnerIO/DequantKernel.swift                — add Swift param structs
Sources/EdgeRunner/Models/LlamaLanguageModel.swift      — update supportedTypes set and init dequant kernels
```

## Implementation Order

Start with the simplest and work up:

1. **Q8_K** — simplest K-quant (just `d * int8`), validates the K-quant pipeline works
2. **Q5_0** — extends Q4_0 pattern with 1 extra bit from a bitmap
3. **Q5_1** — like Q5_0 but with min value (like Q4_1)
4. **Q6_K** — most commonly requested K-quant, moderate complexity
5. **Q5_K** — similar to Q4_K_M but 5-bit
6. **Q2_K** — 2-bit with sub-block scales
7. **Q3_K** — most complex (3-bit with hmask + packed 6-bit scales)

## For Each Quant Type, Follow This TDD Loop

1. Write the CPU reference packer and unpacker functions in the test file
2. Write the test that packs known values → GPU dequant → compares against CPU reference
3. Create the Metal shader
4. Create the Swift kernel wrapper
5. Run `swift test --filter Dequant<Type>Tests` — all must pass
6. Commit with message: `feat: add <type> dequantization kernel`

## After All Kernels Are Done

Update `Sources/EdgeRunner/Models/LlamaLanguageModel.swift`:

1. In `validateQuantizationTypes()`, add the new types to `supportedTypes`:
```swift
let supportedTypes: Set<TensorDataType> = [
    .float32, .float16, .q4_0, .q8_0, .q4_K,
    .q5_0, .q5_1, .q2_K, .q3_K, .q5_K, .q6_K, .q8_K
]
```

2. In the init, create the new dequant kernel instances (follow the pattern of `dequantQ4_0`, `dequantQ8_0`, `dequantQ4KM`)

3. In the weight loading / dequantization dispatch, add cases for the new types

## Verification

After implementation:
```bash
swift test --filter "DequantQ5_0|DequantQ5_1|DequantQ2_K|DequantQ3_K|DequantQ5_K|DequantQ6_K|DequantQ8_K"
swift build -c release
```

If you have a GGUF model with Q6_K quantization (e.g., download one from HuggingFace), verify it loads and runs:
```bash
# Should no longer throw "unsupported quantization" error
swift test --filter coherentStoryWithTemperature
```

## Critical Implementation Notes

- **Byte order:** GGUF uses little-endian throughout. Metal on Apple Silicon is also little-endian, so no byte swapping needed.
- **Scale precision:** Q*_0 and Q*_1 types use `half` (float16) for scales. K-quant types use `half` for `d`/`dmin` master scales and `uint8` or `int8` for sub-block scales.
- **Tolerance:** Tests should verify dequantized values within `1e-3` of the CPU reference. The quantization itself is lossy, but the dequant step must be exact.
- **Thread dispatch:** One thread per block for simple types. For K-quants with 256 weights per super-block, you may want one thread per sub-block or per super-block depending on complexity.
- **The Q4_K_M shader** (`Dequant_Q4_K_M.metal`) is the best reference for K-quant scale decoding — Q5_K uses the same 12-byte scale packing. Study it carefully.
- **llama.cpp reference:** The canonical implementations are in `ggml/src/ggml-quants.c` (CPU) and `ggml/src/ggml-metal/ggml-metal.metal` (GPU). Use these as ground truth for the block formats if anything is ambiguous.

Keep working until all 7 quant types have passing tests and the full build succeeds. Only stop when `swift test` shows all dequant tests green.
