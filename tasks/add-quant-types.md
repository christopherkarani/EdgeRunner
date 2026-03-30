# Plan: Add All Quantization Types (6 New GGUF Quant Types)

## Current Status

**Already in main repo (uncommitted):**
| Type | Metal Shader | Kernel Wrapper | Tests | Status |
|------|-------------|---------------|-------|--------|
| Q6_K | ✓ `Dequant_Q6_K.metal` | ✓ `DequantQ6KKernel.swift` | ✓ `DequantQ6KTests.swift` | ✅ Complete |
| Q5_K | ✓ `Dequant_Q5_K.metal` | ✓ `DequantQ5KKernel.swift` | ✓ `DequantQ5KTests.swift` | ✅ Complete |
| Q3_K | ✗ Missing | ✗ Missing | ✓ `DequantQ3KTests.swift` | 🔴 Incomplete |
| Q2_K | ✗ Missing | ✗ Missing | ✓ `DequantQ2KTests.swift` | 🔴 Incomplete |
| Q5_0 | ✗ Missing | ✗ Missing | ✓ `DequantQ5_0Tests.swift` | 🔴 Incomplete |
| Q5_1 | ✗ Missing | ✗ Missing | ✗ Missing | 🔴 Not started |

**Integration files (not yet updated):**
- `DequantDispatcher.swift` — missing 6 cases
- `LlamaLanguageModel.swift` — `supportedTypes` only has 3 types, error message outdated
- `DequantDispatcherTests.swift` — no tests for new types

---

## Phase 1: K-Quant Family (superblock-based, 256 weights/block)

All use `DequantQ4KParams` struct. Pattern from `DequantQ4KMKernel.swift` + `Dequant_Q4_K_M.metal`.

### 1A. Q6_K — 210 bytes / 256 weights ✅ DONE

- `Dequant_Q6_K.metal` — kernel `dequant_q6_k`
- `DequantQ6KKernel.swift` — blockByteCount=210, weightsPerBlock=256
- `DequantQ6KTests.swift`

### 1B. Q5_K — 176 bytes / 256 weights ✅ DONE

- `Dequant_Q5_K.metal` — kernel `dequant_q5_k`
- `DequantQ5KKernel.swift` — blockByteCount=176, weightsPerBlock=256
- `DequantQ5KTests.swift`

### 1C. Q3_K — 110 bytes / 256 weights 🔴 INCOMPLETE

**Block layout:**
```
Offset 0:  hmask[32]  — high bit mask (1 bit per weight)
Offset 32: qs[64]     — lower 2 bits packed (4 per byte)
Offset 96: scales[12] — quantized 6-bit sub-block scales
Offset 108: d         — float16 master scale
```

**Dequant formula:** `value = d * scale[sub] * (q3_value - 4)` where `q3_value = (qs_2bits | (hmask_bit << 2))`

**Files needed:**
1. `Sources/EdgeRunnerMetal/Shaders/Dequant_Q3_K.metal` — kernel `dequant_q3_k`
2. `Sources/EdgeRunnerIO/DequantQ3KKernel.swift` — blockByteCount=110, weightsPerBlock=256, uses `DequantQ4KParams`
3. Tests already exist at `Tests/EdgeRunnerIOTests/DequantQ3KTests.swift`

**Test tolerance:** < 0.2

### 1D. Q2_K — 84 bytes / 256 weights 🔴 INCOMPLETE

**Block layout:**
```
Offset 0:  scales[16] — combined quantized scales/mins (4 bits each)
Offset 16: qs[64]     — 2-bit quants packed (4 per byte)
Offset 80: d          — float16 master scale
Offset 82: dmin       — float16 master min
```

**Dequant formula:** `value = d * sc * q2_value - dmin * m` where `sc = scales[sub] & 0xF`, `m = scales[sub] >> 4`

**Files needed:**
1. `Sources/EdgeRunnerMetal/Shaders/Dequant_Q2_K.metal` — kernel `dequant_q2_k`
2. `Sources/EdgeRunnerIO/DequantQ2KKernel.swift` — blockByteCount=84, weightsPerBlock=256, uses `DequantQ4KParams`
3. Tests already exist at `Tests/EdgeRunnerIOTests/DequantQ2KTests.swift`

**Test tolerance:** < 0.3

---

## Phase 2: Legacy Family (block-based, 32 weights/block)

All use `DequantParams` struct. Pattern from `DequantQ8_0Kernel.swift` + `Dequant_Q4_0.metal`.

### 2A. Q5_0 — 22 bytes / 32 weights 🔴 INCOMPLETE

**Block layout:**
```
Offset 0: d       — float16 scale (2 bytes)
Offset 2: qh[4]   — 5th bit storage for 32 values (4 bytes)
Offset 6: qs[16]  — lower 4 bits nibble-packed (16 bytes)
```

**Dequant formula:** `value = d * ((q4_nibble | (qh_bit << 4)) - 16)`

**Files needed:**
1. `Sources/EdgeRunnerMetal/Shaders/Dequant_Q5_0.metal` — kernel `dequant_q5_0`
2. `Sources/EdgeRunnerIO/DequantQ5_0Kernel.swift` — blockByteCount=22, weightsPerBlock=32, uses `DequantParams`
3. Tests already exist at `Tests/EdgeRunnerIOTests/DequantQ5_0Tests.swift`

**Test tolerance:** < 1e-2

### 2B. Q5_1 — 24 bytes / 32 weights 🔴 NOT STARTED

**Block layout:**
```
Offset 0: d       — float16 scale (2 bytes)
Offset 2: m       — float16 min (2 bytes)
Offset 4: qh[4]   — 5th bit storage (4 bytes)
Offset 8: qs[16]  — lower 4 bits nibble-packed (16 bytes)
```

**Dequant formula:** `value = d * (q4_nibble | (qh_bit << 4)) + m`

**Files needed:**
1. `Tests/EdgeRunnerIOTests/DequantQ5_1Tests.swift` — tests FIRST (TDD)
2. `Sources/EdgeRunnerMetal/Shaders/Dequant_Q5_1.metal` — kernel `dequant_q5_1`
3. `Sources/EdgeRunnerIO/DequantQ5_1Kernel.swift` — blockByteCount=24, weightsPerBlock=32, uses `DequantParams`

**Test tolerance:** < 1e-2

---

## Phase 3: Integration & Validation

### 3A. Update DequantDispatcher.swift

Add 6 cases to the switch statement:

```swift
case .q6_K:
    let blockByteCount = 210
    let weightsPerBlock = 256
    // ... validate + call DequantQ6KKernel

case .q5_K:
    let blockByteCount = 176
    let weightsPerBlock = 256
    // ... validate + call DequantQ5KKernel

case .q3_K:
    let blockByteCount = 110
    let weightsPerBlock = 256
    // ... validate + call DequantQ3KKernel

case .q2_K:
    let blockByteCount = 84
    let weightsPerBlock = 256
    // ... validate + call DequantQ2KKernel

case .q5_0:
    let blockByteCount = 22
    let weightsPerBlock = 32
    // ... validate + call DequantQ5_0Kernel

case .q5_1:
    let blockByteCount = 24
    let weightsPerBlock = 32
    // ... validate + call DequantQ5_1Kernel
```

### 3B. Update LlamaLanguageModel.swift

Line 320 — add to `supportedTypes`:
```swift
let supportedTypes: Set<TensorDataType> = [
    .float32, .float16, .q4_0, .q8_0, .q4_K,
    .q6_K, .q5_K, .q3_K, .q2_K, .q5_0, .q5_1
]
```

Line 354 — update error message:
```swift
+ "Supported: Q2_K, Q3_K, Q4_0, Q4_K_M, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, F16, F32. "
```

### 3C. Update DequantDispatcherTests.swift

Add 6 dispatch tests verifying correct output count for each new type.

---

## Implementation Order

1. **Q3_K Metal shader** (`Dequant_Q3_K.metal`)
2. **Q3_K Kernel** (`DequantQ3KKernel.swift`)
3. **Q2_K Metal shader** (`Dequant_Q2_K.metal`)
4. **Q2_K Kernel** (`DequantQ2KKernel.swift`)
5. **Q5_0 Metal shader** (`Dequant_Q5_0.metal`)
6. **Q5_0 Kernel** (`DequantQ5_0Kernel.swift`)
7. **Q5_1 Tests** (`DequantQ5_1Tests.swift`) — TDD, write first
8. **Q5_1 Metal shader** (`Dequant_Q5_1.metal`)
9. **Q5_1 Kernel** (`DequantQ5_1Kernel.swift`)
10. **DequantDispatcher** — add 6 cases
11. **LlamaLanguageModel** — update supportedTypes + error msg
12. **DequantDispatcherTests** — add 6 dispatch tests
13. **Run tests** — `swift test --filter Dequant`
14. **Full suite** — `swift test`

---

## Commit Strategy

One commit per type (6 commits), each self-contained with shader + kernel + dispatcher + tests. Enables easy bisection.

| Commit | Contents |
|--------|----------|
| 1 | Q6_K shader + kernel + dispatcher case |
| 2 | Q5_K shader + kernel + dispatcher case |
| 3 | Q3_K shader + kernel + dispatcher case |
| 4 | Q2_K shader + kernel + dispatcher case |
| 5 | Q5_0 shader + kernel + dispatcher case |
| 6 | Q5_1 shader + kernel + tests + dispatcher + supportedTypes |
