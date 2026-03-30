# Code Images — Carbon URLs — 2026-03-24

## Image 1: SIMD Softmax Failure

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=//%20Experiment%2033%3A%20SIMD-optimized%20softmax%0A//%20Hypothesis%3A%20Replace%20threadgroup%20barriers%20with%20simd_sum%0A//%20Predicted%3A%20228%20%E2%86%92%20250%20tok%2Fs%20(%2B10%25)%0A//%20Result%3A%20WRONG%20TOKENS%0A%0Afloat%20globalMax%20%3D%20simd_max(newMax)%3B%0Afloat%20scaleToGlobal%20%3D%20exp(newMax%20-%20globalMax)%3B%0A%0A//%20Bug%3A%20renormalization%20broke%20online%20softmax%0A//%20Tokens%3A%20%5B1%2C%20101828%2C%20122053%5D%20%F0%9F%98%AC%0A%2F%2F%20Expected%3A%20%5B1%2C%201479%2C%2035%2C%205371%2C%201%5D&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 2: Metal ICB Limitation

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20Experiment%2034%3A%20Metal%20Indirect%20Command%20Buffers%0A%2F%2F%20Hypothesis%3A%20Pre-record%20142%20dispatches%0A%2F%2F%20Predicted%3A%20228%20%E2%86%92%20260%20tok%2Fs%20(%2B14%25)%0A%0Alet%20icbDesc%20%3D%20MTLIndirectCommandBufferDescriptor()%0A%0A%2F%2F%20iOS%2FtvOS%3A%20%E2%9C%85%20Works%0A%2F%2F%20macOS%3A%20%E2%9D%8C%20Not%20available%0AicbDesc.commandTypes%20%3D%20.concurrentCompute%0A%0A%2F%2F%20Error%3A%20Type%20'MTLIndirectCommandType'%20has%20no%20member%0A%2F%2F%20'concurrentCompute'%20on%20macOS&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 3: Scientific Method in Code

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20Autoresearch%20Swarm%3A%20Experiment%20Structure%0Astruct%20Experiment%20%7B%0A%20%20%20%20let%20hypothesis%3A%20String%0A%20%20%20%20let%20predictedImprovement%3A%20ClosedRange%3CDouble%3E%0A%20%20%20%20let%20failureMode%3A%20String%20%2F%2F%20What%20falsifies%20this%3F%0A%20%20%20%20%0A%20%20%20%20var%20result%3A%20Result%3C%0A%20%20%20%20%20%20%20%20Benchmark%2C%20%2F%2F%20Success%3A%20tok%2Fs%2C%20hash%0A%20%20%20%20%20%20%20%20ExperimentError%20%2F%2F%20Failure%3A%20why%0A%20%20%20%20%3E%0A%20%20%20%20%0A%20%20%20%20var%20verdict%3A%20Verdict%20%7B%0A%20%20%20%20%20%20%20%20improvement%20%3E%200.03%20%26%26%20pValue%20%3C%200.05%0A%20%20%20%20%20%20%20%20%3F%20.breakthrough%0A%20%20%20%20%20%20%20%20%3A%20.rollback%0A%20%20%20%20%7D%0A%7D&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 4: Tile-Based GEMV Kernel

**Carbon URL:**
```
https://carbon.now.sh/?l=c%2B%2B&t=dracula&bg=rgba(10,10,10,1)&code=kernel%20void%20dequant_q8_0_gemv_tiled(%0A%20%20%20%20device%20const%20uchar%20*weights%2C%0A%20%20%20%20device%20const%20float%20*x%2C%0A%20%20%20%20device%20float%20*dst%2C%0A%20%20%20%20...%0A)%20%7B%0A%20%20%20%20threadgroup%20float%20tile%5BTILE_SIZE%5D%3B%0A%20%20%20%20%0A%20%20%20%20%2F%2F%20Cooperatively%20load%20into%20threadgroup%20memory%0A%20%20%20%20for%20(uint%20i%20%3D%20tid.x%3B%20i%20%3C%20TILE_SIZE%3B%20i%20%2B%3D%2032)%20%7B%0A%20%20%20%20%20%20%20%20tile%5Bi%5D%20%3D%20x%5Bi%5D%3B%20%2F%2F%20Coalesced!%0A%20%20%20%20%7D%0A%20%20%20%20threadgroup_barrier(mem_flags%3A%3Amem_threadgroup)%3B%0A%20%20%20%20%0A%20%20%20%20%2F%2F%20Process%20from%20fast%20SRAM...%0A%20%20%20%20%2F%2F%20Result%3A%20%2B1.2%25%20%F0%9F%98%90%0A%7D&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Notes

To generate images from these URLs:
1. Open URL in browser
2. Screenshot or use Carbon's export feature
3. Save to `marketing/assets/code-images/`

Alternative: Use `carbon-now-cli` if installed:
```bash
npm install -g carbon-now-cli
carbon-now --config carbon-config.json snippet.swift
```
