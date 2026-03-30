# Code Images — Carbon URLs — 2026-03-25

## Image 1: GenerationSession API (3 lines)

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=import%20EdgeRunner%0A%0Alet%20model%20%3D%20try%20await%20ModelLoader.load(from%3A%20modelURL)%0Alet%20session%20%3D%20GenerationSession(model%3A%20model%2C%20maxTokens%3A%201024)%0A%0Afor%20try%20await%20text%20in%20session.stream(prompt%3A%20%22Write%20a%20story%22)%20%7B%0A%20%20%20%20print(text%2C%20terminator%3A%20%22%22)%0A%7D&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 2: Mega-Kernel Phase Structure

**Carbon URL:**
```
https://carbon.now.sh/?l=c%2B%2B&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20dequant_q8_0_fused_ffn_block%20phases%0A%2F%2F%201024%20threads%20per%20threadgroup%0A%0A%2F%2F%20Phase%201%3A%20Wo%20GEMV%20%2B%20residual%20add%0Afor%20each%20thread%20tid%3A%0A%20%20%20%20afterAttn%5Btid%5D%20%3D%20Wo%5Btid%2C%3A%5D%20%C2%B7%20attnOut%20%2B%20residual%5Btid%5D%0A%0A%2F%2F%20Phase%202%3A%20Cooperative%20RMSNorm%20(1024%20elements)%0A%2F%2F%20Each%20SG%3A%20simd_sum%20of%20squares%20-%3E%20partial_sums%5BsgIdx%5D%0A%2F%2F%20SG0%3A%20reduce%2032%20partials%20via%20simd_sum%0A%2F%2F%20Broadcast%20via%20threadgroup%20memory%0A%0A%2F%2F%20Phase%203%3A%20Gate%20%2B%20Up%20%2B%20SwiGLU%0A%2F%2F%20Phase%204%3A%20Down%20GEMV%20%2B%20residual&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 3: Three Inference Modes (Auto-detected)

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20Inference%20mode%20detection%0Alet%20commonPrefixLen%20%3D%20countMatchingPrefix(previousTokenIDs%2C%20tokenIDs)%0A%0Alet%20isDecodeMode%20%3D%0A%20%20%20%20commonPrefixLen%20%3D%3D%20previousTokenIDs.count%20%26%26%0A%20%20%20%20tokenIDs.count%20%3D%3D%20commonPrefixLen%20%2B%201%0A%0Alet%20isPrefixReuseMode%20%3D%0A%20%20%20%20commonPrefixLen%20%3E%200%20%26%26%0A%20%20%20%20commonPrefixLen%20%3D%3D%20previousTokenIDs.count%20%26%26%0A%20%20%20%20tokenIDs.count%20%3E%20commonPrefixLen%20%2B%201%0A%0A%2F%2F%20Otherwise%3A%20Full%20Prefill%20(reset%20KV%20cache)&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 4: KV Cache Circular Buffer

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20KV%20Cache%20circular%20buffer%20state%0Astruct%20LayerState%20%7B%0A%20%20%20%20var%20writePos%20%3D%200%20%20%20%20%20%20%20%20%2F%2F%20Current%20position%20(0..maxSeqLen)%0A%20%20%20%20var%20totalWritten%20%3D%200%20%20%20%20%2F%2F%20Total%20tokens%20ever%20written%0A%7D%0A%0A%2F%2F%20When%20totalWritten%20%3E%20maxSeqLen%3A%0A%2F%2F%20Retrieval%20reads%20in%20two%20chunks%3A%0A%2F%2F%20%20%20%5BwritePos...maxSeqLen%5D%20then%20%5B0...writePos%5D%0A%2F%2F%20This%20correctly%20reconstructs%20causal%20view&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 5: powr vs pow (RoPE hardware optimization)

**Carbon URL:**
```
https://carbon.now.sh/?l=c%2B%2B&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20RoPE%20frequency%20computation%0A%2F%2F%20powr%20uses%20hardware%20reciprocal%20units%0A%2F%2F%20pow%20goes%20through%20the%20generic%20path%0A%0A%2F%2F%20GOOD%3A%20hardware-optimized%0Afloat%20frequency%20%3D%201.0f%20%2F%20powr(theta%2C%20exponent)%3B%0A%0A%2F%2F%20BAD%3A%20generic%20power%20function%0Afloat%20frequency%20%3D%201.0f%20%2F%20pow(theta%2C%20exponent)%3B%0A%0A%2F%2F%20Same%20mathematical%20result.%0A%2F%2F%20Different%20hardware%20units.%0A%2F%2F20Different%20performance.&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 6: Scratch Buffers (Zero-Allocation)

**Carbon URL:**
```
https://carbon.now.sh/?l=swift&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%2019%20pre-allocated%20scratch%20buffers%0A%2F%2F%20Zero%20MTLBuffer%20allocation%20per%20forward%20pass%0A%0Astruct%20ScratchBuffers%20%7B%0A%20%20%20%20let%20normed%20%3A%20MTLBuffer%20%20%20%20%20%20%2F%2F%20RMSNorm%20output%0A%20%20%20%20let%20afterAttn%20%3A%20MTLBuffer%20%20%20%2F%2F%20Post-attention%20residual%0A%20%20%20%20let%20outputA%2C%20outputB%20%3A%20MTLBuffer%20%2F%2F%20Layer%20ping-pong%0A%20%20%20%20let%20allQ%2C%20allK%2C%20allV%20%3A%20MTLBuffer%20%2F%2F%20Full%20QKV%20tensors%0A%20%20%20%20let%20ropeQ%2C%20ropeK%20%3A%20MTLBuffer%20%20%20%20%2F%2F%20RoPE%20output%0A%20%20%20%20let%20gateOut%2C%20upOut%20%3A%20MTLBuffer%20%20%2F%2F%20SwiGLU%20intermediates%0A%20%20%20%20let%20logits%20%3A%20MTLBuffer%20%20%20%20%20%20%20%20%2F%2F%20Final%20vocab%20output%0A%7D%0A%0A%2F%2F%20Ping-pong%20alternates%20every%20layer%0A%2F%2F%20Residual%20add%20%3D%20reading%20from%20previous%20buffer&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Image 7: Fused QKV GEMV (RMSNorm + 3 GEMVs → 1)

**Carbon URL:**
```
https://carbon.now.sh/?l=c%2B%2B&t=dracula&bg=rgba(10,10,10,1)&code=%2F%2F%20dequant_q8_0_fused_qkv%20replaces%3A%0A%2F%2F%20%201.%20RMSNorm%0A%2F%2F%20%202.%20Q%20GEMV%0A%2F%2F%20%203.%20K%20GEMV%0A%2F2F%20%204.%20V%20GEMV%0A%2F%2F%20---%0A%2F%2F%20with%201%20dispatch%0A%0Astruct%20ERFusedQKVParams%20%7B%0A%20%20%20%20uint%20qRows%3B%20%20%20%20%20%2F%2F%20numHeads%20*%20headDim%0A%20%20%20%20uint%20kvRows%3B%20%20%20%20%20%2F%2F%20numKVHeads%20*%20headDim%0A%20%20%20%20uint%20cols%3B%20%20%20%20%20%20%20%2F%2F%20dim%0A%20%20%20%20uint%20blocksPerRow%3B%0A%20%20%20%20float%20rmsEps%3B%0A%7D%3B&wt=none&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=true&fl=1&fm=JetBrains%20Mono&fs=14px&lh=133%25&si=false&es=2x&wm=false&ts=false
```

---

## Notes

To generate PNG images from these URLs:
1. Open each URL in a browser
2. Screenshot or use Carbon's export feature
3. Save to `marketing/assets/code-images/`

To use Carbon API (requires API key):
```bash
npm install -g carbon-now-cli
carbon-now --config carbon-config.json snippet.swift
```
