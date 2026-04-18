# Gemma 4 E4B + Per-Layer Embeddings Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship first-class Gemma 4 E4B-it (5.1B total / 2.3B active via PLE) text-only inference in EdgeRunner with Q4_K_M quant running on iPhone 15 Pro Max and M-series Mac at parity with HuggingFace transformers reference output.

**Architecture:** New `Gemma4LanguageModel` (parallel to `LlamaLanguageModel` / `BonsaiLanguageModel`) registered in `ModelRegistry` via a `Gemma4ArchitectureFactory`. Reuses existing Metal kernels (Q4_K_M / Q8_0 dequant, RMSNorm, GEMV, GEMM, GQA) and adds five new ones (GeGLU, PLE gather, PLE per-layer builder, PLE side-channel, logit softcapping). Hybrid sliding/global attention is implemented by selecting masks per layer in the existing GQA dispatch path. Dual RoPE (local 1e4, global pRoPE 1e6 with `partial_rotary=0.25`) uses two precomputed cos/sin tables. KV sharing (last 18 of 42 layers) reuses cache pointers from earlier layers. PLE table is kept mmap-backed in `MTLResourceStorageModeShared` so only touched rows become resident, mirroring Google's "PLE caching" guidance on unified memory.

**Tech Stack:** Swift 6.2 (strict concurrency, typed throws), Metal Shading Language 3.0+, Swift Testing (`import Testing`), GGUF v3, SentencePiece tokenizer, XCTest harness for Metal-GPU parity, `swift test -c release` for benchmarks.

---

## Non-Goals (explicit)

- Multimodal (vision / audio) — text-only path. Image / audio tokens (id ≥ 262144) are not supported in v1.
- Gemma 4 26B-A4B (MoE) and 31B dense — E4B only.
- Gemma 4 E2B — can be added trivially after E4B ships (same architecture, fewer layers) but not in this plan.
- Training / fine-tuning — inference only.
- Ollama-style engine compatibility shims — we read GGUF directly.

---

## Assumptions & Risks (to validate during Task 1)

1. **Gemma 4 E4B is dense (no AltUp)** — `enable_moe_block=false` in `text_config`, and no `altup_*` hparams present. Confirmed from `huggingface.co/google/gemma-4-E4B/config.json`. Gemma 3n's AltUp mechanism does NOT apply to Gemma 4 E4B. Injection is a simple residual add, not a multi-slot update.
2. **PLE must be loaded at Q8_0 minimum** — empirical constraint from every community GGUF (bartowski / unsloth / ggml-org). We enforce this in the loader: reject if `per_layer_token_embd.weight` is < Q8_0.
3. **KV-sharing target layer mapping** — each of layers 24..41 maps to the nearest preceding non-shared layer of the same mask type (sliding/global). Must be verified against `llama-model.cpp` `build_gemma4` graph during Task 15.
4. **Gemma4RMSNorm uses `(1 + weight)` trick** — Gemma 1/2/3 lineage convention; confirm in Task 3 by loading tensor stats (weights near 0, not near 1) and matching HF parity output.
5. **Chat template is the `<|turn>` sentinel format**, not `<start_of_turn>` / `<end_of_turn>`. Rendered from `chat_template.jinja` in the gated `-it` repo.

---

## File Structure

### New files (`Sources/`)

| Path | Responsibility |
|---|---|
| `Sources/EdgeRunner/Models/Gemma4LanguageModel.swift` | Top-level model type conforming to `EdgeRunnerLanguageModel` + `LogitsModel`. Owns tokenizer, KV cache, prefill/decode orchestration. |
| `Sources/EdgeRunner/Models/Gemma4/Gemma4ModelConfig.swift` | Parsed GGUF hparams (layers, hidden, PLE dims, layer-type schedule, KV-share map, RoPE tables, softcap). |
| `Sources/EdgeRunner/Models/Gemma4/Gemma4Weights.swift` | Tensor handles for all weights (attn QKVO, FFN gate/up/down, RMSNorms, PLE tensors, embeddings). |
| `Sources/EdgeRunner/Models/Gemma4/Gemma4ChatTemplate.swift` | `<\|turn>role\n...<turn\|>` encoder. |
| `Sources/EdgeRunnerIO/Gemma4/Gemma4ArchitectureFactory.swift` | `ArchitectureFactory` registration + GGUF → `Gemma4ModelConfig` binding. |
| `Sources/EdgeRunnerMetal/PLEGatherKernel.swift` | Swift host side of `ple_gather_q8_0`. Single-row dequant per token per layer. |
| `Sources/EdgeRunnerMetal/PLEInputsKernel.swift` | Builds `per_layer_inputs[B,S,L,P]` = `RMSNorm(Wproj·h · 1/√H) + ple_row·√P` all times `1/√2`. |
| `Sources/EdgeRunnerMetal/PLESideChannelKernel.swift` | Per-layer: `RMSNorm(Wproj_out · (GELU-tanh(Wgate·x) ⊙ ple_inp_layer))` → residual add. |
| `Sources/EdgeRunnerMetal/GeGLUKernel.swift` | `gelu_pytorch_tanh(gate) ⊙ up` fused. |
| `Sources/EdgeRunnerMetal/LogitSoftcapKernel.swift` | `tanh(logits/30)·30`. |
| `Sources/EdgeRunnerMetal/SlidingWindowMask.swift` | Precomputed 512-token SWA mask + dispatch helper. |
| `Sources/EdgeRunnerMetal/Shaders/GeGLU.metal` | `gelu_tanh_mul_f32(gate, up, out)`. |
| `Sources/EdgeRunnerMetal/Shaders/PLE.metal` | Three kernels: `ple_gather_q8_0`, `ple_inputs_build`, `ple_side_channel`. |
| `Sources/EdgeRunnerMetal/Shaders/LogitSoftcap.metal` | `logit_softcap_f32`. |
| `Sources/EdgeRunnerMetal/Shaders/SlidingCausalMask.metal` | Generates per-layer additive mask. |

### Modified files

| Path | Change |
|---|---|
| `Sources/EdgeRunner/ModelLoader.swift:46-52` | Route Gemma 4 to `Gemma4LanguageModel.load()` before the Llama-compatible fallthrough. |
| `Sources/EdgeRunnerIO/ModelRegistry.swift:55-68` | Register `Gemma4ArchitectureFactory()` in `Default`. |
| `Sources/EdgeRunnerMetal/KVCache.swift` | Add dual-stride support: per-layer `headDim` (256 sliding / 512 global) + sharing map. |
| `Sources/EdgeRunnerMetal/RoPE.swift` | Accept two RoPE tables (local θ=1e4, global θ=1e6) + `partialRotaryFactor`. |
| `Sources/EdgeRunnerMetal/GQAKernel.swift` | Accept precomputed additive mask buffer. |
| `Sources/EdgeRunnerSharedTypes/include/KVCacheParams.h` | Add `uint32_t sharedKVSourceLayer` per-entry. |

### New test files

| Path | Purpose |
|---|---|
| `Tests/EdgeRunnerTests/Gemma4ConfigParityTest.swift` | GGUF hparam extraction matches `config.json`. |
| `Tests/EdgeRunnerTests/Gemma4ChatTemplateTest.swift` | Sentinel format matches Jinja reference. |
| `Tests/EdgeRunnerMetalTests/GeGLUKernelTest.swift` | GeGLU numerical parity vs `torch.nn.functional.gelu(g, approximate="tanh") * u`. |
| `Tests/EdgeRunnerMetalTests/PLEGatherKernelTest.swift` | Single-row Q8_0 gather parity. |
| `Tests/EdgeRunnerMetalTests/PLEInputsKernelTest.swift` | `per_layer_inputs` builder parity (float tolerance 1e-3). |
| `Tests/EdgeRunnerMetalTests/PLESideChannelKernelTest.swift` | Side-channel forward parity. |
| `Tests/EdgeRunnerMetalTests/LogitSoftcapTest.swift` | Softcap parity. |
| `Tests/EdgeRunnerMetalTests/SlidingWindowMaskTest.swift` | Mask entries are 0 or -inf at expected positions. |
| `Tests/EdgeRunnerTests/Gemma4HelloParityTest.swift` | Single-token logits vs HF `transformers` reference. |
| `Tests/EdgeRunnerTests/Gemma4LongGenerationParityTest.swift` | 128-token greedy parity. |
| `Tests/EdgeRunnerTests/Gemma4iPhoneMemoryBenchmark.swift` | RSS and TTFT on iPhone 15 Pro Max. |

---

## Phase 1 — Config & Loader Foundation

### Task 1: Parse Gemma 4 GGUF hparams into `Gemma4ModelConfig`

**Files:**
- Create: `Sources/EdgeRunner/Models/Gemma4/Gemma4ModelConfig.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4ConfigParityTest.swift`

**Reference values (must match `huggingface.co/google/gemma-4-E4B/config.json`):**
- `num_hidden_layers=42`, `hidden_size=2560`, `intermediate_size=10240`
- `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, `global_head_dim=512`
- `vocab_size=262144`, `max_position_embeddings=131072`
- `rms_norm_eps=1e-6`, `final_logit_softcapping=30.0`
- `hidden_size_per_layer_input=256` (P), `vocab_size_per_layer_input=262144`
- `num_kv_shared_layers=18`, `sliding_window=512`
- `layer_types`: 42-entry array with `full_attention` at indices 5,11,17,23,29,35,41

- [ ] **Step 1.1: Write failing test**

```swift
import Testing
@testable import EdgeRunner

@Suite("Gemma4ModelConfig parsing")
struct Gemma4ModelConfigTests {
    @Test("Parses E4B hparams from GGUF metadata")
    func parsesE4BHparams() throws {
        let metadata = Gemma4ModelConfigTests.makeReferenceMetadata()
        let config = try Gemma4ModelConfig(metadata: metadata)

        #expect(config.numHiddenLayers == 42)
        #expect(config.hiddenSize == 2560)
        #expect(config.intermediateSize == 10240)
        #expect(config.numAttentionHeads == 8)
        #expect(config.numKeyValueHeads == 2)
        #expect(config.headDim == 256)
        #expect(config.globalHeadDim == 512)
        #expect(config.vocabSize == 262144)
        #expect(config.maxPositionEmbeddings == 131072)
        #expect(config.rmsNormEps == 1e-6)
        #expect(config.finalLogitSoftcapping == 30.0)
        #expect(config.perLayerDim == 256)
        #expect(config.perLayerVocabSize == 262144)
        #expect(config.numKVSharedLayers == 18)
        #expect(config.slidingWindow == 512)
        #expect(config.layerTypes.count == 42)

        let globalLayers = config.layerTypes.enumerated()
            .compactMap { $0.element == .global ? $0.offset : nil }
        #expect(globalLayers == [5, 11, 17, 23, 29, 35, 41])
    }

    static func makeReferenceMetadata() -> [String: GGUFValue] {
        [
            "general.architecture": .string("gemma4"),
            "gemma4.block_count": .uint32(42),
            "gemma4.embedding_length": .uint32(2560),
            "gemma4.feed_forward_length": .uint32(10240),
            "gemma4.attention.head_count": .uint32(8),
            "gemma4.attention.head_count_kv": .uint32(2),
            "gemma4.attention.key_length": .uint32(256),
            "gemma4.attention.value_length": .uint32(256),
            "gemma4.attention.key_length_global": .uint32(512),
            "gemma4.attention.value_length_global": .uint32(512),
            "gemma4.vocab_size": .uint32(262144),
            "gemma4.context_length": .uint32(131072),
            "gemma4.attention.layer_norm_rms_epsilon": .float32(1e-6),
            "gemma4.final_logit_softcapping": .float32(30.0),
            "gemma4.embedding_length_per_layer": .uint32(256),
            "gemma4.per_layer_vocab_size": .uint32(262144),
            "gemma4.attention.shared_kv_layers": .uint32(18),
            "gemma4.attention.sliding_window": .uint32(512),
            "gemma4.layer_types": .string(
                "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global,"
                + "sliding,sliding,sliding,sliding,sliding,global"
            ),
        ]
    }
}
```

- [ ] **Step 1.2: Run test — expect FAIL (type not defined)**

```bash
swift test --filter Gemma4ModelConfigTests
```
Expected: `error: cannot find 'Gemma4ModelConfig' in scope`

- [ ] **Step 1.3: Implement `Gemma4ModelConfig`**

```swift
import Foundation
import EdgeRunnerIO

public enum Gemma4LayerType: String, Sendable, Equatable {
    case sliding
    case global
}

public struct Gemma4ModelConfig: Sendable, Equatable {
    public let numHiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let finalLogitSoftcapping: Float
    public let perLayerDim: Int
    public let perLayerVocabSize: Int
    public let numKVSharedLayers: Int
    public let slidingWindow: Int
    public let layerTypes: [Gemma4LayerType]
    public let ropeThetaLocal: Float
    public let ropeThetaGlobal: Float
    public let partialRotaryFactor: Float

    public init(metadata: [String: GGUFValue]) throws {
        func u32(_ key: String) throws -> UInt32 {
            guard case let .uint32(v) = metadata[key] else {
                throw GGUFMetadataError.missingKey(key)
            }
            return v
        }
        func f32(_ key: String) throws -> Float {
            guard case let .float32(v) = metadata[key] else {
                throw GGUFMetadataError.missingKey(key)
            }
            return v
        }
        self.numHiddenLayers = Int(try u32("gemma4.block_count"))
        self.hiddenSize = Int(try u32("gemma4.embedding_length"))
        self.intermediateSize = Int(try u32("gemma4.feed_forward_length"))
        self.numAttentionHeads = Int(try u32("gemma4.attention.head_count"))
        self.numKeyValueHeads = Int(try u32("gemma4.attention.head_count_kv"))
        self.headDim = Int(try u32("gemma4.attention.key_length"))
        self.globalHeadDim = Int(try u32("gemma4.attention.key_length_global"))
        self.vocabSize = Int(try u32("gemma4.vocab_size"))
        self.maxPositionEmbeddings = Int(try u32("gemma4.context_length"))
        self.rmsNormEps = try f32("gemma4.attention.layer_norm_rms_epsilon")
        self.finalLogitSoftcapping = try f32("gemma4.final_logit_softcapping")
        self.perLayerDim = Int(try u32("gemma4.embedding_length_per_layer"))
        self.perLayerVocabSize = Int(try u32("gemma4.per_layer_vocab_size"))
        self.numKVSharedLayers = Int(try u32("gemma4.attention.shared_kv_layers"))
        self.slidingWindow = Int(try u32("gemma4.attention.sliding_window"))

        guard case let .string(raw) = metadata["gemma4.layer_types"] else {
            throw GGUFMetadataError.missingKey("gemma4.layer_types")
        }
        let parsed = raw.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
        self.layerTypes = try parsed.map { token in
            guard let type = Gemma4LayerType(rawValue: token) else {
                throw GGUFMetadataError.invalidValue(
                    key: "gemma4.layer_types", value: token
                )
            }
            return type
        }
        guard layerTypes.count == numHiddenLayers else {
            throw GGUFMetadataError.invalidValue(
                key: "gemma4.layer_types",
                value: "expected \(numHiddenLayers) entries, got \(layerTypes.count)"
            )
        }
        self.ropeThetaLocal = 10_000.0
        self.ropeThetaGlobal = 1_000_000.0
        self.partialRotaryFactor = 0.25
    }
}

public enum GGUFMetadataError: Error, Equatable {
    case missingKey(String)
    case invalidValue(key: String, value: String)
}
```

- [ ] **Step 1.4: Run test — expect PASS**

```bash
swift test --filter Gemma4ModelConfigTests
```
Expected: all `#expect` pass.

- [ ] **Step 1.5: Commit**

```bash
git add Sources/EdgeRunner/Models/Gemma4/Gemma4ModelConfig.swift \
        Tests/EdgeRunnerTests/Gemma4ConfigParityTest.swift
git commit -m "feat(gemma4): parse E4B GGUF hparams into Gemma4ModelConfig"
```

---

### Task 2: Build KV-share source map for layers 24–41

**Rule:** Layers `i` where `i >= numHiddenLayers - numKVSharedLayers` (i.e. 24..41) reuse K/V from the nearest preceding layer with the same `layerType`. Cache this map at init time.

**Files:**
- Modify: `Sources/EdgeRunner/Models/Gemma4/Gemma4ModelConfig.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4ConfigParityTest.swift`

- [ ] **Step 2.1: Add failing test**

```swift
@Test("KV share map routes layers 24..41 to nearest same-type predecessor")
func kvShareMapIsCorrect() throws {
    let metadata = Gemma4ModelConfigTests.makeReferenceMetadata()
    let config = try Gemma4ModelConfig(metadata: metadata)

    #expect(config.kvSourceLayer(for: 0) == 0)
    #expect(config.kvSourceLayer(for: 23) == 23)

    #expect(config.kvSourceLayer(for: 24) == 22)  // sliding; 23 is sliding
    #expect(config.kvSourceLayer(for: 29) == 23)  // global; last global before was 23
    #expect(config.kvSourceLayer(for: 35) == 23)  // global still 23 (29 is shared itself)
    #expect(config.kvSourceLayer(for: 41) == 23)  // global — final layer
}
```

- [ ] **Step 2.2: Run — expect FAIL (no method)**

- [ ] **Step 2.3: Implement `kvSourceLayer(for:)`**

Append to `Gemma4ModelConfig`:
```swift
public func kvSourceLayer(for layer: Int) -> Int {
    let firstSharedLayer = numHiddenLayers - numKVSharedLayers
    guard layer >= firstSharedLayer else { return layer }

    let targetType = layerTypes[layer]
    var probe = layer - 1
    while probe >= 0 {
        if layerTypes[probe] == targetType && probe < firstSharedLayer {
            return probe
        }
        probe -= 1
    }
    return layer  // fallback: own layer
}
```

- [ ] **Step 2.4: Run — expect PASS**

- [ ] **Step 2.5: Commit**

```bash
git commit -am "feat(gemma4): compute KV-share source layer map"
```

---

### Task 3: `Gemma4Weights` tensor handle bundle

**Files:**
- Create: `Sources/EdgeRunner/Models/Gemma4/Gemma4Weights.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4WeightsTest.swift`

- [ ] **Step 3.1: Write failing test**

```swift
import Testing
import Metal
@testable import EdgeRunner
@testable import EdgeRunnerIO

@Suite("Gemma4Weights binding")
struct Gemma4WeightsTests {
    @Test("Binds all required tensor handles from weight map")
    func bindsAllTensors() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        let weightMap = Gemma4WeightsTests.makeStubWeightMap(config: config)
        let device = MTLCreateSystemDefaultDevice()!

        let weights = try Gemma4Weights(
            weightMap: weightMap, config: config, device: device
        )

        #expect(weights.tokenEmbedding != nil)
        #expect(weights.outputNorm != nil)
        #expect(weights.perLayerTokenEmbed != nil)
        #expect(weights.perLayerModelProjection != nil)
        #expect(weights.perLayerProjectionNorm != nil)
        #expect(weights.blocks.count == 42)
        let block0 = weights.blocks[0]
        #expect(block0.attnQ != nil && block0.attnK != nil)
        #expect(block0.attnV != nil && block0.attnO != nil)
        #expect(block0.ffnGate != nil && block0.ffnUp != nil && block0.ffnDown != nil)
        #expect(block0.inputNorm != nil && block0.postAttentionNorm != nil)
        #expect(block0.postFFNNorm != nil)
        #expect(block0.perLayerInputGate != nil)
        #expect(block0.perLayerProjection != nil)
        #expect(block0.postPerLayerInputNorm != nil)
    }

    @Test("Rejects weight map missing PLE tensors")
    func rejectsMissingPLE() throws {
        let config = try Gemma4ModelConfig(
            metadata: Gemma4ModelConfigTests.makeReferenceMetadata()
        )
        var weightMap = Gemma4WeightsTests.makeStubWeightMap(config: config)
        weightMap.removeValue(forKey: "per_layer_token_embd.weight")
        let device = MTLCreateSystemDefaultDevice()!

        #expect(throws: Gemma4LoadError.missingPLETensor("per_layer_token_embd.weight")) {
            try Gemma4Weights(weightMap: weightMap, config: config, device: device)
        }
    }

    static func makeStubWeightMap(config: Gemma4ModelConfig) -> WeightMap {
        // ... builds a minimal valid map with zero-filled tensors; see implementation
        fatalError("to be filled in Step 3.3")
    }
}
```

- [ ] **Step 3.2: Run — expect FAIL**

- [ ] **Step 3.3: Implement `Gemma4Weights`**

```swift
import Foundation
import Metal
import EdgeRunnerIO

public struct Gemma4BlockWeights: Sendable {
    public var inputNorm: TensorStorage
    public var attnQ: TensorStorage
    public var attnK: TensorStorage?
    public var attnV: TensorStorage?
    public var attnO: TensorStorage
    public var postAttentionNorm: TensorStorage
    public var ffnGate: TensorStorage
    public var ffnUp: TensorStorage
    public var ffnDown: TensorStorage
    public var postFFNNorm: TensorStorage
    public var perLayerInputGate: TensorStorage
    public var perLayerProjection: TensorStorage
    public var postPerLayerInputNorm: TensorStorage
}

public enum Gemma4LoadError: Error, Equatable {
    case missingTensor(String)
    case missingPLETensor(String)
    case unsupportedPLEQuant(String)
}

public struct Gemma4Weights: Sendable {
    public let tokenEmbedding: TensorStorage
    public let outputNorm: TensorStorage
    public let perLayerTokenEmbed: TensorStorage
    public let perLayerModelProjection: TensorStorage
    public let perLayerProjectionNorm: TensorStorage
    public let blocks: [Gemma4BlockWeights]

    public init(
        weightMap: WeightMap,
        config: Gemma4ModelConfig,
        device: MTLDevice
    ) throws {
        func require(_ name: String) throws -> TensorStorage {
            guard let t = weightMap[name] else {
                throw Gemma4LoadError.missingTensor(name)
            }
            return t
        }
        func requirePLE(_ name: String) throws -> TensorStorage {
            guard let t = weightMap[name] else {
                throw Gemma4LoadError.missingPLETensor(name)
            }
            return t
        }

        self.tokenEmbedding = try require("token_embd.weight")
        self.outputNorm = try require("output_norm.weight")
        let ple = try requirePLE("per_layer_token_embd.weight")
        let allowedPLEQuants: Set<GGUFTensorType> = [
            .q8_0, .q5_0, .q5_1, .q4_0, .q4_1, .f16, .f32, .bf16
        ]
        guard allowedPLEQuants.contains(ple.quantization) else {
            throw Gemma4LoadError.unsupportedPLEQuant(ple.quantization.rawValue)
        }
        self.perLayerTokenEmbed = ple
        self.perLayerModelProjection = try requirePLE("per_layer_model_proj.weight")
        self.perLayerProjectionNorm = try requirePLE("per_layer_proj_norm.weight")

        self.blocks = try (0..<config.numHiddenLayers).map { i in
            let kvSrc = config.kvSourceLayer(for: i)
            let ownK = kvSrc == i ? try require("blk.\(i).attn_k.weight") : nil
            let ownV = kvSrc == i ? try require("blk.\(i).attn_v.weight") : nil
            return Gemma4BlockWeights(
                inputNorm: try require("blk.\(i).attn_norm.weight"),
                attnQ: try require("blk.\(i).attn_q.weight"),
                attnK: ownK,
                attnV: ownV,
                attnO: try require("blk.\(i).attn_output.weight"),
                postAttentionNorm: try require("blk.\(i).post_attention_norm.weight"),
                ffnGate: try require("blk.\(i).ffn_gate.weight"),
                ffnUp: try require("blk.\(i).ffn_up.weight"),
                ffnDown: try require("blk.\(i).ffn_down.weight"),
                postFFNNorm: try require("blk.\(i).post_ffw_norm.weight"),
                perLayerInputGate: try require("blk.\(i).inp_gate.weight"),
                perLayerProjection: try require("blk.\(i).proj.weight"),
                postPerLayerInputNorm: try require("blk.\(i).post_norm.weight")
            )
        }
    }
}
```

Implement `makeStubWeightMap` in test helper to produce zero-filled `TensorStorage` values for every required name (see `Tests/EdgeRunnerTests/StubWeightMap.swift` pattern from `QwenHelloParityTest.swift`).

- [ ] **Step 3.4: Run — expect PASS**

- [ ] **Step 3.5: Commit**

```bash
git commit -m "feat(gemma4): bind weight tensor handles with PLE quant validation"
```

---

### Task 4: `Gemma4ArchitectureFactory` registration

**Files:**
- Create: `Sources/EdgeRunnerIO/Gemma4/Gemma4ArchitectureFactory.swift`
- Modify: `Sources/EdgeRunnerIO/ModelRegistry.swift`
- Test: `Tests/EdgeRunnerTests/ModelRegistryGemma4Test.swift`

- [ ] **Step 4.1: Failing test**

```swift
@Test("Default registry routes gemma4 to Gemma4ArchitectureFactory")
func defaultRegistryHandlesGemma4() throws {
    let registry = ModelRegistry.default
    let factory = registry.factory(for: "gemma4")
    #expect(factory is Gemma4ArchitectureFactory)
}
```

- [ ] **Step 4.2: Run — expect FAIL**

- [ ] **Step 4.3: Implement factory + registration**

```swift
// Gemma4ArchitectureFactory.swift
import Foundation

public struct Gemma4ArchitectureFactory: ArchitectureFactory {
    public let architectureName = "gemma4"
    public init() {}
    public func supports(architecture: String) -> Bool {
        architecture.lowercased() == "gemma4"
    }
}
```

Modify `ModelRegistry.default` to include:
```swift
registry.register(factory: Gemma4ArchitectureFactory())
```

- [ ] **Step 4.4: Run — expect PASS**

- [ ] **Step 4.5: Commit**

```bash
git commit -m "feat(gemma4): register Gemma4ArchitectureFactory in default registry"
```

---

## Phase 2 — Tokenizer & Chat Template

### Task 5: Gemma 4 chat template (sentinel format)

**Format** (verbatim from `chat_template.jinja` in `google/gemma-4-E4B-it`):
```
<bos><|turn>system
{system}
<turn|>
<|turn>user
{user}
<turn|>
<|turn>model
```

Notes:
- `assistant` role is rewritten to `model`.
- `bos_token` prepended once at top.
- Tool-call sentinels (`<|tool>…<tool|>`, `<|tool_call>call:name{k:v}<tool_call|>`, `<|tool_response>…<tool_response|>`) are wired but **optional in v1** — raise if the caller passes tool messages.

**Files:**
- Create: `Sources/EdgeRunner/Models/Gemma4/Gemma4ChatTemplate.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4ChatTemplateTest.swift`

- [ ] **Step 5.1: Write failing test**

```swift
import Testing
@testable import EdgeRunner

@Suite("Gemma4 chat template")
struct Gemma4ChatTemplateTests {
    @Test("Renders system + user turn with trailing model open")
    func rendersSystemUserThenModelOpen() {
        let rendered = Gemma4ChatTemplate.render(messages: [
            .init(role: .system, content: "You are helpful."),
            .init(role: .user, content: "Hi")
        ], addGenerationPrompt: true)

        #expect(rendered == """
        <bos><|turn>system
        You are helpful.
        <turn|>
        <|turn>user
        Hi
        <turn|>
        <|turn>model

        """)
    }

    @Test("Rewrites assistant role to model")
    func rewritesAssistantToModel() {
        let rendered = Gemma4ChatTemplate.render(messages: [
            .init(role: .user, content: "Hi"),
            .init(role: .assistant, content: "Hello!")
        ], addGenerationPrompt: false)

        #expect(rendered.contains("<|turn>model\nHello!\n<turn|>"))
        #expect(!rendered.contains("assistant"))
    }

    @Test("Throws on tool messages (unsupported in v1)")
    func throwsOnToolMessages() {
        #expect(throws: Gemma4ChatTemplateError.toolsUnsupportedInV1) {
            _ = try Gemma4ChatTemplate.renderThrowing(messages: [
                .init(role: .tool, content: "{\"result\": 42}")
            ], addGenerationPrompt: false)
        }
    }
}
```

- [ ] **Step 5.2: Run — expect FAIL**

- [ ] **Step 5.3: Implement**

```swift
import Foundation

public enum Gemma4ChatRole: String, Sendable, Equatable {
    case system, user, assistant, model, tool
}

public struct Gemma4ChatMessage: Sendable, Equatable {
    public let role: Gemma4ChatRole
    public let content: String
    public init(role: Gemma4ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

public enum Gemma4ChatTemplateError: Error, Equatable {
    case toolsUnsupportedInV1
}

public enum Gemma4ChatTemplate: Sendable {
    public static func render(
        messages: [Gemma4ChatMessage],
        addGenerationPrompt: Bool
    ) -> String {
        (try? renderThrowing(
            messages: messages, addGenerationPrompt: addGenerationPrompt
        )) ?? ""
    }

    public static func renderThrowing(
        messages: [Gemma4ChatMessage],
        addGenerationPrompt: Bool
    ) throws -> String {
        var output = "<bos>"
        for msg in messages {
            let role: String
            switch msg.role {
            case .tool: throw Gemma4ChatTemplateError.toolsUnsupportedInV1
            case .assistant: role = "model"
            case .system: role = "system"
            case .user: role = "user"
            case .model: role = "model"
            }
            output += "<|turn>\(role)\n\(msg.content)\n<turn|>\n"
        }
        if addGenerationPrompt {
            output += "<|turn>model\n"
        }
        return output
    }
}
```

- [ ] **Step 5.4: Run — expect PASS**

- [ ] **Step 5.5: Commit**

```bash
git commit -m "feat(gemma4): chat template renderer with sentinel format"
```

---

### Task 6: Extend `GemmaTokenizerParityTest` to cover Gemma 4 sentinels

Gemma 4 reuses the Gemma 3 SentencePiece model, but adds new special tokens `<|turn>` `<turn|>` etc. These are in `added_tokens_decoder` of `tokenizer_config.json`.

**Files:**
- Modify: `Tests/EdgeRunnerTests/GemmaTokenizerParityTest.swift`
- Add fixtures: `Tests/EdgeRunnerTests/Fixtures/gemma4_special_tokens.json`

- [ ] **Step 6.1: Add failing test**

```swift
@Test("Gemma 4 chat sentinels encode as single tokens")
func gemma4SentinelsAreSingleTokens() throws {
    let tokenizer = try Self.loadGemma4Tokenizer()
    let turnOpen = try tokenizer.encode("<|turn>", addSpecialTokens: false)
    let turnClose = try tokenizer.encode("<turn|>", addSpecialTokens: false)
    #expect(turnOpen.count == 1, "<|turn> must be a single token")
    #expect(turnClose.count == 1, "<turn|> must be a single token")
    #expect(turnOpen != turnClose)
}
```

- [ ] **Step 6.2: Run — expect FAIL (fixture missing)**

- [ ] **Step 6.3: Capture special-token IDs from HF and commit fixture**

```bash
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('google/gemma-4-E4B-it')
import json
print(json.dumps({tok: t.convert_tokens_to_ids(tok) for tok in [
    '<|turn>', '<turn|>', '<|tool>', '<tool|>', '<|tool_call>', '<tool_call|>',
    '<|tool_response>', '<tool_response|>', '<|channel>', '<channel|>',
    '<|think|>', 'model', '<bos>', '<eos>'
]}, indent=2))
" > Tests/EdgeRunnerTests/Fixtures/gemma4_special_tokens.json
```

- [ ] **Step 6.4: Run — expect PASS**

- [ ] **Step 6.5: Commit**

```bash
git commit -am "test(gemma4): verify sentinel tokens encode as single ids"
```

---

## Phase 3 — Metal Kernels

### Task 7: GeGLU kernel (gelu_pytorch_tanh + elementwise multiply)

**Math:** `y[i] = gelu_tanh(gate[i]) * up[i]` where
`gelu_tanh(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/GeGLU.metal`
- Create: `Sources/EdgeRunnerMetal/GeGLUKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/GeGLUKernelTest.swift`

- [ ] **Step 7.1: Write failing test**

```swift
import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("GeGLU kernel")
struct GeGLUKernelTests {
    @Test("Matches reference gelu_tanh(gate)*up within 1e-5")
    func matchesReference() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let kernel = try GeGLUKernel(device: device)
        let gate: [Float] = [-2, -1, 0, 0.5, 1, 2]
        let up: [Float]   = [ 1,  2, 3, 4.0, 5, 6]
        let out = try kernel.run(gate: gate, up: up)
        let expected = zip(gate, up).map { g, u in
            let c: Float = 0.7978845608
            let inner = c * (g + 0.044715 * g * g * g)
            return g * 0.5 * (1 + tanh(inner)) * u
        }
        for i in 0..<out.count {
            #expect(abs(out[i] - expected[i]) < 1e-5)
        }
    }
}
```

- [ ] **Step 7.2: Run — expect FAIL**

- [ ] **Step 7.3: Implement Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/GeGLU.metal
#include <metal_stdlib>
using namespace metal;

struct GeGLUParams { uint count; };

kernel void gelu_tanh_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float *out        [[buffer(2)]],
    constant GeGLUParams &p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float g = gate[gid];
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    float inner = c * (g + 0.044715f * g * g * g);
    float gelu = g * 0.5f * (1.0f + tanh(inner));
    out[gid] = gelu * up[gid];
}
```

Implement `GeGLUKernel.swift` host wrapper following the `swiglu_f32` dispatch pattern in existing `Activations.metal` wrapper (binds three buffers, threadgroup size 256).

- [ ] **Step 7.4: Run — expect PASS**

- [ ] **Step 7.5: Commit**

```bash
git commit -m "feat(metal): add GeGLU (gelu_pytorch_tanh × up) kernel for Gemma 4"
```

---

### Task 8: PLE single-row Q8_0 gather kernel

**Math:** For each token `t` in batch and each layer `ℓ`, gather one row from `per_layer_token_embd[tok_id, ℓ·P : (ℓ+1)·P]`, dequantize Q8_0 block (32 elements per block, 1 scale f16 + 32 int8), multiply by `√P = 16.0`.

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/PLE.metal`
- Create: `Sources/EdgeRunnerMetal/PLEGatherKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/PLEGatherKernelTest.swift`

- [ ] **Step 8.1: Failing test**

```swift
@Test("Gathers single PLE row and scales by sqrt(P)")
func gathersAndScales() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let P = 256
    let L = 42
    let vocab = 8
    let rows = Array(repeating: Float(0), count: vocab * L * P).enumerated().map { i, _ in
        Float(i % 137) / 137.0 - 0.5
    }
    let q8 = Self.quantizeToQ8_0(floats: rows)
    let kernel = try PLEGatherKernel(device: device, perLayerDim: P, numLayers: L)
    let tokens: [Int32] = [3, 7, 1]
    let out = try kernel.run(q8Table: q8, tokens: tokens)  // [tokens.count, L, P]

    let sqrtP: Float = sqrt(Float(P))
    for (tIdx, tok) in tokens.enumerated() {
        for ell in 0..<L {
            for p in 0..<P {
                let srcIdx = Int(tok) * L * P + ell * P + p
                let outIdx = tIdx * L * P + ell * P + p
                let expected = rows[srcIdx] * sqrtP
                #expect(abs(out[outIdx] - expected) < 1e-2)  // Q8_0 tolerance
            }
        }
    }
}
```

- [ ] **Step 8.2: Run — expect FAIL**

- [ ] **Step 8.3: Implement shader + host**

```metal
// Appends to PLE.metal
struct PLEGatherParams {
    uint perLayerDim;   // P
    uint numLayers;     // L
    uint numTokens;
    uint rowStrideBytes; // bytes per (token, L*P row) in Q8_0 storage
};

// Q8_0 block layout: f16 scale (2B) + 32 int8 quants
constant uint kQ8BlockBytes = 34;
constant uint kQ8BlockElems = 32;

kernel void ple_gather_q8_0(
    device const uchar *q8Table       [[buffer(0)]],
    device const int *tokens          [[buffer(1)]],
    device float *out                 [[buffer(2)]],
    constant PLEGatherParams &params  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tIdx = gid.y;
    uint elem = gid.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    if (tIdx >= params.numTokens || elem >= totalElems) return;

    int tokenId = tokens[tIdx];
    uint rowBase = uint(tokenId) * params.rowStrideBytes;
    uint blockIndex = elem / kQ8BlockElems;
    uint inBlock = elem % kQ8BlockElems;
    device const uchar *blockPtr = q8Table + rowBase + blockIndex * kQ8BlockBytes;

    half scale = *reinterpret_cast<device const half *>(blockPtr);
    int8_t q = *reinterpret_cast<device const int8_t *>(blockPtr + 2 + inBlock);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = float(scale) * float(q) * sqrtP;
}
```

- [ ] **Step 8.4: Run — expect PASS**

- [ ] **Step 8.5: Commit**

```bash
git commit -m "feat(metal): PLE Q8_0 single-row gather + sqrt(P) scale kernel"
```

---

### Task 9: `per_layer_inputs` builder kernel

**Math:** `per_layer_inputs[b,s,ℓ,p] = (RMSNorm(Wproj·h[b,s] · 1/√H)[ℓ,p] + ple_row[b,s,ℓ,p]) · 1/√2`

Depends on: matmul (`Wproj @ h` → `[B,S,L·P]`), reshape, RMSNorm along last dim, elementwise add, elementwise scale.

**Approach:** Implement as **two dispatches** for simplicity + existing kernels:
1. GEMV (batched) `h @ Wprojᵀ` → `[B*S, L*P]` with accumulator scaled by `1/√H`.
2. Fused kernel `ple_inputs_build` that takes the projection output + PLE rows, applies RMSNorm along P within each (batch, seq, layer) slice, adds the PLE row, and scales by `1/√2`.

**Files:**
- Append: `Sources/EdgeRunnerMetal/Shaders/PLE.metal`
- Create: `Sources/EdgeRunnerMetal/PLEInputsKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/PLEInputsKernelTest.swift`

- [ ] **Step 9.1: Failing test** (compares against pure-Swift reference on tiny shapes B=1, S=2, L=3, P=4, H=8):

```swift
@Test("Builds per_layer_inputs matching reference math")
func buildsPerLayerInputs() throws {
    let H = 8, L = 3, P = 4, BS = 2
    let h = (0..<(BS*H)).map { Float($0) * 0.01 - 0.05 }
    let wProj = (0..<(H*L*P)).map { Float($0) * 0.001 - 0.02 }   // [H, L*P]
    let normWeight = (0..<P).map { _ in Float(0.0) }             // (1+w) trick → identity
    let pleRows = (0..<(BS*L*P)).map { Float($0) * 0.02 - 0.1 }  // pre-scaled by sqrt(P)

    let device = MTLCreateSystemDefaultDevice()!
    let kernel = try PLEInputsKernel(device: device, hidden: H, perLayerDim: P, numLayers: L)
    let out = try kernel.run(
        hidden: h, perLayerProj: wProj, projNormWeight: normWeight, pleRows: pleRows, batchSeq: BS
    )

    let expected = PLEInputsKernelTests.referenceForward(
        h: h, wProj: wProj, norm: normWeight, pleRows: pleRows,
        H: H, L: L, P: P, BS: BS
    )
    for i in 0..<out.count {
        #expect(abs(out[i] - expected[i]) < 1e-3)
    }
}

static func referenceForward(
    h: [Float], wProj: [Float], norm: [Float], pleRows: [Float],
    H: Int, L: Int, P: Int, BS: Int
) -> [Float] {
    let scaleProj = 1.0 / sqrt(Float(H))
    let scaleMix = 1.0 / sqrt(Float(2))
    var out = [Float](repeating: 0, count: BS * L * P)
    for b in 0..<BS {
        // h @ wProj → [L*P]
        var proj = [Float](repeating: 0, count: L*P)
        for j in 0..<(L*P) {
            var acc: Float = 0
            for k in 0..<H { acc += h[b*H + k] * wProj[k*L*P + j] }
            proj[j] = acc * scaleProj
        }
        // RMSNorm along P within each layer-slice
        for ell in 0..<L {
            var sumSq: Float = 0
            for p in 0..<P {
                let v = proj[ell*P + p]
                sumSq += v * v
            }
            let rms = sqrt(sumSq / Float(P) + 1e-6)
            for p in 0..<P {
                let w = Float(1) + norm[p]
                let normed = proj[ell*P + p] / rms * w
                let mixed = (normed + pleRows[b*L*P + ell*P + p]) * scaleMix
                out[b*L*P + ell*P + p] = mixed
            }
        }
    }
    return out
}
```

- [ ] **Step 9.2: Run — expect FAIL**

- [ ] **Step 9.3: Implement kernel**

```metal
// Appends to PLE.metal
struct PLEInputsParams {
    uint hidden;
    uint perLayerDim;
    uint numLayers;
    uint batchSeq;
    float rmsEps;
    float scaleProj;  // 1/sqrt(H)
    float scaleMix;   // 1/sqrt(2)
};

// Input: projection buffer [BS, L*P] (already computed via existing GEMV with scaleProj applied)
// Output: per_layer_inputs [BS, L, P]
kernel void ple_inputs_build(
    device const float *proj     [[buffer(0)]],
    device const float *normW    [[buffer(1)]],
    device const float *pleRows  [[buffer(2)]],
    device float *out            [[buffer(3)]],
    constant PLEInputsParams &p  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint layerIdx = gid.y % p.numLayers;
    uint batchSeq = gid.y / p.numLayers;
    uint pIdx = gid.x;
    if (batchSeq >= p.batchSeq || pIdx >= p.perLayerDim) return;

    uint sliceBase = batchSeq * p.numLayers * p.perLayerDim + layerIdx * p.perLayerDim;

    // RMS along P for this (bs, layer) slice — naive per-thread fallback; production uses simdgroup reduce
    float sumSq = 0;
    for (uint i = 0; i < p.perLayerDim; ++i) {
        float v = proj[sliceBase + i];
        sumSq += v * v;
    }
    float rms = sqrt(sumSq / float(p.perLayerDim) + p.rmsEps);

    float v = proj[sliceBase + pIdx];
    float w = 1.0f + normW[pIdx];
    float normed = (v / rms) * w;
    float ple = pleRows[batchSeq * p.numLayers * p.perLayerDim
                      + layerIdx * p.perLayerDim + pIdx];
    out[sliceBase + pIdx] = (normed + ple) * p.scaleMix;
}
```

Production note: replace the per-thread RMS loop with a simdgroup reduction once parity lands. Track in `tasks/todo.md`.

- [ ] **Step 9.4: Run — expect PASS**

- [ ] **Step 9.5: Commit**

```bash
git commit -m "feat(metal): build per_layer_inputs (proj + RMSNorm + PLE row + mix)"
```

---

### Task 10: PLE side-channel per-layer kernel

**Math** (per decoder layer ℓ):
```
gate = Wgate_ℓ · h              # [B*S, P]
gate = gelu_pytorch_tanh(gate)
gate = gate ⊙ per_layer_inputs[:, ℓ, :]   # [B*S, P]
proj = Wproj_ℓ · gate            # [B*S, H]
proj = RMSNorm(proj)             # along H, with post_per_layer_input_norm weights
h_out = h + proj                 # residual add
```

**Approach:** Reuse existing GEMV kernel for the two matmuls. Write `ple_side_channel_finalize` as the fused (mul + RMSNorm + residual-add) kernel.

**Files:**
- Append: `Sources/EdgeRunnerMetal/Shaders/PLE.metal`
- Create: `Sources/EdgeRunnerMetal/PLESideChannelKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/PLESideChannelKernelTest.swift`

- [ ] **Step 10.1: Failing test** mirrors Task 9's reference pattern (see plan Task 9 test for template).

- [ ] **Step 10.2: Run — expect FAIL**

- [ ] **Step 10.3: Implement**

```metal
// Appends to PLE.metal — expects gelu-then-mul output already produced via GeGLU kernel
struct PLESideChannelParams {
    uint hidden;
    uint batchSeq;
    float rmsEps;
};

// Input:
//   proj:    [BS, H]  — output of Wproj_ℓ · (gelu(Wgate·h) ⊙ ple_inputs[:,ℓ,:])
//   postNormW: [H]    — post_per_layer_input_norm weight
// In/out:
//   h:       [BS, H]  — residual stream; updated in place
kernel void ple_side_channel_finalize(
    device float *h                [[buffer(0)]],
    device const float *proj       [[buffer(1)]],
    device const float *postNormW  [[buffer(2)]],
    constant PLESideChannelParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint bs = gid.y;
    uint hIdx = gid.x;
    if (bs >= p.batchSeq || hIdx >= p.hidden) return;

    uint base = bs * p.hidden;
    float sumSq = 0;
    for (uint i = 0; i < p.hidden; ++i) {
        float v = proj[base + i];
        sumSq += v * v;
    }
    float rms = sqrt(sumSq / float(p.hidden) + p.rmsEps);
    float v = proj[base + hIdx] / rms * (1.0f + postNormW[hIdx]);
    h[base + hIdx] += v;
}
```

- [ ] **Step 10.4: Run — expect PASS**

- [ ] **Step 10.5: Commit**

```bash
git commit -m "feat(metal): PLE side-channel finalize (RMSNorm + residual add)"
```

---

### Task 11: Logit softcap kernel

**Math:** `logits[i] = tanh(logits[i] / 30) * 30`.

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/LogitSoftcap.metal`
- Create: `Sources/EdgeRunnerMetal/LogitSoftcapKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/LogitSoftcapTest.swift`

- [ ] **Step 11.1: Failing test**

```swift
@Test("Softcaps logits with cap=30 to match tanh(x/30)*30")
func softcapsLogits() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let kernel = try LogitSoftcapKernel(device: device)
    let inp: [Float] = [-100, -30, -1, 0, 1, 30, 100]
    let out = try kernel.run(logits: inp, cap: 30)
    for (i, x) in inp.enumerated() {
        let expected = tanh(x / 30) * 30
        #expect(abs(out[i] - expected) < 1e-5)
    }
}
```

- [ ] **Step 11.2: Run — expect FAIL**

- [ ] **Step 11.3: Implement**

```metal
kernel void logit_softcap_f32(
    device float *logits [[buffer(0)]],
    constant float &cap  [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = logits[gid];
    logits[gid] = tanh(x / cap) * cap;
}
```

- [ ] **Step 11.4: Run — expect PASS**

- [ ] **Step 11.5: Commit**

```bash
git commit -m "feat(metal): final logit softcap kernel (tanh(x/30)*30)"
```

---

## Phase 4 — Hybrid Attention (Dual RoPE, Sliding Window, KV Share)

### Task 12: Sliding-window causal mask kernel

**Math:** For query position `q` attending to key position `k`:
- If `k > q` → `-inf` (causal)
- If `k < q - window + 1` → `-inf` (outside SWA window)
- Else → `0.0`

Window size = 512 for sliding layers; global layers get a standard causal mask (the same kernel with `window = maxSeqLen`).

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/SlidingCausalMask.metal`
- Create: `Sources/EdgeRunnerMetal/SlidingWindowMask.swift`
- Test: `Tests/EdgeRunnerMetalTests/SlidingWindowMaskTest.swift`

- [ ] **Step 12.1: Failing test**

```swift
@Test("SWA mask masks positions outside [q-window+1, q]")
func swaMaskIsCorrect() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let maker = try SlidingWindowMask(device: device)
    let mask = try maker.build(seqLen: 10, window: 3)  // [10, 10]
    // For q=5, allowed k = {3, 4, 5}
    #expect(mask[5 * 10 + 2] == -.infinity)
    #expect(mask[5 * 10 + 3] == 0)
    #expect(mask[5 * 10 + 4] == 0)
    #expect(mask[5 * 10 + 5] == 0)
    #expect(mask[5 * 10 + 6] == -.infinity)
}
```

- [ ] **Step 12.2: Run — expect FAIL**

- [ ] **Step 12.3: Implement**

```metal
struct MaskParams { uint seqLen; uint window; };

kernel void sliding_causal_mask_f32(
    device float *mask   [[buffer(0)]],
    constant MaskParams &p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q = gid.y;
    uint k = gid.x;
    if (q >= p.seqLen || k >= p.seqLen) return;
    bool outOfWindow = (k > q) || (q - k >= p.window && q >= p.window);
    mask[q * p.seqLen + k] = outOfWindow ? -INFINITY : 0.0f;
}
```

Edge case: when `q < window`, the `q - k >= window` check underflows if unsigned — guard with `q >= p.window`.

- [ ] **Step 12.4: Run — expect PASS**

- [ ] **Step 12.5: Commit**

```bash
git commit -m "feat(metal): sliding-window causal mask generator"
```

---

### Task 13: Dual RoPE tables (local θ=1e4, global pRoPE θ=1e6 partial=0.25)

**Local** (sliding layers): standard RoPE, full head-dim rotated, `base=10000`.
**Global** (global layers): pRoPE — only first `partial_rotary_factor × head_dim_global = 0.25 × 512 = 128` channels rotated; rest pass through. `base=1,000,000`.

**Files:**
- Modify: `Sources/EdgeRunnerMetal/RoPE.swift`
- Modify: `Sources/EdgeRunnerMetal/Shaders/RoPE.metal`
- Test: `Tests/EdgeRunnerMetalTests/RoPEDualTableTest.swift`

- [ ] **Step 13.1: Failing test**

```swift
@Test("Global pRoPE rotates only first partial*head_dim channels")
func pRoPERotatesOnlyPartial() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let rope = try RoPE(device: device, headDim: 512, base: 1_000_000, partialRotaryFactor: 0.25)
    var q = [Float](repeating: 1.0, count: 512)
    try rope.applyInPlace(&q, position: 10)
    // First 128 channels should be rotated (not equal to 1.0)
    #expect(q[0] != 1.0 || q[1] != 1.0)
    // Last 384 channels should pass through (unchanged)
    for i in 128..<512 { #expect(q[i] == 1.0) }
}

@Test("Local RoPE rotates all channels with base 10000")
func localRotatesAll() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let rope = try RoPE(device: device, headDim: 256, base: 10_000, partialRotaryFactor: 1.0)
    var q = [Float](repeating: 1.0, count: 256)
    try rope.applyInPlace(&q, position: 10)
    for i in 0..<256 { #expect(q[i] != 1.0) }
}
```

- [ ] **Step 13.2: Run — expect FAIL**

- [ ] **Step 13.3: Extend `RoPE` to take `partialRotaryFactor`**

Modify `Sources/EdgeRunnerMetal/RoPE.swift` — add `partialRotaryFactor` to init. In shader, skip rotation for channel pairs where `2*pair >= headDim * partialRotaryFactor`.

```metal
kernel void rope_apply_f32(
    device float *x [[buffer(0)]],
    constant RoPEParams &p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint headIdx = gid.y;
    uint pair = gid.x;
    if (headIdx >= p.numHeads || pair >= p.halfHeadDim) return;
    uint rotatedPairs = uint(float(p.halfHeadDim) * p.partialRotaryFactor);
    if (pair >= rotatedPairs) return;  // pass-through

    float theta = p.position * pow(p.base, -float(2 * pair) / float(p.headDim));
    float c = cos(theta), s = sin(theta);
    uint i0 = headIdx * p.headDim + 2 * pair;
    float x0 = x[i0];
    float x1 = x[i0 + 1];
    x[i0]     = x0 * c - x1 * s;
    x[i0 + 1] = x0 * s + x1 * c;
}
```

- [ ] **Step 13.4: Run — expect PASS**

- [ ] **Step 13.5: Commit**

```bash
git commit -m "feat(metal): dual RoPE with partial_rotary_factor for Gemma 4 global layers"
```

---

### Task 14: Extend `KVCache` for dual stride + share map

`globalHeadDim=512` vs `slidingHeadDim=256`. Per-layer bytes-per-token differ. Share map: for layers ≥ 24, point to source layer's buffer (no allocation, no write).

**Files:**
- Modify: `Sources/EdgeRunnerMetal/KVCache.swift`
- Modify: `Sources/EdgeRunnerSharedTypes/include/KVCacheParams.h`
- Test: `Tests/EdgeRunnerMetalTests/KVCacheGemma4Test.swift`

- [ ] **Step 14.1: Failing test**

```swift
@Test("Allocates per-layer buffers with correct head dim and shares for layers >= 24")
func kvCacheGemma4Layout() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let config = try Gemma4ModelConfig(metadata: Gemma4ModelConfigTests.makeReferenceMetadata())
    let cache = try KVCache.gemma4(
        device: device, config: config, maxSeqLen: 2048, compression: .disabled
    )
    #expect(cache.keyBuffer(forLayer: 0).length == 2 * 256 * 2048 * 4)  // 2 KV heads * 256 dim * 2048 tokens * float32
    #expect(cache.keyBuffer(forLayer: 5).length == 2 * 512 * 2048 * 4)  // global head dim
    // Layer 24 is sliding and shares from layer 22 (nearest sliding before it that isn't itself shared).
    #expect(cache.keyBuffer(forLayer: 24) === cache.keyBuffer(forLayer: 22))
    #expect(cache.keyBuffer(forLayer: 41) === cache.keyBuffer(forLayer: 23))  // global source
}
```

- [ ] **Step 14.2: Run — expect FAIL**

- [ ] **Step 14.3: Extend `KVCache` with `.gemma4(...)` factory that allocates per-layer buffers with per-layer head dim and stores a share-pointer map.**

Signature:
```swift
public extension KVCache {
    static func gemma4(
        device: MTLDevice,
        config: Gemma4ModelConfig,
        maxSeqLen: Int,
        compression: KVCacheCompression
    ) throws -> KVCache
}
```

Internal logic:
```swift
for layer in 0..<config.numHiddenLayers {
    let srcLayer = config.kvSourceLayer(for: layer)
    if srcLayer != layer {
        keyBuffers.append(keyBuffers[srcLayer])
        valueBuffers.append(valueBuffers[srcLayer])
    } else {
        let headDim = config.layerTypes[layer] == .global
            ? config.globalHeadDim : config.headDim
        let bytes = config.numKeyValueHeads * headDim * maxSeqLen * MemoryLayout<Float>.stride
        keyBuffers.append(try allocateBuffer(bytes: bytes))
        valueBuffers.append(try allocateBuffer(bytes: bytes))
    }
}
```

- [ ] **Step 14.4: Run — expect PASS**

- [ ] **Step 14.5: Commit**

```bash
git commit -m "feat(kvcache): Gemma 4 dual head-dim + KV sharing for last 18 layers"
```

---

### Task 15: GQA dispatch path accepts precomputed mask

**Files:**
- Modify: `Sources/EdgeRunnerMetal/GQAKernel.swift`
- Modify: `Sources/EdgeRunnerMetal/Shaders/GQA.metal`
- Test: `Tests/EdgeRunnerMetalTests/GQAWithMaskTest.swift`

- [ ] **Step 15.1: Failing test** — verify that with an SWA mask of window=3, attention at position 5 only attends to positions {3,4,5}; output matches hand-computed softmax.

- [ ] **Step 15.2: Run — expect FAIL**

- [ ] **Step 15.3: Add `mask` buffer arg to `GQA.metal` kernel signature; add `mask[q*S + k]` before softmax.**

In `GQAKernel.swift`, extend the dispatch `encode(...)` method to accept an `additiveMask: MTLBuffer?` parameter; when nil, fall back to current implicit causal mask.

- [ ] **Step 15.4: Run — expect PASS**

- [ ] **Step 15.5: Commit**

```bash
git commit -m "feat(metal): GQA kernel accepts optional additive mask for SWA support"
```

---

## Phase 5 — Forward Pass Integration

### Task 16: Decoder-layer forward block (single layer parity)

Orchestrate a single Gemma 4 decoder block end-to-end. This task is the parity anchor for everything below.

**Per-layer forward:**
```
// pre-attention norm
h_in = RMSNorm(h, input_norm_weight)
// attention
q = Wq(h_in); k = Wk(h_in) [if not shared]; v = Wv(h_in) [if not shared]
q = RoPE(q, layerType); k = RoPE(k, layerType)
// update KV cache (owned layers only)
kv_cache.append(layer: ℓ, k: k, v: v) if kvSourceLayer(ℓ) == ℓ
// attention
attn_out = GQA(q, kv_cache.read(srcLayer), mask=mask_for(layerType))
attn_out = Wo(attn_out)
// post-attention norm + residual
h = h + RMSNorm(attn_out, post_attn_norm_weight)
// FFN
f_in = RMSNorm(h, pre_ffn_norm_weight)    // note: Gemma 4 uses pre-ffn norm
up = W_up(f_in); gate = W_gate(f_in)
ffn_out = W_down(GeGLU(gate, up))
h = h + RMSNorm(ffn_out, post_ffn_norm_weight)
// PLE side-channel
gated = GELU-tanh(W_gate_ple(h))         // H → P
gated = gated ⊙ per_layer_inputs[:, ℓ, :]
proj = W_proj_ple(gated)                 // P → H
h = h + RMSNorm(proj, post_ple_norm_weight)
```

**Note on pre/post FFN norm:** verify during implementation that Gemma 4 uses Gemma-2/3 style double-norm (both pre-attn + post-attn, pre-ffn + post-ffn). The Gemma 3 lineage has four norms per block; reference `modeling_gemma4.py` lines 1745–1956 confirms this.

**Files:**
- Create: `Sources/EdgeRunner/Models/Gemma4/Gemma4DecoderLayer.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4DecoderLayerParityTest.swift`

- [ ] **Step 16.1: Prepare reference output**

Generate reference hidden state after layer 0 using HF transformers:
```bash
python3 Tests/scripts/gen_gemma4_layer0_reference.py \
    --model google/gemma-4-E4B-it --out Tests/EdgeRunnerTests/Fixtures/gemma4_layer0_ref.npz
```

The script loads the model with `torch_dtype=torch.float32`, runs a single forward pass on `input_ids=[2, 2023, 1234]` (bos + two arbitrary tokens), captures `model.layers[0]` output, and saves `{input_hidden, output_hidden, per_layer_inputs[0]}` to .npz.

- [ ] **Step 16.2: Failing parity test**

```swift
@Test("Layer 0 forward matches HF reference within 1e-2", .enabled(if: ProcessInfo.processInfo.environment["EDGERUNNER_RUN_GEMMA4_PARITY"] == "1"))
func layer0ForwardParity() throws {
    let fixture = try Gemma4Fixtures.loadLayer0Reference()
    let config = try Gemma4ModelConfig(metadata: fixture.metadata)
    let weights = try Gemma4Weights.load(from: fixture.weightMap, config: config, device: device)
    let layer = Gemma4DecoderLayer(index: 0, config: config, weights: weights.blocks[0], device: device)

    let out = try layer.forward(
        h: fixture.inputHidden,
        perLayerInputsForThisLayer: fixture.perLayerInputsLayer0,
        kvCache: kvCache,
        position: 0
    )
    for i in 0..<out.count {
        #expect(abs(out[i] - fixture.outputHidden[i]) < 1e-2)
    }
}
```

- [ ] **Step 16.3: Run — expect FAIL**

- [ ] **Step 16.4: Implement `Gemma4DecoderLayer.forward(...)`** — wires all subkernels in the order listed above.

- [ ] **Step 16.5: Run — expect PASS (may require iterating on norm placement and scale placement)**

- [ ] **Step 16.6: Commit**

```bash
git commit -m "feat(gemma4): single decoder layer forward with PLE side-channel"
```

---

### Task 17: Full-model prefill parity (42 layers)

**Files:**
- Create: `Sources/EdgeRunner/Models/Gemma4LanguageModel.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4FullForwardParityTest.swift`

- [ ] **Step 17.1: Reference from HF**

```bash
python3 Tests/scripts/gen_gemma4_full_prefill_ref.py \
    --prompt "Hello" --out Tests/EdgeRunnerTests/Fixtures/gemma4_prefill_hello.npz
```
Captures logits for input_ids=[2, 9259] (bos + "Hello").

- [ ] **Step 17.2: Failing test**

```swift
@Test("Full 42-layer prefill + softcap matches HF logits for 'Hello' within 1e-2",
      .enabled(if: ProcessInfo.processInfo.environment["EDGERUNNER_RUN_GEMMA4_PARITY"] == "1"))
func fullPrefillHelloParity() async throws {
    let fixture = try Gemma4Fixtures.loadPrefillHello()
    let model = try await Gemma4LanguageModel.load(
        from: Gemma4Fixtures.ggufURL, configuration: ModelConfiguration()
    )
    let logits = try await model.logits(for: fixture.inputIds)

    let topKActual = logits.argtop(5)
    let topKRef = fixture.refLogits.argtop(5)
    #expect(topKActual == topKRef, "Top-5 token mismatch")

    let maxAbsDiff = zip(logits, fixture.refLogits).map { abs($0 - $1) }.max()!
    #expect(maxAbsDiff < 1e-2)
}
```

- [ ] **Step 17.3: Run — expect FAIL**

- [ ] **Step 17.4: Implement `Gemma4LanguageModel.prefill(_:)`** — orchestrates:
  1. Token embed + scale by √H
  2. Build `per_layer_inputs[B,S,L,P]` once (gather PLE rows + project from h + mix)
  3. For each layer 0..41: run `Gemma4DecoderLayer.forward()`
  4. Final output norm → logits via tied embedding
  5. Apply `logit_softcap_f32(cap=30)`

- [ ] **Step 17.5: Run — iterate until PASS** (most likely source of bugs: norm-weight (1+w) trick, embedding scale by √H, or per_layer_inputs axis ordering). Capture intermediate tensor snapshots if diverging.

- [ ] **Step 17.6: Commit**

```bash
git commit -m "feat(gemma4): full prefill parity with HF transformers reference"
```

---

### Task 18: Decode path (single-token KV-cache incremental)

**Files:**
- Modify: `Sources/EdgeRunner/Models/Gemma4LanguageModel.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4DecodeParityTest.swift`

- [ ] **Step 18.1: Failing test** — 16-token greedy decode from "Hello" prompt matches HF greedy decode token-for-token.

- [ ] **Step 18.2: Run — expect FAIL**

- [ ] **Step 18.3: Implement incremental decode** — reuses KV cache written during prefill, rebuilds only `per_layer_inputs` for the single new token, runs 42 layers with seq=1 queries.

- [ ] **Step 18.4: Run — expect PASS** (tokens identical)

- [ ] **Step 18.5: Commit**

```bash
git commit -m "feat(gemma4): incremental decode path with KV cache reuse"
```

---

### Task 19: Route `ModelLoader.load()` to Gemma 4 before Llama fallthrough

**Files:**
- Modify: `Sources/EdgeRunner/ModelLoader.swift`

- [ ] **Step 19.1: Failing integration test**

```swift
@Test("ModelLoader.load() returns Gemma4LanguageModel for gemma4 architecture")
func loadsGemma4Model() async throws {
    let model = try await ModelLoader.load(from: Gemma4Fixtures.ggufURL)
    #expect(model is Gemma4LanguageModel)
}
```

- [ ] **Step 19.2: Run — expect FAIL (returns `LlamaLanguageModel`)**

- [ ] **Step 19.3: Edit `ModelLoader.swift`**

```swift
if Gemma4LanguageModel.supports(modelConfig: modelConfig) {
    return try await Gemma4LanguageModel.load(from: url, configuration: configuration)
}
```
Insert before the `BonsaiLanguageModel.supports(...)` check on line 46.

- [ ] **Step 19.4: Run — expect PASS**

- [ ] **Step 19.5: Commit**

```bash
git commit -m "feat(loader): route gemma4 GGUFs to Gemma4LanguageModel"
```

---

## Phase 6 — iPhone Integration + Memory Strategy

### Task 20: `mmap`-backed PLE storage with `MTLResourceStorageModeShared`

**Goal:** The 2.5 GB PLE table (`per_layer_token_embd` at Q8_0) must never be fully resident. We use `mmap(PROT_READ, MAP_PRIVATE)` on the GGUF file, then create a Metal buffer over the mmap region with `newBufferWithBytesNoCopy:length:options:deallocator:` using `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`. iOS pages in only touched 16 KB pages.

**Files:**
- Modify: `Sources/EdgeRunnerIO/GGUF/MemoryMappedFile.swift`
- Modify: `Sources/EdgeRunner/Models/Gemma4LanguageModel.swift`
- Test: `Tests/EdgeRunnerTests/Gemma4PLEPagingTest.swift`

- [ ] **Step 20.1: Failing test** — loads PLE table, runs 3 tokens through decode, asserts that RSS increase is < 50 MB (loose bound: 3 tokens × 42 layers × 16 KB page rounding ≈ 2 MB). Use `task_info()` via Darwin API to measure RSS.

```swift
@Test("PLE gather on 3 tokens keeps RSS increase under 50 MB",
      .enabled(if: ProcessInfo.processInfo.environment["EDGERUNNER_RUN_GEMMA4_MEMORY"] == "1"))
func pleTableStaysPaged() async throws {
    let rssBefore = Darwin.residentSetSizeBytes()
    let model = try await Gemma4LanguageModel.load(from: Gemma4Fixtures.ggufURL)
    let rssAfterLoad = Darwin.residentSetSizeBytes()
    _ = try await model.generate(prompt: "Hi", maxTokens: 3)
    let rssAfterDecode = Darwin.residentSetSizeBytes()

    let pleContribution = rssAfterDecode - rssAfterLoad
    #expect(pleContribution < 50 * 1024 * 1024, "PLE paging leaked \(pleContribution) bytes")
}
```

- [ ] **Step 20.2: Run — expect FAIL** (current loader likely eagerly copies PLE to a private buffer)

- [ ] **Step 20.3: Implement**

In `MemoryMappedFile.swift`, add `makeSharedMetalBuffer(offset:length:)` that creates a `MTLResourceStorageModeShared` buffer via `newBufferWithBytesNoCopy`. In `Gemma4Weights.init`, route the `per_layer_token_embd` tensor through this shared buffer instead of the default private-buffer path.

- [ ] **Step 20.4: Run — expect PASS**

- [ ] **Step 20.5: Commit**

```bash
git commit -m "feat(gemma4): mmap-backed shared-storage PLE table for memory paging"
```

---

### Task 21: iPhone 15 Pro Max TTFT + memory benchmark

**Files:**
- Create: `Tests/EdgeRunnerTests/Gemma4iPhoneMemoryBenchmark.swift`
- Create: `benchmarks/gemma4_e4b_iphone.json` (result file)

- [ ] **Step 21.1: Implement benchmark**

```swift
@Suite("Gemma4 iPhone benchmark")
struct Gemma4iPhoneBenchmark {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["EDGERUNNER_RUN_IPHONE_BENCH"] == "1"))
    func benchmark() async throws {
        let model = try await Gemma4LanguageModel.load(
            from: Gemma4Fixtures.ggufURL,
            configuration: ModelConfiguration(maxTokens: 128, contextWindowSize: 4096)
        )
        let prompt = "Explain quantum entanglement briefly:"

        let t0 = ContinuousClock.now
        let warmOutput = try await model.generate(prompt: prompt, maxTokens: 1)
        let ttft = ContinuousClock.now - t0
        _ = warmOutput

        let t1 = ContinuousClock.now
        let full = try await model.generate(prompt: prompt, maxTokens: 128)
        let decodeElapsed = ContinuousClock.now - t1
        let tokensPerSec = Double(full.tokenCount) / decodeElapsed.seconds

        let rss = Darwin.residentSetSizeBytes()

        try Gemma4BenchmarkReport(
            ttftMs: ttft.milliseconds,
            decodeTokPerSec: tokensPerSec,
            peakRSSMB: Double(rss) / (1024 * 1024)
        ).save(to: URL(fileURLWithPath: "benchmarks/gemma4_e4b_iphone.json"))

        // Acceptance gates
        #expect(ttft.milliseconds < 3000, "TTFT > 3s")
        #expect(tokensPerSec > 8, "Decode < 8 tok/s")
        #expect(Double(rss) / (1024 * 1024) < 4500, "RSS > 4.5 GB")
    }
}
```

- [ ] **Step 21.2: Run on iPhone 15 Pro Max via Xcode test plan**

- [ ] **Step 21.3: Commit baseline report**

```bash
git add benchmarks/gemma4_e4b_iphone.json \
        Tests/EdgeRunnerTests/Gemma4iPhoneMemoryBenchmark.swift
git commit -m "bench(gemma4): iPhone 15 Pro Max TTFT + decode + RSS baseline"
```

---

## Phase 7 — Public API + Docs

### Task 22: Expose Gemma 4 in `EdgeRunnerFacade`

**Files:**
- Modify: `Sources/EdgeRunner/EdgeRunnerFacade.swift`
- Test: `Tests/EdgeRunnerTests/FacadeGemma4Test.swift`

- [ ] **Step 22.1: Failing test** — `EdgeRunner.load(modelURL:)` returns usable instance for Gemma 4 GGUF.

- [ ] **Step 22.2: Run — expect FAIL** (may already pass via `ModelLoader`; if so, skip).

- [ ] **Step 22.3: Ensure facade re-exports `Gemma4LanguageModel` as a typed option if relevant.**

- [ ] **Step 22.4: Commit**

```bash
git commit -m "feat(api): expose Gemma 4 via EdgeRunnerFacade"
```

---

### Task 23: Update `Examples/EdgeRunnerChat` to support Gemma 4 model picker

**Files:**
- Modify: `Examples/EdgeRunnerChat/EdgeRunnerChatApp.swift`
- Modify: `Examples/EdgeRunnerChat/ChatViewModel.swift` (or equivalent)

- [ ] **Step 23.1: Add Gemma 4 entry to model picker + chat template handoff.**

- [ ] **Step 23.2: Run example app on iPhone 15 Pro Max simulator + device, verify a 3-turn conversation renders correctly with sentinel-format tokens.**

- [ ] **Step 23.3: Commit**

```bash
git commit -m "feat(example): add Gemma 4 E4B to EdgeRunnerChat picker"
```

---

### Task 24: Update `docs/ROADMAP.md` Phase 3, `docs/arch/public_api.md`, and `README.md`

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/arch/public_api.md`
- Modify: `README.md`

- [ ] **Step 24.1: Mark Gemma 4 E4B as supported in ROADMAP.**

- [ ] **Step 24.2: Document `Gemma4LanguageModel`, chat template helpers, benchmark numbers.**

- [ ] **Step 24.3: Commit**

```bash
git commit -m "docs: mark Gemma 4 E4B as supported + add API + benchmark docs"
```

---

## Phase 8 — Long-Context & Robustness

### Task 25: 128K context stress test

**Files:**
- Create: `Tests/EdgeRunnerTests/Gemma4LongContextTest.swift`

- [ ] **Step 25.1: Failing test** — prefill a 64K-token prompt on M3 Max (not iPhone; iPhone will OOM at 128K without KV sharing tuning), assert that decoding 1 token after the prompt returns finite, non-NaN logits.

- [ ] **Step 25.2: Run — expect FAIL if KV cache layout doesn't account for 128K + dual head dim.**

- [ ] **Step 25.3: Fix any discovered issues (buffer sizing, pRoPE position overflow at 131K, etc).**

- [ ] **Step 25.4: Run — expect PASS**

- [ ] **Step 25.5: Commit**

```bash
git commit -m "test(gemma4): 64K-prompt long-context stress test passes on M-series"
```

---

### Task 26: Quantized (Q4_K_M) end-to-end parity

**Goal:** Verify the Q4_K_M GGUF from `unsloth/gemma-4-E4B-it-GGUF` produces sane output (non-garbled English for a simple prompt) and top-1 tokens diverge from the F16 reference by no more than a few tokens over 32 greedy steps.

**Files:**
- Create: `Tests/EdgeRunnerTests/Gemma4Q4KMSmokeTest.swift`

- [ ] **Step 26.1: Failing test**

```swift
@Test("Q4_K_M decodes coherent English for 'The capital of France is'",
      .enabled(if: ProcessInfo.processInfo.environment["EDGERUNNER_RUN_GEMMA4_Q4KM"] == "1"))
func q4kmDecodesCoherently() async throws {
    let modelURL = URL(fileURLWithPath: "/tmp/edgerunner-models/gemma-4-E4B-it-Q4_K_M.gguf")
    let model = try await Gemma4LanguageModel.load(from: modelURL)
    let output = try await model.generate(
        prompt: "The capital of France is",
        maxTokens: 8,
        samplingStrategy: .greedy
    )
    #expect(output.text.lowercased().contains("paris"))
}
```

- [ ] **Step 26.2: Run — expect FAIL or PASS depending on quant fidelity. Iterate on PLE quant handling if output is garbled.**

- [ ] **Step 26.3: Commit**

```bash
git commit -m "test(gemma4): Q4_K_M end-to-end coherence smoke test"
```

---

## Self-Review (run before handoff)

### 1. Spec coverage

| Requirement from research briefs | Task(s) |
|---|---|
| 42-layer topology, hidden=2560, FFN=10240 | Task 1 |
| GQA 8/2, head_dim 256 local / 512 global | Task 14 |
| Hybrid 5:1 SWA:global schedule | Task 1, 12, 15 |
| Dual RoPE (θ=1e4 local, θ=1e6 global, partial=0.25) | Task 13 |
| KV sharing for last 18 layers | Task 2, 14 |
| PLE gather + projection + side-channel | Task 8, 9, 10, 16 |
| GeGLU with gelu_pytorch_tanh | Task 7 |
| Final logit softcap tanh(x/30)·30 | Task 11 |
| Tied embeddings + √H scaling | Task 17 |
| Sentinel chat template | Task 5, 6 |
| PLE paged via mmap + shared Metal buffer | Task 20 |
| iPhone 15 Pro Max TTFT + RSS | Task 21 |
| 128K context correctness | Task 25 |
| Q4_K_M coherence | Task 26 |

### 2. Placeholder scan

Run `grep -nE 'TBD|TODO|fill in|similar to task|add appropriate' docs/superpowers/plans/2026-04-18-gemma-4-e4b-ple-support.md` — expect zero hits after final pass. If any exist, inline the missing content.

### 3. Type consistency

- `Gemma4ModelConfig.kvSourceLayer(for:)` used in Task 2, 14 ✓
- `Gemma4Weights.blocks[i]` field names match across Task 3, 16, 17 ✓
- `Gemma4ChatMessage.role` enum used identically in Task 5 ✓
- `PLEGatherKernel.run(q8Table:tokens:)` signature matches Task 8 + Task 17 (`buildPerLayerInputs`) ✓
- `GeGLUKernel.run(gate:up:)` matches Task 7 + Task 16 FFN path ✓

### 4. Known gaps (explicit, to resolve during execution)

- **KV-share target layer algorithm** in Task 2 is inferred; verify against `llama-model.cpp` `build_gemma4` during Task 14 and patch if incorrect.
- **Pre/post RMSNorm placement** in Gemma 4 decoder — plan assumes 4-norm Gemma 2/3 style. Task 16 parity test will reveal if the ordering differs; adjust in place.
- **`(1+weight)` RMSNorm trick** — plan assumes it everywhere for Gemma lineage weights. If parity fails with large offsets, test with and without this trick.
- **BF16 PLE path** is defined but not benchmarked; v1 ships Q8_0-only. Q4_0 / Q4_1 / Q5_0 / Q5_1 PLE acceptance is wired into `Gemma4LoadError.unsupportedPLEQuant` but unverified on real GGUFs.
- **Multimodal token IDs (≥ 262144)** — we don't mask to 0 during gather in v1 because we don't accept them; add a precondition check in `Gemma4LanguageModel.generate(prompt:)`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-gemma-4-e4b-ple-support.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?

