# EdgeRunner Inference Pipeline

This document provides a comprehensive overview of EdgeRunner's end-to-end inference pipeline, from prompt input to generated token output.

## Architecture Overview

EdgeRunner implements a Metal-accelerated inference engine for LLM models (Llama, Qwen, Mistral family) with quantized weight support (Q4_0, Q8_0, Q4_K_M, Q2_K-Q6_K variants). The pipeline is designed for low-latency autoregressive generation with KV caching, prefix reuse, and multiple sampling strategies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EdgeRunner Inference Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Prompt  │───▶│ Tokenize + BOS   │───▶│   Generation Session   │  │
│  └──────────┘    └──────────────────┘    └────────────────────────┘  │
│                                                      │               │
│                     ┌─────────────────────────────────┼───────────┐  │
│                     ▼                                 ▼           │  │
│            ┌───────────────┐              ┌─────────────────────┐  │  │
│            │  SamplingConfig│              │  EdgeRunnerLanguage │  │  │
│            │  temperature  │              │      Model          │  │  │
│            │  topK/topP    │              │  (LogitsModel)     │  │  │
│            │  repetition   │              └──────────┬──────────┘  │  │
│            └───────────────┘                         │            │  │
│                                                       ▼            │  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FORWARD PASS                              │  │
│  │  ┌─────────┐  ┌─────────────┐  ┌────────────────────────┐  │  │
│  │  │ Prefill │─▶│Decode Mode  │─▶│  Prefix Reuse (Hint)   │  │  │
│  │  │  (N)    │  │   (1 token) │  │   (suffix only)       │  │  │
│  │  └────┬────┘  └──────┬──────┘  └────────────────────────┘  │  │
│  │       │              │                                      │  │
│  │       └──────────────┼──────────────────────────────────────┘  │
│  │                      ▼                                         │
│  │  ┌─────────────────────────────────────────────────────────┐  │
│  │  │              Transformer Layers (N layers)              │  │
│  │  │  RMSNorm → RoPE → GQA + KVCache → SwiGLU FFN           │  │
│  │  └─────────────────────────────────────────────────────────┘  │
│  │                      │                                         │
│  │                      ▼                                         │
│  │  ┌─────────────────────────────────────────────────────────┐  │
│  │  │           Final RMSNorm → LM Head → Logits               │  │
│  │  └─────────────────────────────────────────────────────────┘  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SAMPLING PIPELINE                          │  │
│  │  Logits ──▶ RepetitionPenalty ──▶ Temperature ──▶ TopK/P    │  │
│  │                                ──▶ Stochastic/Greedy Select   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│                      ┌──────────────┐                                │
│                      │  Next Token  │                                │
│                      └──────────────┘                                │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Streaming Output                           │  │
│  │         AsyncThrowingStream<String, Error>                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Entry Points

### 1.1 GenerationSession

`Sources/EdgeRunner/Streaming/GenerationSession.swift`

The `GenerationSession<Model: EdgeRunnerLanguageModel>` struct provides the primary interface for text generation with streaming support.

```swift
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    private let model: Model
    private let sampling: SamplingConfiguration
    public let maxTokens: Int
    private let onToken: (@Sendable (Int, String) -> Void)?
}
```

Key methods:
- `stream(prompt: String) -> AsyncThrowingStream<String, Error>` - Main streaming generation
- `generate(prompt: String) async throws -> String` - Non-streaming convenience wrapper

### 1.2 EdgeRunnerLanguageModel Protocol

`Sources/EdgeRunner/EdgeRunnerLanguageModel.swift`

The core protocol defining the model interface:

```swift
public protocol EdgeRunnerLanguageModel: Sendable {
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self
    func tokenize(_ text: String) -> [Int]
    func detokenize(_ ids: [Int]) -> String
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}
```

### 1.3 LogitsModel Extension

For Metal-accelerated models (like `LlamaLanguageModel`), the `LogitsModel` sub-protocol exposes raw logits access:

```swift
public protocol LogitsModel: EdgeRunnerLanguageModel {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}

extension LogitsModel {
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let logitsArray = try await logits(for: tokenIDs)
        let pipeline = sampling.toPipeline()
        return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
    }
}
```

## 2. Sampling Configuration

`Sources/EdgeRunner/SamplingConfiguration.swift`

Controls token selection strategy:

```swift
public struct SamplingConfiguration: Sendable {
    public var temperature: Float  // Controls randomness (0 = greedy)
    public var topK: Int           // Limit to top K tokens
    public var topP: Float        // Nucleus sampling threshold
    public var repetitionPenalty: Float  // Penalize repeated tokens
    public var seed: UInt64?      // Random seed for reproducibility
}
```

### 2.1 Sampling Pipeline Construction

`SamplingConfiguration.toPipeline()` builds a composable sampling pipeline:

```swift
public func toPipeline() -> SamplingPipeline
```

**Pipeline construction logic:**

| Condition | Pipeline |
|-----------|----------|
| `temperature <= 0` | Greedy + optional RepetitionPenalty |
| `temperature > 0` | TemperatureSampler → TopKSampler → TopPSampler → StochasticSampler |

## 3. Sampling Pipeline Architecture

`Sources/EdgeRunnerCore/Sampling/SamplingPipeline.swift`

The `SamplingPipeline` composes transforms and a selector:

```swift
public struct SamplingPipeline: Sendable {
    private let transforms: [any LogitsTransform]
    private let selector: any TokenSelector
    private let repetitionPenalty: RepetitionPenalty?
}
```

### 3.1 Sampling Flow

```
Input Logits
    │
    ▼
┌─────────────────────────┐
│  RepetitionPenalty      │ (if penalty > 1.0)
│  - Divide positive vals│
│  - Multiply negative vals
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  TemperatureSampler     │ (if temperature != 1.0)
│  - logits / temperature │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  TopKSampler            │ (if topK > 0)
│  - Mask all but top K  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  TopPSampler            │ (if topP < 1.0)
│  - Mask below nucleus  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  TokenSelector          │
│  GreedySampler OR       │
│  StochasticSampler     │
└─────────────────────────┘
    │
    ▼
  Token ID
```

### 3.2 LogitsTransform Protocol

```swift
public protocol LogitsTransform: Sendable {
    func transformLogits(_ logits: [Float]) -> [Float]
}
```

**Implementations:**

| Transform | Behavior |
|-----------|----------|
| `TemperatureSampler` | Divides logits by temperature (scales distribution sharpness) |
| `TopKSampler` | Sets all logits below top-K threshold to `-infinity` |
| `TopPSampler` | Sets tokens outside cumulative probability nucleus to `-infinity` |
| `RepetitionPenalty` | Applies penalty based on frequency in previous tokens |

### 3.3 TokenSelector Protocol

```swift
public protocol TokenSelector: Sendable {
    func sample(logits: [Float]) -> Int
}
```

**Implementations:**

| Selector | Behavior |
|----------|----------|
| `GreedySampler` | Returns index of maximum logit |
| `StochasticSampler<RNG>` | Samples from softmax distribution using RNG |

### 3.4 Repetition Penalty

`Sources/EdgeRunnerCore/Sampling/RepetitionPenalty.swift`

```swift
public struct RepetitionPenalty: Sendable {
    public let penalty: Float        // Typically 1.0-1.5
    public let frequencyPenalty: Float  // Additional frequency-based penalty
}
```

**Algorithm:**
```
for each token in previousTokens:
    if logits[token] > 0:
        logits[token] /= penalty
    else:
        logits[token] *= penalty
    if frequencyPenalty > 0:
        logits[token] -= frequencyPenalty * count(token)
```

## 4. Prefill vs Decode vs Prefix Reuse

`Sources/EdgeRunner/Models/LlamaLanguageModel.swift` - `forwardLogitsBuffer()`

The system automatically detects which mode to use based on input sequence comparison:

### 4.1 Mode Detection Logic

```swift
let commonPrefixLen = countMatchingPrefix(previousTokenIDs, tokenIDs)
let isDecodeMode = commonPrefixLen == previousTokenIDs.count
    && tokenIDs.count == commonPrefixLen + 1
    && tokenIDs.count > 1

let isPrefixReuseMode = commonPrefixLen > 0
    && commonPrefixLen == previousTokenIDs.count
    && tokenIDs.count > commonPrefixLen + 1

// Otherwise: Full Prefill Mode
```

### 4.2 Full Prefill Mode

**Triggered when:** No useful KV cache exists (new conversation, cache miss)

**Behavior:**
1. Reset KV cache (`kvCache.reset()`)
2. Embed ALL tokens in the sequence
3. Run all N transformer layers over full sequence
4. Store K/V in cache at positions 0...N-1
5. Output logits for final position

**Use case:** First token of generation, long jumps in context

### 4.3 Decode Mode

**Triggered when:** `tokenIDs` is exactly 1 token longer than cached sequence

**Behavior:**
1. Embed ONLY the single new token
2. Run transformer layers with KV cache:
   - Query attends to all cached K/V (causal mask)
   - Only write new K/V at current position
3. Output logits for single position

**Use case:** Normal autoregressive generation (token-by-token)

**Decode Optimization - GPU Pipeline Warmup:**
On first prefill, the system runs 5 dummy decode passes to warm up GPU JIT compilation, then re-runs the actual prefill to populate the KV cache correctly.

### 4.4 Prefix Reuse Mode (Hint Mode)

**Triggered when:** New sequence extends cached sequence by multiple tokens

**Behavior:**
1. Detect common prefix length
2. Embed ONLY suffix tokens (new tokens)
3. Run transformer with:
   - RoPE positions offset by `commonPrefixLen`
   - GQA attends over full KV cache (prefix + suffix)
   - Causal mask offset accordingly
4. Update decoder state

**Use case:** Multi-turn conversations where system prompt + conversation history forms a prefix

## 5. KV Cache

`Sources/EdgeRunnerMetal/KVCache.swift`

### 5.1 Structure

```swift
public final class KVCache: Sendable {
    public let maxSeqLen: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let precision: Precision  // .float32, .float16, .float8

    private let keyBuffers: [MetalBufferHandle]  // Per-layer
    private let valueBuffers: [MetalBufferHandle] // Per-layer
    private let layerStates: Mutex<[LayerState]>
}

struct LayerState: Sendable {
    var writePos = 0    // Circular buffer position
    var totalWritten = 0
}
```

### 5.2 Key Operations

| Method | Purpose |
|--------|---------|
| `reset()` | Clear all layers, reset positions to 0 |
| `setPosition(_:)` | Set write position for all layers (used after prefill) |
| `advanceWritePosition(layer:count:)` | Increment position after GPU write |
| `metalBuffers(layer:)` | Get raw MTLBuffers for GPU kernel access |
| `cacheParams(layer:)` | Get `ERKVCacheParams` for kernel dispatch |

### 5.3 Circular Buffer Behavior

The KV cache uses a circular buffer to support sequences up to `maxSeqLen`:
- `writePos` cycles from 0 to maxSeqLen-1
- `totalWritten` tracks absolute token count (for position calculation)
- When `totalWritten > maxSeqLen`, oldest tokens are overwritten

## 6. Transformer Forward Pass

### 6.1 Architecture

Llama-family models use this transformer architecture:

```
Input Hidden States (seqLen × dim)
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                    PER LAYER (N layers)                         │
│                                                                 │
│  1. RMSNorm (attention input)                                   │
│     └── weight: attentionNorm.weight                            │
│                                                                 │
│  2. Q/K/V Projections + RMSNorm (fused)                        │
│     ├── Q = input @ wq (qDim = headCount × headDim)             │
│     ├── K = input @ wk (kvDim = kvHeadCount × headDim)         │
│     ├── V = input @ wv                                          │
│     └── Cache K/V at (startPos + token) positions               │
│                                                                 │
│  3. RoPE (Rotary Position Embedding)                            │
│     └── Apply rotations to Q and K                              │
│                                                                 │
│  4. GQA (Grouped Query Attention) + KV Cache                    │
│     └── Query attends to cached K/V                             │
│                                                                 │
│  5. Output Projection                                           │
│     └── attn_output @ wo                                        │
│                                                                 │
│  6. Residual Add                                                │
│     └── output = input + attn_output                            │
│                                                                 │
│  7. RMSNorm (FFN input)                                         │
│     └── weight: ffnNorm.weight                                  │
│                                                                 │
│  8. SwiGLU FFN                                                  │
│     ├── gate = input @ gate_weight                              │
│     ├── up = input @ up_weight                                  │
│     ├── SiLU(gate) * up                                         │
│     └── down = (SiLU(gate) * up) @ down_weight                  │
│                                                                 │
│  9. Residual Add                                                │
│     └── output = input + ffn_output                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → LM Head → Logits (vocabSize)
```

### 6.2 Fused Operations

The Metal implementation uses several fused kernels to reduce GPU synchronization:

| Kernel | Fuses |
|--------|-------|
| `fusedQKVPipeline` | RMSNorm + Q + K + V projections |
| `fusedFinalNormGemvPipeline` | Final RMSNorm + LM head GEMV |
| `fusedQKNormRoPEPipeline` | Q/K Norm + RoPE |
| `fusedNormRoPEGQAPipeline` | Norm + RoPE + GQA |

### 6.3 Quantization Support

On-the-fly dequantization during GEMV operations:

| Type | Format | Kernel |
|------|--------|--------|
| Q8_0 | 8-bit block quantization, block size 32 | `dequantQ8_0Kernel` + `fusedQ8GemvPipeline` |
| Q4_0 | 4-bit block quantization | `dequantQ4_0Kernel` |
| Q4_K_M | 4-bit with metadata | `dequantQ4KMKernel` |
| Q5_K, Q6_K, Q3_K, Q2_K, Q5_0, Q5_1 | Various K-quants | `dequantQ5KKernel`, `dequantQ6KKernel`, etc. |

## 7. LlamaLanguageModel Implementation

`Sources/EdgeRunner/Models/LlamaLanguageModel.swift`

### 7.1 Key State

```swift
public struct LlamaLanguageModel: LogitsModel, @unchecked Sendable {
    // Metal infrastructure
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Per-layer KV cache MTLBuffers
    private let layerKCaches: [MTLBuffer]
    private let layerVCaches: [MTLBuffer]

    // Decoder state for cache detection
    private let decoderState: DecoderStateStore

    // Pre-loaded weight buffers (Q8_0 raw)
    private let preloadedWeights: PreloadedWeightsStore

    // Pre-allocated scratch buffers
    private let scratch: ScratchBuffers

    // Debug options
    private let decodeDebugOptions: DecodeDebugOptions
}
```

### 7.2 DecoderStateStore

Tracks previously processed tokens for KV cache mode detection:

```swift
/// Tracks the previously processed token sequence for KV cache decode detection.
struct DecoderStateStore {
    var previousTokenIDs: [Int] = []
    var cachedLogits: [Float]?
    var cachedLogitsInput: [Int]?
    var decodeWarmedUp: Bool = false
}
```

### 7.3 ScratchBuffers

Pre-allocated MTLBuffers to avoid per-call allocations:

```swift
struct ScratchBuffers {
    let normed: MTLBuffer      // After RMSNorm
    let afterAttn: MTLBuffer   // After attention + residual
    let ffnNormed: MTLBuffer   // Before FFN
    let outputA, outputB: MTLBuffer  // Layer output ping-pong
    let allQ, allK, allV: MTLBuffer  // Attention inputs
    let ropeQ, ropeK: MTLBuffer      // After RoPE
    let attnOut: MTLBuffer     // Attention output
    let proj: MTLBuffer        // After projection
    let gateOut, upOut: MTLBuffer  // FFN intermediate
    let activ: MTLBuffer       // SiLU activation
    let downOut: MTLBuffer     // FFN down projection
    let logits: MTLBuffer      // Final output
    let decodeHidden: MTLBuffer // Single-token embedding
}
```

## 8. Generation Loop

### 8.1 Stream Generation Flow

```
1. Tokenize prompt + add BOS if needed
2. For each step in 0..<maxTokens:
   a. Call model.nextToken(tokenIDs, sampling)
   b. If token == EOS: break
   c. Append token to tokenIDs
   d. Detokenize token to text
   e. Call onToken callback (if provided)
   f. Yield text through AsyncThrowingStream
3. Finish stream
```

### 8.2 nextToken Implementation

```swift
public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
    // Fast path: pure greedy (no logits materialization needed)
    if isPureGreedy {
        let result = try await greedyToken(for: tokenIDs)
        return result.token
    }

    // Standard path: logits + sampling
    let logitsArray: [Float]
    if tokenIDs == decoderState.cachedLogitsInput {
        logitsArray = decoderState.cachedLogits!
    } else {
        logitsArray = try await self.logits(for: tokenIDs)
    }

    let pipeline = sampling.toPipeline()
    return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
}
```

## 9. Conversation/Chat Integration

`Sources/EdgeRunner/Conversation.swift`

For multi-turn conversations:

```swift
public struct Conversation: Sendable {
    public private(set) var messages: [ChatMessage]

    public mutating func addUser(_ content: String)
    public mutating func addAssistant(_ content: String)
    public mutating func reset(keepSystem: Bool = true)
}
```

**Usage pattern:**
```swift
var convo = Conversation(systemPrompt: "You are helpful.")
convo.addUser("What is 2+2?")
let prompt = model.applyChatTemplate(messages: convo.messages, addGenerationPrompt: true)
// Generate response...
convo.addAssistant(response)
// Next turn: KV cache prefix reuse happens automatically
```

## 10. Error Handling

`Sources/EdgeRunnerCore/Generation/GenerationError.swift`

| Error | Cause |
|-------|-------|
| `modelLoadFailed` | GPU unavailable, buffer allocation failed |
| `decodingFailed` | NaN/Inf in logits during greedy decode |
| `tokenizationFailed` | Invalid UTF-8, tokenizer error |
| `unsupportedModel` | Unknown architecture |
| `cancelled` | Task was cancelled |

## 11. File Map

| File | Purpose |
|------|---------|
| `EdgeRunner/EdgeRunnerLanguageModel.swift` | Core model protocol |
| `EdgeRunner/Models/LlamaLanguageModel.swift` | Main Metal-accelerated implementation |
| `EdgeRunner/Streaming/GenerationSession.swift` | Streaming generation session |
| `EdgeRunner/Streaming/TokenStream.swift` | Token stream types |
| `EdgeRunner/SamplingConfiguration.swift` | Sampling config |
| `EdgeRunner/Conversation.swift` | Chat history management |
| `EdgeRunnerCore/Sampling/SamplingPipeline.swift` | Composable sampling pipeline |
| `EdgeRunnerCore/Sampling/GreedySampler.swift` | Argmax selection |
| `EdgeRunnerCore/Sampling/StochasticSampler.swift` | Random sampling |
| `EdgeRunnerCore/Sampling/TemperatureSampler.swift` | Temperature scaling |
| `EdgeRunnerCore/Sampling/TopKSampler.swift` | Top-K filtering |
| `EdgeRunnerCore/Sampling/TopPSampler.swift` | Nucleus filtering |
| `EdgeRunnerCore/Sampling/RepetitionPenalty.swift` | Repetition penalty |
| `EdgeRunnerMetal/KVCache.swift` | KV cache implementation |
| `EdgeRunner/Transformer/TransformerBlock.swift` | CPU fallback transformer |
