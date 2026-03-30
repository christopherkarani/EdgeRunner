# EdgeRunner Public API Documentation

## Overview

EdgeRunner is a Swift package for on-device LLM inference with Metal GPU acceleration. It supports GGUF model files with automatic architecture detection (Llama, Qwen, Gemma, Mistral, Phi3, and more).

**Products:**
- `EdgeRunner` - Main package re-exporting all public types
- `EspressoEdgeRunner` - Separate package for Espresso/ANE integration

**Platforms:** iOS 26+, macOS 26+

---

## Package Structure

```
EdgeRunner
├── EdgeRunner           (main product, re-exports all modules)
├── EdgeRunnerCore       (tensor computation, sampling, tokenizers)
├── EdgeRunnerIO         (model loading, GGUF parsing, quantization)
├── EdgeRunnerMetal      (Metal compute kernels)
├── EdgeRunnerSharedTypes (C interop headers)
└── ANEInteropIO        (Apple Neural Engine integration)
```

---

## Core Protocols

### `EdgeRunnerLanguageModel`

The central protocol for all language model implementations. All implementations are `Sendable` and thread-safe.

```swift
public protocol EdgeRunnerLanguageModel: Sendable {
    /// Model type identifier (e.g., "llama")
    static var modelIdentifier: String { get }

    /// Load a model from a GGUF file
    static func load(from url: URL, configuration: ModelConfiguration) async throws -> Self

    /// Convert text to token IDs
    func tokenize(_ text: String) -> [Int]

    /// Convert token IDs to text
    func detokenize(_ ids: [Int]) -> String

    /// End-of-sequence token ID
    var eosTokenID: Int { get }

    /// Beginning-of-sequence token ID
    var bosTokenID: Int? { get }

    /// Vocabulary size
    var vocabularySize: Int { get }

    /// Apply chat template to format messages
    func applyChatTemplate(messages: [ChatMessage], addGenerationPrompt: Bool) -> String?

    /// Generate next token given token IDs
    func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int

    /// Stream generated text as async sequence
    func stream(_ prompt: String) -> AsyncThrowingStream<String, Error>
}
```

**Default Implementations:**
- `applyChatTemplate` returns `nil` by default (no template support)
- `stream(prompt:)` has a default implementation using `nextToken(for:sampling:)` loop

### `LogitsModel`

Sub-protocol for Metal-accelerated models that expose raw logits access. Foundation Models backends do NOT conform to this.

```swift
public protocol LogitsModel: EdgeRunnerLanguageModel {
    /// Returns raw logits for given token IDs
    func logits(for tokenIDs: [Int]) async throws -> [Float]
}
```

### `EdgeRunnerModule`

Protocol for composable neural network modules.

```swift
public protocol EdgeRunnerModule: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable

    /// Forward computation
    func forward(_ input: Input) async throws -> Output

    /// Learnable parameters keyed by name
    var parameters: [String: any TensorBox] { get }
}
```

### `Tokenizer`

Protocol for tokenizer implementations.

```swift
public protocol Tokenizer: Sendable {
    func encode(_ text: String, addBOS: Bool) -> [Int]
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String
    var vocabularySize: Int { get }
    var eosTokenID: Int { get }
    var bosTokenID: Int? { get }
    var padTokenID: Int? { get }
    var shouldAddBOS: Bool { get }
    func applyChatTemplate(messages: [ChatMessage], addGenerationPrompt: Bool) throws -> String?
}
```

### `EdgeRunnerTool`

Protocol for tool/function calling implementations.

```swift
public protocol EdgeRunnerTool: Sendable {
    static var name: String { get }
    static var description: String { get }
    static var parameters: [ToolParameter] { get }
    func invoke(arguments: [String: Any]) async throws -> String
}
```

### `LoadableModel`

Protocol for model weight loading.

```swift
public protocol LoadableModel: Sendable {
    var parameterNames: [String] { get }
    mutating func loadWeights(from map: WeightMap) throws
}
```

### `LocalModelBackend`

Protocol for local Metal-accelerated model backends.

```swift
public protocol LocalModelBackend: EdgeRunnerLanguageModel {
    static var supportedFormat: String { get }
    func estimatedMemoryUsage() -> Int
}
```

### `SystemModelBackend`

Protocol for system-integrated model backends (e.g., Apple Foundation Models).

```swift
public protocol SystemModelBackend: EdgeRunnerLanguageModel {
    var supportsGuidedGeneration: Bool { get }
    func generateStream(prompt: String) -> AsyncThrowingStream<String, Error>
}
```

---

## Public Structs and Types

### Configuration Types

#### `ModelConfiguration`

Configuration for model loading and generation behavior.

```swift
public struct ModelConfiguration: Sendable {
    /// Maximum tokens to generate (default: 2048)
    public var maxTokens: Int

    /// Maximum sequence length (default: 4096)
    public var contextWindowSize: Int

    /// Use memory-mapped file I/O (default: true)
    public var useMemoryMapping: Bool

    /// Optional external tokenizer URL
    public var tokenizerURL: URL?

    public init(
        maxTokens: Int = 2048,
        contextWindowSize: Int = 4096,
        useMemoryMapping: Bool = true,
        tokenizerURL: URL? = nil
    )
}
```

#### `SamplingConfiguration`

Configuration for token sampling during generation.

```swift
public struct SamplingConfiguration: Sendable {
    public var temperature: Float       // Default: 1.0
    public var topK: Int               // Default: 40
    public var topP: Float             // Default: 0.9
    public var repetitionPenalty: Float // Default: 1.0
    public var seed: UInt64?           // Optional, for reproducibility

    public init(
        temperature: Float = 1.0,
        topK: Int = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil
    )

    /// Convert to composable SamplingPipeline
    public func toPipeline() -> SamplingPipeline
}
```

#### `TransformerConfig`

Configuration for decoder-only transformer models.

```swift
public struct TransformerConfig: Sendable {
    public let hiddenDim: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let intermediateSize: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float

    public var headDim: Int { hiddenDim / numHeads }
    public var kvGroupSize: Int { numHeads / numKVHeads }
}
```

#### `EdgeRunnerMemoryPolicy`

Memory management policy for model loading.

```swift
public struct EdgeRunnerMemoryPolicy: Sendable, Equatable {
    public let fallbackChain: [QuantisationLevel]
    public let evictBufferCacheOnPressure: Bool
    public let maxMemoryBytes: Int

    public static let `default` = EdgeRunnerMemoryPolicy(
        fallbackChain: [.q8_0, .q4_k_m, .q4_0],
        evictBufferCacheOnPressure: true
    )
}

public enum QuantisationLevel: String, Sendable, Equatable, CaseIterable {
    case q8_0   // 8 bits
    case q4_k_m // 4.5 bits
    case q4_0   // 4 bits
}
```

#### `LlamaConfig`

Llama-specific model configuration parsed from GGUF metadata.

```swift
public struct LlamaConfig: Sendable, Equatable {
    public let embeddingDim: Int
    public let layerCount: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let vocabSize: Int
    public let intermediateDim: Int
    public let ropeFreqBase: Double
    public let rmsNormEpsilon: Double
    public let explicitHeadDim: Int?  // Qwen 3 uses non-standard head dim

    public var headDim: Int
    public var gqaRatio: Int

    public init(fromGGUFMetadata metadata: [String: MetadataValue]) throws
}
```

#### `ModelConfig`

Generic model configuration with metadata accessors.

```swift
public struct ModelConfig: Sendable, Equatable {
    public let architectureName: String
    public let metadata: [String: MetadataValue]

    public func string(forKey key: String) -> String?
    public func int(forKey key: String) -> Int?
    public func float(forKey key: String) -> Float?
    public func bool(forKey key: String) -> Bool?
    public func array(forKey key: String) -> [MetadataValue]?
}
```

### Chat and Conversation Types

#### `ChatMessage`

A single message in a chat conversation.

```swift
public struct ChatMessage: Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public var content: String
    public let timestamp: Date

    public enum MessageRole: String, Sendable {
        case user, assistant, system
    }

    public init(role: MessageRole, content: String)
}
```

#### `Conversation`

Lightweight message history manager for multi-turn conversations.

```swift
public struct Conversation: Sendable {
    public private(set) var messages: [ChatMessage]

    public init(systemPrompt: String? = nil)
    public mutating func addUser(_ content: String)
    public mutating func addAssistant(_ content: String)
    public mutating func addSystem(_ content: String)
    public mutating func reset(keepSystem: Bool = true)
    public var messageCount: Int { messages.count }
    public var isEmpty: Bool { messages.isEmpty }
}
```

#### `ChatViewModelState`

Pure state container for chat UI (testable without SwiftUI).

```swift
public struct ChatViewModelState: Sendable {
    public var messages: [ChatMessage]
    public var isGenerating: Bool
    public var currentInput: String
    public var selectedModel: ModelInfo?
    public var memoryUsedMB: Double
    public var memoryTotalMB: Double
    public var tokensPerSecond: Double
    public var error: String?

    public var memoryUsagePercent: Double
    public mutating func addUserMessage(_ content: String)
    public mutating func addAssistantMessage(_ content: String)
    public mutating func appendToLastMessage(_ text: String)
    public mutating func clearMessages()
    public mutating func updateMemoryUsage(usedMB: Double, totalMB: Double)
}
```

### Model Information Types

#### `ModelInfo`

Metadata about an available model file.

```swift
public struct ModelInfo: Identifiable, Sendable {
    public let id: UUID
    public let name: String
    public let path: URL
    public let format: String
    public let parameterCount: String
    public let quantization: String
    public let fileSizeBytes: Int64

    public var fileSizeFormatted: String  // e.g., "4.2 GB"
}
```

### Tool Calling Types

#### `ToolParameter`

Definition of a tool parameter.

```swift
public struct ToolParameter: Sendable {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let required: Bool
}

public enum ToolParameterType: String, Sendable {
    case string, integer, number, boolean, array, object
}
```

#### `ToolCall`

Represents a tool call request.

```swift
public struct ToolCall: Sendable {
    public let name: String
    public let arguments: [String: any Sendable]
}
```

#### `ToolChoice`

Strategy for tool selection in function calling.

```swift
public enum ToolChoice: Sendable, Equatable {
    case auto
    case required
    case none
    case specific(String)  // Specific tool name
}
```

#### `ToolExecutor`

Executes tool calls.

```swift
public struct ToolExecutor: Sendable {
    public init(tools: [any EdgeRunnerTool])
    public func execute(_ call: ToolCall) async throws -> String
    public func executeAll(_ calls: [ToolCall]) async throws -> [String]
    public func toolDescriptions() -> String
    public func shouldAttemptToolCall(choice: ToolChoice, modelOutput: String) -> Bool
}
```

### Streaming and Generation Types

#### `GenerationSession`

Manages a single text generation session with streaming output.

```swift
public struct GenerationSession<Model: EdgeRunnerLanguageModel>: Sendable {
    public let maxTokens: Int

    public init(
        model: Model,
        sampling: SamplingConfiguration = SamplingConfiguration(),
        maxTokens: Int = 2048,
        onToken: (@Sendable (Int, String) -> Void)? = nil
    )

    /// Stream generated tokens
    public func stream(prompt: String) -> AsyncThrowingStream<String, Error>

    /// Generate complete response (non-streaming)
    public func generate(prompt: String) async throws -> String
}
```

#### `StreamToken`

A token emitted during streaming generation.

```swift
public struct StreamToken: Sendable {
    public let id: Int
    public let text: String
    public let isEOS: Bool
}
```

#### `GenerationStats`

Statistics collected during a generation session.

```swift
public struct GenerationStats: Sendable {
    public var tokenCount: Int = 0
    public var timeToFirstToken: Double = 0
    public var totalTime: Double = 0
    public var tokensPerSecond: Double
}
```

### Module Types

#### `Sequential`

Container that chains modules in sequence.

```swift
public struct Sequential<M: EdgeRunnerModule>: EdgeRunnerModule
where M.Input == M.Output {
    public init(_ modules: M...)
    public init(_ modules: [M])
    public func forward(_ input: Input) async throws -> Output
    public var parameters: [String: any TensorBox]
}
```

#### `AnyModule`

Type-erased module wrapper for heterogeneous module composition.

```swift
public struct AnyModule<Value: Sendable>: EdgeRunnerModule, Sendable {
    public init<M: EdgeRunnerModule>(_ module: M)
    public func forward(_ input: Value) async throws -> Value
    public var parameters: [String: any TensorBox]
}
```

#### `LinearModule`

A fully-connected linear layer: `y = x @ W^T + b`.

```swift
public struct LinearModule: EdgeRunnerModule, Sendable {
    public typealias Input = [Float]
    public typealias Output = [Float]

    public let inFeatures: Int
    public let outFeatures: Int

    public init(
        inFeatures: Int,
        outFeatures: Int,
        weight: [Float],
        bias: [Float]?
    ) throws

    public func forward(_ input: [Float]) async throws -> [Float]
    public var parameters: [String: any TensorBox]
}
```

#### `TensorBox`

Type-erased container for tensor parameter data.

```swift
public protocol TensorBox: Sendable {
    var elementCount: Int { get }
    var floatArray: [Float] { get }
    var shape: [Int] { get }
}

public struct ScalarTensorBox: TensorBox, Sendable {
    public let value: Float
    public var elementCount: Int { 1 }
    public var floatArray: [Float] { [value] }
    public var shape: [Int] { [] }
}

public struct ArrayTensorBox: TensorBox, Sendable {
    public let data: [Float]
    public let shape: [Int]
    public var elementCount: Int { data.count }
    public var floatArray: [Float] { data }
}
```

### Transformer Types

#### `TransformerBlockInput`

Input to a single transformer block.

```swift
public struct TransformerBlockInput: Sendable {
    public let hidden: [Float]
    public let seqLen: Int
    public let startPos: Int
}
```

#### `TransformerBlockOutput`

Output from a single transformer block.

```swift
public struct TransformerBlockOutput: Sendable {
    public let hidden: [Float]
}
```

### Metrics Types

#### `Perplexity`

Perplexity computation utilities for language model evaluation.

```swift
public enum Perplexity: Sendable {
    public static func negLogLikelihood(logits: [Float], targetId: Int) -> Float
    public static func compute(logitsPerToken: [[Float]], targetIds: [Int]) -> Float
}
```

---

## Error Types

### `GenerationError`

Errors during model loading and text generation.

```swift
public enum GenerationError: Error, Sendable, CustomStringConvertible {
    case modelLoadFailed(reason: String)
    case contextWindowExceeded(requested: Int, maximum: Int)
    case invalidTokenID(Int)
    case decodingFailed(String)
    case cancelled
    case samplingFailed(String)
    case toolCallFailed(name: String, reason: String)
    case structuredOutputFailed(reason: String)
}
```

### `ModelLoadError`

Errors specific to model loading.

```swift
public enum ModelLoadError: Error, Sendable, Equatable {
    case unsupportedFormat(String)
    case unknownArchitecture(String)
    case loadFailed(description: String)
}
```

### `WeightLoaderError`

Errors during weight loading from GGUF files.

```swift
public enum WeightLoaderError: Error, Sendable, Equatable {
    case deviceNotAvailable
    case fileNotFound(URL)
    case invalidFormat(String)
    case unsupportedVersion(UInt32)
    case unsupportedDataType(UInt32)
    case allocationFailed(byteCount: Int)
    case mmapFailed(errno: Int32)
    case missingMetadata(String)
    case tensorNotFound(String)
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
    case checksumMismatch(name: String)
}
```

### `LlamaConfigError`

Errors during Llama configuration parsing.

```swift
public enum LlamaConfigError: Error, Sendable, Equatable {
    case missingMetadataKey(String)
    case invalidMetadataValue(key: String, description: String)
}
```

### `TokenizerFactoryError`

Tokenizer creation errors.

```swift
public enum TokenizerFactoryError: Error, Sendable, Equatable
```

### `ShapeError`

Tensor shape errors.

```swift
public enum ShapeError: Error, Sendable
```

### `TensorStorageError`

Tensor storage errors.

```swift
public enum TensorStorageError: Error, Sendable
```

---

## Backend Registry

### `BackendRegistry`

Registry for loading models by format.

```swift
public final class BackendRegistry: Sendable {
    public init()

    public func register<T: EdgeRunnerLanguageModel>(_ type: T.Type, for format: String)
    public func backend(for format: String) -> (any EdgeRunnerLanguageModel.Type)?
    public var availableBackends: Set<String>
    public func load(
        from url: URL,
        format: String,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws -> any EdgeRunnerLanguageModel
}
```

---

## Foundation Models Backend

### `FoundationModelsAvailability`

Availability check for Foundation Models integration.

```swift
public enum FoundationModelsAvailability {
    public static var isAvailable: Bool
}
```

---

## Model Loading

### `ModelLoader`

High-level model loading with automatic architecture detection.

```swift
public enum ModelLoader: Sendable {
    /// Load a GGUF model with automatic architecture detection
    public static func load(
        from url: URL,
        configuration: ModelConfiguration = ModelConfiguration()
    ) async throws -> any EdgeRunnerLanguageModel
}
```

**Supported Architectures:**
- `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`
- `phi3`, `mistral`, `starcoder`, `starcoder2`
- `internlm2`, `yi`, `deepseek`, `deepseek2`
- `command-r`, `falcon`

---

## Sampling Pipeline

### `SamplingPipeline`

Composable sampling pipeline with transforms and selectors.

```swift
public struct SamplingPipeline: Sendable {
    public init(
        transforms: [any LogitsTransform],
        selector: any TokenSelector,
        repetitionPenalty: RepetitionPenalty? = nil
    )

    public func sample(logits: [Float], previousTokens: [Int] = []) -> Int

    public static var greedy: SamplingPipeline
    public static func nucleus(temperature: Float = 0.8, topP: Float = 0.9, seed: UInt64 = 0) -> SamplingPipeline
    public static func topK(k: Int = 40, temperature: Float = 0.8, seed: UInt64 = 0) -> SamplingPipeline
}
```

### `LogitsTransform`

Protocol for logits transformation (temperature, top-k, top-p).

```swift
public protocol LogitsTransform: Sendable {
    func transformLogits(_ logits: [Float]) -> [Float]
}
```

### `TokenSelector`

Protocol for token selection (greedy, stochastic).

```swift
public protocol TokenSelector: Sendable {
    func sample(logits: [Float]) -> Int
}
```

### Built-in Transforms

```swift
public struct TemperatureSampler: LogitsTransform, Sendable
public struct TopKSampler: LogitsTransform, Sendable
public struct TopPSampler: LogitsTransform, Sendable
public struct MinPSampler: LogitsTransform, Sendable
public struct GreedySampler: TokenSelector, Sendable
public struct RepetitionPenalty: Sendable
public struct SeededRandomSource: RandomNumberGenerator, Sendable
```

### Stochastic Sampler

```swift
public final class StochasticSampler<RNG: RandomNumberGenerator & Sendable>: TokenSelector, @unchecked Sendable {
    public init(randomSource: inout RNG)
    public func sample(logits: [Float]) -> Int
}
```

---

## Concrete Model Implementations

### `LlamaLanguageModel`

Full Llama inference engine conforming to `LogitsModel`.

```swift
public struct LlamaLanguageModel: LogitsModel, @unchecked Sendable {
    public static let modelIdentifier = "llama"

    /// Load from GGUF file
    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> LlamaLanguageModel

    /// Tokenize text
    public func tokenize(_ text: String) -> [Int]

    /// Detokenize tokens
    public func detokenize(_ ids: [Int]) -> String

    public var eosTokenID: Int
    public var bosTokenID: Int?
    public var vocabularySize: Int

    /// Apply chat template
    public func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool
    ) -> String?

    /// Generate next token
    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int

    /// Raw logits access
    public func logits(for tokenIDs: [Int]) async throws -> [Float]
}
```

**Architecture Support:**
- Llama 2, Llama 3
- Qwen 2, Qwen 3
- Mistral, Gemma
- Any GGUF model with standard Llama architecture

**Quantization Support:**
- Q2_K, Q3_K, Q4_0, Q4_K_M, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0

---

## Usage Examples

### Basic Generation

```swift
import EdgeRunner

// Load a model
let model = try await ModelLoader.load(
    from: modelURL,
    configuration: ModelConfiguration()
)

// Simple generation
let tokens = model.tokenize("Hello, world!")
for _ in 0..<50 {
    let next = try await model.nextToken(for: tokens, sampling: SamplingConfiguration())
    tokens.append(next)
}
let text = model.detokenize(tokens)
```

### Streaming Generation

```swift
import EdgeRunner

let model = try await ModelLoader.load(from: modelURL)

// Using AsyncThrowingStream
let stream = model.stream("Tell me a story")
for try await text in stream {
    print(text, terminator: "")
}
```

### Using GenerationSession

```swift
import EdgeRunner

let model = try await ModelLoader.load(from: modelURL)
let session = GenerationSession(
    model: model,
    sampling: SamplingConfiguration(temperature: 0.7, topP: 0.9),
    maxTokens: 1024
)

// Streaming
let stream = session.stream(prompt: "Write a haiku")
for try await text in stream {
    print(text, terminator: "")
}

// Non-streaming
let result = try await session.generate(prompt: "Write a haiku")
```

### Chat with Conversation

```swift
import EdgeRunner
import EdgeRunnerCore

var convo = Conversation(systemPrompt: "You are a helpful assistant.")
convo.addUser("What is 2+2?")
let prompt = model.applyChatTemplate(
    messages: convo.messages,
    addGenerationPrompt: true
)

// Generate response...
convo.addAssistant(response)
convo.addUser("And 3+3?")
// Continue conversation...
```

### Tool Calling

```swift
import EdgeRunner

// Define a tool
struct CalculatorTool: EdgeRunnerTool {
    static let name = "calculator"
    static let description = "Perform calculations"
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "expression", type: .string, description: "Math expression", required: true)
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let expr = arguments["expression"] as! String
        // Evaluate...
        return "\(result)"
    }
}

// Execute tools
let executor = ToolExecutor(tools: [CalculatorTool()])
if executor.shouldAttemptToolCall(choice: .auto, modelOutput: modelOutput) {
    let results = try await executor.executeAll(parsedToolCalls)
}
```

### Custom Sampling Pipeline

```swift
import EdgeRunner

// Greedy sampling
let pipeline = SamplingPipeline.greedy

// Nucleus sampling with seed
let pipeline = SamplingPipeline.nucleus(temperature: 0.8, topP: 0.95, seed: 42)

// Custom pipeline
let pipeline = SamplingPipeline(
    transforms: [
        TemperatureSampler(temperature: 0.7),
        TopPSampler(p: 0.9)
    ],
    selector: StochasticSampler(randomSource: &rng),
    repetitionPenalty: RepetitionPenalty(penalty: 1.1)
)
```

---

## Module Organization

### EdgeRunner (Main Product)

Re-exports:
- `EdgeRunnerCore`
- `EdgeRunnerIO`
- `EdgeRunnerMetal`
- `EdgeRunnerSharedTypes`

### EdgeRunnerCore

- `Tensor`, `Shape`, `Strides`
- `ComputeGraph`, `TensorOp`, `FusionEngine`, `AutoTuner`
- `SamplingPipeline`, `SamplingConfiguration`
- All sampling transforms and selectors
- `Tokenizer`, `BPETokenizer`, `SentencePieceTokenizer`
- `ChatMessage`, `ChatTemplateEngine`
- `GenerationError`, `SpeculativeDecoder`
- `StructuredGenerator`, `GrammarState`

### EdgeRunnerIO

- `ModelLoader`, `ModelConfiguration`, `ModelConfig`
- `GGUFLoader`, `GGUFParser`, `GGUFMetadata`
- `LlamaConfig`, `LlamaModel`, `LlamaBlock`
- `WeightLoader`, `WeightMap`, `TensorDataType`
- `SafeTensorLoader`, `NPZLoader`, `NPYParser`
- `EdgeRunnerMemoryPolicy`
- `LoadableModel` protocol

### EdgeRunnerMetal

- `MetalBackend`
- `GEMMKernel`, `GEMVKernel`
- `RMSNormKernel`, `LayerNormKernel`
- `RoPEKernel`, `GQAKernel`
- `FlashAttentionKernel`, `SoftmaxKernel`
- `DequantQ4_0Kernel`, `DequantQ8_0Kernel`, etc.
- `KVCache`, `BufferCache`
- All Metal shaders

---

## Type Aliases and Re-exports

The main `EdgeRunner` module re-exports all public types from its dependencies via `@_exported import`, so users typically only need:

```swift
import EdgeRunner
```

Key re-exported types from EdgeRunnerCore:
- `ChatMessage` (from EdgeRunnerCore.Tokenizer.ChatMessage)
- `GenerationError`
- `SamplingPipeline`, `SamplingConfiguration`
- `Tokenizer`, `BPETokenizer`, `SentencePieceTokenizer`
- All sampling types
