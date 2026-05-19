# EdgeRunner

[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2026%2B%20%7C%20iOS%2026%2B-blue.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, local LLM inference for Apple Silicon. Built in Swift/Metal from the ground up.**

Run large language models entirely on your Mac or iPhone—no network, no API keys, no data leaving your device. EdgeRunner loads GGUF models, fuses GPU kernels, and streams tokens at speeds that rival (and sometimes beat) the heavyweights.

```swift
import EdgeRunner

let runner = try await EdgeRunner(modelPath: "Qwen3-0.6B-Q8_0.gguf")

// Streaming
for try await token in runner.stream("Once upon a time") {
    print(token, terminator: "")
}

// Or one-shot
let text = try await runner.generate("What is Swift?", maxTokens: 100)
```

---

## What You Can Build

- **Private chatbots** — Talk to an AI that lives on your device. Your conversations never leave your Mac.
- **Code assistants** — Autocomplete, explain code, or generate snippets offline in Xcode.
- **On-device agents** — Give your app reasoning, tool-calling, and long-context memory with zero cloud costs.
- **Embedded intelligence** — Ship an LLM inside your iOS or macOS app. No backend required.

---

## Features

- **⚡ ~230+ tok/s** median decode on Qwen3-0.6B (Apple M3 Max, publishable benchmark)
- **🚀 3.5 ms time-to-first-token** — start reading output before you blink
- **🧠 Metal-native** — Custom compute kernels optimized for Apple Silicon (Metal 3 & Metal 4)
- **📦 GGUF support** — Load quantized models directly: Q8_0, Q4_0, Q4_K_M, Q2_K, Q3_K, Q5_0, Q5_1, Q5_K, Q6_K, F16, F32
- **💾 Memory-mapped loading** — Instant startup, minimal memory pressure
- **🔒 Private** — Runs entirely on-device, no network required
- **💬 Chat templates** — Built-in Jinja2-style chat formatting for multi-turn conversations
- **🛠️ Tool calling** — Let models invoke your Swift code with structured JSON arguments
- **📱 iOS + macOS** — Drop into any Swift project as a package dependency

---

## Requirements

- **macOS 26.0+** or **iOS 26.0+** (required for Metal 4 argument-table dispatch and residency sets)
- Apple Silicon (M1 or later)
- Swift 6.2+
- Xcode 26 beta or newer

> **Note:** EdgeRunner pushes the bleeding edge of Apple’s GPU APIs. The macOS 26 requirement unlocks zero-dispatch-overhead inference via Metal 4. If you’re on an older OS, watch this repo—backwards compatibility is on the roadmap.

---

## Installation

### Swift Package Manager

Add EdgeRunner to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/EdgeRunner.git", from: "0.1.0")
]
```

Or add it in Xcode: **File → Add Package Dependencies...**

---

## Quick Start

### 1. Download a Model

EdgeRunner uses **GGUF** format models. Grab a small one to test (~640 MB):

```bash
mkdir -p ~/edgerunner-models
curl -L -o ~/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

**Recommended models:**

| Model | Size | Best For |
|-------|------|----------|
| Qwen3 0.6B Q8_0 | ~640 MB | Fast tests, low latency |
| Qwen3 1.7B Q8_0 | ~1.8 GB | Better quality, still fast |
| Qwen3 4B Q8_0 | ~4.3 GB | Strong reasoning |
| Llama 3.1 8B Q4_K_M | ~4.7 GB | General purpose, best accuracy |

### 2. Generate Text

```swift
import EdgeRunner

@main
struct HelloEdgeRunner {
    static func main() async throws {
        let runner = try await EdgeRunner(
            modelPath: "~/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
        )

        print("🤖 Generating...")
        let text = try await runner.generate(
            "The capital of France is",
            maxTokens: 10
        )
        print(text)
        // Output: " Paris. It is known for the Eiffel Tower..."
    }
}
```

### 3. Stream Tokens in Real Time

```swift
for try await token in runner.stream("Write a haiku about Swift:") {
    print(token, terminator: "")
}
// Output appears token-by-token as the GPU generates it
```

---

## Examples

### Chat with History

EdgeRunner understands chat templates built into the model. Just pass messages:

```swift
import EdgeRunner
import EdgeRunnerCore

let runner = try await EdgeRunner(modelPath: "Qwen3-0.6B-Q8_0.gguf")

let messages = [
    ChatMessage(role: "system", content: "You are a helpful coding assistant."),
    ChatMessage(role: "user", content: "How do I read a file in Swift?")
]

let model = try await ModelLoader.load(
    from: URL(fileURLWithPath: "~/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"),
    configuration: ModelConfiguration()
)

if let prompt = model.applyChatTemplate(messages: messages, addGenerationPrompt: true) {
    for try await token in runner.stream(prompt, maxTokens: 200) {
        print(token, terminator: "")
    }
}
```

### SwiftUI Chat Interface

```swift
import SwiftUI
import EdgeRunner

struct ChatView: View {
    @State private var runner: EdgeRunner?
    @State private var messages = [String]()
    @State private var input = ""
    @State private var isGenerating = false

    var body: some View {
        VStack {
            List(messages, id: \.self) { Text($0) }

            HStack {
                TextField("Message...", text: $input)
                Button("Send") {
                    Task { await send() }
                }
                .disabled(isGenerating)
            }
            .padding()
        }
        .task {
            runner = try? await EdgeRunner(
                modelPath: "~/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"
            )
        }
    }

    func send() async {
        guard let runner else { return }
        isGenerating = true
        messages.append("You: \(input)")
        var reply = "Bot: "
        for try await token in runner.stream(input, maxTokens: 100) {
            reply += token
        }
        messages.append(reply)
        input = ""
        isGenerating = false
    }
}
```

### Custom Sampling

Control creativity and repetition:

```swift
let text = try await runner.generate(
    "Write a story about a robot",
    maxTokens: 200,
    sampling: SamplingConfiguration(
        temperature: 0.7,   // More randomness
        topP: 0.9,          // Nucleus sampling
        topK: 40,           // Top-k filtering
        repetitionPenalty: 1.1
    )
)
```

### Tool Calling

Give your model access to Swift functions:

```swift
import EdgeRunner
import EdgeRunnerCore

struct WeatherTool: EdgeRunnerTool {
    static let name = "get_weather"
    static let description = "Get the current weather"
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "location", type: .string, description: "City name", required: true)
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let city = arguments["location"] as? String ?? "unknown"
        return "Weather in \(city): 72°F, sunny"
    }
}

let tools = [WeatherTool()]
let executor = ToolExecutor(tools: tools)

// Include tool descriptions in your system prompt, then:
let output = try await runner.generate(
    "What's the weather in Tokyo?",
    maxTokens: 50
)

// Parse tool calls from the model's response
let calls = try ToolCallParser.parse(modelOutput: output)
for call in calls {
    let result = try await executor.execute(call)
    print(result)  // "Weather in Tokyo: 72°F, sunny"
}
```

---

## Performance

Benchmarks on Apple M3 Max (128-token greedy decode):

| Model | EdgeRunner | llama.cpp | MLX |
|-------|-----------|-----------|-----|
| Qwen3-0.6B-Q8_0 | **~230+ tok/s** | ~198 tok/s | ~286 tok/s |
| Time to First Token | **3.5 ms** | n/a | n/a |

**Why EdgeRunner wins on TTFT:**
While MLX is slightly faster on sustained decode, EdgeRunner starts emitting tokens in **3.5 ms**—over **20× faster** than MLX’s typical 70–80 ms startup. For interactive apps (chat, autocomplete), perceived responsiveness matters more than raw throughput.

Memory usage (with memory mapping):

| Model | Weights | KV Cache (2K ctx) | Total |
|-------|---------|-------------------|-------|
| Qwen3-0.6B-Q8_0 | ~640 MB | ~70 MB | ~710 MB |
| Qwen3-1.7B-Q8_0 | ~1.8 GB | ~150 MB | ~2.0 GB |
| Qwen3-4B-Q8_0 | ~4.3 GB | ~280 MB | ~4.6 GB |

---

## Supported Models

EdgeRunner auto-detects architecture from GGUF metadata and supports any model in the Llama-family:

- **Qwen3** (0.6B, 1.7B, 4B) — primary test target
- **Llama 3 / 3.1** (8B, 70B)
- **Mistral / Mixtral**
- **Gemma / Gemma 2 / Gemma 4**
- **Phi-3**
- **DeepSeek**
- **Yi**, **InternLM2**, **StarCoder**, **Falcon**, **Command-R**

Plus dedicated loaders for:
- **Gemma 4** (multimodal vision-language)
- **Bonsai** (experimental architecture)

---

## Project Status

EdgeRunner is in **beta**. Core inference is solid and fast. Here is the honest state of play:

| Feature | Status | Notes |
|---------|--------|-------|
| Fast GPU inference | ✅ Production-grade | 230+ tok/s on Qwen3-0.6B |
| GGUF Q8_0 / Q4_K_M / etc. | ✅ Supported | 9 quantization types |
| KV cache + incremental decode | ✅ Production-grade | Auto prefix-reuse |
| Streaming generation | ✅ Works | `AsyncThrowingStream<String, Error>` |
| BPE tokenizer | ✅ Production-grade | Validated against HuggingFace |
| Chat templates (Jinja2) | ✅ Production-grade | Qwen3, Llama, Mistral templates |
| Sampling (temp, top-p, top-k) | ✅ Works | `SamplingConfiguration` |
| Tool calling protocol | ✅ Works | `EdgeRunnerTool` + `ToolExecutor` |
| Multi-model support | ✅ 10+ architectures | Auto-detected from GGUF |
| Memory-mapped loading | ✅ Works | Instant model startup |
| Metal 4 dispatch | ✅ macOS 26+ | Argument tables, residency sets |
| F16 / F32 weights | ✅ Supported | Slower than quantized |

**Known rough edges:**
- Only Apple Silicon is supported (no Intel Macs, no simulator).
- macOS 26 / iOS 26 are required today because of Metal 4 API usage.
- Very large contexts (>8K) on small-memory devices can OOM; reduce `contextWindowSize` as needed.
- Non-deterministic output can occur across process restarts (GPU driver JIT variance).

---

## Building from Source

```bash
git clone https://github.com/christopherkarani/EdgeRunner.git
cd EdgeRunner
swift build

# Run the full test suite
swift test

# Run the canonical publishable benchmark (release build)
swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```

---

## Documentation

- **[Architecture Deep Dive](docs/arch/core_architecture.md)** — Metal kernels, memory layout, fused pipelines
- **[Public API Reference](docs/arch/public_api.md)** — Every type and method
- **[Inference Pipeline](docs/arch/inference_pipeline.md)** — Prefill vs decode vs prefix-reuse
- **[Metal Shaders](docs/arch/metal_shaders.md)** — Kernel design and dispatch strategy
- **[Troubleshooting](TROUBLESHOOTING.md)** — Common errors and fixes
- **[Roadmap](docs/ROADMAP.md)** — Where we’re headed next

---

## License

MIT License. See [LICENSE](LICENSE) for details.
