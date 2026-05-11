# EdgeRunner

Fast, local LLM inference for Apple Silicon. Built in Swift/Metal from the ground up

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

## Features

- **~230+ tok/s** median decode on Qwen3-0.6B (Apple M3 Max, publishable benchmark)
- **Metal-native**: Custom compute kernels optimized for Apple Silicon
- **GGUF support**: Load quantized models directly (Q8_0, Q4_0, Q4_K_M, Q2_K, Q3_K, Q5_0, Q5_1, Q5_K, Q6_K)
- **Memory-mapped loading**: Instant startup, minimal memory pressure
- **Private**: Runs entirely on-device, no network required
- **iOS + macOS**: Drop into any Swift project as a package dependency

## Requirements

- macOS 26.0+ or iOS 26.0+
- Apple Silicon (M1 or later)
- Swift 6.2+
- Xcode 26 beta or newer

## Installation

### Swift Package Manager

Add EdgeRunner to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/EdgeRunner.git", from: "0.1.0")
]
```

Or add it in Xcode: **File → Add Package Dependencies...**

## Quick Start

### 1. Download a Model

EdgeRunner uses GGUF format models. Download a compatible model:

```bash
# Qwen3 0.6B (recommended for testing)
curl -L -o Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

Supported models:
- Qwen3 0.6B, 1.7B, 4B (Q8_0)
- More models coming soon (Llama 3, Mistral, Phi-3)

### 2. Load and Run

```swift
import EdgeRunner

@main
struct MyApp {
    static func main() async throws {
        let runner = try await EdgeRunner(
            modelPath: "Qwen3-0.6B-Q8_0.gguf"
        )

        // Streaming output
        for try await token in runner.stream("Hello, world!", maxTokens: 50) {
            print(token, terminator: "")
        }
    }
}
```

## Architecture

EdgeRunner is organized into layered modules:

```
EdgeRunnerMetal    # Low-level Metal compute kernels
EdgeRunnerIO       # Model loading (GGUF, SafeTensors)
EdgeRunnerCore     # Sampling, tokenization, graph execution
EdgeRunner         # Public API facade
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `EdgeRunner` | Simple entry point — load, stream, generate |
| `LlamaLanguageModel` | Full inference engine (advanced use) |
| `KVCache` | Efficient autoregressive generation |
| `GGUFLoader` | Memory-mapped model loading |
| `SamplingPipeline` | Temperature, top-p, top-k, repetition penalty |

## Performance

Benchmarks on Apple M3 Max (128-token decode):

| Model | EdgeRunner | llama.cpp | MLX |
|-------|-----------|-----------|-----|
| Qwen3-0.6B-Q8_0 | **~230+ tok/s** | ~198 tok/s | ~286 tok/s |
| Time to First Token | **3.5 ms** | n/a in `llama-bench` | n/a in `mlx_lm.benchmark` |

Memory usage:
- Qwen3-0.6B: ~700MB (with memory mapping)
- Qwen3-4B: ~4.5GB

## Advanced Usage

### Custom Sampling

```swift
let text = try await runner.generate(
    "Write a story about",
    maxTokens: 200,
    sampling: SamplingConfiguration(
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repetitionPenalty: 1.1
    )
)
```

### Low-Level Control

For fine-grained control over the inference loop, use `LlamaLanguageModel` directly:

```swift
let model = try await LlamaLanguageModel.load(
    from: modelURL,
    configuration: ModelConfiguration(contextWindowSize: 2048)
)

var tokens = [model.bosTokenID ?? 1]
for _ in 0..<100 {
    let next = try await model.nextToken(
        for: tokens,
        sampling: SamplingConfiguration(temperature: 0)
    )
    guard next != model.eosTokenID else { break }
    tokens.append(next)
}
print(model.detokenize(tokens))
```

### Tool Calling

Tool calling is available through concrete `EdgeRunnerTool` conformers and streaming/generation sessions.
See `Sources/EdgeRunner/ToolCalling/EdgeRunnerTool.swift` for the protocol surface and the streaming APIs for orchestration.

## Building from Source

```bash
git clone https://github.com/christopherkarani/EdgeRunner.git
cd EdgeRunner
swift build

# Run tests
swift test

# Run benchmark (publishable, release build)
swift test -c release --filter "PublishableBenchmark/fullBenchmark"
```

## Project Status

EdgeRunner is in **beta**. Core features work well:

- ✅ Fast inference (~230+ tok/s publishable decode on Qwen3-0.6B)
- ✅ GGUF Q8_0 quantization
- ✅ KV cache for efficient generation
- ⚠️  Tokenizer (BPE) - basic implementation
- 🚧 Multi-model support (Qwen only for now)

See [ROADMAP.md](docs/ROADMAP.md) for planned features.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## License

MIT License. See [LICENSE](LICENSE) for details.

