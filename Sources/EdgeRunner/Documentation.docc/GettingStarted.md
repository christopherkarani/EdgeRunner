# Getting Started with EdgeRunner

Load a model, configure sampling, and generate text in a few lines of Swift.

## Overview

EdgeRunner is designed for simplicity. Load a GGUF model, create a generation session, and start streaming tokens.

### Load a Model

```swift
import EdgeRunner

let model = try await MyModel.load(
    from: URL(fileURLWithPath: "path/to/model.gguf"),
    configuration: ModelConfiguration(
        maxTokens: 1024,
        contextWindowSize: 4096
    )
)
```

### Stream Tokens

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .nucleus(temperature: 0.7, topP: 0.9),
    maxTokens: 512
)

for try await token in session.stream(prompt: "Explain quantum computing:") {
    print(token, terminator: "")
}
```

### Generate a Complete Response

```swift
let response = try await session.generate(prompt: "What is Swift?")
print(response)
```

### Backend Swapping

```swift
let model: any EdgeRunnerLanguageModel
if FoundationModelsAvailability.isAvailable {
    model = try await SystemModel.load(from: url, configuration: config)
} else {
    model = try await LocalGGUFModel.load(from: url, configuration: config)
}
```
