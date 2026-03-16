# Model Loading

Load models from GGUF, SafeTensor, or NPZ formats.

## Overview

EdgeRunner supports multiple model formats through its backend system. The ``BackendRegistry`` maps file formats to the appropriate loader.

### Direct Loading

```swift
let model = try await LlamaModel.load(
    from: URL(fileURLWithPath: "llama-3-8b-q4_0.gguf"),
    configuration: ModelConfiguration(useMemoryMapping: true)
)
```

### Registry-Based Loading

```swift
let registry = BackendRegistry()
registry.register(LlamaModel.self, for: "gguf")

let model = try await registry.load(from: modelURL, format: "gguf")
```

### Memory Configuration

For devices with limited memory:

```swift
let config = ModelConfiguration(
    maxTokens: 512,
    contextWindowSize: 2048,
    useMemoryMapping: true
)
```
