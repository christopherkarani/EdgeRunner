# Streaming Generation

Generate text token-by-token using Swift's structured concurrency.

## Overview

EdgeRunner uses `AsyncThrowingStream` for streaming token output with built-in backpressure handling and cancellation support.

### Basic Streaming

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .greedy,
    maxTokens: 256
)

for try await token in session.stream(prompt: "Once upon a time") {
    print(token, terminator: "")
}
```

### Token Callbacks

```swift
let session = GenerationSession(
    model: model,
    samplingPipeline: .nucleus(temperature: 0.8, topP: 0.9),
    maxTokens: 512,
    onToken: { tokenID, text in
        print("Token \(tokenID): '\(text)'")
    }
)
```

### Cancellation

```swift
let task = Task {
    for try await token in session.stream(prompt: "...") {
        updateUI(with: token)
    }
}
task.cancel()
```
