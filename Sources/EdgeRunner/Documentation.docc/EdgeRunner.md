# ``EdgeRunner``

A Metal-native Swift 6.2 inference engine for running large language models on Apple Silicon.

## Overview

EdgeRunner provides a high-performance, on-device inference engine for LLMs. It runs entirely on Apple Silicon GPUs using Metal 4 compute shaders, with no external dependencies.

Key capabilities:
- **Streaming generation** via `AsyncThrowingStream`
- **Structured output** with constrained decoding from `Decodable` types
- **Tool calling** with automatic schema generation
- **Composable sampling** with temperature, top-k, top-p, min-p, and repetition penalty
- **Backend swapping** between local Metal inference and Apple Foundation Models
- **Speculative decoding** for faster generation with draft models

## Topics

### Essentials
- <doc:GettingStarted>
- ``EdgeRunnerLanguageModel``
- ``ModelConfiguration``
- ``GenerationSession``

### Model Loading
- <doc:ModelLoading>
- ``BackendRegistry``
- ``LocalModelBackend``

### Generation
- <doc:StreamingGeneration>
- ``GenerationError``

### Structured Output
- <doc:StructuredOutput>

### Tool Calling
- <doc:ToolCalling>
- ``EdgeRunnerTool``
- ``ToolExecutor``
- ``ToolChoice``

### Sampling
- <doc:Sampling>

### Backend Swapping
- ``FoundationModelsAvailability``
- ``SystemModelBackend``
