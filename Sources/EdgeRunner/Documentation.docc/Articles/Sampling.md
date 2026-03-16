# Sampling Strategies

Configure how tokens are selected during generation.

## Overview

EdgeRunner uses a composable sampling pipeline that chains logit transforms before token selection.

### Quick Start

```swift
let pipeline = SamplingPipeline.greedy
let pipeline = SamplingPipeline.nucleus(temperature: 0.8, topP: 0.9)
let pipeline = SamplingPipeline.topK(k: 40, temperature: 0.7)
```

### Custom Pipeline

```swift
let pipeline = SamplingPipeline(
    transforms: [
        TemperatureSampler(temperature: 0.7),
        TopKSampler(k: 50),
        TopPSampler(p: 0.9),
        MinPSampler(minP: 0.05),
    ],
    selector: StochasticSampler(randomSource: &myRNG),
    repetitionPenalty: RepetitionPenalty(penalty: 1.2, frequencyPenalty: 0.5)
)
```

### Deterministic Output

```swift
var rng = SeededRandomSource(seed: 42)
let pipeline = SamplingPipeline(
    transforms: [TemperatureSampler(temperature: 0.5)],
    selector: StochasticSampler(randomSource: &rng)
)
```
