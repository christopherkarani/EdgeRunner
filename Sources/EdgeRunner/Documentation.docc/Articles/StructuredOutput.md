# Structured Output

Generate typed Swift objects from model output using constrained decoding.

## Overview

EdgeRunner can generate JSON that conforms to a specific `Decodable` type using grammar-guided sampling.

### Define Your Type

```swift
struct MovieReview: Codable {
    let title: String
    let rating: Int
    let summary: String
}
```

### Generate Structured Output

```swift
let schema = try JSONSchemaExtractor.extractSchema(for: MovieReview.self)
let json = try await session.generate(prompt: "Review the movie Inception:")
let review: MovieReview = try StructuredGenerator.parse(json: json)
```

### Schema Extraction

The schema extractor supports:
- Primitive types: `String`, `Int`, `Double`, `Bool`
- Nested objects
- Arrays
- Optional fields
