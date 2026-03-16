# EdgeRunner Milestone 3: Weight Loading & Quantisation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add loaders for GGUF, SafeTensor, and NPZ weight formats, implement quantisation/dequantisation kernels, build the Llama 3 architecture, and add memory pressure handling.

**Architecture:** Protocol-based weight loading (EdgeRunnerWeightLoader) decoupled from model architectures. Memory-mapped file access via mmap. Metal dequantisation kernels fused with GEMV. Runtime precision selection based on memory pressure.

**Tech Stack:** Swift 6.2, Metal Shading Language 4.0, Swift Testing

**Depends on:** Milestone 2 (docs/plans/2026-03-16-edgerunner-m2-implementation.md)

---

## Task 1: EdgeRunnerWeightLoader Protocol & WeightMap

**Files:**
- Create: `Sources/EdgeRunnerIO/WeightLoader.swift`
- Create: `Sources/EdgeRunnerIO/WeightMap.swift`
- Create: `Sources/EdgeRunnerIO/ModelConfig.swift`
- Create: `Sources/EdgeRunnerIO/WeightLoaderError.swift`
- Update: `Package.swift` — add `EdgeRunnerIO` target and `EdgeRunnerIOTests` test target
- Test: `Tests/EdgeRunnerIOTests/WeightLoaderTests.swift`

**Step 1: Update Package.swift**

Add the new `EdgeRunnerIO` target and test target to `Package.swift`:

```swift
// Add to targets array in Package.swift:

// Weight loading & model I/O
.target(
    name: "EdgeRunnerIO",
    dependencies: ["EdgeRunnerMetal"],
    path: "Sources/EdgeRunnerIO"
),

// Update EdgeRunnerCore to depend on EdgeRunnerIO
// EdgeRunnerCore dependencies: ["EdgeRunnerMetal", "EdgeRunnerIO"]

// Add test target
.testTarget(
    name: "EdgeRunnerIOTests",
    dependencies: ["EdgeRunnerIO", "EdgeRunnerMetal"]
),
```

**Step 2: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/WeightLoaderTests.swift
import Testing
import Metal
@testable import EdgeRunnerIO

@Suite("WeightMap")
struct WeightMapTests {

    @Test func emptyWeightMapHasZeroCount() {
        let map = WeightMap()
        #expect(map.count == 0)
        #expect(map.tensorNames.isEmpty)
    }

    @Test func insertAndRetrieveTensorStorage() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }
        var map = WeightMap()
        let byteCount = 128 * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: byteCount,
                                             options: .storageModeShared) else {
            throw WeightLoaderError.allocationFailed(byteCount: byteCount)
        }
        let storage = TensorStorage(
            buffer: buffer,
            dataType: .float32,
            shape: [4, 32],
            name: "model.layers.0.attention.wq.weight"
        )
        map["model.layers.0.attention.wq.weight"] = storage

        #expect(map.count == 1)
        #expect(map.tensorNames == ["model.layers.0.attention.wq.weight"])

        let retrieved = map["model.layers.0.attention.wq.weight"]
        #expect(retrieved != nil)
        #expect(retrieved?.shape == [4, 32])
        #expect(retrieved?.dataType == .float32)
    }

    @Test func subscriptReturnsNilForMissingKey() {
        let map = WeightMap()
        #expect(map["nonexistent"] == nil)
    }

    @Test func multipleInsertionsTrackAllNames() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }
        var map = WeightMap()
        for i in 0..<5 {
            let byteCount = 64 * MemoryLayout<Float>.stride
            guard let buffer = device.makeBuffer(length: byteCount,
                                                 options: .storageModeShared) else {
                throw WeightLoaderError.allocationFailed(byteCount: byteCount)
            }
            let storage = TensorStorage(
                buffer: buffer,
                dataType: .float16,
                shape: [8, 8],
                name: "layer.\(i).weight"
            )
            map["layer.\(i).weight"] = storage
        }
        #expect(map.count == 5)
    }
}

@Suite("ModelConfig")
struct ModelConfigTests {

    @Test func initWithRequiredFields() {
        let config = ModelConfig(
            architecture: .llama,
            vocabularySize: 32000,
            embeddingDimension: 4096,
            layerCount: 32,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 11008,
            contextLength: 4096,
            ropeTheta: 500000.0,
            normEpsilon: 1e-5
        )
        #expect(config.architecture == .llama)
        #expect(config.vocabularySize == 32000)
        #expect(config.headCount == 32)
        #expect(config.kvHeadCount == 8)
        #expect(config.headDimension == 128) // 4096 / 32
    }

    @Test func headDimensionComputed() {
        let config = ModelConfig(
            architecture: .llama,
            vocabularySize: 128256,
            embeddingDimension: 2048,
            layerCount: 16,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 8192,
            contextLength: 8192,
            ropeTheta: 500000.0,
            normEpsilon: 1e-5
        )
        #expect(config.headDimension == 64) // 2048 / 32
    }
}

@Suite("WeightLoaderProtocol")
struct WeightLoaderProtocolTests {

    @Test func protocolRequiresLoadMethod() {
        // Compile-time check: MockWeightLoader conforms to EdgeRunnerWeightLoader
        let _: any EdgeRunnerWeightLoader = MockWeightLoader()
    }

    @Test func protocolRequiresCanLoadMethod() {
        let loader: any EdgeRunnerWeightLoader = MockWeightLoader()
        #expect(loader.canLoad(url: URL(fileURLWithPath: "/tmp/test.mock")))
        #expect(!loader.canLoad(url: URL(fileURLWithPath: "/tmp/test.bin")))
    }
}

/// Mock loader for protocol conformance tests.
private struct MockWeightLoader: EdgeRunnerWeightLoader {
    func canLoad(url: URL) -> Bool {
        url.pathExtension == "mock"
    }

    func load(from url: URL, on device: MTLDevice) async throws -> (WeightMap, ModelConfig) {
        let config = ModelConfig(
            architecture: .llama,
            vocabularySize: 100,
            embeddingDimension: 64,
            layerCount: 1,
            headCount: 1,
            kvHeadCount: 1,
            intermediateSize: 128,
            contextLength: 512,
            ropeTheta: 10000.0,
            normEpsilon: 1e-5
        )
        return (WeightMap(), config)
    }
}
```

**Step 3: Run tests to verify they fail**

Run: `swift test --filter WeightMapTests 2>&1`
Expected: FAIL — types not defined.

**Step 4: Implement WeightLoaderError**

```swift
// Sources/EdgeRunnerIO/WeightLoaderError.swift
import Foundation

/// Errors that can occur during weight loading.
public enum WeightLoaderError: Error, Sendable {
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

**Step 5: Implement TensorStorage**

```swift
// Sources/EdgeRunnerIO/WeightMap.swift
import Metal
import Foundation

/// The data type of tensor elements stored on disk or in GPU memory.
public enum TensorDataType: UInt32, Sendable {
    case float32  = 0
    case float16  = 1
    case q4_0     = 2
    case q4_1     = 3
    case q5_0     = 6
    case q5_1     = 7
    case q8_0     = 8
    case q8_1     = 9
    case q2_K     = 10
    case q3_K     = 11
    case q4_K     = 12
    case q5_K     = 13
    case q6_K     = 14
    case q8_K     = 15
    case i8       = 16
    case i16      = 17
    case i32      = 18
    case i64      = 19
    case f64      = 20
    case bfloat16 = 30
}

/// A single tensor's GPU-resident storage, including shape and type metadata.
public struct TensorStorage: Sendable {
    /// The Metal buffer holding the raw weight data.
    public let buffer: MTLBuffer

    /// The element data type (may be quantised).
    public let dataType: TensorDataType

    /// The logical shape of the tensor (e.g. [4096, 4096] for a weight matrix).
    public let shape: [Int]

    /// The tensor's name as it appears in the model file.
    public let name: String

    /// Total number of logical elements.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public init(buffer: MTLBuffer, dataType: TensorDataType, shape: [Int], name: String) {
        self.buffer = buffer
        self.dataType = dataType
        self.shape = shape
        self.name = name
    }
}

/// A dictionary mapping tensor names to their GPU-resident storage.
/// This is the primary output of any weight loader.
public struct WeightMap: Sendable {
    private var storage: [String: TensorStorage] = [:]

    public init() {}

    /// Number of tensors in the map.
    public var count: Int { storage.count }

    /// All tensor names, sorted alphabetically.
    public var tensorNames: [String] { storage.keys.sorted() }

    /// Access a tensor by name.
    public subscript(name: String) -> TensorStorage? {
        get { storage[name] }
        set { storage[name] = newValue }
    }

    /// Total GPU memory used by all tensors in bytes.
    public var totalBytes: Int {
        storage.values.reduce(0) { $0 + $1.buffer.length }
    }
}
```

**Step 6: Implement ModelConfig**

```swift
// Sources/EdgeRunnerIO/ModelConfig.swift
import Foundation

/// Supported model architectures.
public enum ModelArchitecture: String, Sendable, CaseIterable {
    case llama   = "llama"
    case gpt2    = "gpt2"
    case mistral = "mistral"
    case phi     = "phi"
    case gemma   = "gemma"
    case qwen2   = "qwen2"
}

/// Configuration metadata extracted from a model file.
/// Describes the model's architecture so the correct computation graph can be built.
public struct ModelConfig: Sendable {
    public let architecture: ModelArchitecture
    public let vocabularySize: Int
    public let embeddingDimension: Int
    public let layerCount: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let intermediateSize: Int
    public let contextLength: Int
    public let ropeTheta: Float
    public let normEpsilon: Float

    /// Computed dimension per attention head.
    public var headDimension: Int {
        embeddingDimension / headCount
    }

    /// Grouped-query attention ratio.
    public var gqaGroupSize: Int {
        headCount / kvHeadCount
    }

    public init(
        architecture: ModelArchitecture,
        vocabularySize: Int,
        embeddingDimension: Int,
        layerCount: Int,
        headCount: Int,
        kvHeadCount: Int,
        intermediateSize: Int,
        contextLength: Int,
        ropeTheta: Float,
        normEpsilon: Float
    ) {
        self.architecture = architecture
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
        self.layerCount = layerCount
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.intermediateSize = intermediateSize
        self.contextLength = contextLength
        self.ropeTheta = ropeTheta
        self.normEpsilon = normEpsilon
    }
}
```

**Step 7: Implement the WeightLoader protocol**

```swift
// Sources/EdgeRunnerIO/WeightLoader.swift
import Metal
import Foundation

/// Protocol for loading model weights from disk into GPU memory.
///
/// Implementations handle specific file formats (GGUF, SafeTensor, etc.)
/// and produce a `WeightMap` of GPU-resident tensors plus a `ModelConfig`
/// describing the model architecture.
public protocol EdgeRunnerWeightLoader: Sendable {
    /// Returns `true` if this loader can handle the file at the given URL.
    func canLoad(url: URL) -> Bool

    /// Load weights from disk into GPU buffers.
    ///
    /// - Parameters:
    ///   - url: Path to the model file.
    ///   - device: The Metal device to allocate buffers on.
    /// - Returns: A tuple of (WeightMap, ModelConfig).
    /// - Throws: `WeightLoaderError` on failure.
    func load(from url: URL, on device: MTLDevice) async throws -> (WeightMap, ModelConfig)
}
```

**Step 8: Run tests and verify they pass**

Run: `swift test --filter WeightMapTests 2>&1`
Expected: All tests PASS.

**Step 9: Commit**

```
feat(io): add EdgeRunnerWeightLoader protocol, WeightMap, ModelConfig, TensorStorage
```

---

## Task 2: GGUF Header Parser

**Files:**
- Create: `Sources/EdgeRunnerIO/GGUF/GGUFParser.swift`
- Create: `Sources/EdgeRunnerIO/GGUF/GGUFMetadata.swift`
- Create: `Sources/EdgeRunnerIO/GGUF/GGUFTypes.swift`
- Test: `Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/GGUFHeaderTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

/// Helper to build a synthetic GGUF binary blob for testing.
private struct GGUFBuilder {
    var data = Data()

    mutating func writeUInt32(_ v: UInt32) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeUInt64(_ v: UInt64) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeInt32(_ v: Int32) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeInt64(_ v: Int64) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeFloat32(_ v: Float) {
        withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
    }
    mutating func writeFloat64(_ v: Double) {
        withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
    }
    mutating func writeBool(_ v: Bool) {
        data.append(v ? 1 : 0)
    }
    mutating func writeString(_ s: String) {
        let utf8 = Array(s.utf8)
        writeUInt64(UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }

    /// Write a metadata KV entry: string key, typed value.
    mutating func writeKV(key: String, stringValue: String) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.string.rawValue)
        writeString(stringValue)
    }
    mutating func writeKV(key: String, uint32Value: UInt32) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.uint32.rawValue)
        writeUInt32(uint32Value)
    }
    mutating func writeKV(key: String, int32Value: Int32) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.int32.rawValue)
        writeInt32(int32Value)
    }
    mutating func writeKV(key: String, float32Value: Float) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.float32.rawValue)
        writeFloat32(float32Value)
    }
    mutating func writeKV(key: String, boolValue: Bool) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.bool.rawValue)
        writeBool(boolValue)
    }
    mutating func writeKV(key: String, uint64Value: UInt64) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.uint64.rawValue)
        writeUInt64(uint64Value)
    }
    mutating func writeKV(key: String, int64Value: Int64) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.int64.rawValue)
        writeInt64(int64Value)
    }
    mutating func writeKV(key: String, float64Value: Double) {
        writeString(key)
        writeUInt32(GGUFMetadataValueType.float64.rawValue)
        writeFloat64(float64Value)
    }

    /// Build a minimal valid GGUF header with the given metadata KV pairs.
    static func minimalHeader(
        version: UInt32 = 3,
        tensorCount: UInt64 = 0,
        metadataKVCount: UInt64 = 0,
        build: (inout GGUFBuilder) -> Void = { _ in }
    ) -> Data {
        var b = GGUFBuilder()
        // Magic: "GGUF" = 0x46475547 little-endian
        b.writeUInt32(0x46475547)
        b.writeUInt32(version)
        b.writeUInt64(tensorCount)
        b.writeUInt64(metadataKVCount)
        build(&b)
        return b.data
    }
}

@Suite("GGUF Header Parsing")
struct GGUFHeaderTests {

    @Test func parseMagicAndVersion() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 0)
        let header = try GGUFHeader.parse(from: data)
        #expect(header.version == 3)
        #expect(header.tensorCount == 0)
        #expect(header.metadataKVCount == 0)
    }

    @Test func rejectInvalidMagic() {
        var data = Data()
        withUnsafeBytes(of: UInt32(0xDEADBEEF).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: [UInt8](repeating: 0, count: 20))
        #expect(throws: WeightLoaderError.self) {
            try GGUFHeader.parse(from: data)
        }
    }

    @Test func rejectUnsupportedVersion() {
        let data = GGUFBuilder.minimalHeader(version: 1, tensorCount: 0, metadataKVCount: 0)
        #expect(throws: WeightLoaderError.self) {
            try GGUFHeader.parse(from: data)
        }
    }

    @Test func parseVersion2() throws {
        let data = GGUFBuilder.minimalHeader(version: 2, tensorCount: 5, metadataKVCount: 2)
        let header = try GGUFHeader.parse(from: data)
        #expect(header.version == 2)
        #expect(header.tensorCount == 5)
        #expect(header.metadataKVCount == 2)
    }

    @Test func parseStringMetadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 1) { b in
            b.writeKV(key: "general.architecture", stringValue: "llama")
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["general.architecture"]?.stringValue == "llama")
    }

    @Test func parseUInt32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 1) { b in
            b.writeKV(key: "llama.embedding_length", uint32Value: 4096)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["llama.embedding_length"]?.uint32Value == 4096)
    }

    @Test func parseInt32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 1) { b in
            b.writeKV(key: "test.signed", int32Value: -42)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["test.signed"]?.int32Value == -42)
    }

    @Test func parseFloat32Metadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 1) { b in
            b.writeKV(key: "llama.rope.freq_base", float32Value: 500000.0)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["llama.rope.freq_base"]?.float32Value == 500000.0)
    }

    @Test func parseBoolMetadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 1) { b in
            b.writeKV(key: "general.use_parallel", boolValue: true)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata["general.use_parallel"]?.boolValue == true)
    }

    @Test func parseMultipleMetadataEntries() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 3) { b in
            b.writeKV(key: "general.architecture", stringValue: "llama")
            b.writeKV(key: "llama.block_count", uint32Value: 32)
            b.writeKV(key: "llama.attention.head_count", uint32Value: 32)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        #expect(metadata.count == 3)
        #expect(metadata["general.architecture"]?.stringValue == "llama")
        #expect(metadata["llama.block_count"]?.uint32Value == 32)
        #expect(metadata["llama.attention.head_count"]?.uint32Value == 32)
    }

    @Test func extractModelConfigFromMetadata() throws {
        let data = GGUFBuilder.minimalHeader(version: 3, tensorCount: 0, metadataKVCount: 9) { b in
            b.writeKV(key: "general.architecture", stringValue: "llama")
            b.writeKV(key: "llama.vocab_size", uint32Value: 32000)
            b.writeKV(key: "llama.embedding_length", uint32Value: 4096)
            b.writeKV(key: "llama.block_count", uint32Value: 32)
            b.writeKV(key: "llama.attention.head_count", uint32Value: 32)
            b.writeKV(key: "llama.attention.head_count_kv", uint32Value: 8)
            b.writeKV(key: "llama.feed_forward_length", uint32Value: 11008)
            b.writeKV(key: "llama.context_length", uint32Value: 4096)
            b.writeKV(key: "llama.rope.freq_base", float32Value: 500000.0)
        }
        let reader = GGUFReader(data: data)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let config = try ModelConfig.from(ggufMetadata: metadata)

        #expect(config.architecture == .llama)
        #expect(config.vocabularySize == 32000)
        #expect(config.embeddingDimension == 4096)
        #expect(config.layerCount == 32)
        #expect(config.headCount == 32)
        #expect(config.kvHeadCount == 8)
        #expect(config.intermediateSize == 11008)
        #expect(config.contextLength == 4096)
        #expect(config.ropeTheta == 500000.0)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GGUFHeaderTests 2>&1`
Expected: FAIL — `GGUFHeader`, `GGUFReader`, `GGUFMetadataValueType` not defined.

**Step 3: Implement GGUFTypes**

```swift
// Sources/EdgeRunnerIO/GGUF/GGUFTypes.swift
import Foundation

/// GGUF magic bytes: "GGUF" = 0x46475547 (little-endian).
public let ggufMagic: UInt32 = 0x46475547

/// Supported GGUF versions.
public let ggufSupportedVersions: ClosedRange<UInt32> = 2...3

/// GGUF metadata value types.
public enum GGUFMetadataValueType: UInt32, Sendable {
    case uint8   = 0
    case int8    = 1
    case uint16  = 2
    case int16   = 3
    case uint32  = 4
    case int32   = 5
    case float32 = 6
    case bool    = 7
    case string  = 8
    case array   = 9
    case uint64  = 10
    case int64   = 11
    case float64 = 12
}

/// GGUF tensor data types (matches the on-disk enum).
public enum GGUFTensorType: UInt32, Sendable {
    case f32    = 0
    case f16    = 1
    case q4_0   = 2
    case q4_1   = 3
    case q5_0   = 6
    case q5_1   = 7
    case q8_0   = 8
    case q8_1   = 9
    case q2_K   = 10
    case q3_K   = 11
    case q4_K   = 12
    case q5_K   = 13
    case q6_K   = 14
    case q8_K   = 15
    case iq2_xxs = 16
    case iq2_xs  = 17
    case iq3_xxs = 18
    case iq1_s   = 19
    case iq4_nl  = 20
    case iq3_s   = 21
    case iq2_s   = 22
    case iq4_xs  = 23
}

/// Parsed GGUF file header (fixed-size preamble).
public struct GGUFHeader: Sendable {
    public let version: UInt32
    public let tensorCount: UInt64
    public let metadataKVCount: UInt64

    /// Parse the GGUF header from the first bytes of the file.
    public static func parse(from data: Data) throws -> GGUFHeader {
        guard data.count >= 24 else {
            throw WeightLoaderError.invalidFormat("GGUF header too short: \(data.count) bytes")
        }
        let magic = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        guard magic == ggufMagic else {
            throw WeightLoaderError.invalidFormat(
                "Invalid GGUF magic: 0x\(String(magic, radix: 16)), expected 0x46475547"
            )
        }
        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        guard ggufSupportedVersions.contains(version) else {
            throw WeightLoaderError.unsupportedVersion(version)
        }
        let tensorCount = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) }
        let metadataKVCount = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt64.self) }

        return GGUFHeader(
            version: version,
            tensorCount: tensorCount,
            metadataKVCount: metadataKVCount
        )
    }
}

/// Tensor info entry from the GGUF tensor table.
public struct GGUFTensorInfo: Sendable {
    public let name: String
    public let dimensions: [UInt64]
    public let type: GGUFTensorType
    public let offset: UInt64
}
```

**Step 4: Implement GGUFMetadata**

```swift
// Sources/EdgeRunnerIO/GGUF/GGUFMetadata.swift
import Foundation

/// A parsed GGUF metadata value supporting all GGUF KV types.
public enum GGUFMetadataValue: Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case array([GGUFMetadataValue])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    // Convenience accessors
    public var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }
    public var uint32Value: UInt32? {
        if case .uint32(let v) = self { return v }
        return nil
    }
    public var int32Value: Int32? {
        if case .int32(let v) = self { return v }
        return nil
    }
    public var float32Value: Float? {
        if case .float32(let v) = self { return v }
        return nil
    }
    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }
    public var uint64Value: UInt64? {
        if case .uint64(let v) = self { return v }
        return nil
    }
    public var int64Value: Int64? {
        if case .int64(let v) = self { return v }
        return nil
    }
    public var float64Value: Double? {
        if case .float64(let v) = self { return v }
        return nil
    }

    /// Coerce any integer/float type to Int, for convenience.
    public var intValue: Int? {
        switch self {
        case .uint8(let v):  return Int(v)
        case .int8(let v):   return Int(v)
        case .uint16(let v): return Int(v)
        case .int16(let v):  return Int(v)
        case .uint32(let v): return Int(v)
        case .int32(let v):  return Int(v)
        case .uint64(let v): return Int(v)
        case .int64(let v):  return Int(v)
        default: return nil
        }
    }

    /// Coerce any numeric type to Float.
    public var floatValue: Float? {
        switch self {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        case .uint32(let v):  return Float(v)
        case .int32(let v):   return Float(v)
        default: return nil
        }
    }
}

/// Sequential binary reader for GGUF files.
public final class GGUFReader: Sendable {
    private let data: Data
    // Using a class with nonisolated(unsafe) for the mutable offset within a Sendable type.
    // This is safe because GGUFReader is used single-threaded during parsing.
    nonisolated(unsafe) private var offset: Int = 0

    public init(data: Data) {
        self.data = data
    }

    /// Read the fixed-size header.
    public func readHeader() throws -> GGUFHeader {
        let header = try GGUFHeader.parse(from: data)
        offset = 24 // Past magic(4) + version(4) + tensorCount(8) + metadataKVCount(8)
        return header
    }

    /// Read `count` metadata KV entries from current position.
    public func readMetadata(count: Int) throws -> [String: GGUFMetadataValue] {
        var result: [String: GGUFMetadataValue] = [:]
        result.reserveCapacity(count)
        for _ in 0..<count {
            let key = try readString()
            let value = try readTypedValue()
            result[key] = value
        }
        return result
    }

    /// Read tensor info entries from current position.
    public func readTensorInfos(count: Int) throws -> [GGUFTensorInfo] {
        var infos: [GGUFTensorInfo] = []
        infos.reserveCapacity(count)
        for _ in 0..<count {
            let name = try readString()
            let nDim = try readUInt32()
            var dims: [UInt64] = []
            for _ in 0..<nDim {
                dims.append(try readUInt64())
            }
            let typeRaw = try readUInt32()
            guard let type = GGUFTensorType(rawValue: typeRaw) else {
                throw WeightLoaderError.unsupportedDataType(typeRaw)
            }
            let tensorOffset = try readUInt64()
            infos.append(GGUFTensorInfo(name: name, dimensions: dims, type: type, offset: tensorOffset))
        }
        return infos
    }

    /// Current read position (useful for computing data section start).
    public var currentOffset: Int { offset }

    // MARK: - Primitive Reads

    private func readUInt8() throws -> UInt8 {
        guard offset + 1 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data[data.startIndex + offset]
        offset += 1
        return v
    }

    private func readInt8() throws -> Int8 {
        Int8(bitPattern: try readUInt8())
    }

    private func readUInt16() throws -> UInt16 {
        guard offset + 2 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt16.self) }
        offset += 2
        return v
    }

    private func readInt16() throws -> Int16 {
        guard offset + 2 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int16.self) }
        offset += 2
        return v
    }

    private func readUInt32() throws -> UInt32 {
        guard offset + 4 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
        offset += 4
        return v
    }

    private func readInt32() throws -> Int32 {
        guard offset + 4 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int32.self) }
        offset += 4
        return v
    }

    private func readUInt64() throws -> UInt64 {
        guard offset + 8 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        return v
    }

    private func readInt64() throws -> Int64 {
        guard offset + 8 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int64.self) }
        offset += 8
        return v
    }

    private func readFloat32() throws -> Float {
        guard offset + 4 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
        offset += 4
        return v
    }

    private func readFloat64() throws -> Double {
        guard offset + 8 <= data.count else {
            throw WeightLoaderError.invalidFormat("Unexpected end of data at offset \(offset)")
        }
        let v = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Double.self) }
        offset += 8
        return v
    }

    private func readBool() throws -> Bool {
        let v = try readUInt8()
        return v != 0
    }

    private func readString() throws -> String {
        let length = try readUInt64()
        guard offset + Int(length) <= data.count else {
            throw WeightLoaderError.invalidFormat("String length \(length) exceeds data at offset \(offset)")
        }
        let range = offset..<(offset + Int(length))
        offset += Int(length)
        guard let str = String(data: data[range], encoding: .utf8) else {
            throw WeightLoaderError.invalidFormat("Invalid UTF-8 string at offset \(range.lowerBound)")
        }
        return str
    }

    private func readTypedValue() throws -> GGUFMetadataValue {
        let typeRaw = try readUInt32()
        guard let type = GGUFMetadataValueType(rawValue: typeRaw) else {
            throw WeightLoaderError.invalidFormat("Unknown metadata type: \(typeRaw)")
        }
        return try readValue(ofType: type)
    }

    private func readValue(ofType type: GGUFMetadataValueType) throws -> GGUFMetadataValue {
        switch type {
        case .uint8:   return .uint8(try readUInt8())
        case .int8:    return .int8(try readInt8())
        case .uint16:  return .uint16(try readUInt16())
        case .int16:   return .int16(try readInt16())
        case .uint32:  return .uint32(try readUInt32())
        case .int32:   return .int32(try readInt32())
        case .float32: return .float32(try readFloat32())
        case .bool:    return .bool(try readBool())
        case .string:  return .string(try readString())
        case .uint64:  return .uint64(try readUInt64())
        case .int64:   return .int64(try readInt64())
        case .float64: return .float64(try readFloat64())
        case .array:
            let elemTypeRaw = try readUInt32()
            guard let elemType = GGUFMetadataValueType(rawValue: elemTypeRaw) else {
                throw WeightLoaderError.invalidFormat("Unknown array element type: \(elemTypeRaw)")
            }
            let count = try readUInt64()
            var elements: [GGUFMetadataValue] = []
            elements.reserveCapacity(Int(count))
            for _ in 0..<count {
                elements.append(try readValue(ofType: elemType))
            }
            return .array(elements)
        }
    }
}

// MARK: - ModelConfig extraction from GGUF metadata

extension ModelConfig {
    /// Build a ModelConfig from parsed GGUF metadata.
    public static func from(ggufMetadata metadata: [String: GGUFMetadataValue]) throws -> ModelConfig {
        guard let archStr = metadata["general.architecture"]?.stringValue else {
            throw WeightLoaderError.missingMetadata("general.architecture")
        }
        guard let architecture = ModelArchitecture(rawValue: archStr) else {
            throw WeightLoaderError.invalidFormat("Unknown architecture: \(archStr)")
        }
        let prefix = archStr // e.g. "llama"

        func requiredInt(_ key: String) throws -> Int {
            guard let val = metadata[key]?.intValue ?? metadata[key]?.uint32Value.map({ Int($0) }) else {
                throw WeightLoaderError.missingMetadata(key)
            }
            return val
        }

        func optionalFloat(_ key: String, default defaultVal: Float) -> Float {
            metadata[key]?.floatValue ?? defaultVal
        }

        let vocabSize = try requiredInt("\(prefix).vocab_size")
        let embDim = try requiredInt("\(prefix).embedding_length")
        let layerCount = try requiredInt("\(prefix).block_count")
        let headCount = try requiredInt("\(prefix).attention.head_count")
        let kvHeadCount = try requiredInt("\(prefix).attention.head_count_kv")
        let ffnSize = try requiredInt("\(prefix).feed_forward_length")
        let ctxLen = try requiredInt("\(prefix).context_length")
        let ropeTheta = optionalFloat("\(prefix).rope.freq_base", default: 10000.0)
        let normEps = optionalFloat("\(prefix).attention.layer_norm_rms_epsilon", default: 1e-5)

        return ModelConfig(
            architecture: architecture,
            vocabularySize: vocabSize,
            embeddingDimension: embDim,
            layerCount: layerCount,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            intermediateSize: ffnSize,
            contextLength: ctxLen,
            ropeTheta: ropeTheta,
            normEpsilon: normEps
        )
    }
}
```

**Step 5: Run tests and verify they pass**

Run: `swift test --filter GGUFHeaderTests 2>&1`
Expected: All tests PASS.

**Step 6: Commit**

```
feat(io): add GGUF header parser with metadata KV support and ModelConfig extraction
```

---

## Task 3: GGUF Tensor Table & Memory Mapping

**Files:**
- Create: `Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift`
- Create: `Sources/EdgeRunnerIO/GGUF/MemoryMappedFile.swift`
- Test: `Tests/EdgeRunnerIOTests/GGUFTensorTableTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/GGUFTensorTableTests.swift
import Testing
import Foundation
import Metal
@testable import EdgeRunnerIO

@Suite("GGUF Tensor Table Parsing")
struct GGUFTensorTableTests {

    @Test func parseSingleTensorInfo() throws {
        var b = GGUFBuilder()
        // Magic + version + tensorCount=1 + metadataKVCount=0
        b.writeUInt32(0x46475547)
        b.writeUInt32(3)
        b.writeUInt64(1)
        b.writeUInt64(0)
        // Tensor info: name, ndim, dims[], type, offset
        b.writeString("output.weight")
        b.writeUInt32(2)        // ndim
        b.writeUInt64(4096)     // dim[0]
        b.writeUInt64(32000)    // dim[1]
        b.writeUInt32(1)        // type = f16
        b.writeUInt64(0)        // offset into data section

        let reader = GGUFReader(data: b.data)
        let header = try reader.readHeader()
        let _ = try reader.readMetadata(count: Int(header.metadataKVCount))
        let infos = try reader.readTensorInfos(count: Int(header.tensorCount))

        #expect(infos.count == 1)
        #expect(infos[0].name == "output.weight")
        #expect(infos[0].dimensions == [4096, 32000])
        #expect(infos[0].type == .f16)
        #expect(infos[0].offset == 0)
    }

    @Test func parseMultipleTensorInfos() throws {
        var b = GGUFBuilder()
        b.writeUInt32(0x46475547)
        b.writeUInt32(3)
        b.writeUInt64(3)
        b.writeUInt64(0)

        let tensors: [(String, [UInt64], UInt32, UInt64)] = [
            ("token_embd.weight", [32000, 4096], 1, 0),
            ("blk.0.attn_q.weight", [4096, 4096], 8, 262144000),
            ("blk.0.ffn_gate.weight", [4096, 11008], 2, 295698432),
        ]

        for (name, dims, type, off) in tensors {
            b.writeString(name)
            b.writeUInt32(UInt32(dims.count))
            for d in dims { b.writeUInt64(d) }
            b.writeUInt32(type)
            b.writeUInt64(off)
        }

        let reader = GGUFReader(data: b.data)
        let header = try reader.readHeader()
        let _ = try reader.readMetadata(count: Int(header.metadataKVCount))
        let infos = try reader.readTensorInfos(count: Int(header.tensorCount))

        #expect(infos.count == 3)
        #expect(infos[0].name == "token_embd.weight")
        #expect(infos[0].type == .f16)
        #expect(infos[1].name == "blk.0.attn_q.weight")
        #expect(infos[1].type == .q8_0)
        #expect(infos[2].name == "blk.0.ffn_gate.weight")
        #expect(infos[2].type == .q4_0)
    }

    @Test func rejectUnknownTensorType() {
        var b = GGUFBuilder()
        b.writeUInt32(0x46475547)
        b.writeUInt32(3)
        b.writeUInt64(1)
        b.writeUInt64(0)
        b.writeString("bad.weight")
        b.writeUInt32(1)
        b.writeUInt64(100)
        b.writeUInt32(0xFF) // Invalid type
        b.writeUInt64(0)

        let reader = GGUFReader(data: b.data)
        #expect(throws: WeightLoaderError.self) {
            let header = try reader.readHeader()
            let _ = try reader.readMetadata(count: Int(header.metadataKVCount))
            let _ = try reader.readTensorInfos(count: Int(header.tensorCount))
        }
    }
}

@Suite("Memory-Mapped File")
struct MemoryMappedFileTests {

    @Test func mmapReadOnlyFile() throws {
        // Create a temporary file with known content
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_mmap_\(UUID().uuidString).bin")
        let testData = Data(repeating: 0xAB, count: 4096)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        #expect(mapped.size == 4096)
        #expect(mapped.data[0] == 0xAB)
        #expect(mapped.data[4095] == 0xAB)
    }

    @Test func mmapAlignment() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_align_\(UUID().uuidString).bin")
        // 32-byte aligned size
        let testData = Data(repeating: 0xCD, count: 32 * 100)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        // Verify the base pointer is page-aligned (mmap guarantees this)
        let ptr = mapped.basePointer
        #expect(Int(bitPattern: ptr) % 4096 == 0)
    }

    @Test func mmapNonexistentFileThrows() {
        let url = URL(fileURLWithPath: "/tmp/nonexistent_\(UUID().uuidString).bin")
        #expect(throws: WeightLoaderError.self) {
            _ = try MemoryMappedFile(url: url)
        }
    }

    @Test func mmapSliceAtOffset() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_slice_\(UUID().uuidString).bin")
        var testData = Data(repeating: 0x00, count: 256)
        // Write known pattern at offset 128
        for i in 128..<192 {
            testData[i] = UInt8(i - 128)
        }
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        let slice = mapped.slice(offset: 128, length: 64)
        #expect(slice.count == 64)
        #expect(slice[0] == 0)
        #expect(slice[1] == 1)
        #expect(slice[63] == 63)
    }
}

@Suite("GGUF Loader Integration")
struct GGUFLoaderTests {

    @Test func loaderConformsToProtocol() {
        let loader: any EdgeRunnerWeightLoader = GGUFLoader()
        #expect(loader.canLoad(url: URL(fileURLWithPath: "/tmp/model.gguf")))
        #expect(!loader.canLoad(url: URL(fileURLWithPath: "/tmp/model.safetensors")))
    }

    @Test func createMTLBufferFromMmapNoCopy() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_nocopy_\(UUID().uuidString).bin")
        // Must be page-aligned for bytesNoCopy
        let pageSize = Int(getpagesize())
        let bufferSize = pageSize * 4
        let testData = Data(repeating: 0x42, count: bufferSize)
        try testData.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mapped = try MemoryMappedFile(url: url)
        let buffer = try mapped.makeMetalBuffer(
            device: device,
            offset: 0,
            length: bufferSize
        )
        #expect(buffer.length == bufferSize)

        // Verify contents
        let ptr = buffer.contents().bindMemory(to: UInt8.self, capacity: bufferSize)
        #expect(ptr[0] == 0x42)
        #expect(ptr[bufferSize - 1] == 0x42)
    }
}

/// Reuse the builder from header tests.
private struct GGUFBuilder {
    var data = Data()
    mutating func writeUInt32(_ v: UInt32) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeUInt64(_ v: UInt64) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func writeString(_ s: String) {
        let utf8 = Array(s.utf8)
        writeUInt64(UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter GGUFTensorTableTests 2>&1`
Expected: FAIL — `MemoryMappedFile`, `GGUFLoader` not defined.

**Step 3: Implement MemoryMappedFile**

```swift
// Sources/EdgeRunnerIO/GGUF/MemoryMappedFile.swift
import Foundation
import Metal

/// A read-only memory-mapped file for zero-copy weight loading.
///
/// Uses POSIX `mmap` to map the file into the process address space.
/// Metal buffers can be created with `MTLDevice.makeBuffer(bytesNoCopy:)`
/// pointing directly into the mapped region, avoiding any data copies.
public final class MemoryMappedFile: @unchecked Sendable {
    // @unchecked because the mapped memory is immutable after init.

    /// The mapped memory region as `UnsafeRawPointer`.
    public let basePointer: UnsafeRawPointer

    /// Total size of the mapped file in bytes.
    public let size: Int

    /// Access to the mapped bytes.
    public var data: UnsafeBufferPointer<UInt8> {
        UnsafeBufferPointer(
            start: basePointer.assumingMemoryBound(to: UInt8.self),
            count: size
        )
    }

    private let rawPointer: UnsafeMutableRawPointer // For munmap

    /// Memory-map a file at the given URL.
    ///
    /// - Parameter url: Path to the file.
    /// - Throws: `WeightLoaderError.fileNotFound` or `WeightLoaderError.mmapFailed`.
    public init(url: URL) throws {
        let path = url.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw WeightLoaderError.fileNotFound(url)
        }
        defer { close(fd) }

        var stat = stat()
        guard fstat(fd, &stat) == 0 else {
            throw WeightLoaderError.mmapFailed(errno: errno)
        }
        let fileSize = Int(stat.st_size)
        guard fileSize > 0 else {
            throw WeightLoaderError.invalidFormat("Empty file: \(url.lastPathComponent)")
        }

        let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mapped != MAP_FAILED else {
            throw WeightLoaderError.mmapFailed(errno: errno)
        }

        self.rawPointer = mapped!
        self.basePointer = UnsafeRawPointer(mapped!)
        self.size = fileSize
    }

    deinit {
        munmap(rawPointer, size)
    }

    /// Return a slice of the mapped data as `UnsafeBufferPointer<UInt8>`.
    public func slice(offset: Int, length: Int) -> UnsafeBufferPointer<UInt8> {
        precondition(offset >= 0 && offset + length <= size,
                     "Slice out of bounds: offset=\(offset), length=\(length), fileSize=\(size)")
        let ptr = basePointer.advanced(by: offset).assumingMemoryBound(to: UInt8.self)
        return UnsafeBufferPointer(start: ptr, count: length)
    }

    /// Create a Metal buffer backed by the mmap'd region (zero-copy).
    ///
    /// The offset must be page-aligned. If the region is not page-aligned,
    /// this falls back to a copy.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - offset: Byte offset into the mapped file.
    ///   - length: Number of bytes.
    /// - Returns: An `MTLBuffer` backed by the mapped memory.
    public func makeMetalBuffer(
        device: MTLDevice,
        offset: Int,
        length: Int
    ) throws -> MTLBuffer {
        let pageSize = Int(getpagesize())
        let ptr = rawPointer.advanced(by: offset)

        // Try zero-copy first (requires page-aligned pointer)
        if Int(bitPattern: ptr) % pageSize == 0 && length % pageSize == 0 {
            if let buffer = device.makeBuffer(
                bytesNoCopy: ptr,
                length: length,
                options: [.storageModeShared],
                deallocator: nil // mmap owns the memory
            ) {
                return buffer
            }
        }

        // Fallback: copy the data into a new buffer
        guard let buffer = device.makeBuffer(
            bytes: basePointer.advanced(by: offset),
            length: length,
            options: [.storageModeShared]
        ) else {
            throw WeightLoaderError.allocationFailed(byteCount: length)
        }
        return buffer
    }
}
```

**Step 4: Implement GGUFLoader**

```swift
// Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift
import Foundation
import Metal

/// Loads model weights from GGUF files.
///
/// GGUF (GGML Unified Format) stores quantised model weights with metadata.
/// This loader memory-maps the file and creates MTLBuffers pointing directly
/// into the mapped region where possible.
public struct GGUFLoader: EdgeRunnerWeightLoader, Sendable {

    public init() {}

    public func canLoad(url: URL) -> Bool {
        url.pathExtension.lowercased() == "gguf"
    }

    public func load(from url: URL, on device: MTLDevice) async throws -> (WeightMap, ModelConfig) {
        let mapped = try MemoryMappedFile(url: url)

        // Read the full file data for header/metadata parsing
        // (the reader operates on the header portion; actual weights are mmap'd)
        let headerData = Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: mapped.basePointer),
            count: mapped.size,
            deallocator: .none
        )

        let reader = GGUFReader(data: headerData)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let tensorInfos = try reader.readTensorInfos(count: Int(header.tensorCount))
        let config = try ModelConfig.from(ggufMetadata: metadata)

        // Data section starts after the header, aligned to 32 bytes.
        let headerEnd = reader.currentOffset
        let alignment = 32
        let dataOffset = (headerEnd + alignment - 1) / alignment * alignment

        // Build the WeightMap
        var weightMap = WeightMap()
        for info in tensorInfos {
            let tensorOffset = dataOffset + Int(info.offset)
            let byteCount = Self.tensorByteCount(info: info)
            let buffer = try mapped.makeMetalBuffer(
                device: device,
                offset: tensorOffset,
                length: byteCount
            )
            let shape = info.dimensions.map { Int($0) }
            let dataType = Self.convertDataType(info.type)
            let storage = TensorStorage(
                buffer: buffer,
                dataType: dataType,
                shape: shape,
                name: info.name
            )
            weightMap[info.name] = storage
        }

        return (weightMap, config)
    }

    // MARK: - Helpers

    /// Compute the byte count for a tensor given its info.
    static func tensorByteCount(info: GGUFTensorInfo) -> Int {
        let elementCount = info.dimensions.reduce(1) { $0 * Int($1) }
        switch info.type {
        case .f32:     return elementCount * 4
        case .f16:     return elementCount * 2
        case .q4_0:    return elementCount / 32 * 18  // 32 weights -> 18 bytes
        case .q4_1:    return elementCount / 32 * 20
        case .q5_0:    return elementCount / 32 * 22
        case .q5_1:    return elementCount / 32 * 24
        case .q8_0:    return elementCount / 32 * 34  // 32 weights -> 34 bytes
        case .q8_1:    return elementCount / 32 * 36
        case .q2_K:    return elementCount / 256 * 84
        case .q3_K:    return elementCount / 256 * 110
        case .q4_K:    return elementCount / 256 * 144  // Q4_K_M
        case .q5_K:    return elementCount / 256 * 176
        case .q6_K:    return elementCount / 256 * 210
        case .q8_K:    return elementCount / 256 * 292
        case .iq2_xxs: return elementCount / 256 * 66
        case .iq2_xs:  return elementCount / 256 * 74
        case .iq3_xxs: return elementCount / 256 * 98
        case .iq1_s:   return elementCount / 256 * 50
        case .iq4_nl:  return elementCount / 32 * 18
        case .iq3_s:   return elementCount / 256 * 110
        case .iq2_s:   return elementCount / 256 * 82
        case .iq4_xs:  return elementCount / 256 * 136
        }
    }

    /// Convert GGUF tensor type to our TensorDataType enum.
    static func convertDataType(_ type: GGUFTensorType) -> TensorDataType {
        switch type {
        case .f32:   return .float32
        case .f16:   return .float16
        case .q4_0:  return .q4_0
        case .q4_1:  return .q4_1
        case .q5_0:  return .q5_0
        case .q5_1:  return .q5_1
        case .q8_0:  return .q8_0
        case .q8_1:  return .q8_1
        case .q2_K:  return .q2_K
        case .q3_K:  return .q3_K
        case .q4_K:  return .q4_K
        case .q5_K:  return .q5_K
        case .q6_K:  return .q6_K
        case .q8_K:  return .q8_K
        default:     return .float16 // Fallback for exotic types
        }
    }
}
```

**Step 5: Run tests and verify they pass**

Run: `swift test --filter "GGUFTensorTableTests|MemoryMappedFileTests|GGUFLoaderTests" 2>&1`
Expected: All tests PASS.

**Step 6: Commit**

```
feat(io): add GGUF tensor table parsing, memory-mapped file loader, GGUFLoader
```

---

## Task 4: Q4_0 Dequantisation Kernel

**Files:**
- Create: `Sources/EdgeRunnerSharedTypes/include/DequantParams.h`
- Create: `Sources/EdgeRunnerMetal/Shaders/Dequant_Q4_0.metal`
- Create: `Sources/EdgeRunnerMetal/DequantKernel.swift`
- Test: `Tests/EdgeRunnerMetalTests/DequantQ4_0Tests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/DequantQ4_0Tests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference: pack 32 float values into a Q4_0 block.
/// Q4_0 block: 1 FP16 scale (2 bytes) + 16 bytes packed nibbles = 18 bytes.
/// Each nibble = round(value / scale) + 8, clamped to [0, 15].
private func packQ4_0Block(values: [Float]) -> (blockData: [UInt8], scale: Float) {
    precondition(values.count == 32)
    let absMax = values.map { abs($0) }.max()!
    let scale = absMax / 7.0  // Map [-7*scale, 7*scale]
    var blockData = [UInt8](repeating: 0, count: 18)

    // Write FP16 scale (2 bytes, little-endian)
    let f16Scale = Float16(scale)
    withUnsafeBytes(of: f16Scale) { ptr in
        blockData[0] = ptr[0]
        blockData[1] = ptr[1]
    }

    // Pack nibbles: 2 values per byte, low nibble first
    for i in 0..<16 {
        let v0 = values[i]
        let v1 = values[i + 16]
        let q0 = min(15, max(0, Int(round(v0 / scale)) + 8))
        let q1 = min(15, max(0, Int(round(v1 / scale)) + 8))
        blockData[2 + i] = UInt8(q0) | (UInt8(q1) << 4)
    }
    return (blockData, scale)
}

/// CPU reference: dequantise a Q4_0 block back to floats.
private func dequantQ4_0Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == 18)
    let scale: Float16 = blockData.withUnsafeBufferPointer { buf in
        buf.baseAddress!.withMemoryRebound(to: Float16.self, capacity: 1) { $0.pointee }
    }
    var result = [Float](repeating: 0, count: 32)
    for i in 0..<16 {
        let byte = blockData[2 + i]
        let low = Int(byte & 0x0F)
        let high = Int(byte >> 4)
        result[i] = Float(scale) * Float(low - 8)
        result[i + 16] = Float(scale) * Float(high - 8)
    }
    return result
}

@Suite("Q4_0 Dequantisation Kernel")
struct DequantQ4_0Tests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        self.commandQueue = q
    }

    @Test func singleBlockDequant() async throws {
        let values: [Float] = (0..<32).map { Float($0) - 15.5 }
        let (blockData, _) = packQ4_0Block(values: values)
        let expected = dequantQ4_0Block(blockData: blockData)

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == 32)
        for i in 0..<32 {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "Block 0, elem \(i): GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func multipleBlocksDequant() async throws {
        let blockCount = 16
        var allBlockData = [UInt8]()
        var allExpected = [Float]()

        for b in 0..<blockCount {
            let values = (0..<32).map { i in
                Float(b * 32 + i) * 0.1 - Float(blockCount) * 1.6
            }
            let (blockData, _) = packQ4_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ4_0Block(blockData: blockData))
        }

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            blockCount: blockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == blockCount * 32)
        for i in 0..<result.count {
            #expect(abs(result[i] - allExpected[i]) < 1e-3,
                    "Elem \(i): GPU=\(result[i]) CPU=\(allExpected[i])")
        }
    }

    @Test func zeroScaleBlock() async throws {
        let values = [Float](repeating: 0, count: 32)
        let (blockData, _) = packQ4_0Block(values: values)
        let expected = dequantQ4_0Block(blockData: blockData)

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for i in 0..<32 {
            #expect(abs(result[i] - expected[i]) < 1e-6,
                    "Zero block elem \(i): GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func fusedDequantGEMVSingleRow() async throws {
        // Test the fused dequant+GEMV: y = quantised_W * x
        // W is 1 row of 64 elements = 2 Q4_0 blocks
        let cols = 64
        let blockCount = cols / 32

        var allBlockData = [UInt8]()
        var allDequantised = [Float]()
        for _ in 0..<blockCount {
            let values = (0..<32).map { _ in Float.random(in: -1...1) }
            let (blockData, _) = packQ4_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allDequantised.append(contentsOf: dequantQ4_0Block(blockData: blockData))
        }

        let x = (0..<cols).map { _ in Float.random(in: -1...1) }

        // CPU reference: dot product
        var expected: Float = 0
        for i in 0..<cols {
            expected += allDequantised[i] * x[i]
        }

        let kernel = try DequantQ4_0Kernel(device: device)
        let result = try await kernel.fusedDequantGEMV(
            quantisedRows: allBlockData,
            x: x,
            rows: 1,
            cols: cols,
            commandQueue: commandQueue
        )

        #expect(result.count == 1)
        #expect(abs(result[0] - expected) < 0.05,
                "Fused GEMV: GPU=\(result[0]) CPU=\(expected)")
    }
}

enum MetalTestError: Error {
    case noMetal
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter DequantQ4_0Tests 2>&1`
Expected: FAIL — `DequantQ4_0Kernel` not defined.

**Step 3: Implement DequantParams.h**

```c
// Sources/EdgeRunnerSharedTypes/include/DequantParams.h
#ifndef DEQUANT_PARAMS_H
#define DEQUANT_PARAMS_H

#include <stdint.h>

/// Parameters for Q4_0 / Q8_0 dequantisation kernels.
typedef struct {
    uint32_t blockCount;    // Number of quantisation blocks
    uint32_t outputOffset;  // Offset into output buffer (for batched dispatch)
} ERDequantParams;

/// Parameters for fused dequant + GEMV.
/// y[rows] = Q_W[rows, cols] * x[cols]
typedef struct {
    uint32_t rows;          // Number of output rows
    uint32_t cols;          // Number of columns (must be multiple of 32)
    uint32_t blocksPerRow;  // cols / 32
} ERDequantGEMVParams;

/// Parameters for Q4_K_M dequantisation.
typedef struct {
    uint32_t superBlockCount;  // Number of 256-weight super-blocks
    uint32_t outputOffset;
} ERDequantQ4KParams;

#endif /* DEQUANT_PARAMS_H */
```

**Step 4: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Dequant_Q4_0.metal
#include <metal_stdlib>
using namespace metal;

// Q4_0 block layout: 2 bytes FP16 scale + 16 bytes packed nibbles = 18 bytes per 32 weights.
// Nibble layout: byte[i] low nibble = weight[i], high nibble = weight[i+16].
// Dequant: value = scale * (nibble - 8)

struct ERDequantParams {
    uint blockCount;
    uint outputOffset;
};

struct ERDequantGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint Q4_0_BLOCK_SIZE = 18;  // bytes per block
constant uint Q4_0_WEIGHTS_PER_BLOCK = 32;

/// Dequantise Q4_0 blocks to FP32.
/// One thread per block. Each thread produces 32 float values.
kernel void dequant_q4_0(
    device const uint8_t* input    [[buffer(0)]],
    device float*         output   [[buffer(1)]],
    constant ERDequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;

    // Locate this block
    device const uint8_t* block = input + tid * Q4_0_BLOCK_SIZE;
    uint outBase = params.outputOffset + tid * Q4_0_WEIGHTS_PER_BLOCK;

    // Read FP16 scale
    half scale = *reinterpret_cast<device const half*>(block);

    // Dequantise 32 weights from 16 nibble-packed bytes
    for (uint i = 0; i < 16; i++) {
        uint8_t byte = block[2 + i];
        int low  = int(byte & 0x0F) - 8;
        int high = int(byte >> 4) - 8;
        output[outBase + i]      = float(scale) * float(low);
        output[outBase + i + 16] = float(scale) * float(high);
    }
}

/// Fused dequant Q4_0 + GEMV: y[row] = sum_j( dequant(W[row,j]) * x[j] )
/// Dispatch: one threadgroup per row. Threads within the group cooperate to reduce.
kernel void dequant_q4_0_gemv(
    device const uint8_t* quantisedW  [[buffer(0)]],
    device const float*   x           [[buffer(1)]],
    device float*         y           [[buffer(2)]],
    constant ERDequantGEMVParams& params [[buffer(3)]],
    uint row       [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    if (row >= params.rows) return;

    float sum = 0.0;
    uint rowBlockOffset = row * params.blocksPerRow;

    // Each thread handles a subset of blocks
    for (uint b = lane; b < params.blocksPerRow; b += groupSize) {
        device const uint8_t* block = quantisedW + (rowBlockOffset + b) * Q4_0_BLOCK_SIZE;
        half scale = *reinterpret_cast<device const half*>(block);
        float fscale = float(scale);
        uint colBase = b * Q4_0_WEIGHTS_PER_BLOCK;

        for (uint i = 0; i < 16; i++) {
            uint8_t byte = block[2 + i];
            float low  = fscale * float(int(byte & 0x0F) - 8);
            float high = fscale * float(int(byte >> 4) - 8);
            sum += low  * x[colBase + i];
            sum += high * x[colBase + i + 16];
        }
    }

    // Reduce across simdgroup
    sum = simd_sum(sum);

    // First lane writes the result
    if (lane == 0) {
        y[row] = sum;
    }
}
```

**Step 5: Implement the Swift kernel wrapper**

```swift
// Sources/EdgeRunnerMetal/DequantKernel.swift
import Metal
import Foundation

/// Swift wrapper for the Q4_0 dequantisation Metal kernels.
public struct DequantQ4_0Kernel: Sendable {
    private let dequantPipeline: MTLComputePipelineState
    private let gemvPipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        guard let dequantFn = library.makeFunction(name: "dequant_q4_0"),
              let gemvFn = library.makeFunction(name: "dequant_q4_0_gemv") else {
            fatalError("Failed to find dequant_q4_0 kernel functions in Metal library")
        }
        self.dequantPipeline = try device.makeComputePipelineState(function: dequantFn)
        self.gemvPipeline = try device.makeComputePipelineState(function: gemvFn)
        self.device = device
    }

    /// Dequantise Q4_0 blocks to Float32.
    public func dequantise(
        blockData: [UInt8],
        blockCount: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let inputBuffer = device.makeBuffer(
            bytes: blockData,
            length: blockData.count,
            options: .storageModeShared
        )!

        let outputCount = blockCount * 32
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERDequantParams(blockCount: UInt32(blockCount), outputOffset: 0)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERDequantParams>.stride, index: 2)

        let threadsPerGroup = min(dequantPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outputCount))
    }

    /// Fused dequant Q4_0 + GEMV: y = W_q * x.
    public func fusedDequantGEMV(
        quantisedRows: [UInt8],
        x: [Float],
        rows: Int,
        cols: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let blocksPerRow = cols / 32

        let wBuffer = device.makeBuffer(
            bytes: quantisedRows,
            length: quantisedRows.count,
            options: .storageModeShared
        )!
        let xBuffer = device.makeBuffer(
            bytes: x,
            length: x.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let yBuffer = device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERDequantGEMVParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            blocksPerRow: UInt32(blocksPerRow)
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(gemvPipeline)
        encoder.setBuffer(wBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<ERDequantGEMVParams>.stride, index: 3)

        // One threadgroup per row, 32 threads per group (one simdgroup)
        let threadsPerGroup = 32
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: ptr, count: rows))
    }
}
```

**Step 6: Run tests and verify they pass**

Run: `swift test --filter DequantQ4_0Tests 2>&1`
Expected: All tests PASS.

**Step 7: Commit**

```
feat(metal): add Q4_0 dequantisation kernel with fused dequant+GEMV variant
```

---

## Task 5: Q8_0 Dequantisation Kernel

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_0.metal`
- Extend: `Sources/EdgeRunnerMetal/DequantKernel.swift` (add `DequantQ8_0Kernel`)
- Test: `Tests/EdgeRunnerMetalTests/DequantQ8_0Tests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/DequantQ8_0Tests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

/// CPU reference: pack 32 float values into a Q8_0 block.
/// Q8_0 block: 2 bytes FP16 scale + 32 bytes int8 = 34 bytes.
/// Each int8 = round(value / scale), clamped to [-128, 127].
private func packQ8_0Block(values: [Float]) -> (blockData: [UInt8], scale: Float) {
    precondition(values.count == 32)
    let absMax = values.map { abs($0) }.max()!
    let scale = absMax / 127.0
    var blockData = [UInt8](repeating: 0, count: 34)

    let f16Scale = Float16(scale)
    withUnsafeBytes(of: f16Scale) { ptr in
        blockData[0] = ptr[0]
        blockData[1] = ptr[1]
    }

    for i in 0..<32 {
        let q = scale > 0 ? min(127, max(-128, Int(round(values[i] / scale)))) : 0
        blockData[2 + i] = UInt8(bitPattern: Int8(q))
    }
    return (blockData, scale)
}

/// CPU reference: dequantise a Q8_0 block.
private func dequantQ8_0Block(blockData: [UInt8]) -> [Float] {
    precondition(blockData.count == 34)
    let scale: Float16 = blockData.withUnsafeBufferPointer { buf in
        buf.baseAddress!.withMemoryRebound(to: Float16.self, capacity: 1) { $0.pointee }
    }
    var result = [Float](repeating: 0, count: 32)
    for i in 0..<32 {
        let q = Int8(bitPattern: blockData[2 + i])
        result[i] = Float(scale) * Float(q)
    }
    return result
}

@Suite("Q8_0 Dequantisation Kernel")
struct DequantQ8_0Tests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        self.commandQueue = q
    }

    @Test func singleBlockDequant() async throws {
        let values: [Float] = (0..<32).map { Float($0) * 0.1 - 1.6 }
        let (blockData, _) = packQ8_0Block(values: values)
        let expected = dequantQ8_0Block(blockData: blockData)

        let kernel = try DequantQ8_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == 32)
        for i in 0..<32 {
            #expect(abs(result[i] - expected[i]) < 1e-3,
                    "Elem \(i): GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func multipleBlocksDequant() async throws {
        let blockCount = 32
        var allBlockData = [UInt8]()
        var allExpected = [Float]()

        for b in 0..<blockCount {
            let values = (0..<32).map { i in Float.random(in: -2...2) }
            let (blockData, _) = packQ8_0Block(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ8_0Block(blockData: blockData))
        }

        let kernel = try DequantQ8_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            blockCount: blockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == blockCount * 32)
        for i in 0..<result.count {
            #expect(abs(result[i] - allExpected[i]) < 1e-3,
                    "Elem \(i): GPU=\(result[i]) CPU=\(allExpected[i])")
        }
    }

    @Test func zeroScaleBlock() async throws {
        let values = [Float](repeating: 0, count: 32)
        let (blockData, _) = packQ8_0Block(values: values)
        let expected = dequantQ8_0Block(blockData: blockData)

        let kernel = try DequantQ8_0Kernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            blockCount: 1,
            commandQueue: commandQueue
        )

        for i in 0..<32 {
            #expect(abs(result[i] - expected[i]) < 1e-6)
        }
    }

    @Test func higherPrecisionThanQ4() async throws {
        // Q8_0 should have strictly less quantisation error than Q4_0
        let values: [Float] = (0..<32).map { _ in Float.random(in: -1...1) }
        let (q8Data, _) = packQ8_0Block(values: values)
        let q8Dequant = dequantQ8_0Block(blockData: q8Data)

        var q8Error: Float = 0
        for i in 0..<32 {
            q8Error += abs(values[i] - q8Dequant[i])
        }

        // Q8_0 mean absolute error should be < 0.02 for values in [-1, 1]
        let mae = q8Error / 32.0
        #expect(mae < 0.02, "Q8_0 MAE \(mae) too high")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter DequantQ8_0Tests 2>&1`
Expected: FAIL — `DequantQ8_0Kernel` not defined.

**Step 3: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Dequant_Q8_0.metal
#include <metal_stdlib>
using namespace metal;

// Q8_0 block layout: 2 bytes FP16 scale + 32 bytes int8 = 34 bytes per 32 weights.
// Dequant: value = scale * int8_value

struct ERDequantParams {
    uint blockCount;
    uint outputOffset;
};

constant uint Q8_0_BLOCK_SIZE = 34;
constant uint Q8_0_WEIGHTS_PER_BLOCK = 32;

/// Dequantise Q8_0 blocks to FP32.
/// One thread per block. Each thread produces 32 float values.
kernel void dequant_q8_0(
    device const uint8_t* input    [[buffer(0)]],
    device float*         output   [[buffer(1)]],
    constant ERDequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;

    device const uint8_t* block = input + tid * Q8_0_BLOCK_SIZE;
    uint outBase = params.outputOffset + tid * Q8_0_WEIGHTS_PER_BLOCK;

    // Read FP16 scale
    half scale = *reinterpret_cast<device const half*>(block);
    float fscale = float(scale);

    // Dequantise 32 int8 values
    for (uint i = 0; i < 32; i++) {
        int8_t q = as_type<int8_t>(block[2 + i]);
        output[outBase + i] = fscale * float(q);
    }
}
```

**Step 4: Implement the Swift kernel wrapper**

Add `DequantQ8_0Kernel` to `Sources/EdgeRunnerMetal/DequantKernel.swift`:

```swift
// Append to Sources/EdgeRunnerMetal/DequantKernel.swift

/// Swift wrapper for the Q8_0 dequantisation Metal kernel.
public struct DequantQ8_0Kernel: Sendable {
    private let dequantPipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        guard let fn = library.makeFunction(name: "dequant_q8_0") else {
            fatalError("Failed to find dequant_q8_0 kernel function in Metal library")
        }
        self.dequantPipeline = try device.makeComputePipelineState(function: fn)
        self.device = device
    }

    /// Dequantise Q8_0 blocks to Float32.
    public func dequantise(
        blockData: [UInt8],
        blockCount: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let inputBuffer = device.makeBuffer(
            bytes: blockData,
            length: blockData.count,
            options: .storageModeShared
        )!

        let outputCount = blockCount * 32
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERDequantParams(blockCount: UInt32(blockCount), outputOffset: 0)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERDequantParams>.stride, index: 2)

        let threadsPerGroup = min(dequantPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outputCount))
    }
}
```

**Step 5: Run tests and verify they pass**

Run: `swift test --filter DequantQ8_0Tests 2>&1`
Expected: All tests PASS.

**Step 6: Commit**

```
feat(metal): add Q8_0 dequantisation kernel
```

---

## Task 6: Q4_K_M Dequantisation Kernel

**Files:**
- Create: `Sources/EdgeRunnerMetal/Shaders/Dequant_Q4_K_M.metal`
- Extend: `Sources/EdgeRunnerMetal/DequantKernel.swift` (add `DequantQ4KMKernel`)
- Test: `Tests/EdgeRunnerMetalTests/DequantQ4KMTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerMetalTests/DequantQ4KMTests.swift
import Testing
import Metal
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

// Q4_K_M super-block layout (256 weights, 144 bytes total):
//   - d:    FP16 (2 bytes) — master scale
//   - dmin: FP16 (2 bytes) — master min scale
//   - scales: 12 bytes — 6-bit quantised scales and mins for 8 sub-blocks
//   - qs:   128 bytes — packed 4-bit quantised weights (2 per byte)
// Total: 2 + 2 + 12 + 128 = 144 bytes
//
// Each super-block has 8 sub-blocks of 32 weights each.
// Sub-block j: dequant[i] = d * scale_j * (nibble - 8) - dmin * min_j
// where scale_j and min_j are 6-bit values encoded in the scales array.

private let Q4_K_BLOCK_SIZE = 144
private let Q4_K_WEIGHTS_PER_BLOCK = 256

/// CPU reference: build a Q4_K_M super-block from float values.
private func packQ4KBlock(values: [Float]) -> [UInt8] {
    precondition(values.count == 256)
    var block = [UInt8](repeating: 0, count: Q4_K_BLOCK_SIZE)

    // Find per-sub-block scales and mins
    var subScales = [Float](repeating: 0, count: 8)
    var subMins = [Float](repeating: 0, count: 8)

    for j in 0..<8 {
        let subValues = Array(values[(j*32)..<((j+1)*32)])
        let maxVal = subValues.max()!
        let minVal = subValues.min()!
        // scale maps [min, max] to [0, 15]
        if maxVal - minVal > 0 {
            subScales[j] = (maxVal - minVal) / 15.0
            subMins[j] = -minVal
        } else {
            subScales[j] = 0
            subMins[j] = -minVal
        }
    }

    // Compute master d and dmin (max of sub-block scales and mins)
    let maxScale = subScales.max()!
    let maxMin = subMins.max()!
    let d: Float = maxScale > 0 ? maxScale / 63.0 : 0
    let dmin: Float = maxMin > 0 ? maxMin / 63.0 : 0

    // Write d and dmin as FP16
    let f16d = Float16(d)
    let f16dmin = Float16(dmin)
    withUnsafeBytes(of: f16d) { block[0] = $0[0]; block[1] = $0[1] }
    withUnsafeBytes(of: f16dmin) { block[2] = $0[0]; block[3] = $0[1] }

    // Quantise sub-block scales and mins to 6-bit
    var qscales = [UInt8](repeating: 0, count: 8)
    var qmins = [UInt8](repeating: 0, count: 8)
    for j in 0..<8 {
        qscales[j] = d > 0 ? UInt8(min(63, round(subScales[j] / d))) : 0
        qmins[j] = dmin > 0 ? UInt8(min(63, round(subMins[j] / dmin))) : 0
    }

    // Encode scales array (12 bytes for 8 x 6-bit scales + 8 x 6-bit mins)
    // Layout: first 4 bytes = low 4 bits of scales[0..7] and mins[0..7]
    //         next 4 bytes  = low 4 bits of scales[4..7] and mins[4..7] (upper slots)
    //         final 4 bytes = high 2 bits of everything
    // Simplified encoding matching llama.cpp:
    for j in 0..<4 {
        block[4 + j] = (qscales[j] & 0x3F) | ((qmins[j] & 0x3F) << 0) // simplified
    }
    // For test simplicity, use a flat encoding:
    // bytes 4..7:  low 6 bits of scales[0..3] packed as (scale & 0x3F)
    // bytes 8..11: low 6 bits of mins[0..3]
    // bytes 12..15: low 6 bits of scales[4..7]
    // ... This follows the actual llama.cpp k_quants layout:
    for j in 0..<8 {
        // Interleaved: scales in low nibble, mins in high nibble for first 4 bytes
        // Then remaining bits in next bytes
        // We'll use the actual k_quants encoding:
        if j < 4 {
            block[4 + j] = (qscales[j] & 0x3F) | ((qscales[j + 4] & 0x03) << 6)
            block[4 + 4 + j] = (qmins[j] & 0x3F) | ((qmins[j + 4] & 0x03) << 6)
        }
    }
    // High bits of scales[4..7] and mins[4..7]
    for j in 0..<4 {
        block[4 + 8 + j] = ((qscales[j + 4] >> 2) & 0x0F) | (((qmins[j + 4] >> 2) & 0x0F) << 4)
    }

    // Quantise weights to 4-bit nibbles
    for j in 0..<8 {
        let sc = Float(f16d) * Float(qscales[j])
        let mn = Float(f16dmin) * Float(qmins[j])
        for i in 0..<32 {
            let idx = j * 32 + i
            var q: Int
            if sc > 0 {
                q = Int(round((values[idx] + mn) / sc))
                q = min(15, max(0, q))
            } else {
                q = 0
            }
            let byteIdx = 16 + (j * 16 + i / 2) // offset past header (4+12=16 bytes)
            if i % 2 == 0 {
                block[byteIdx] = (block[byteIdx] & 0xF0) | UInt8(q & 0x0F)
            } else {
                block[byteIdx] = (block[byteIdx] & 0x0F) | UInt8((q & 0x0F) << 4)
            }
        }
    }

    return block
}

/// CPU reference: dequantise a Q4_K_M super-block.
private func dequantQ4KBlock(block: [UInt8]) -> [Float] {
    precondition(block.count == Q4_K_BLOCK_SIZE)

    let d: Float16 = block.withUnsafeBufferPointer { buf in
        buf.baseAddress!.withMemoryRebound(to: Float16.self, capacity: 1) { $0.pointee }
    }
    let dmin: Float16 = block.withUnsafeBufferPointer { buf in
        (buf.baseAddress! + 2).withMemoryRebound(to: Float16.self, capacity: 1) { $0.pointee }
    }

    // Decode 6-bit scales and mins
    var scales = [UInt8](repeating: 0, count: 8)
    var mins = [UInt8](repeating: 0, count: 8)

    for j in 0..<4 {
        scales[j] = block[4 + j] & 0x3F
        scales[j + 4] = ((block[4 + j] >> 6) & 0x03) | (((block[4 + 8 + j]) & 0x0F) << 2)
        mins[j] = block[4 + 4 + j] & 0x3F
        mins[j + 4] = ((block[4 + 4 + j] >> 6) & 0x03) | (((block[4 + 8 + j] >> 4) & 0x0F) << 2)
    }

    var result = [Float](repeating: 0, count: 256)
    for j in 0..<8 {
        let sc = Float(d) * Float(scales[j])
        let mn = Float(dmin) * Float(mins[j])
        for i in 0..<32 {
            let byteIdx = 16 + (j * 16 + i / 2)
            let nibble: UInt8
            if i % 2 == 0 {
                nibble = block[byteIdx] & 0x0F
            } else {
                nibble = (block[byteIdx] >> 4) & 0x0F
            }
            result[j * 32 + i] = sc * Float(nibble) - mn
        }
    }
    return result
}

@Suite("Q4_K_M Dequantisation Kernel")
struct DequantQ4KMTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else {
            throw MetalTestError.noMetal
        }
        self.device = d
        self.commandQueue = q
    }

    @Test func singleSuperBlockDequant() async throws {
        let values: [Float] = (0..<256).map { Float($0) * 0.01 - 1.28 }
        let blockData = packQ4KBlock(values: values)
        let expected = dequantQ4KBlock(block: blockData)

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        #expect(result.count == 256)
        for i in 0..<256 {
            #expect(abs(result[i] - expected[i]) < 1e-2,
                    "Elem \(i): GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func multipleSuperBlocksDequant() async throws {
        let superBlockCount = 8
        var allBlockData = [UInt8]()
        var allExpected = [Float]()

        for b in 0..<superBlockCount {
            let values = (0..<256).map { i in
                Float.random(in: -2...2)
            }
            let blockData = packQ4KBlock(values: values)
            allBlockData.append(contentsOf: blockData)
            allExpected.append(contentsOf: dequantQ4KBlock(block: blockData))
        }

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: allBlockData,
            superBlockCount: superBlockCount,
            commandQueue: commandQueue
        )

        #expect(result.count == superBlockCount * 256)
        for i in 0..<result.count {
            #expect(abs(result[i] - allExpected[i]) < 1e-2,
                    "Elem \(i): GPU=\(result[i]) CPU=\(allExpected[i])")
        }
    }

    @Test func zeroBlock() async throws {
        let values = [Float](repeating: 0, count: 256)
        let blockData = packQ4KBlock(values: values)
        let expected = dequantQ4KBlock(block: blockData)

        let kernel = try DequantQ4KMKernel(device: device)
        let result = try await kernel.dequantise(
            blockData: blockData,
            superBlockCount: 1,
            commandQueue: commandQueue
        )

        for i in 0..<256 {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "Zero block elem \(i): GPU=\(result[i]) CPU=\(expected[i])")
        }
    }

    @Test func quantisationErrorWithinTolerance() async throws {
        // Q4_K_M should have lower error than Q4_0 due to per-sub-block scales
        let values: [Float] = (0..<256).map { _ in Float.random(in: -1...1) }
        let blockData = packQ4KBlock(values: values)
        let dequantised = dequantQ4KBlock(block: blockData)

        var totalError: Float = 0
        for i in 0..<256 {
            totalError += abs(values[i] - dequantised[i])
        }
        let mae = totalError / 256.0

        // Tolerance from master plan: < 5e-2 relative error
        #expect(mae < 0.15, "Q4_K_M MAE \(mae) exceeds tolerance")
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter DequantQ4KMTests 2>&1`
Expected: FAIL — `DequantQ4KMKernel` not defined.

**Step 3: Implement the Metal shader**

```metal
// Sources/EdgeRunnerMetal/Shaders/Dequant_Q4_K_M.metal
#include <metal_stdlib>
using namespace metal;

// Q4_K_M super-block: 256 weights, 144 bytes
// Layout:
//   [0..1]   FP16 d (master scale)
//   [2..3]   FP16 dmin (master min scale)
//   [4..15]  12 bytes: encoded 6-bit scales and mins for 8 sub-blocks
//   [16..143] 128 bytes: packed 4-bit weights (2 per byte)

struct ERDequantQ4KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q4_K_BLOCK_SIZE = 144;
constant uint Q4_K_WEIGHTS_PER_BLOCK = 256;

/// Dequantise Q4_K_M super-blocks to FP32.
/// One thread per super-block.
kernel void dequant_q4_k_m(
    device const uint8_t*       input  [[buffer(0)]],
    device float*               output [[buffer(1)]],
    constant ERDequantQ4KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) return;

    device const uint8_t* block = input + tid * Q4_K_BLOCK_SIZE;
    uint outBase = params.outputOffset + tid * Q4_K_WEIGHTS_PER_BLOCK;

    // Read master scales
    half d    = *reinterpret_cast<device const half*>(block);
    half dmin = *reinterpret_cast<device const half*>(block + 2);
    float fd    = float(d);
    float fdmin = float(dmin);

    // Decode 6-bit scales and mins for 8 sub-blocks
    float scales[8];
    float mins[8];

    // Decode: bytes 4..7 contain low 6 bits of scales[0..3] + 2 high bits of scales[4..7]
    //         bytes 8..11 contain low 6 bits of mins[0..3] + 2 high bits of mins[4..7]
    //         bytes 12..15 contain bits 2..5 of scales[4..7] (low nibble) and mins[4..7] (high nibble)
    for (uint j = 0; j < 4; j++) {
        uint8_t sb = block[4 + j];
        uint8_t mb = block[4 + 4 + j];
        uint8_t hb = block[4 + 8 + j];

        scales[j]     = fd * float(sb & 0x3F);
        scales[j + 4] = fd * float(((sb >> 6) & 0x03) | ((hb & 0x0F) << 2));
        mins[j]       = fdmin * float(mb & 0x3F);
        mins[j + 4]   = fdmin * float(((mb >> 6) & 0x03) | (((hb >> 4) & 0x0F) << 2));
    }

    // Dequantise 256 weights across 8 sub-blocks of 32
    for (uint j = 0; j < 8; j++) {
        float sc = scales[j];
        float mn = mins[j];
        for (uint i = 0; i < 32; i++) {
            uint byteIdx = 16 + (j * 16 + i / 2);
            uint8_t byte = block[byteIdx];
            uint8_t nibble;
            if (i % 2 == 0) {
                nibble = byte & 0x0F;
            } else {
                nibble = (byte >> 4) & 0x0F;
            }
            output[outBase + j * 32 + i] = sc * float(nibble) - mn;
        }
    }
}
```

**Step 4: Implement the Swift kernel wrapper**

Append to `Sources/EdgeRunnerMetal/DequantKernel.swift`:

```swift
/// Swift wrapper for the Q4_K_M dequantisation Metal kernel.
public struct DequantQ4KMKernel: Sendable {
    private let dequantPipeline: MTLComputePipelineState
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        guard let fn = library.makeFunction(name: "dequant_q4_k_m") else {
            fatalError("Failed to find dequant_q4_k_m kernel function in Metal library")
        }
        self.dequantPipeline = try device.makeComputePipelineState(function: fn)
        self.device = device
    }

    /// Dequantise Q4_K_M super-blocks to Float32.
    public func dequantise(
        blockData: [UInt8],
        superBlockCount: Int,
        commandQueue: MTLCommandQueue
    ) async throws -> [Float] {
        let inputBuffer = device.makeBuffer(
            bytes: blockData,
            length: blockData.count,
            options: .storageModeShared
        )!

        let outputCount = superBlockCount * 256
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        var params = ERDequantQ4KParams(
            superBlockCount: UInt32(superBlockCount),
            outputOffset: 0
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ERDequantQ4KParams>.stride, index: 2)

        let threadsPerGroup = min(dequantPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: superBlockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outputCount))
    }
}
```

**Step 5: Run tests and verify they pass**

Run: `swift test --filter DequantQ4KMTests 2>&1`
Expected: All tests PASS.

**Step 6: Commit**

```
feat(metal): add Q4_K_M two-level dequantisation kernel for 256-weight super-blocks
```

---

## Summary: Tasks 1-6

| Task | Deliverable | Tests | Key Types |
|------|------------|-------|-----------|
| 1 | Weight loading protocol & types | ~8 | `EdgeRunnerWeightLoader`, `WeightMap`, `TensorStorage`, `ModelConfig` |
| 2 | GGUF header & metadata parser | ~10 | `GGUFHeader`, `GGUFReader`, `GGUFMetadataValue` |
| 3 | GGUF tensor table & mmap loader | ~8 | `GGUFLoader`, `MemoryMappedFile`, `GGUFTensorInfo` |
| 4 | Q4_0 dequant kernel | ~4 | `DequantQ4_0Kernel`, `dequant_q4_0.metal`, fused GEMV |
| 5 | Q8_0 dequant kernel | ~4 | `DequantQ8_0Kernel`, `dequant_q8_0.metal` |
| 6 | Q4_K_M dequant kernel | ~4 | `DequantQ4KMKernel`, `dequant_q4_k_m.metal` |

**Running total: ~38 tests across Tasks 1-6.**

Tasks 7-12 (second half) will cover: SafeTensor loader, NPZ loader, Llama 3 architecture, convenience load API, memory pressure handler, and end-to-end integration.

---

## Task 7: SafeTensor Loader

**Files:**
- Create: `Sources/EdgeRunnerIO/SafeTensorLoader.swift`
- Create: `Sources/EdgeRunnerIO/SafeTensorHeader.swift`
- Test: `Tests/EdgeRunnerIOTests/SafeTensorLoaderTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/SafeTensorLoaderTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

// MARK: - Synthetic SafeTensor Builder

/// Builds a minimal .safetensors binary blob for testing.
private struct SyntheticSafeTensor: Sendable {
    struct TensorSpec: Sendable {
        let name: String
        let dtype: String   // "F32", "F16", "I8"
        let shape: [Int]
        let data: Data
    }

    static func build(tensors: [TensorSpec]) -> Data {
        var dataSection = Data()
        var headerDict: [String: Any] = [:]

        for tensor in tensors {
            let begin = dataSection.count
            dataSection.append(tensor.data)
            let end = dataSection.count
            headerDict[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [begin, end]
            ] as [String: Any]
        }

        let jsonData = try! JSONSerialization.data(
            withJSONObject: headerDict,
            options: [.sortedKeys]
        )
        var result = Data()
        var headerSize = UInt64(jsonData.count)
        result.append(Data(bytes: &headerSize, count: 8))
        result.append(jsonData)
        result.append(dataSection)
        return result
    }
}

@Suite("SafeTensor Loader Tests")
struct SafeTensorLoaderTests: Sendable {

    @Test("Parse JSON header from 8-byte size prefix")
    func parseHeader() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        let spec = SyntheticSafeTensor.TensorSpec(
            name: "weight",
            dtype: "F32",
            shape: [2, 2],
            data: floatData
        )
        let blob = SyntheticSafeTensor.build(tensors: [spec])
        let header = try SafeTensorHeader.parse(from: blob)

        #expect(header.tensors.count == 1)
        #expect(header.tensors["weight"] != nil)
    }

    @Test("Extract tensor metadata: dtype, shape, data_offsets")
    func extractMetadata() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        let spec = SyntheticSafeTensor.TensorSpec(
            name: "layer.0.weight",
            dtype: "F32",
            shape: [2, 3],
            data: floatData
        )
        let blob = SyntheticSafeTensor.build(tensors: [spec])
        let header = try SafeTensorHeader.parse(from: blob)

        let meta = try #require(header.tensors["layer.0.weight"])
        #expect(meta.dtype == .float32)
        #expect(meta.shape == [2, 3])
        #expect(meta.dataOffsets.end - meta.dataOffsets.begin == 24) // 6 * 4 bytes
    }

    @Test("Memory-map binary data section and read float32 values")
    func mmapDataSection() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        let spec = SyntheticSafeTensor.TensorSpec(
            name: "embed",
            dtype: "F32",
            shape: [4],
            data: floatData
        )
        let blob = SyntheticSafeTensor.build(tensors: [spec])

        // Write to temp file for mmap testing
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try SafeTensorLoader(url: tmpURL)
        let storage = try loader.loadTensor(named: "embed")

        #expect(storage.shape == [4])
        #expect(storage.dtype == .float32)
        #expect(storage.byteCount == 16)
    }

    @Test("Load multiple tensors from single file")
    func multipleTensors() throws {
        let w1Data = [Float](repeating: 1.0, count: 6)
            .withUnsafeBufferPointer { Data(buffer: $0) }
        let w2Data = [Float](repeating: 2.0, count: 4)
            .withUnsafeBufferPointer { Data(buffer: $0) }

        let blob = SyntheticSafeTensor.build(tensors: [
            .init(name: "attn.weight", dtype: "F32", shape: [2, 3], data: w1Data),
            .init(name: "ffn.weight", dtype: "F32", shape: [2, 2], data: w2Data),
        ])

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try SafeTensorLoader(url: tmpURL)
        let names = loader.tensorNames
        #expect(names.contains("attn.weight"))
        #expect(names.contains("ffn.weight"))
        #expect(names.count == 2)
    }

    @Test("Float16 dtype parsing")
    func float16Dtype() throws {
        // 4 float16 values = 8 bytes
        let data = Data(repeating: 0, count: 8)
        let spec = SyntheticSafeTensor.TensorSpec(
            name: "half_tensor",
            dtype: "F16",
            shape: [4],
            data: data
        )
        let blob = SyntheticSafeTensor.build(tensors: [spec])
        let header = try SafeTensorHeader.parse(from: blob)
        let meta = try #require(header.tensors["half_tensor"])
        #expect(meta.dtype == .float16)
    }

    @Test("Invalid header throws descriptive error")
    func invalidHeader() throws {
        let garbage = Data([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        #expect(throws: SafeTensorError.self) {
            _ = try SafeTensorHeader.parse(from: garbage)
        }
    }
}
```

**Step 2: Define the header parser**

```swift
// Sources/EdgeRunnerIO/SafeTensorHeader.swift
import Foundation

/// Errors from SafeTensor parsing.
public enum SafeTensorError: Error, Sendable {
    case fileTooSmall
    case headerSizeExceedsFile
    case invalidJSON(description: String)
    case missingField(tensor: String, field: String)
    case unknownDtype(String)
}

/// Supported data types in SafeTensor format.
public enum SafeTensorDtype: String, Sendable, Equatable {
    case float32 = "F32"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case int8 = "I8"
    case int32 = "I32"

    public var byteSize: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .bfloat16: return 2
        case .int8: return 1
        }
    }
}

/// Offset range into the binary data section.
public struct DataOffsets: Sendable, Equatable {
    public let begin: Int
    public let end: Int
}

/// Metadata for one tensor in a SafeTensor file.
public struct SafeTensorTensorMeta: Sendable {
    public let dtype: SafeTensorDtype
    public let shape: [Int]
    public let dataOffsets: DataOffsets
}

/// Parsed SafeTensor header.
public struct SafeTensorHeader: Sendable {
    /// Byte offset where the data section begins (8 + headerSize).
    public let dataOffset: Int
    /// Tensor name -> metadata mapping.
    public let tensors: [String: SafeTensorTensorMeta]

    /// Parse the JSON header from raw SafeTensor bytes.
    public static func parse(from data: Data) throws -> SafeTensorHeader {
        guard data.count >= 8 else {
            throw SafeTensorError.fileTooSmall
        }

        let headerSize: UInt64 = data.withUnsafeBytes { buf in
            buf.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        }
        let headerSizeInt = Int(headerSize)
        let dataOffset = 8 + headerSizeInt

        guard data.count >= dataOffset else {
            throw SafeTensorError.headerSizeExceedsFile
        }

        let jsonData = data[8..<dataOffset]

        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: jsonData)
        } catch {
            throw SafeTensorError.invalidJSON(underlying: error)
        }

        guard let dict = parsed as? [String: Any] else {
            throw SafeTensorError.invalidJSON(
                underlying: NSError(domain: "SafeTensor", code: 0,
                    userInfo: [NSLocalizedDescriptionKey: "Top-level is not a dictionary"])
            )
        }

        var tensors: [String: SafeTensorTensorMeta] = [:]

        for (name, value) in dict {
            // Skip __metadata__ key
            if name == "__metadata__" { continue }

            guard let tensorDict = value as? [String: Any] else { continue }

            guard let dtypeStr = tensorDict["dtype"] as? String else {
                throw SafeTensorError.missingField(tensor: name, field: "dtype")
            }
            guard let dtype = SafeTensorDtype(rawValue: dtypeStr) else {
                throw SafeTensorError.unknownDtype(dtypeStr)
            }
            guard let shape = tensorDict["shape"] as? [Int] else {
                throw SafeTensorError.missingField(tensor: name, field: "shape")
            }
            guard let offsets = tensorDict["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                throw SafeTensorError.missingField(tensor: name, field: "data_offsets")
            }

            tensors[name] = SafeTensorTensorMeta(
                dtype: dtype,
                shape: shape,
                dataOffsets: DataOffsets(begin: offsets[0], end: offsets[1])
            )
        }

        return SafeTensorHeader(dataOffset: dataOffset, tensors: tensors)
    }
}
```

**Step 3: Implement the SafeTensor loader with mmap**

```swift
// Sources/EdgeRunnerIO/SafeTensorLoader.swift
import Foundation

/// Loads tensors from a .safetensors file using memory-mapped I/O.
public final class SafeTensorLoader: Sendable, EdgeRunnerWeightLoader {
    private let header: SafeTensorHeader
    private let mappedFile: MemoryMappedFile
    private let _tensorNames: [String]

    public var tensorNames: [String] { _tensorNames }

    public init(url: URL) throws {
        let mappedFile = try MemoryMappedFile(url: url)
        self.mappedFile = mappedFile

        // Parse header from mapped data
        let data = Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: mappedFile.baseAddress),
            count: mappedFile.size,
            deallocator: .none
        )
        self.header = try SafeTensorHeader.parse(from: data)
        self._tensorNames = Array(header.tensors.keys).sorted()
    }

    /// Load a single tensor by name.
    public func loadTensor(named name: String) throws -> TensorStorage {
        guard let meta = header.tensors[name] else {
            throw SafeTensorError.missingField(tensor: name, field: "(not found)")
        }

        let absoluteBegin = header.dataOffset + meta.dataOffsets.begin
        let byteCount = meta.dataOffsets.end - meta.dataOffsets.begin
        let pointer = mappedFile.baseAddress.advanced(by: absoluteBegin)

        return TensorStorage(
            name: name,
            shape: meta.shape,
            dtype: tensorDtype(from: meta.dtype),
            pointer: UnsafeRawPointer(pointer),
            byteCount: byteCount,
            owner: mappedFile
        )
    }

    /// Map SafeTensorDtype to EdgeRunner's TensorDtype.
    private func tensorDtype(from st: SafeTensorDtype) -> TensorDtype {
        switch st {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int8: return .int8
        case .int32: return .int32
        }
    }

    public func loadWeightMap() throws -> WeightMap {
        var map = WeightMap()
        for name in tensorNames {
            map[name] = try loadTensor(named: name)
        }
        return map
    }
}
```

**Step 4: Run tests and verify they pass**

Run: `swift test --filter SafeTensorLoaderTests 2>&1`
Expected: All 6 tests PASS.

**Step 5: Commit**

```
feat(weights): add SafeTensor loader with JSON header parsing and mmap data access
```

---

## Task 8: NPZ Loader

**Files:**
- Create: `Sources/EdgeRunnerIO/NPZLoader.swift`
- Create: `Sources/EdgeRunnerIO/NPYParser.swift`
- Test: `Tests/EdgeRunnerIOTests/NPZLoaderTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/NPZLoaderTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

// MARK: - Synthetic NPZ Builder

/// Builds a minimal .npz (zip of .npy files) for testing.
private struct SyntheticNPZ: Sendable {
    struct ArraySpec: Sendable {
        let name: String
        let dtype: NPYDtype
        let shape: [Int]
        let data: Data
    }

    /// Build an NPY v1.0 binary blob for a single array.
    static func buildNPY(dtype: NPYDtype, shape: [Int], data: Data) -> Data {
        let descrString: String
        switch dtype {
        case .float32: descrString = "<f4"
        case .float16: descrString = "<f2"
        case .int8: descrString = "|i1"
        }

        // NumPy header: magic + version + HEADER_LEN + header string
        let headerStr = "{'descr': '\(descrString)', 'fortran_order': False, 'shape': (\(shape.map(String.init).joined(separator: ", "))\(shape.count == 1 ? "," : "")), }"
        // Pad to multiple of 64 bytes (including magic(6) + version(2) + headerLen(2) = 10)
        let preambleLen = 10
        var paddedHeader = headerStr
        let totalSoFar = preambleLen + paddedHeader.utf8.count + 1 // +1 for newline
        let padding = (64 - (totalSoFar % 64)) % 64
        paddedHeader += String(repeating: " ", count: padding) + "\n"

        var result = Data()
        // Magic: \x93NUMPY
        result.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        // Version 1.0
        result.append(contentsOf: [0x01, 0x00])
        // HEADER_LEN (little-endian UInt16)
        var headerLen = UInt16(paddedHeader.utf8.count)
        result.append(Data(bytes: &headerLen, count: 2))
        // Header string
        result.append(paddedHeader.data(using: .ascii)!)
        // Data
        result.append(data)
        return result
    }
}

@Suite("NPZ Loader Tests")
struct NPZLoaderTests: Sendable {

    @Test("Parse .npy header: magic, version, dtype, shape")
    func parseNPYHeader() throws {
        let floats: [Float] = [1.0, 2.0, 3.0]
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let npy = SyntheticNPZ.buildNPY(dtype: .float32, shape: [3], data: data)

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .float32)
        #expect(header.shape == [3])
        #expect(header.isFortranOrder == false)
    }

    @Test("Load float32 tensor from .npy")
    func loadFloat32() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let npy = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 3], data: data)

        let (header, dataOffset) = try NPYHeader.parseWithOffset(from: npy)
        #expect(header.shape == [2, 3])
        #expect(npy.count - dataOffset == 24) // 6 * 4 bytes
    }

    @Test("Load float16 tensor from .npy")
    func loadFloat16() throws {
        // 4 float16 values = 8 bytes
        let data = Data(repeating: 0, count: 8)
        let npy = SyntheticNPZ.buildNPY(dtype: .float16, shape: [4], data: data)

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .float16)
        #expect(header.shape == [4])
    }

    @Test("Load int8 tensor from .npy")
    func loadInt8() throws {
        let bytes: [UInt8] = [1, 2, 3, 4, 5, 6, 7, 8]
        let data = Data(bytes)
        let npy = SyntheticNPZ.buildNPY(dtype: .int8, shape: [8], data: data)

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .int8)
        #expect(header.shape == [8])
    }

    @Test("Load tensors from NPZ (zip of .npy)")
    func loadNPZ() throws {
        let w1 = [Float](repeating: 1.0, count: 4)
            .withUnsafeBufferPointer { Data(buffer: $0) }
        let w2 = [Float](repeating: 2.0, count: 6)
            .withUnsafeBufferPointer { Data(buffer: $0) }

        let npy1 = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 2], data: w1)
        let npy2 = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 3], data: w2)

        // Build a zip manually with two entries
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try npy1.write(to: tmpDir.appendingPathComponent("weight_a.npy"))
        try npy2.write(to: tmpDir.appendingPathComponent("weight_b.npy"))

        // Create zip using ditto (macOS built-in)
        let npzURL = tmpDir.appendingPathComponent("model.npz")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = [
            "-c", "-k", "--sequesterRsrc",
            tmpDir.appendingPathComponent("weight_a.npy").path,
            tmpDir.appendingPathComponent("weight_b.npy").path,
            npzURL.path
        ]
        // Alternative: use zip from individual npy files
        let zipProcess = Process()
        zipProcess.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        zipProcess.arguments = ["-j", npzURL.path,
            tmpDir.appendingPathComponent("weight_a.npy").path,
            tmpDir.appendingPathComponent("weight_b.npy").path]
        zipProcess.currentDirectoryURL = tmpDir
        try zipProcess.run()
        zipProcess.waitUntilExit()

        let loader = try NPZLoader(url: npzURL)
        let names = loader.tensorNames
        #expect(names.contains("weight_a"))
        #expect(names.contains("weight_b"))
        #expect(names.count == 2)

        let storage = try loader.loadTensor(named: "weight_a")
        #expect(storage.shape == [2, 2])
        #expect(storage.dtype == .float32)
    }

    @Test("Invalid .npy magic throws error")
    func invalidMagic() {
        let garbage = Data([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        #expect(throws: NPYError.self) {
            _ = try NPYHeader.parse(from: garbage)
        }
    }
}
```

**Step 2: Implement the NPY parser**

```swift
// Sources/EdgeRunnerIO/NPYParser.swift
import Foundation

/// Errors from NPY/NPZ parsing.
public enum NPYError: Error, Sendable {
    case fileTooSmall
    case invalidMagic
    case unsupportedVersion(major: UInt8, minor: UInt8)
    case invalidHeader(String)
    case unsupportedDtype(String)
    case entryNotFound(String)
}

/// Supported NumPy dtypes.
public enum NPYDtype: String, Sendable, Equatable {
    case float32  // <f4
    case float16  // <f2
    case int8     // |i1

    public var byteSize: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .int8: return 1
        }
    }

    init(descrString: String) throws {
        switch descrString {
        case "<f4", "=f4": self = .float32
        case "<f2", "=f2": self = .float16
        case "|i1", "=i1", "<i1": self = .int8
        default: throw NPYError.unsupportedDtype(descrString)
        }
    }
}

/// Parsed .npy file header.
public struct NPYHeader: Sendable {
    public let dtype: NPYDtype
    public let shape: [Int]
    public let isFortranOrder: Bool

    /// Parse header from .npy data, returning the header.
    public static func parse(from data: Data) throws -> NPYHeader {
        let (header, _) = try parseWithOffset(from: data)
        return header
    }

    /// Parse header and return both header and data offset.
    public static func parseWithOffset(from data: Data) throws -> (NPYHeader, Int) {
        guard data.count >= 10 else {
            throw NPYError.fileTooSmall
        }

        // Verify magic: \x93NUMPY
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
        for i in 0..<6 {
            guard data[i] == magic[i] else {
                throw NPYError.invalidMagic
            }
        }

        let major = data[6]
        let minor = data[7]

        let headerLen: Int
        let headerStart: Int

        if major == 1 {
            // Version 1.0: 2-byte HEADER_LEN (little-endian)
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else if major == 2 {
            // Version 2.0: 4-byte HEADER_LEN (little-endian)
            guard data.count >= 12 else { throw NPYError.fileTooSmall }
            headerLen = Int(data[8]) | (Int(data[9]) << 8) |
                       (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        } else {
            throw NPYError.unsupportedVersion(major: major, minor: minor)
        }

        let dataOffset = headerStart + headerLen
        guard data.count >= dataOffset else {
            throw NPYError.fileTooSmall
        }

        let headerData = data[headerStart..<dataOffset]
        guard let headerString = String(data: headerData, encoding: .ascii) else {
            throw NPYError.invalidHeader("Cannot decode header as ASCII")
        }

        // Parse the Python dict-like header string
        let header = try parseHeaderDict(headerString)
        return (header, dataOffset)
    }

    /// Minimal parser for NumPy header dict strings like:
    /// {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
    private static func parseHeaderDict(_ s: String) throws -> NPYHeader {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)

        // Extract descr
        guard let descrRange = trimmed.range(of: "'descr':\\s*'([^']+)'", options: .regularExpression) else {
            throw NPYError.invalidHeader("Missing 'descr' field")
        }
        let descrMatch = trimmed[descrRange]
        let descrValue = String(descrMatch).components(separatedBy: "'")
            .filter { !$0.contains("descr") && !$0.contains(":") && !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .first ?? ""
        let dtype = try NPYDtype(descrString: descrValue.trimmingCharacters(in: .whitespaces))

        // Extract fortran_order
        let isFortran = trimmed.contains("'fortran_order': True")

        // Extract shape
        guard let shapeRange = trimmed.range(of: "'shape':\\s*\\(([^)]*)\\)", options: .regularExpression) else {
            throw NPYError.invalidHeader("Missing 'shape' field")
        }
        let shapeStr = String(trimmed[shapeRange])
        let parenStart = shapeStr.firstIndex(of: "(")!
        let parenEnd = shapeStr.lastIndex(of: ")")!
        let innerShape = String(shapeStr[shapeStr.index(after: parenStart)..<parenEnd])
        let shape = innerShape
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .map { Int($0)! }

        return NPYHeader(dtype: dtype, shape: shape, isFortranOrder: isFortran)
    }
}
```

**Step 3: Implement the NPZ loader**

```swift
// Sources/EdgeRunnerIO/NPZLoader.swift
import Foundation

/// Loads tensors from .npz files (zip archives of .npy arrays).
public final class NPZLoader: Sendable, EdgeRunnerWeightLoader {
    private let archiveURL: URL
    private let entries: [String: Data]  // name -> npy data
    private let _tensorNames: [String]

    public var tensorNames: [String] { _tensorNames }

    public init(url: URL) throws {
        self.archiveURL = url

        // Use Foundation's zip reading via FileWrapper or manual zip parsing
        // For zero external deps, use Process to unzip to temp dir then read
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("npz-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", url.path, "-d", tmpDir.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()

        // Enumerate .npy files
        let contents = try FileManager.default.contentsOfDirectory(
            at: tmpDir,
            includingPropertiesForKeys: nil
        )

        var entries: [String: Data] = [:]
        for fileURL in contents where fileURL.pathExtension == "npy" {
            let name = fileURL.deletingPathExtension().lastPathComponent
            entries[name] = try Data(contentsOf: fileURL)
        }

        // Cleanup temp dir
        try? FileManager.default.removeItem(at: tmpDir)

        self.entries = entries
        self._tensorNames = Array(entries.keys).sorted()
    }

    /// Load a single tensor by name.
    public func loadTensor(named name: String) throws -> TensorStorage {
        guard let npyData = entries[name] else {
            throw NPYError.entryNotFound(name)
        }

        let (header, dataOffset) = try NPYHeader.parseWithOffset(from: npyData)
        let tensorData = npyData[dataOffset...]

        let dtype: TensorDtype
        switch header.dtype {
        case .float32: dtype = .float32
        case .float16: dtype = .float16
        case .int8: dtype = .int8
        }

        return TensorStorage(
            name: name,
            shape: header.shape,
            dtype: dtype,
            data: Data(tensorData)
        )
    }

    public func loadWeightMap() throws -> WeightMap {
        var map = WeightMap()
        for name in tensorNames {
            map[name] = try loadTensor(named: name)
        }
        return map
    }
}
```

**Step 4: Run tests and verify they pass**

Run: `swift test --filter NPZLoaderTests 2>&1`
Expected: All 6 tests PASS.

**Step 5: Commit**

```
feat(weights): add NPZ/NPY loader with dtype parsing and zip archive extraction
```

---

## Task 9: Llama 3 Architecture

**Files:**
- Create: `Sources/EdgeRunnerIO/LlamaConfig.swift`
- Create: `Sources/EdgeRunnerIO/LlamaModel.swift`
- Create: `Sources/EdgeRunnerIO/LlamaBlock.swift`
- Create: `Sources/EdgeRunnerIO/Protocols/EdgeRunnerModule.swift`
- Create: `Sources/EdgeRunnerIO/Protocols/LoadableModel.swift`
- Test: `Tests/EdgeRunnerIOTests/LlamaConfigTests.swift`
- Test: `Tests/EdgeRunnerIOTests/LlamaModelTests.swift`
- Test: `Tests/EdgeRunnerIOTests/LlamaBlockTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/LlamaConfigTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO
@testable import EdgeRunnerIO

@Suite("Llama Config Tests")
struct LlamaConfigTests: Sendable {

    @Test("Parse LlamaConfig from GGUF metadata")
    func parseFromGGUF() throws {
        let metadata: [String: MetadataValue] = [
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.vocab_size": 128256,
            "llama.feed_forward_length": 14336,
            "llama.rope.freq_base": 500000.0,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
        ]

        let config = try LlamaConfig(fromGGUFMetadata: metadata)
        #expect(config.embeddingDim == 4096)
        #expect(config.layerCount == 32)
        #expect(config.headCount == 32)
        #expect(config.kvHeadCount == 8)
        #expect(config.vocabSize == 128256)
        #expect(config.intermediateDim == 14336)
        #expect(config.ropeFreqBase == 500000.0)
        #expect(config.rmsNormEpsilon == 1e-5)
    }

    @Test("Computed properties: headDim, GQA ratio")
    func computedProperties() throws {
        let config = LlamaConfig(
            embeddingDim: 4096,
            layerCount: 32,
            headCount: 32,
            kvHeadCount: 8,
            vocabSize: 128256,
            intermediateDim: 14336,
            ropeFreqBase: 500000.0,
            rmsNormEpsilon: 1e-5
        )

        #expect(config.headDim == 128)  // 4096 / 32
        #expect(config.gqaRatio == 4)   // 32 / 8
    }

    @Test("Missing metadata key throws descriptive error")
    func missingKey() {
        let metadata: [String: MetadataValue] = [
            "llama.embedding_length": 4096,
            // Missing other required fields
        ]
        #expect(throws: LlamaConfigError.self) {
            _ = try LlamaConfig(fromGGUFMetadata: metadata)
        }
    }
}

// Tests/EdgeRunnerIOTests/LlamaBlockTests.swift
import Testing
import Metal
@testable import EdgeRunnerIO

@Suite("Llama Block Tests")
struct LlamaBlockTests: Sendable {

    @Test("LlamaBlock has correct sub-module structure")
    func blockStructure() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 1,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let block = LlamaBlock(config: config, layerIndex: 0)

        // Verify structure
        #expect(block.attentionNorm != nil)
        #expect(block.ffnNorm != nil)
        #expect(block.attention != nil)
        #expect(block.feedForward != nil)
    }

    @Test("Weight name mapping from GGUF tensor names")
    func weightNameMapping() {
        let mapping = LlamaWeightNameMapper.mapGGUFName("blk.0.attn_q.weight")
        #expect(mapping == "layers.0.attention.wq.weight")

        let normMapping = LlamaWeightNameMapper.mapGGUFName("blk.5.attn_norm.weight")
        #expect(normMapping == "layers.5.attentionNorm.weight")

        let ffnMapping = LlamaWeightNameMapper.mapGGUFName("blk.3.ffn_gate.weight")
        #expect(ffnMapping == "layers.3.feedForward.gate.weight")

        let outputMapping = LlamaWeightNameMapper.mapGGUFName("output.weight")
        #expect(outputMapping == "lmHead.weight")

        let embedMapping = LlamaWeightNameMapper.mapGGUFName("token_embd.weight")
        #expect(embedMapping == "embedding.weight")
    }
}

// Tests/EdgeRunnerIOTests/LlamaModelTests.swift
import Testing
import Metal
@testable import EdgeRunnerIO

@Suite("Llama Model Tests")
struct LlamaModelTests: Sendable {

    @Test("LlamaModel initialises with correct layer count")
    func modelInit() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 4,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        #expect(model.layers.count == 4)
        #expect(model.config.vocabSize == 100)
    }

    @Test("LlamaModel conforms to LoadableModel")
    func conformance() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 1,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        let lm: any LoadableModel = model
        #expect(lm.parameterNames.contains("embedding.weight"))
        #expect(lm.parameterNames.contains("lmHead.weight"))
    }

    @Test("Weight name list covers all parameters")
    func weightNames() {
        let config = LlamaConfig(
            embeddingDim: 64,
            layerCount: 2,
            headCount: 4,
            kvHeadCount: 2,
            vocabSize: 100,
            intermediateDim: 128,
            ropeFreqBase: 10000.0,
            rmsNormEpsilon: 1e-5
        )

        let model = LlamaModel(config: config)
        let names = model.parameterNames

        // Embedding + lmHead + finalNorm + 2 layers * (attNorm, ffnNorm, wq, wk, wv, wo, gate, up, down)
        #expect(names.contains("embedding.weight"))
        #expect(names.contains("lmHead.weight"))
        #expect(names.contains("finalNorm.weight"))
        #expect(names.contains("layers.0.attention.wq.weight"))
        #expect(names.contains("layers.0.attention.wk.weight"))
        #expect(names.contains("layers.0.attention.wv.weight"))
        #expect(names.contains("layers.0.attention.wo.weight"))
        #expect(names.contains("layers.0.feedForward.gate.weight"))
        #expect(names.contains("layers.0.feedForward.up.weight"))
        #expect(names.contains("layers.0.feedForward.down.weight"))
        #expect(names.contains("layers.1.attention.wq.weight"))
    }
}
```

**Step 2: Implement protocols**

```swift
// Sources/EdgeRunnerIO/Protocols/EdgeRunnerModule.swift
import Foundation

/// Base protocol for all EdgeRunner neural network modules.
public protocol EdgeRunnerModule: Sendable {
    /// All named parameters in this module and its children.
    var parameterNames: [String] { get }
}

// Sources/EdgeRunnerIO/Protocols/LoadableModel.swift
import Foundation

/// M3-scoped protocol for models that can receive weights from a WeightMap.
public protocol LoadableModel: Sendable {
    var parameterNames: [String] { get }
    mutating func loadWeights(from map: WeightMap) throws
}
```

**Step 3: Implement LlamaConfig**

```swift
// Sources/EdgeRunnerIO/LlamaConfig.swift
import Foundation

public enum LlamaConfigError: Error, Sendable {
    case missingMetadataKey(String)
    case invalidValue(key: String, value: Any)
}

/// Configuration for Llama 3 architecture.
public struct LlamaConfig: Sendable, Equatable {
    public let embeddingDim: Int
    public let layerCount: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let vocabSize: Int
    public let intermediateDim: Int
    public let ropeFreqBase: Double
    public let rmsNormEpsilon: Double

    /// Dimension per attention head.
    public var headDim: Int { embeddingDim / headCount }

    /// GQA group ratio (queries per KV head).
    public var gqaRatio: Int { headCount / kvHeadCount }

    public init(
        embeddingDim: Int,
        layerCount: Int,
        headCount: Int,
        kvHeadCount: Int,
        vocabSize: Int,
        intermediateDim: Int,
        ropeFreqBase: Double,
        rmsNormEpsilon: Double
    ) {
        self.embeddingDim = embeddingDim
        self.layerCount = layerCount
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.vocabSize = vocabSize
        self.intermediateDim = intermediateDim
        self.ropeFreqBase = ropeFreqBase
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    /// Initialise from GGUF metadata dictionary.
    public init(fromGGUFMetadata metadata: [String: MetadataValue]) throws {
        func require<T>(_ key: String, as type: T.Type) throws -> T {
            guard let value = metadata[key] else {
                throw LlamaConfigError.missingMetadataKey(key)
            }
            guard let typed = value as? T else {
                throw LlamaConfigError.invalidValue(key: key, value: value)
            }
            return typed
        }

        self.embeddingDim = try require("llama.embedding_length", as: Int.self)
        self.layerCount = try require("llama.block_count", as: Int.self)
        self.headCount = try require("llama.attention.head_count", as: Int.self)
        self.kvHeadCount = try require("llama.attention.head_count_kv", as: Int.self)
        self.vocabSize = try require("llama.vocab_size", as: Int.self)
        self.intermediateDim = try require("llama.feed_forward_length", as: Int.self)
        self.ropeFreqBase = try require("llama.rope.freq_base", as: Double.self)
        self.rmsNormEpsilon = try require("llama.attention.layer_norm_rms_epsilon", as: Double.self)
    }
}
```

**Step 4: Implement LlamaBlock and LlamaModel**

```swift
// Sources/EdgeRunnerIO/LlamaBlock.swift
import Foundation

/// RMSNorm -> GQA Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual
public final class LlamaBlock: EdgeRunnerModule, Sendable {
    public let layerIndex: Int
    public let config: LlamaConfig

    // Sub-modules (public for structure inspection in tests)
    public let attentionNorm: RMSNorm?
    public let ffnNorm: RMSNorm?
    public let attention: GQAAttention?
    public let feedForward: SwiGLUFFN?

    private let prefix: String

    public init(config: LlamaConfig, layerIndex: Int) {
        self.config = config
        self.layerIndex = layerIndex
        self.prefix = "layers.\(layerIndex)"

        self.attentionNorm = RMSNorm(dim: config.embeddingDim, epsilon: config.rmsNormEpsilon)
        self.ffnNorm = RMSNorm(dim: config.embeddingDim, epsilon: config.rmsNormEpsilon)
        self.attention = GQAAttention(config: config)
        self.feedForward = SwiGLUFFN(
            inputDim: config.embeddingDim,
            hiddenDim: config.intermediateDim
        )
    }

    public var parameterNames: [String] {
        [
            "\(prefix).attentionNorm.weight",
            "\(prefix).ffnNorm.weight",
            "\(prefix).attention.wq.weight",
            "\(prefix).attention.wk.weight",
            "\(prefix).attention.wv.weight",
            "\(prefix).attention.wo.weight",
            "\(prefix).feedForward.gate.weight",
            "\(prefix).feedForward.up.weight",
            "\(prefix).feedForward.down.weight",
        ]
    }
}

/// Placeholder for RMSNorm module.
public struct RMSNorm: Sendable {
    public let dim: Int
    public let epsilon: Double

    public init(dim: Int, epsilon: Double) {
        self.dim = dim
        self.epsilon = epsilon
    }
}

/// Placeholder for Grouped-Query Attention module.
public struct GQAAttention: Sendable {
    public let config: LlamaConfig

    public init(config: LlamaConfig) {
        self.config = config
    }
}

/// Placeholder for SwiGLU Feed-Forward Network module.
public struct SwiGLUFFN: Sendable {
    public let inputDim: Int
    public let hiddenDim: Int

    public init(inputDim: Int, hiddenDim: Int) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
    }
}

// Sources/EdgeRunnerIO/LlamaModel.swift
import Foundation

/// Llama 3 decoder-only transformer model.
public final class LlamaModel: LoadableModel, Sendable {
    public let config: LlamaConfig
    public let layers: [LlamaBlock]

    public var vocabSize: Int { config.vocabSize }
    public var embeddingDim: Int { config.embeddingDim }

    public init(config: LlamaConfig) {
        self.config = config
        self.layers = (0..<config.layerCount).map { i in
            LlamaBlock(config: config, layerIndex: i)
        }
    }

    public var parameterNames: [String] {
        var names: [String] = [
            "embedding.weight",
        ]

        for layer in layers {
            names.append(contentsOf: layer.parameterNames)
        }

        names.append("finalNorm.weight")
        names.append("lmHead.weight")
        return names
    }

    public func loadWeights(from map: WeightMap) throws {
        for name in parameterNames {
            guard map[name] != nil else {
                throw ModelLoadError.loadFailed(
                    underlying: NSError(domain: "EdgeRunner", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Missing weight: \(name)"])
                )
            }
            // Apply weight data to corresponding parameter tensor
        }
    }
}
```

**Step 5: Implement weight name mapper**

```swift
// Add to LlamaBlock.swift or a separate file:

/// Maps GGUF tensor names to EdgeRunner internal parameter names.
public enum LlamaWeightNameMapper: Sendable {
    private static let mappings: [(pattern: String, replacement: String)] = [
        ("token_embd.weight", "embedding.weight"),
        ("output.weight", "lmHead.weight"),
        ("output_norm.weight", "finalNorm.weight"),
    ]

    private static let blockMappings: [(gguf: String, er: String)] = [
        ("attn_q.weight", "attention.wq.weight"),
        ("attn_k.weight", "attention.wk.weight"),
        ("attn_v.weight", "attention.wv.weight"),
        ("attn_output.weight", "attention.wo.weight"),
        ("attn_norm.weight", "attentionNorm.weight"),
        ("ffn_gate.weight", "feedForward.gate.weight"),
        ("ffn_up.weight", "feedForward.up.weight"),
        ("ffn_down.weight", "feedForward.down.weight"),
        ("ffn_norm.weight", "ffnNorm.weight"),
    ]

    /// Convert a GGUF tensor name to an EdgeRunner parameter path.
    public static func mapGGUFName(_ ggufName: String) -> String {
        // Check top-level mappings first
        for mapping in mappings {
            if ggufName == mapping.pattern {
                return mapping.replacement
            }
        }

        // Check block-level mappings: "blk.N.suffix"
        if ggufName.hasPrefix("blk.") {
            let parts = ggufName.split(separator: ".", maxSplits: 2)
            guard parts.count >= 3,
                  let layerIndex = Int(parts[1]) else {
                return ggufName
            }
            let suffix = String(parts[2])

            for mapping in blockMappings {
                if suffix == mapping.gguf {
                    return "layers.\(layerIndex).\(mapping.er)"
                }
            }
        }

        return ggufName // Unmapped, return as-is
    }
}
```

**Step 6: Run tests and verify they pass**

Run: `swift test --filter "LlamaConfigTests|LlamaBlockTests|LlamaModelTests" 2>&1`
Expected: All tests PASS.

**Step 7: Commit**

```
feat(models): add Llama 3 architecture with GQA, SwiGLU FFN, and GGUF weight name mapping
```

---

## Task 10: Convenience Load API

**Files:**
- Create: `Sources/EdgeRunnerIO/EdgeRunnerModelLoader.swift`
- Create: `Sources/EdgeRunnerIO/ModelRegistry.swift`
- Test: `Tests/EdgeRunnerIOTests/ModelLoaderTests.swift`
- Test: `Tests/EdgeRunnerIOTests/ModelRegistryTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/ModelRegistryTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

/// Minimal test model for registry testing.
private struct StubModel: LoadableModel, @unchecked Sendable {
    var parameterNames: [String] { ["stub.weight"] }
    var weightsLoaded = false

    mutating func loadWeights(from map: WeightMap) throws {
        weightsLoaded = true
    }
}

/// Stub architecture factory.
private struct StubArchitectureFactory: ArchitectureFactory, Sendable {
    let architectureName: String = "stub"

    func create(config: ModelConfig) throws -> any LoadableModel {
        return StubModel()
    }
}

@Suite("Model Registry Tests")
struct ModelRegistryTests: Sendable {

    @Test("Register and retrieve architecture factory")
    func registerAndRetrieve() {
        let registry = ModelRegistry()
        let factory = StubArchitectureFactory()

        registry.register(factory)

        let retrieved = registry.factory(for: "stub")
        #expect(retrieved != nil)
        #expect(retrieved?.architectureName == "stub")
    }

    @Test("Unknown architecture returns nil")
    func unknownArchitecture() {
        let registry = ModelRegistry()
        let retrieved = registry.factory(for: "nonexistent")
        #expect(retrieved == nil)
    }

    @Test("Default registry includes Llama")
    func defaultRegistryHasLlama() {
        let registry = ModelRegistry.default
        let llama = registry.factory(for: "llama")
        #expect(llama != nil)
    }

    @Test("Overwrite existing registration")
    func overwrite() {
        let registry = ModelRegistry()
        let factory1 = StubArchitectureFactory()
        let factory2 = StubArchitectureFactory()

        registry.register(factory1)
        registry.register(factory2)

        // Should not crash, latest wins
        let retrieved = registry.factory(for: "stub")
        #expect(retrieved != nil)
    }
}

// Tests/EdgeRunnerIOTests/ModelLoaderTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO
@testable import EdgeRunnerIO

@Suite("EdgeRunner Model Loader Tests")
struct ModelLoaderTests: Sendable {

    @Test("Detect GGUF format from extension")
    func detectGGUF() {
        let format = ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.gguf"))
        #expect(format == .gguf)
    }

    @Test("Detect SafeTensor format from extension")
    func detectSafeTensor() {
        let format = ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.safetensors"))
        #expect(format == .safetensors)
    }

    @Test("Detect NPZ format from extension")
    func detectNPZ() {
        let format = ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.npz"))
        #expect(format == .npz)
    }

    @Test("Unknown extension returns nil")
    func unknownExtension() {
        let format = ModelFormat.detect(from: URL(fileURLWithPath: "/tmp/model.bin"))
        #expect(format == nil)
    }

    @Test("load(from:) loads weights into model and returns LoadableModel")
    func loadReturnsTypeErasedWithWeights() async throws {
        // Build a synthetic GGUF with a known weight
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_stub_\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let syntheticWeight: [Float] = [1.0, 2.0, 3.0, 4.0]
        try SyntheticGGUFBuilder.write(
            to: tmpURL,
            architecture: "stub",
            tensors: ["stub.weight": syntheticWeight]
        )

        // Register the StubArchitectureFactory
        let registry = ModelRegistry()
        registry.register(StubArchitectureFactory())

        let model = try await EdgeRunnerModel.load(from: tmpURL, registry: registry)

        #expect(model.parameterNames == ["stub.weight"])
        // Verify weight loading actually occurred (StubModel tracks this)
        let stub = model as? StubModel
        #expect(stub != nil)
        #expect(stub?.weightsLoaded == true)
    }
}
```

**Step 2: Implement ModelFormat and ModelRegistry**

```swift
// Sources/EdgeRunnerIO/ModelRegistry.swift
import Foundation

/// Supported model file formats.
public enum ModelFormat: String, Sendable, Equatable {
    case gguf
    case safetensors
    case npz

    /// Detect format from file URL extension.
    public static func detect(from url: URL) -> ModelFormat? {
        switch url.pathExtension.lowercased() {
        case "gguf": return .gguf
        case "safetensors": return .safetensors
        case "npz": return .npz
        default: return nil
        }
    }
}

/// A Sendable metadata value extracted from model weight files.
public enum MetadataValue: Sendable, Equatable {
    case string(String)
    case int(Int)
    case float(Float)
    case bool(Bool)
    case array([MetadataValue])
}

/// Generic model configuration extracted from weight file metadata.
public struct ModelConfig: Sendable {
    public let architectureName: String
    public let metadata: [String: MetadataValue]

    public init(architectureName: String, metadata: [String: MetadataValue]) {
        self.architectureName = architectureName
        self.metadata = metadata
    }

    /// Convenience accessor for string metadata values.
    public func string(forKey key: String) -> String? {
        if case .string(let v) = metadata[key] { return v }
        return nil
    }

    /// Convenience accessor for integer metadata values.
    public func int(forKey key: String) -> Int? {
        if case .int(let v) = metadata[key] { return v }
        return nil
    }
}

/// Factory protocol for creating model instances from config.
public protocol ArchitectureFactory: Sendable {
    var architectureName: String { get }
    func create(config: ModelConfig) throws -> any LoadableModel
}

/// Llama architecture factory.
public struct LlamaArchitectureFactory: ArchitectureFactory, Sendable {
    public let architectureName: String = "llama"

    public init() {}

    public func create(config: ModelConfig) throws -> any LoadableModel {
        let llamaConfig = try LlamaConfig(fromGGUFMetadata: config.metadata)
        return LlamaModel(config: llamaConfig)
    }
}

/// Registry for model architectures.
public final class ModelRegistry: @unchecked Sendable {
    private var factories: [String: any ArchitectureFactory] = [:]
    private let lock = NSLock()

    public init() {}

    /// Default registry with built-in architectures.
    public static let `default`: ModelRegistry = {
        let registry = ModelRegistry()
        registry.register(LlamaArchitectureFactory())
        return registry
    }()

    /// Register an architecture factory.
    public func register(_ factory: any ArchitectureFactory) {
        lock.lock()
        defer { lock.unlock() }
        factories[factory.architectureName] = factory
    }

    /// Retrieve a factory by architecture name.
    public func factory(for name: String) -> (any ArchitectureFactory)? {
        lock.lock()
        defer { lock.unlock() }
        return factories[name]
    }
}
```

**Step 3: Implement the convenience load API**

```swift
// Sources/EdgeRunnerIO/EdgeRunnerModelLoader.swift
import Foundation

/// Errors from model loading.
public enum ModelLoadError: Error, Sendable {
    case unsupportedFormat(String)
    case unknownArchitecture(String)
    case loadFailed(description: String)
}

/// Convenience API for loading models.
public enum EdgeRunnerModel {
    /// Load a model from a file URL, auto-detecting format.
    ///
    /// Returns a type-erased `any LoadableModel` ready for inference.
    public static func load(
        from url: URL,
        registry: ModelRegistry = .default
    ) async throws -> any LoadableModel {
        guard let format = ModelFormat.detect(from: url) else {
            throw ModelLoadError.unsupportedFormat(url.pathExtension)
        }

        let weightLoader: any EdgeRunnerWeightLoader
        let architectureName: String
        var metadata: [String: String] = [:]

        switch format {
        case .gguf:
            let loader = try GGUFLoader(url: url)
            weightLoader = loader
            metadata = loader.stringMetadata
            architectureName = metadata["general.architecture"] ?? "llama"

        case .safetensors:
            let loader = try SafeTensorLoader(url: url)
            weightLoader = loader
            // SafeTensors may carry __metadata__.architecture; otherwise caller
            // must pass architecture explicitly via the registry parameter.
            architectureName = loader.metadata?["architecture"] ?? {
                // Infer from weight names if possible (e.g., "model.layers.0" → llama)
                let names = loader.tensorNames
                if names.contains(where: { $0.hasPrefix("model.layers.") }) {
                    return "llama"
                }
                throw ModelLoadError.unknownArchitecture(
                    "SafeTensor file has no architecture metadata. " +
                    "Pass a registry with the correct factory, or add __metadata__.architecture to the file."
                )
            }()

        case .npz:
            let loader = try NPZLoader(url: url)
            weightLoader = loader
            architectureName = loader.metadata?["architecture"] ?? {
                throw ModelLoadError.unknownArchitecture(
                    "NPZ file has no architecture metadata. " +
                    "Pass a registry with the correct factory."
                )
            }()
        }

        guard let factory = registry.factory(for: architectureName) else {
            throw ModelLoadError.unknownArchitecture(architectureName)
        }

        let config = ModelConfig(
            architectureName: architectureName,
            metadata: metadata
        )

        var model = try factory.create(config: config)
        let weightMap = try await weightLoader.load(from: url)
        try model.loadWeights(from: weightMap)
        return model
    }
}
```

**Step 4: Run tests and verify they pass**

Run: `swift test --filter "ModelRegistryTests|ModelLoaderTests" 2>&1`
Expected: All tests PASS.

**Step 5: Commit**

```
feat(models): add convenience load API with format detection and ModelRegistry
```

---

## Task 11: Memory Pressure Handler

**Files:**
- Create: `Sources/EdgeRunnerIO/MemoryPressureHandler.swift`
- Create: `Sources/EdgeRunnerIO/EdgeRunnerMemoryPolicy.swift`
- Test: `Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift`
- Test: `Tests/EdgeRunnerIOTests/MemoryPolicyTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/EdgeRunnerIOTests/MemoryPressureHandlerTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

@Suite("Memory Pressure Handler Tests")
struct MemoryPressureHandlerTests: Sendable {

    @Test("Default fallback chain: Q8 -> Q4_K_M -> Q4_0")
    func defaultFallbackChain() {
        let policy = EdgeRunnerMemoryPolicy.default
        #expect(policy.fallbackChain == [.q8_0, .q4_k_m, .q4_0])
    }

    @Test("Handler triggers fallback on simulated pressure")
    func simulatedPressure() async throws {
        let handler = MemoryPressureHandler(
            policy: .default
        )

        // Start at highest quality
        #expect(handler.currentQuantisation == .q8_0)

        // Simulate memory pressure
        await handler.simulateMemoryWarning()

        // Should fall back one level
        #expect(handler.currentQuantisation == .q4_k_m)

        // Another pressure event
        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Handler does not fall back past minimum")
    func minimumFallback() async throws {
        let handler = MemoryPressureHandler(
            policy: .default
        )

        // Push all the way down
        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()
        await handler.simulateMemoryWarning()

        // Should stay at Q4_0
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Custom policy with restricted fallback chain")
    func customPolicy() async throws {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0, .q4_0],
            evictBufferCacheOnPressure: false
        )
        let handler = MemoryPressureHandler(policy: policy)

        #expect(handler.currentQuantisation == .q8_0)
        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_0)
    }

    @Test("Buffer cache eviction flag is respected")
    func bufferCacheEviction() async throws {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0, .q4_0],
            evictBufferCacheOnPressure: true
        )
        let handler = MemoryPressureHandler(policy: policy)

        var evictionCount = 0
        handler.onBufferCacheEviction = {
            evictionCount += 1
        }

        await handler.simulateMemoryWarning()
        #expect(evictionCount == 1)
    }

    @Test("Handler can reset to highest quality")
    func reset() async throws {
        let handler = MemoryPressureHandler(policy: .default)

        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_k_m)

        handler.reset()
        #expect(handler.currentQuantisation == .q8_0)
    }
}

// Tests/EdgeRunnerIOTests/MemoryPolicyTests.swift
import Testing
@testable import EdgeRunnerIO

@Suite("Memory Policy Tests")
struct MemoryPolicyTests: Sendable {

    @Test("QuantisationLevel ordering")
    func quantisationOrdering() {
        #expect(QuantisationLevel.q8_0.bitsPerWeight > QuantisationLevel.q4_k_m.bitsPerWeight)
        #expect(QuantisationLevel.q4_k_m.bitsPerWeight > QuantisationLevel.q4_0.bitsPerWeight)
    }

    @Test("Policy with empty fallback chain uses Q4_0")
    func emptyFallbackChain() {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [],
            evictBufferCacheOnPressure: false
        )
        let handler = MemoryPressureHandler(policy: policy)
        #expect(handler.currentQuantisation == .q4_0) // Absolute minimum
    }

    @Test("Custom eviction threshold")
    func customEvictionThreshold() {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0],
            evictBufferCacheOnPressure: true,
            maxMemoryBytes: 2 * 1024 * 1024 * 1024 // 2 GB
        )
        #expect(policy.maxMemoryBytes == 2 * 1024 * 1024 * 1024)
    }
}
```

**Step 2: Implement QuantisationLevel and EdgeRunnerMemoryPolicy**

```swift
// Sources/EdgeRunnerIO/EdgeRunnerMemoryPolicy.swift
import Foundation

/// Available quantisation levels for runtime precision selection.
public enum QuantisationLevel: String, Sendable, Equatable, CaseIterable {
    case q8_0
    case q4_k_m
    case q4_0

    /// Approximate bits per weight for memory estimation.
    public var bitsPerWeight: Double {
        switch self {
        case .q8_0: return 8.0
        case .q4_k_m: return 4.5
        case .q4_0: return 4.0
        }
    }
}

/// Developer-configurable memory management policy.
public struct EdgeRunnerMemoryPolicy: Sendable {
    /// Quantisation levels to try, in order from highest quality to lowest.
    public let fallbackChain: [QuantisationLevel]

    /// Whether to evict Metal buffer caches under memory pressure.
    public let evictBufferCacheOnPressure: Bool

    /// Maximum memory budget in bytes (0 = no limit, use system heuristics).
    public let maxMemoryBytes: Int

    public init(
        fallbackChain: [QuantisationLevel],
        evictBufferCacheOnPressure: Bool,
        maxMemoryBytes: Int = 0
    ) {
        self.fallbackChain = fallbackChain
        self.evictBufferCacheOnPressure = evictBufferCacheOnPressure
        self.maxMemoryBytes = maxMemoryBytes
    }

    /// Default policy: Q8 -> Q4_K_M -> Q4_0 with buffer eviction enabled.
    public static let `default` = EdgeRunnerMemoryPolicy(
        fallbackChain: [.q8_0, .q4_k_m, .q4_0],
        evictBufferCacheOnPressure: true
    )
}
```

**Step 3: Implement MemoryPressureHandler**

```swift
// Sources/EdgeRunnerIO/MemoryPressureHandler.swift
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Handles memory pressure events and adjusts quantisation level.
public final class MemoryPressureHandler: @unchecked Sendable {
    private let policy: EdgeRunnerMemoryPolicy
    private let lock = NSLock()
    private var currentIndex: Int

    /// Callback invoked when buffer cache eviction is triggered.
    public var onBufferCacheEviction: (() -> Void)?

    /// The current quantisation level.
    public var currentQuantisation: QuantisationLevel {
        lock.lock()
        defer { lock.unlock() }
        if policy.fallbackChain.isEmpty { return .q4_0 }
        return policy.fallbackChain[currentIndex]
    }

    public init(policy: EdgeRunnerMemoryPolicy) {
        self.policy = policy
        self.currentIndex = 0

        #if canImport(UIKit) && !targetEnvironment(simulator)
        registerForSystemMemoryWarnings()
        #endif
    }

    /// Register for iOS memory warning notifications.
    #if canImport(UIKit)
    private func registerForSystemMemoryWarnings() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { [weak self] in
                await self?.handleMemoryPressure()
            }
        }
    }
    #endif

    /// Handle a memory pressure event: step down the fallback chain.
    @MainActor
    public func handleMemoryPressure() {
        lock.lock()
        let canFallBack = currentIndex < policy.fallbackChain.count - 1
        if canFallBack {
            currentIndex += 1
        }
        let shouldEvict = policy.evictBufferCacheOnPressure
        lock.unlock()

        if shouldEvict {
            onBufferCacheEviction?()
        }
    }

    /// Simulate a memory warning for testing.
    public func simulateMemoryWarning() async {
        await MainActor.run {
            handleMemoryPressure()
        }
    }

    /// Reset to highest quality (first in fallback chain).
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        currentIndex = 0
    }
}
```

**Step 4: Run tests and verify they pass**

Run: `swift test --filter "MemoryPressureHandlerTests|MemoryPolicyTests" 2>&1`
Expected: All tests PASS.

**Step 5: Commit**

```
feat(runtime): add memory pressure handler with quantisation fallback chain and cache eviction
```

---

## Task 12: End-to-End Integration & Verification

**Files:**
- Create: `Tests/EdgeRunnerIntegrationTests/EndToEndLoadTests.swift`
- Create: `Tests/EdgeRunnerIntegrationTests/PerformanceBenchmarkTests.swift`
- Create: `Tests/EdgeRunnerIntegrationTests/PerplexityVerificationTests.swift`

**Step 1: Write integration tests**

```swift
// Tests/EdgeRunnerIntegrationTests/EndToEndLoadTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO
@testable import EdgeRunnerIO
@testable import EdgeRunnerIO

@Suite("End-to-End Load Tests")
struct EndToEndLoadTests: Sendable {

    @Test("Load GGUF -> construct model -> verify parameter names")
    func ggufToModel() throws {
        // Build a minimal synthetic GGUF with Llama metadata
        let metadata: [String: MetadataValue] = [
            "general.architecture": "llama",
            "llama.embedding_length": 64,
            "llama.block_count": 2,
            "llama.attention.head_count": 4,
            "llama.attention.head_count_kv": 2,
            "llama.vocab_size": 100,
            "llama.feed_forward_length": 128,
            "llama.rope.freq_base": 10000.0,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
        ]

        let config = try LlamaConfig(fromGGUFMetadata: metadata)
        let model = LlamaModel(config: config)

        // Verify it constructed correctly
        #expect(model.vocabSize == 100)
        #expect(model.layers.count == 2)

        // Verify all expected parameter names exist
        let names = model.parameterNames
        #expect(names.contains("embedding.weight"))
        #expect(names.contains("lmHead.weight"))
        #expect(names.contains("layers.0.attention.wq.weight"))
        #expect(names.contains("layers.1.feedForward.down.weight"))
    }

    @Test("SafeTensor round-trip: write -> load -> verify tensor data")
    func safetensorRoundTrip() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let floatData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        // Build synthetic safetensor header + data
        var headerDict: [String: Any] = [:]
        headerDict["test_tensor"] = [
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        ] as [String: Any]

        let jsonData = try JSONSerialization.data(
            withJSONObject: headerDict,
            options: [.sortedKeys]
        )
        var blob = Data()
        var headerSize = UInt64(jsonData.count)
        blob.append(Data(bytes: &headerSize, count: 8))
        blob.append(jsonData)
        blob.append(floatData)

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try SafeTensorLoader(url: tmpURL)
        let storage = try loader.loadTensor(named: "test_tensor")

        #expect(storage.shape == [2, 2])
        #expect(storage.dtype == .float32)
        #expect(storage.byteCount == 16)
    }

    @Test("Format detection works for all supported formats")
    func formatDetection() {
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "m.gguf")) == .gguf)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "m.safetensors")) == .safetensors)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "m.npz")) == .npz)
        #expect(ModelFormat.detect(from: URL(fileURLWithPath: "m.bin")) == nil)
    }

    @Test("Memory pressure integration: handler adjusts during model lifecycle")
    func memoryPressureIntegration() async throws {
        let policy = EdgeRunnerMemoryPolicy(
            fallbackChain: [.q8_0, .q4_k_m, .q4_0],
            evictBufferCacheOnPressure: true
        )
        let handler = MemoryPressureHandler(policy: policy)

        // Model starts at Q8
        #expect(handler.currentQuantisation == .q8_0)

        // Pressure forces fallback
        await handler.simulateMemoryWarning()
        #expect(handler.currentQuantisation == .q4_k_m)

        // Reset recovers
        handler.reset()
        #expect(handler.currentQuantisation == .q8_0)
    }

    @Test("Weight name mapping round-trip: GGUF names -> EdgeRunner names")
    func weightNameMappingRoundTrip() {
        let ggufNames = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
            "output.weight",
        ]

        let expectedNames = [
            "embedding.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.feedForward.gate.weight",
            "layers.0.feedForward.up.weight",
            "layers.0.feedForward.down.weight",
            "layers.0.attentionNorm.weight",
            "layers.0.ffnNorm.weight",
            "finalNorm.weight",
            "lmHead.weight",
        ]

        for (gguf, expected) in zip(ggufNames, expectedNames) {
            let mapped = LlamaWeightNameMapper.mapGGUFName(gguf)
            #expect(mapped == expected, "Mapping \(gguf) -> expected \(expected), got \(mapped)")
        }
    }
}
```

```swift
// Tests/EdgeRunnerIntegrationTests/PerformanceBenchmarkTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO
@testable import EdgeRunnerIO

@Suite("Performance Benchmark Tests")
struct PerformanceBenchmarkTests: Sendable {

    @Test("SafeTensor loader performance: 1000 tensors under 1 second")
    func safetensorLoadPerformance() throws {
        // Build a synthetic safetensor with many small tensors
        var headerDict: [String: Any] = [:]
        var dataSection = Data()

        for i in 0..<1000 {
            let tensorSize = 64 * 4 // 64 float32s = 256 bytes
            let begin = dataSection.count
            dataSection.append(Data(repeating: 0, count: tensorSize))
            let end = dataSection.count
            headerDict["tensor_\(i)"] = [
                "dtype": "F32",
                "shape": [8, 8],
                "data_offsets": [begin, end]
            ] as [String: Any]
        }

        let jsonData = try JSONSerialization.data(
            withJSONObject: headerDict,
            options: [.sortedKeys]
        )
        var blob = Data()
        var headerSize = UInt64(jsonData.count)
        blob.append(Data(bytes: &headerSize, count: 8))
        blob.append(jsonData)
        blob.append(dataSection)

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try blob.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let start = CFAbsoluteTimeGetCurrent()
        let loader = try SafeTensorLoader(url: tmpURL)
        let names = loader.tensorNames
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(names.count == 1000)
        #expect(elapsed < 1.0, "Header parsing took \(elapsed)s, expected < 1s")
    }

    @Test("LlamaModel construction performance: 32 layers under 100ms")
    func modelConstructionPerformance() throws {
        let config = LlamaConfig(
            embeddingDim: 4096,
            layerCount: 32,
            headCount: 32,
            kvHeadCount: 8,
            vocabSize: 128256,
            intermediateDim: 14336,
            ropeFreqBase: 500000.0,
            rmsNormEpsilon: 1e-5
        )

        let start = CFAbsoluteTimeGetCurrent()
        let model = LlamaModel(config: config)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(model.layers.count == 32)
        #expect(elapsed < 0.1, "Model construction took \(elapsed)s, expected < 100ms")
    }

    @Test("Memory estimate for Llama 3 8B at different quantisation levels")
    func memoryEstimation() {
        let paramCount: Double = 8_000_000_000 // 8B params

        let q8Memory = paramCount * QuantisationLevel.q8_0.bitsPerWeight / 8.0
        let q4kmMemory = paramCount * QuantisationLevel.q4_k_m.bitsPerWeight / 8.0
        let q4Memory = paramCount * QuantisationLevel.q4_0.bitsPerWeight / 8.0

        // Q8_0: ~8 GB
        #expect(q8Memory > 7_500_000_000 && q8Memory < 8_500_000_000)
        // Q4_K_M: ~4.5 GB
        #expect(q4kmMemory > 4_000_000_000 && q4kmMemory < 5_000_000_000)
        // Q4_0: ~4 GB
        #expect(q4Memory > 3_500_000_000 && q4Memory < 4_500_000_000)
    }
}
```

```swift
// Tests/EdgeRunnerIntegrationTests/PerplexityVerificationTests.swift
import Testing
import Foundation
@testable import EdgeRunnerIO

@Suite("Perplexity Verification Tests")
struct PerplexityVerificationTests: Sendable {

    @Test("Softmax produces valid probability distribution")
    func softmaxValidation() {
        // Reference softmax for perplexity computation
        let logits: [Float] = [2.0, 1.0, 0.1, -1.0, 3.0]
        let maxLogit = logits.max()!
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExp }

        // Sum of probabilities should be ~1.0
        let sum = probs.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5)

        // All probabilities should be positive
        for p in probs {
            #expect(p > 0)
        }
    }

    @Test("Perplexity computation from logits is numerically stable")
    func perplexityComputation() {
        // Simulate logits for 10 tokens
        let tokenCount = 10
        var totalNLL: Float = 0

        for i in 0..<tokenCount {
            // Fake logits: correct token gets highest score
            var logits = [Float](repeating: -5.0, count: 100)
            logits[i % 100] = 5.0

            let maxLogit = logits.max()!
            let exps = logits.map { exp($0 - maxLogit) }
            let sumExp = exps.reduce(0, +)
            let logProb = (logits[i % 100] - maxLogit) - log(sumExp)
            totalNLL -= logProb
        }

        let perplexity = exp(totalNLL / Float(tokenCount))

        // With near-perfect predictions, perplexity should be close to 1
        #expect(perplexity > 0.9 && perplexity < 2.0,
                "Perplexity \(perplexity) should be near 1.0 for confident predictions")
    }

    @Test("Cross-entropy loss matches expected value")
    func crossEntropyLoss() {
        // Uniform distribution over 100 classes
        let logits = [Float](repeating: 0.0, count: 100)
        let targetIndex = 42

        let maxLogit = logits.max()!
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        let logProb = (logits[targetIndex] - maxLogit) - log(sumExp)
        let loss = -logProb

        // Loss for uniform = log(100) ~ 4.605
        #expect(abs(loss - log(Float(100))) < 1e-4)
    }
}
```

**Step 2: Run all integration tests**

Run: `swift test --filter "EndToEndLoadTests|PerformanceBenchmarkTests|PerplexityVerificationTests" 2>&1`
Expected: All tests PASS.

**Step 3: Run full M3 test suite**

Run: `swift test 2>&1`
Expected: All M3 tests pass alongside M1 and M2 tests.

**Step 4: Commit**

```
test(integration): add end-to-end load, performance benchmark, and perplexity verification tests
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 7 | SafeTensor Loader | `SafeTensorHeader.swift`, `SafeTensorLoader.swift`, `SafeTensorLoaderTests.swift` | ~6 |
| 8 | NPZ Loader | `NPYParser.swift`, `NPZLoader.swift`, `NPZLoaderTests.swift` | ~6 |
| 9 | Llama 3 Architecture | `LlamaConfig.swift`, `LlamaModel.swift`, `LlamaBlock.swift`, `EdgeRunnerModule.swift`, `LoadableModel.swift`, `LlamaConfigTests.swift`, `LlamaBlockTests.swift`, `LlamaModelTests.swift` | ~8 |
| 10 | Convenience Load API | `EdgeRunnerModelLoader.swift`, `ModelRegistry.swift`, `ModelLoaderTests.swift`, `ModelRegistryTests.swift` | ~9 |
| 11 | Memory Pressure Handler | `MemoryPressureHandler.swift`, `EdgeRunnerMemoryPolicy.swift`, `MemoryPressureHandlerTests.swift`, `MemoryPolicyTests.swift` | ~9 |
| 12 | End-to-End Integration | `EndToEndLoadTests.swift`, `PerformanceBenchmarkTests.swift`, `PerplexityVerificationTests.swift` | ~11 |

**Total: ~22 files, ~49 tests, 6 commits (Tasks 7-12) + 6 commits (Tasks 1-6) = 12 commits**
