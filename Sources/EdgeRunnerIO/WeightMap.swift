import Foundation
import Metal

public enum MetadataValue: Sendable, Equatable,
    ExpressibleByStringLiteral,
    ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral,
    ExpressibleByBooleanLiteral,
    ExpressibleByArrayLiteral
{
    case string(String)
    case int(Int)
    case float(Float)
    case bool(Bool)
    case array([MetadataValue])

    public init(stringLiteral value: String) {
        self = .string(value)
    }

    public init(integerLiteral value: Int) {
        self = .int(value)
    }

    public init(floatLiteral value: Double) {
        self = .float(Float(value))
    }

    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }

    public init(arrayLiteral elements: MetadataValue...) {
        self = .array(elements)
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var intValue: Int? {
        guard case .int(let value) = self else { return nil }
        return value
    }

    public var floatValue: Float? {
        guard case .float(let value) = self else { return nil }
        return value
    }

    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    public var arrayValue: [MetadataValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    public var floatArrayValue: [Float]? {
        guard let values = arrayValue else { return nil }
        let floats = values.compactMap(\.floatValue)
        return floats.count == values.count ? floats : nil
    }
}

public enum TensorDataType: UInt32, Sendable, Equatable {
    case float32 = 0
    case float16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case q8_K = 15
    case i8 = 16
    case i16 = 17
    case i32 = 18
    case i64 = 19
    case f64 = 20
    case bfloat16 = 30
    case q1_0_g128 = 41
}

public struct TensorStorage: @unchecked Sendable {
    // @unchecked Sendable: wraps an immutable MTLBuffer (non-Sendable protocol) plus
    // a Sendable owner reference. All fields are let-bound and never mutated after init.
    public let buffer: MTLBuffer
    public let byteOffset: Int
    public let dataType: TensorDataType
    public let shape: [Int]
    public let name: String

    private let owner: (any Sendable)?

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public var byteCount: Int {
        buffer.length
    }

    public init(
        buffer: MTLBuffer,
        byteOffset: Int = 0,
        dataType: TensorDataType,
        shape: [Int],
        name: String,
        owner: (any Sendable & AnyObject)? = nil
    ) {
        self.buffer = buffer
        self.byteOffset = byteOffset
        self.dataType = dataType
        self.shape = shape
        self.name = name
        self.owner = owner
    }
}

public struct WeightMap: Sendable {
    private var storage: [String: TensorStorage] = [:]

    public init() {}

    public var count: Int {
        storage.count
    }

    public var tensorNames: [String] {
        storage.keys.sorted()
    }

    public var totalBytes: Int {
        storage.values.reduce(0) { partialResult, tensor in
            partialResult + tensor.byteCount
        }
    }

    public subscript(name: String) -> TensorStorage? {
        get { storage[name] }
        set { storage[name] = newValue }
    }
}
