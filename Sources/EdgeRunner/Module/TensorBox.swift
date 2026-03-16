/// Type-erased container for tensor parameter data.
/// Enables heterogeneous parameter dictionaries across module types.
public protocol TensorBox: Sendable {
    /// Total number of scalar elements in the tensor.
    var elementCount: Int { get }

    /// Returns a flat array of Float values.
    var floatArray: [Float] { get }

    /// Shape of the underlying tensor.
    var shape: [Int] { get }
}

/// A single scalar value wrapped as a TensorBox.
public struct ScalarTensorBox: TensorBox, Sendable {
    public let value: Float

    public init(value: Float) {
        self.value = value
    }

    public var elementCount: Int { 1 }
    public var floatArray: [Float] { [value] }
    public var shape: [Int] { [] }
}

/// A flat array of floats wrapped as a TensorBox.
public struct ArrayTensorBox: TensorBox, Sendable {
    public let data: [Float]
    public let shape: [Int]

    public init(data: [Float], shape: [Int]) {
        precondition(
            data.count == shape.reduce(1, *),
            "Data count \(data.count) doesn't match shape \(shape)"
        )
        self.data = data
        self.shape = shape
    }

    public var elementCount: Int { data.count }
    public var floatArray: [Float] { data }
}
