import Metal
import EdgeRunnerMetal

public struct Tensor<T: TensorScalar>: Sendable {
    private var _storage: TensorStorage
    public let shape: Shape
    public let strides: Strides

    public var elementCount: Int { shape.elementCount }
    public var byteCount: Int { elementCount * T.byteSize }

    public init(data: [T], shape: Shape) {
        precondition(
            data.count == shape.elementCount,
            "Data count \(data.count) doesn't match shape \(shape) (expected \(shape.elementCount))"
        )
        self._storage = TensorStorage.from(data)
        self.shape = shape
        self.strides = Strides.contiguous(for: shape)
    }

    public init(scalar: T) {
        self.init(data: [scalar], shape: Shape([]))
    }

    init(storage: TensorStorage, shape: Shape, strides: Strides) {
        self._storage = storage
        self.shape = shape
        self.strides = strides
    }

    public static func zeros(shape: Shape) -> Tensor {
        let storage = TensorStorage.zeros(byteCount: shape.elementCount * T.byteSize)
        return Tensor(storage: storage, shape: shape, strides: .contiguous(for: shape))
    }

    public static func ones(shape: Shape) -> Tensor {
        let data = [T](repeating: T.zero, count: shape.elementCount)
            .enumerated()
            .map { _ in _one() }
        return Tensor(data: data, shape: shape)
    }

    public func toArray() -> [T] {
        _storage.toArray(count: elementCount)
    }

    var metalBuffer: MTLBuffer { _storage.buffer.rawValue }

    public func reshape(_ newShape: Shape) throws -> Tensor {
        guard newShape.elementCount == shape.elementCount else {
            throw ShapeError.invalidReshape(from: shape, to: newShape)
        }
        return Tensor(storage: _storage, shape: newShape, strides: .contiguous(for: newShape))
    }

    mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&_storage) {
            _storage = _storage.copy()
        }
    }

    private static func _one() -> T {
        if T.self == Float.self { return unsafeBitCast(Float(1.0), to: T.self) }
        if T.self == Float16.self { return unsafeBitCast(Float16(1.0), to: T.self) }
        if T.self == Int8.self { return unsafeBitCast(Int8(1), to: T.self) }
        if T.self == UInt8.self { return unsafeBitCast(UInt8(1), to: T.self) }
        return T.zero
    }
}
