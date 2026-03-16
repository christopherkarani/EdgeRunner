public enum ShapeError: Error, Sendable {
    case incompatibleBroadcast(Shape, Shape)
    case invalidReshape(from: Shape, to: Shape)
}

public struct Shape: Sendable, Equatable, Hashable, CustomStringConvertible {
    public let dimensions: [Int]

    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions
    }

    public var rank: Int { dimensions.count }

    public var elementCount: Int {
        dimensions.isEmpty ? 1 : dimensions.reduce(1, *)
    }

    public var description: String {
        "Shape(\(dimensions))"
    }

    public func broadcastCompatible(with other: Shape) -> Bool {
        let maxRank = max(rank, other.rank)
        for i in 0..<maxRank {
            let dimA = i < rank ? dimensions[rank - 1 - i] : 1
            let dimB = i < other.rank ? other.dimensions[other.rank - 1 - i] : 1
            if dimA != dimB && dimA != 1 && dimB != 1 {
                return false
            }
        }
        return true
    }

    public func broadcasted(with other: Shape) throws -> Shape {
        let maxRank = max(rank, other.rank)
        var result = [Int]()
        result.reserveCapacity(maxRank)

        for i in 0..<maxRank {
            let dimA = i < rank ? dimensions[rank - 1 - i] : 1
            let dimB = i < other.rank ? other.dimensions[other.rank - 1 - i] : 1
            if dimA == dimB {
                result.append(dimA)
            } else if dimA == 1 {
                result.append(dimB)
            } else if dimB == 1 {
                result.append(dimA)
            } else {
                throw ShapeError.incompatibleBroadcast(self, other)
            }
        }
        result.reverse()
        return Shape(result)
    }
}

public struct Strides: Sendable, Equatable, CustomStringConvertible {
    public let values: [Int]

    public init(values: [Int]) {
        self.values = values
    }

    public var description: String {
        "Strides(\(values))"
    }

    public static func contiguous(for shape: Shape) -> Strides {
        let dims = shape.dimensions
        guard !dims.isEmpty else { return Strides(values: []) }

        var strides = [Int](repeating: 1, count: dims.count)
        for i in stride(from: dims.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * dims[i + 1]
        }
        return Strides(values: strides)
    }

    public func isContiguous(for shape: Shape) -> Bool {
        self == Strides.contiguous(for: shape)
    }
}
