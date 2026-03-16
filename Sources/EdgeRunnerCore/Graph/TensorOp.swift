import Foundation

public enum UnaryOp: String, Sendable {
    case relu, sigmoid, gelu, silu
    case neg, abs, sqrt, exp, log, tanh
}

public enum BinaryOp: String, Sendable {
    case add, sub, mul, div
}

public enum ReductionOp: String, Sendable {
    case sum, mean, max
}

public final class TensorOp: Sendable {
    public let id: UUID
    public let kind: Kind
    public let outputShape: Shape

    public enum Kind: Sendable {
        case input
        case unary(UnaryOp, input: TensorOp)
        case binary(BinaryOp, lhs: TensorOp, rhs: TensorOp)
        case reduction(ReductionOp, input: TensorOp)
    }

    init(kind: Kind, outputShape: Shape) {
        self.id = UUID()
        self.kind = kind
        self.outputShape = outputShape
    }

    public static func input(id: Int, shape: Shape) -> TensorOp {
        TensorOp(kind: .input, outputShape: shape)
    }

    public static func unary(_ op: UnaryOp, input: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .unary(op, input: input), outputShape: outputShape)
    }

    public static func binary(_ op: BinaryOp, lhs: TensorOp, rhs: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .binary(op, lhs: lhs, rhs: rhs), outputShape: outputShape)
    }

    public static func reduction(_ op: ReductionOp, input: TensorOp, outputShape: Shape) -> TensorOp {
        TensorOp(kind: .reduction(op, input: input), outputShape: outputShape)
    }

    public var isElementwise: Bool {
        switch kind {
        case .input: return false
        case .unary: return true
        case .binary: return true
        case .reduction: return false
        }
    }

    public var inputs: [TensorOp] {
        switch kind {
        case .input: return []
        case .unary(_, let input): return [input]
        case .binary(_, let lhs, let rhs): return [lhs, rhs]
        case .reduction(_, let input): return [input]
        }
    }
}

extension TensorOp: Equatable {
    public static func == (lhs: TensorOp, rhs: TensorOp) -> Bool { lhs.id == rhs.id }
}

extension TensorOp: Hashable {
    public func hash(into hasher: inout Hasher) { hasher.combine(id) }
}
