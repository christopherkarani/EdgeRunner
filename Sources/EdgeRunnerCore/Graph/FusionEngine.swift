public enum FusionTier: Sendable {
    case functionConstants(kernelName: String, activationType: Int32)
    case stitched(ops: [UnaryOp])
    case jit(source: String)

    public var isHotPath: Bool {
        if case .functionConstants = self { return true }
        return false
    }
}

public enum FusionEngine {
    private static let hotPathPatterns: Set<String> = [
        "add", "add+relu", "add+sigmoid", "add+gelu", "add+silu",
        "mul", "mul+relu", "mul+sigmoid", "mul+gelu", "mul+silu",
        "relu", "sigmoid", "gelu", "silu",
    ]

    private static let activationMap: [UnaryOp: Int32] = [
        .relu: 1, .sigmoid: 2, .gelu: 3, .silu: 4,
    ]

    public static func selectTier(for ops: [TensorOp]) -> FusionTier {
        let pattern = patternKey(for: ops)

        if hotPathPatterns.contains(pattern) {
            return buildFunctionConstantsTier(ops: ops, pattern: pattern)
        }

        // Stitched: any group that contains at least one unary op
        // (may also contain a leading binary op — e.g. add+relu+sigmoid+neg)
        let unaryOps = extractUnaryOps(from: ops)
        if let unaryOps {
            return .stitched(ops: unaryOps)
        }

        let source = generateMSL(for: ops)
        return .jit(source: source)
    }

    // MARK: - Private helpers

    private static func patternKey(for ops: [TensorOp]) -> String {
        ops.compactMap { op in
            switch op.kind {
            case .unary(let unaryOp, _): return unaryOp.rawValue
            case .binary(let binOp, _, _): return binOp.rawValue
            default: return nil
            }
        }.joined(separator: "+")
    }

    private static func buildFunctionConstantsTier(ops: [TensorOp], pattern: String) -> FusionTier {
        let parts = pattern.split(separator: "+")
        if parts.count == 1 {
            let opName = String(parts[0])
            if let activation = UnaryOp(rawValue: opName), let actValue = activationMap[activation] {
                return .functionConstants(kernelName: "fused_activate_float", activationType: actValue)
            }
            return .functionConstants(kernelName: "elementwise_\(opName)_float", activationType: 0)
        }
        let binOp = String(parts[0])
        let activation = parts.count > 1 ? UnaryOp(rawValue: String(parts[1])) : nil
        let actValue = activation.flatMap { activationMap[$0] } ?? 0
        return .functionConstants(kernelName: "fused_\(binOp)_activate_float", activationType: actValue)
    }

    /// Collects all unary ops from the group.
    /// Returns `nil` only when there are no unary ops at all (e.g. pure reduction group).
    private static func extractUnaryOps(from ops: [TensorOp]) -> [UnaryOp]? {
        var unaryOps = [UnaryOp]()
        for op in ops {
            switch op.kind {
            case .unary(let unaryOp, _):
                unaryOps.append(unaryOp)
            case .binary:
                // Binary ops are allowed as the leading element; skip them.
                continue
            default:
                // Reductions or inputs — cannot be stitched.
                return nil
            }
        }
        return unaryOps.isEmpty ? nil : unaryOps
    }

    private static func generateMSL(for ops: [TensorOp]) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void jit_fused(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
            if (tid >= count) return;
            float x = in[tid];
            out[tid] = x;
        }
        """
    }
}
