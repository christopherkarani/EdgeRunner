import Testing
@testable import EdgeRunnerCore

@Suite("FusionEngine")
struct FusionEngineTests {

    @Test func selectsHotPathForAddRelu() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let tier = FusionEngine.selectTier(for: [add, relu])
        switch tier {
        case .functionConstants: break
        default: Issue.record("Expected .functionConstants, got \(tier)")
        }
    }

    @Test func selectsStitchForLongChain() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let sigmoid = TensorOp.unary(.sigmoid, input: relu, outputShape: Shape([4]))
        let neg = TensorOp.unary(.neg, input: sigmoid, outputShape: Shape([4]))
        let tier = FusionEngine.selectTier(for: [add, relu, sigmoid, neg])
        switch tier {
        case .stitched: break
        default: Issue.record("Expected .stitched, got \(tier)")
        }
    }

    @Test func selectsSingleOpAsFunctionConstant() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: a, outputShape: Shape([4]))
        let tier = FusionEngine.selectTier(for: [relu])
        switch tier {
        case .functionConstants: break
        default: Issue.record("Expected .functionConstants, got \(tier)")
        }
    }

    @Test func tierDescriptions() {
        let hot = FusionTier.functionConstants(kernelName: "fused_add_activate_float", activationType: 1)
        let warm = FusionTier.stitched(ops: [.relu, .sigmoid])
        let cold = FusionTier.jit(source: "kernel void ...")
        #expect(hot.isHotPath)
        #expect(!warm.isHotPath)
        #expect(!cold.isHotPath)
    }
}
