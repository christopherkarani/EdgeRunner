import Testing
@testable import EdgeRunnerCore

@Suite("ComputeGraph")
struct ComputeGraphTests {

    @Test func singleOp() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let sorted = ComputeGraph.topologicalSort(root: add)
        #expect(sorted.count == 3)
        #expect(sorted[2].id == add.id)
    }

    @Test func chainedOps() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let sorted = ComputeGraph.topologicalSort(root: relu)
        #expect(sorted.count == 4)
        #expect(sorted.last?.id == relu.id)
    }

    @Test func fusionGroupIdentification() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))
        let sigmoid = TensorOp.unary(.sigmoid, input: relu, outputShape: Shape([4]))
        let sorted = ComputeGraph.topologicalSort(root: sigmoid)
        let groups = ComputeGraph.identifyFusionGroups(sorted)
        let fusedGroup = groups.first(where: { $0.count > 1 })
        #expect(fusedGroup != nil)
        #expect(fusedGroup!.count == 3)
    }

    @Test func fusionRespectsDifferentShapes() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let sum = TensorOp.reduction(.sum, input: add, outputShape: Shape([1]))
        let sorted = ComputeGraph.topologicalSort(root: sum)
        let groups = ComputeGraph.identifyFusionGroups(sorted)
        let fusedGroup = groups.first(where: { $0.count > 1 })
        #expect(fusedGroup == nil)
    }

    @Test func fusionRespectsDepthLimit() throws {
        var current = TensorOp.input(id: 0, shape: Shape([4]))
        for _ in 0..<15 {
            current = TensorOp.unary(.relu, input: current, outputShape: Shape([4]))
        }
        let sorted = ComputeGraph.topologicalSort(root: current)
        let groups = ComputeGraph.identifyFusionGroups(sorted)
        let fusedGroups = groups.filter { $0.count > 1 }
        #expect(fusedGroups.count >= 2)
        for group in fusedGroups {
            #expect(group.count <= ComputeGraph.maxFusionDepth)
        }
    }
}
