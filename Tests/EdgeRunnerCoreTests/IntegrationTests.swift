import Testing
import Foundation
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Integration")
struct IntegrationTests {

    @Test func tensorAddOnGPU() async throws {
        let backend = MetalBackend.shared
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [5, 6, 7, 8]
        let result = try await backend.elementwiseAddFloat(a, b)
        #expect(result == [6, 8, 10, 12])
    }

    @Test func computeGraphBuildsCorrectly() throws {
        let a = TensorOp.input(id: 0, shape: Shape([4]))
        let b = TensorOp.input(id: 1, shape: Shape([4]))
        let add = TensorOp.binary(.add, lhs: a, rhs: b, outputShape: Shape([4]))
        let relu = TensorOp.unary(.relu, input: add, outputShape: Shape([4]))

        let sorted = ComputeGraph.topologicalSort(root: relu)
        let groups = ComputeGraph.identifyFusionGroups(sorted)
        let tier = FusionEngine.selectTier(for: groups.last!)

        switch tier {
        case .functionConstants(let name, let activation):
            #expect(name == "fused_add_activate_float")
            #expect(activation == 1)
        default:
            Issue.record("Expected function constants tier")
        }
    }

    @Test func bufferCacheRoundTrip() async throws {
        let backend = MetalBackend.shared
        let size = 512
        let actualSize = try await backend.acquireBufferSize(size: size)
        #expect(actualSize >= size)
    }

    @Test func cpuReferenceAdd() {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [5, 6, 7, 8]
        let expected: [Float] = zip(a, b).map(+)
        #expect(expected == [6, 8, 10, 12])
    }

    @Test func cpuReferenceRelu() {
        let input: [Float] = [-2, -1, 0, 1, 2]
        let expected: [Float] = input.map { max($0, 0) }
        #expect(expected == [0, 0, 0, 1, 2])
    }

    @Test func cpuReferenceSigmoid() {
        let input: [Float] = [0]
        let expected = 1.0 / (1.0 + Foundation.exp(-input[0]))
        #expect(abs(expected - 0.5) < 1e-6)
    }
}
