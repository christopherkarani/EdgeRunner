import Testing
@testable import EdgeRunnerCore

@Suite("Shape")
struct ShapeTests {

    @Test func initAndProperties() {
        let s = Shape([2, 3, 4])
        #expect(s.rank == 3)
        #expect(s.elementCount == 24)
        #expect(s.dimensions == [2, 3, 4])
    }

    @Test func scalarShape() {
        let s = Shape([])
        #expect(s.rank == 0)
        #expect(s.elementCount == 1)
    }

    @Test func contiguousStrides() {
        let strides = Strides.contiguous(for: Shape([2, 3, 4]))
        #expect(strides.values == [12, 4, 1])
    }

    @Test func contiguousStridesVector() {
        let strides = Strides.contiguous(for: Shape([5]))
        #expect(strides.values == [1])
    }

    @Test func isContiguous() {
        let s = Strides(values: [12, 4, 1])
        #expect(s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func isNotContiguous() {
        let s = Strides(values: [12, 1, 3])
        #expect(!s.isContiguous(for: Shape([2, 3, 4])))
    }

    @Test func broadcastCompatible() throws {
        let a = Shape([2, 3, 4])
        let b = Shape([1, 3, 4])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastCompatibleScalar() throws {
        let a = Shape([2, 3])
        let b = Shape([])
        #expect(a.broadcastCompatible(with: b))
    }

    @Test func broadcastIncompatible() throws {
        let a = Shape([2, 3])
        let b = Shape([2, 4])
        #expect(!a.broadcastCompatible(with: b))
    }

    @Test func broadcastedShape() throws {
        let a = Shape([2, 1, 4])
        let b = Shape([3, 4])
        let result = try a.broadcasted(with: b)
        #expect(result.dimensions == [2, 3, 4])
    }

    @Test func broadcastedShapeError() throws {
        let a = Shape([2, 3])
        let b = Shape([2, 4])
        #expect(throws: ShapeError.self) {
            try a.broadcasted(with: b)
        }
    }
}
