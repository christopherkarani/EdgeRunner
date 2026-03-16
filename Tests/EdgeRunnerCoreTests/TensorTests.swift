import Testing
import Metal
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Tensor")
struct TensorTests {

    @Test func createFromArray() async throws {
        let t = Tensor<Float>(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: Shape([2, 3]))
        #expect(t.shape == Shape([2, 3]))
        #expect(t.strides == Strides.contiguous(for: Shape([2, 3])))
        #expect(t.elementCount == 6)
    }

    @Test func createZeros() async throws {
        let t = Tensor<Float>.zeros(shape: Shape([3, 4]))
        #expect(t.shape == Shape([3, 4]))
        #expect(t.elementCount == 12)
    }

    @Test func createOnes() async throws {
        let t = Tensor<Float>.ones(shape: Shape([2, 2]))
        #expect(t.shape == Shape([2, 2]))
        let data = t.toArray()
        #expect(data == [1.0, 1.0, 1.0, 1.0])
    }

    @Test func toArrayRoundTrip() async throws {
        let original: [Float] = [1.0, 2.0, 3.0, 4.0]
        let t = Tensor<Float>(data: original, shape: Shape([4]))
        let result = t.toArray()
        #expect(result == original)
    }

    @Test func scalarTensor() async throws {
        let t = Tensor<Float>(scalar: 42.0)
        #expect(t.shape == Shape([]))
        #expect(t.elementCount == 1)
        let data = t.toArray()
        #expect(data == [42.0])
    }

    @Test func copyOnWriteSharing() async throws {
        let a = Tensor<Float>(data: [1.0, 2.0, 3.0], shape: Shape([3]))
        let b = a
        #expect(a.toArray() == b.toArray())
    }

    @Test func reshape() async throws {
        let t = Tensor<Float>(data: [1, 2, 3, 4, 5, 6], shape: Shape([2, 3]))
        let reshaped = try t.reshape(Shape([3, 2]))
        #expect(reshaped.shape == Shape([3, 2]))
        #expect(reshaped.toArray() == t.toArray())
    }

    @Test func reshapeInvalidThrows() async throws {
        let t = Tensor<Float>(data: [1, 2, 3, 4], shape: Shape([2, 2]))
        #expect(throws: ShapeError.self) {
            try t.reshape(Shape([3, 2]))
        }
    }
}
