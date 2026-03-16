import EdgeRunnerMetal

public struct Tensor<T: Sendable>: Sendable {
    public let shape: [Int]

    public init(shape: [Int]) {
        self.shape = shape
    }
}
