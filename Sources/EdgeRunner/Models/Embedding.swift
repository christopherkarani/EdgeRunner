/// Embedding lookup table: maps integer token IDs to dense vectors.
public struct Embedding: EdgeRunnerModule, Sendable {
    public typealias Input = [Int]
    public typealias Output = [Float]

    private let weight: [Float]
    private let vocabSize: Int
    private let dim: Int

    public init(weight: [Float], vocabSize: Int, dim: Int) {
        precondition(weight.count == vocabSize * dim)
        self.weight = weight
        self.vocabSize = vocabSize
        self.dim = dim
    }

    public func forward(_ input: [Int]) async throws -> [Float] {
        var output: [Float] = []
        output.reserveCapacity(input.count * dim)
        for tokenID in input {
            precondition(
                tokenID >= 0 && tokenID < vocabSize,
                "Token ID \(tokenID) out of range [0, \(vocabSize))"
            )
            let offset = tokenID * dim
            output.append(contentsOf: weight[offset..<(offset + dim)])
        }
        return output
    }

    public var parameters: [String: any TensorBox] {
        ["weight": ArrayTensorBox(data: weight, shape: [vocabSize, dim])]
    }
}
