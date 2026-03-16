import Foundation
import Testing
@testable import EdgeRunnerIO

@Suite("Perplexity Verification Tests")
struct PerplexityVerificationTests: Sendable {
    @Test("Softmax produces valid probability distribution")
    func softmaxValidation() {
        let probabilities = stableSoftmax([2.0, 1.0, 0.1, -1.0, 3.0])
        let sum = probabilities.reduce(0.0, +)

        #expect(abs(sum - 1.0) < 1e-9)
        for probability in probabilities {
            #expect(probability > 0.0)
            #expect(probability < 1.0)
        }
    }

    @Test("Perplexity computation from logits is numerically stable")
    func perplexityComputation() {
        let logits = (0..<10).map { index -> [Double] in
            var row = [Double](repeating: -5.0, count: 100)
            row[index] = 5.0
            return row
        }
        let targets = Array(0..<10)

        let perplexity = perplexity(for: logits, targets: targets)
        #expect(perplexity > 0.9 && perplexity < 2.0)
    }

    @Test("Cross-entropy loss matches expected value")
    func crossEntropyLoss() {
        let logits = [Double](repeating: 0.0, count: 100)
        let loss = crossEntropy(logits: logits, targetIndex: 42)

        #expect(abs(loss - Foundation.log(100.0)) < 1e-9)
    }
}

private func stableSoftmax(_ logits: [Double]) -> [Double] {
    let maxLogit = logits.max() ?? 0.0
    let exponentials = logits.map { Foundation.exp($0 - maxLogit) }
    let sum = exponentials.reduce(0.0, +)
    return exponentials.map { $0 / sum }
}

private func crossEntropy(logits: [Double], targetIndex: Int) -> Double {
    let probabilities = stableSoftmax(logits)
    return -Foundation.log(probabilities[targetIndex])
}

private func perplexity(for logits: [[Double]], targets: [Int]) -> Double {
    let losses = zip(logits, targets).map { row, target in
        crossEntropy(logits: row, targetIndex: target)
    }
    let meanLoss = losses.reduce(0.0, +) / Double(losses.count)
    return Foundation.exp(meanLoss)
}
