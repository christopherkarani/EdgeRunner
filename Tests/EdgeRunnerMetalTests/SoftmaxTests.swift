import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuSoftmax1D(_ input: [Float]) -> [Float] {
    let maxVal = input.max() ?? 0
    let exps = input.map { Foundation.exp($0 - maxVal) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

private func cpuSoftmax2D(_ input: [Float], rows: Int, cols: Int) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for row in 0..<rows {
        let offset = row * cols
        let rowValues = Array(input[offset..<(offset + cols)])
        let softmaxed = cpuSoftmax1D(rowValues)
        for col in 0..<cols {
            output[offset + col] = softmaxed[col]
        }
    }
    return output
}

@Suite("Softmax Kernel")
struct SoftmaxTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = commandQueue
    }

    @Test func softmax1DSmall() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let expected = cpuSoftmax1D(input)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: 1,
            cols: 4,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func softmax1DSumsToOne() async throws {
        let input = (0..<128).map { _ in Float.random(in: -5...5) }
        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: 1,
            cols: 128,
            commandQueue: commandQueue
        )

        let sum = result.reduce(0, +)
        #expect(abs(sum - 1) < 1e-5)
    }

    @Test func softmax1DAllEqual() async throws {
        let count = 64
        let input = [Float](repeating: 3, count: count)
        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: 1,
            cols: count,
            commandQueue: commandQueue
        )

        let expected = 1 / Float(count)
        for value in result {
            #expect(abs(value - expected) < 1e-5)
        }
    }

    @Test func softmax1DNumericalStability() async throws {
        let input: [Float] = [1000, 1001, 1002, 1003]
        let expected = cpuSoftmax1D(input)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: 1,
            cols: 4,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(result[index].isFinite)
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func softmax2DRows() async throws {
        let rows = 4
        let cols = 32
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -3...3) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }

        for row in 0..<rows {
            let sum = (0..<cols).reduce(Float.zero) { partialResult, col in
                partialResult + result[row * cols + col]
            }
            #expect(abs(sum - 1) < 1e-5)
        }
    }

    @Test func softmax2DLargeRows() async throws {
        let rows = 8
        let cols = 512
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -2...2) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func softmaxNonAlignedCols() async throws {
        let rows = 3
        let cols = 37
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -1...1) }
        let expected = cpuSoftmax2D(input, rows: rows, cols: cols)

        let kernel = try SoftmaxKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            rows: rows,
            cols: cols,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }
}
