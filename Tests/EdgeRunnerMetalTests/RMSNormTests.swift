import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuRMSNorm(
    _ input: [Float],
    weight: [Float],
    eps: Float = 1e-5
) -> [Float] {
    let meanSq = input.reduce(0.0) { partialResult, value in
        partialResult + value * value
    } / Float(input.count)
    let rms = 1.0 / sqrt(meanSq + eps)
    return zip(input, weight).map { $0 * $1 * rms }
}

private func cpuRMSNormBatched(
    _ input: [Float],
    weight: [Float],
    rows: Int,
    cols: Int,
    eps: Float = 1e-5
) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for row in 0..<rows {
        let offset = row * cols
        let rowValues = Array(input[offset..<(offset + cols)])
        let normalized = cpuRMSNorm(rowValues, weight: weight, eps: eps)
        for col in 0..<cols {
            output[offset + col] = normalized[col]
        }
    }
    return output
}

@Suite("RMSNorm Kernel")
struct RMSNormTests {
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

    @Test func singleTokenKnownValues() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [1, 1, 1, 1]
        let expected = cpuRMSNorm(input, weight: weight)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            weight: weight,
            rows: 1,
            cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func singleTokenWithLearnableWeight() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [0.5, 1, 1.5, 2]
        let expected = cpuRMSNorm(input, weight: weight)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            weight: weight,
            rows: 1,
            cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func batchedTokens() async throws {
        let rows = 8
        let cols = 64
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -2...2) }
        let weight = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }
        let expected = cpuRMSNormBatched(input, weight: weight, rows: rows, cols: cols)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            weight: weight,
            rows: rows,
            cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func largeHiddenDimension() async throws {
        let rows = 4
        let cols = 4096
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -1...1) }
        let weight = (0..<cols).map { _ in Float.random(in: 0.8...1.2) }
        let expected = cpuRMSNormBatched(input, weight: weight, rows: rows, cols: cols)

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            weight: weight,
            rows: rows,
            cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func allZerosInput() async throws {
        let input: [Float] = [0, 0, 0, 0]
        let weight: [Float] = [1, 1, 1, 1]

        let kernel = try RMSNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            weight: weight,
            rows: 1,
            cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for value in result {
            #expect(abs(value) < 1e-5)
        }
    }
}
