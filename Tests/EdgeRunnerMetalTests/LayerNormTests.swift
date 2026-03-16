import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuLayerNorm(
    _ input: [Float],
    gamma: [Float],
    beta: [Float],
    eps: Float = 1e-5
) -> [Float] {
    let mean = input.reduce(0, +) / Float(input.count)
    let variance = input.reduce(0.0) { partialResult, value in
        let delta = value - mean
        return partialResult + delta * delta
    } / Float(input.count)
    let invStd = 1.0 / sqrt(variance + eps)
    return zip(zip(input, gamma), beta).map { pair, betaValue in
        let (value, gammaValue) = pair
        return (value - mean) * invStd * gammaValue + betaValue
    }
}

private func cpuLayerNormBatched(
    _ input: [Float],
    gamma: [Float],
    beta: [Float],
    rows: Int,
    cols: Int,
    eps: Float = 1e-5
) -> [Float] {
    var output = [Float](repeating: 0, count: rows * cols)
    for row in 0..<rows {
        let offset = row * cols
        let normalized = cpuLayerNorm(Array(input[offset..<(offset + cols)]), gamma: gamma, beta: beta, eps: eps)
        for col in 0..<cols {
            output[offset + col] = normalized[col]
        }
    }
    return output
}

@Suite("LayerNorm Kernel")
struct LayerNormTests {
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
        let gamma: [Float] = [1, 1, 1, 1]
        let beta: [Float] = [0, 0, 0, 0]
        let expected = cpuLayerNorm(input, gamma: gamma, beta: beta)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            gamma: gamma,
            beta: beta,
            rows: 1,
            cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func singleTokenWithGammaBeta() async throws {
        let input: [Float] = [1, 2, 3, 4]
        let gamma: [Float] = [0.5, 1, 1.5, 2]
        let beta: [Float] = [0.1, 0.2, 0.3, 0.4]
        let expected = cpuLayerNorm(input, gamma: gamma, beta: beta)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            gamma: gamma,
            beta: beta,
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
        let cols = 128
        let input = (0..<(rows * cols)).map { _ in Float.random(in: -2...2) }
        let gamma = (0..<cols).map { _ in Float.random(in: 0.5...1.5) }
        let beta = (0..<cols).map { _ in Float.random(in: -0.5...0.5) }
        let expected = cpuLayerNormBatched(input, gamma: gamma, beta: beta, rows: rows, cols: cols)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            gamma: gamma,
            beta: beta,
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
        let gamma = (0..<cols).map { _ in Float.random(in: 0.8...1.2) }
        let beta = (0..<cols).map { _ in Float.random(in: -0.1...0.1) }
        let expected = cpuLayerNormBatched(input, gamma: gamma, beta: beta, rows: rows, cols: cols)

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            gamma: gamma,
            beta: beta,
            rows: rows,
            cols: cols,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<(rows * cols) {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func constantInput() async throws {
        let input: [Float] = [5, 5, 5, 5]
        let gamma: [Float] = [1, 1, 1, 1]
        let beta: [Float] = [0.1, 0.2, 0.3, 0.4]

        let kernel = try LayerNormKernel(device: device)
        let result = try await kernel.execute(
            input: input,
            gamma: gamma,
            beta: beta,
            rows: 1,
            cols: 4,
            eps: 1e-5,
            commandQueue: commandQueue
        )

        for index in 0..<4 {
            #expect(abs(result[index] - beta[index]) < 1e-5)
        }
    }
}
