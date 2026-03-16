import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal
@testable import EdgeRunnerSharedTypes

private func cpuSigmoid(_ value: Float) -> Float {
    1 / (1 + Foundation.exp(-value))
}

private func cpuSiLU(_ value: Float) -> Float {
    value * cpuSigmoid(value)
}

private func cpuGELU(_ value: Float) -> Float {
    let coefficient: Float = sqrt(2 / .pi)
    return value * 0.5 * (1 + tanh(coefficient * (value + 0.044715 * value * value * value)))
}

private func cpuSwiGLU(gate: [Float], up: [Float]) -> [Float] {
    zip(gate, up).map { cpuSiLU($0) * $1 }
}

@Suite("Activation Kernels")
struct ActivationTests {
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

    @Test func sigmoidKnownValues() async throws {
        let input: [Float] = [-10, -1, 0, 1, 10]
        let expected = input.map(cpuSigmoid)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.sigmoid(input: input, commandQueue: commandQueue)

        for index in 0..<input.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func sigmoidRandomBatch() async throws {
        let input = (0..<1024).map { _ in Float.random(in: -5...5) }
        let expected = input.map(cpuSigmoid)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.sigmoid(input: input, commandQueue: commandQueue)

        for index in 0..<input.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func geluKnownValues() async throws {
        let input: [Float] = [-3, -1, 0, 1, 3]
        let expected = input.map(cpuGELU)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(input: input, commandQueue: commandQueue)

        for index in 0..<input.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func geluZeroIsZero() async throws {
        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(input: [0], commandQueue: commandQueue)
        #expect(abs(result[0]) < 1e-6)
    }

    @Test func geluRandomBatch() async throws {
        let input = (0..<4096).map { _ in Float.random(in: -5...5) }
        let expected = input.map(cpuGELU)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.gelu(input: input, commandQueue: commandQueue)

        for index in 0..<input.count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func swigluKnownValues() async throws {
        let gate: [Float] = [1, -1, 0, 2]
        let up: [Float] = [1, 1, 1, 0.5]
        let expected = cpuSwiGLU(gate: gate, up: up)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(gate: gate, up: up, commandQueue: commandQueue)

        for index in 0..<gate.count {
            #expect(abs(result[index] - expected[index]) < 1e-5)
        }
    }

    @Test func swigluRandomBatch() async throws {
        let count = 2048
        let gate = (0..<count).map { _ in Float.random(in: -3...3) }
        let up = (0..<count).map { _ in Float.random(in: -3...3) }
        let expected = cpuSwiGLU(gate: gate, up: up)

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(gate: gate, up: up, commandQueue: commandQueue)

        for index in 0..<count {
            #expect(abs(result[index] - expected[index]) < 1e-4)
        }
    }

    @Test func swigluZeroGate() async throws {
        let gate: [Float] = [0, 0, 0, 0]
        let up: [Float] = [1, 2, 3, 4]

        let kernels = try ActivationKernels(device: device)
        let result = try await kernels.swiglu(gate: gate, up: up, commandQueue: commandQueue)

        for value in result {
            #expect(abs(value) < 1e-6)
        }
    }
}
