import Testing
@testable import EdgeRunner

// MARK: - Mock module for testing

/// A simple doubling module for protocol conformance tests.
struct DoublingModule: EdgeRunnerModule {
    typealias Input = [Float]
    typealias Output = [Float]

    let scale: Float

    var parameters: [String: any TensorBox] {
        ["scale": ScalarTensorBox(value: scale)]
    }

    func forward(_ input: [Float]) async throws -> [Float] {
        input.map { $0 * scale }
    }
}

/// A simple offset module for composability tests.
struct OffsetModule: EdgeRunnerModule {
    typealias Input = [Float]
    typealias Output = [Float]

    let offset: Float

    var parameters: [String: any TensorBox] {
        ["offset": ScalarTensorBox(value: offset)]
    }

    func forward(_ input: [Float]) async throws -> [Float] {
        input.map { $0 + offset }
    }
}

@Suite("EdgeRunnerModule Protocol")
struct ModuleTests {

    @Test func moduleForwardPass() async throws {
        let module = DoublingModule(scale: 2.0)
        let input: [Float] = [1, 2, 3]
        let output = try await module.forward(input)
        #expect(output == [2, 4, 6])
    }

    @Test func moduleParametersAccess() throws {
        let module = DoublingModule(scale: 3.0)
        let params = module.parameters
        #expect(params.count == 1)
        #expect(params.keys.contains("scale"))
        let box = params["scale"] as? ScalarTensorBox
        #expect(box?.value == 3.0)
    }

    @Test func sequentialForward() async throws {
        let seq = Sequential(
            DoublingModule(scale: 2.0),
            OffsetModule(offset: 10.0)
        )
        let input: [Float] = [1, 2, 3]
        let output = try await seq.forward(input)
        #expect(output == [12, 14, 16])
    }

    @Test func sequentialParameters() throws {
        let seq = Sequential(
            DoublingModule(scale: 2.0),
            OffsetModule(offset: 10.0)
        )
        let params = seq.parameters
        #expect(params.count == 2)
        #expect(params.keys.contains("0.scale"))
        #expect(params.keys.contains("1.offset"))
    }

    @Test func emptySequential() async throws {
        let seq = Sequential<DoublingModule>()
        let input: [Float] = [1, 2, 3]
        let output = try await seq.forward(input)
        #expect(output == input)
    }

    @Test func tensorBoxProtocol() throws {
        let box = ScalarTensorBox(value: 42.0)
        #expect(box.elementCount == 1)
        #expect(box.floatArray == [42.0])
    }

    @Test func linearModuleForward() async throws {
        let linear = try LinearModule(
            inFeatures: 2,
            outFeatures: 3,
            weight: [1, 0, 0, 1, 1, 1],
            bias: [0.1, 0.2, 0.3]
        )
        let output = try await linear.forward([1, 2])
        #expect(output.count == 3)
        for (index, expected) in [Float(1.1), 2.2, 3.3].enumerated() {
            #expect(
                abs(output[index] - expected) < 1e-5,
                "Linear mismatch at [\(index)]: got \(output[index]) expected \(expected)"
            )
        }
    }

    @Test func linearModuleNoBias() async throws {
        let linear = try LinearModule(
            inFeatures: 2,
            outFeatures: 2,
            weight: [1, 0, 0, 1],
            bias: nil
        )
        let output = try await linear.forward([3, 4])
        #expect(output.count == 2)
        #expect(abs(output[0] - 3.0) < 1e-5)
        #expect(abs(output[1] - 4.0) < 1e-5)
    }
}
