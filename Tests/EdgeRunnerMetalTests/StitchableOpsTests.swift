import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("StitchableOps")
struct StitchableOpsTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func fusedAddRelu() throws {
        let a: [Float] = [-1.0, 2.0, -3.0, 4.0]
        let b: [Float] = [0.5, -3.0, 4.0, -1.0]
        // a+b = [-0.5, -1.0, 1.0, 3.0] -> relu -> [0.0, 0.0, 1.0, 3.0]
        let expected: [Float] = [0.0, 0.0, 1.0, 3.0]
        let result = try dispatchFusedBinary(a: a, b: b, count: 4, activation: .relu)
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-6)
        }
    }

    @Test func fusedAddSigmoid() throws {
        let a: [Float] = [0.0, 0.0, 0.0, 0.0]
        let b: [Float] = [0.0, 0.0, 0.0, 0.0]
        // a+b = [0,0,0,0] -> sigmoid(0) = 0.5
        let expected: [Float] = [0.5, 0.5, 0.5, 0.5]
        let result = try dispatchFusedBinary(a: a, b: b, count: 4, activation: .sigmoid)
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-6)
        }
    }

    @Test func fusedAddNoActivation() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]
        let expected: [Float] = [6.0, 8.0, 10.0, 12.0]
        let result = try dispatchFusedBinary(a: a, b: b, count: 4, activation: .none)
        #expect(result == expected)
    }

    enum Activation: Int {
        case none = 0, relu = 1, sigmoid = 2, gelu = 3, silu = 4
    }

    private func dispatchFusedBinary(
        a: [Float],
        b: [Float],
        count: Int,
        activation: Activation
    ) throws -> [Float] {
        let byteCount = count * MemoryLayout<Float>.size
        let bufA   = device.makeBuffer(bytes: a, length: byteCount, options: .storageModeShared)!
        let bufB   = device.makeBuffer(bytes: b, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        // Use function constants to specialise the activation path.
        let constants = MTLFunctionConstantValues()
        var activationType = Int32(activation.rawValue)
        constants.setConstantValue(&activationType, type: .int, index: 0)

        let library  = registry.metalLibrary
        let function = try library.makeFunction(
            name: "fused_add_activate_float",
            constantValues: constants
        )
        let pipeline = try device.makeComputePipelineState(function: function)

        let cmdBuf  = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA,   offset: 0, index: 0)
        encoder.setBuffer(bufB,   offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var elemCount = UInt32(count)
        encoder.setBytes(&elemCount, length: MemoryLayout<UInt32>.size, index: 3)

        let tpg = MTLSize(
            width: min(count, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        let tg = MTLSize(
            width: (count + tpg.width - 1) / tpg.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tpg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
