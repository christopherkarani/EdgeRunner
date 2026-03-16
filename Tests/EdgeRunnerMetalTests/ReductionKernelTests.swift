import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("ReductionKernels")
struct ReductionKernelTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func sumAll() throws {
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = try dispatchReduction(name: "reduce_sum_float", input: input, reductionSize: 4, outerSize: 1)
        #expect(result.count == 1)
        #expect(abs(result[0] - 10.0) < 1e-6)
    }

    @Test func sumRows() throws {
        let input: [Float] = [1, 2, 3, 4, 5, 6]
        let result = try dispatchReduction(name: "reduce_sum_float", input: input, reductionSize: 3, outerSize: 2)
        #expect(result.count == 2)
        #expect(abs(result[0] - 6.0) < 1e-6)
        #expect(abs(result[1] - 15.0) < 1e-6)
    }

    @Test func maxAll() throws {
        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        let result = try dispatchReduction(name: "reduce_max_float", input: input, reductionSize: 6, outerSize: 1)
        #expect(result.count == 1)
        #expect(abs(result[0] - 9.0) < 1e-6)
    }

    private func dispatchReduction(name: String, input: [Float], reductionSize: Int, outerSize: Int) throws -> [Float] {
        let byteCount = input.count * MemoryLayout<Float>.size
        let outBytes = outerSize * MemoryLayout<Float>.size
        let bufIn = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: outBytes, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: name)
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var params = ERReductionParams(elementCount: UInt32(input.count), reductionSize: UInt32(reductionSize), outerSize: UInt32(outerSize))
        encoder.setBytes(&params, length: MemoryLayout<ERReductionParams>.size, index: 2)

        let tpg = MTLSize(width: min(outerSize, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let tg = MTLSize(width: (outerSize + tpg.width - 1) / tpg.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tpg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: outerSize)
        return Array(UnsafeBufferPointer(start: ptr, count: outerSize))
    }
}
