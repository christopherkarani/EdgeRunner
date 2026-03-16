import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("ElementwiseKernels")
struct ElementwiseKernelTests {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
        guard let q = d.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.commandQueue = q
        self.registry = try KernelRegistry(device: d)
    }

    @Test func addFloat32() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]
        let expected: [Float] = [6.0, 8.0, 10.0, 12.0]
        let result = try dispatchBinary(name: "elementwise_add_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func subtractFloat32() throws {
        let a: [Float] = [5.0, 6.0, 7.0, 8.0]
        let b: [Float] = [1.0, 2.0, 3.0, 4.0]
        let expected: [Float] = [4.0, 4.0, 4.0, 4.0]
        let result = try dispatchBinary(name: "elementwise_sub_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func multiplyFloat32() throws {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [2.0, 3.0, 4.0, 5.0]
        let expected: [Float] = [2.0, 6.0, 12.0, 20.0]
        let result = try dispatchBinary(name: "elementwise_mul_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    @Test func divideFloat32() throws {
        let a: [Float] = [10.0, 20.0, 30.0, 40.0]
        let b: [Float] = [2.0, 4.0, 5.0, 8.0]
        let expected: [Float] = [5.0, 5.0, 6.0, 5.0]
        let result = try dispatchBinary(name: "elementwise_div_float", a: a, b: b, count: 4)
        #expect(result == expected)
    }

    private func dispatchBinary(name: String, a: [Float], b: [Float], count: Int) throws -> [Float] {
        let byteCount = count * MemoryLayout<Float>.size
        let bufA = device.makeBuffer(bytes: a, length: byteCount, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: name)
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var params = ERElementwiseParams(elementCount: UInt32(count))
        encoder.setBytes(&params, length: MemoryLayout<ERElementwiseParams>.size, index: 3)

        let threadsPerGroup = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
