import Testing
import Metal
@testable import EdgeRunnerMetal
import EdgeRunnerSharedTypes

@Suite("TransposeKernels")
struct TransposeKernelTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let registry: KernelRegistry

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw TestError.noMetal }
        self.device = d
        self.commandQueue = d.makeCommandQueue()!
        self.registry = try KernelRegistry(device: d)
    }

    @Test func transpose2x3() throws {
        let input: [Float] = [1, 2, 3, 4, 5, 6]
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        let result = try dispatchTranspose(input: input, rows: 2, cols: 3)
        #expect(result == expected)
    }

    @Test func transposeSquare() throws {
        let input: [Float] = [1, 2, 3, 4]
        let expected: [Float] = [1, 3, 2, 4]
        let result = try dispatchTranspose(input: input, rows: 2, cols: 2)
        #expect(result == expected)
    }

    private func dispatchTranspose(input: [Float], rows: Int, cols: Int) throws -> [Float] {
        let count = rows * cols
        let byteCount = count * MemoryLayout<Float>.size
        let bufIn = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let pipeline = try registry.pipeline(for: "transpose_float")
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var params = ERTransposeParams(rows: UInt32(rows), cols: UInt32(cols))
        encoder.setBytes(&params, length: MemoryLayout<ERTransposeParams>.size, index: 2)

        let tpg = MTLSize(width: 16, height: 16, depth: 1)
        let tg = MTLSize(width: (cols + 15) / 16, height: (rows + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tpg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
