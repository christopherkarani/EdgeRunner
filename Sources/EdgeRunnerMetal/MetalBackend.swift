import Metal
import Synchronization

public enum MetalBackendError: Error, Sendable {
    case allocationFailed
    case encoderCreationFailed
}

public actor MetalBackend {
    public static let shared = MetalBackend()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let kernelRegistry: KernelRegistry
    public let bufferCache: BufferCache
    public let residencyManager: ResidencyManager
    public let commandBatcher: CommandBatcher
    public let barrierTracker: BarrierTracker

    public var deviceName: String { device.name }

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue

        do {
            self.kernelRegistry = try KernelRegistry(device: device)
        } catch {
            fatalError("Failed to initialize KernelRegistry: \(error)")
        }

        let maxCacheBytes = Int(Double(device.recommendedMaxWorkingSetSize) * 0.5)
        self.bufferCache = BufferCache(device: device, maxBytes: max(maxCacheBytes, 64 * 1024 * 1024))

        self.residencyManager = ResidencyManager(device: device, commandQueue: queue)
        self.commandBatcher = CommandBatcher(commandQueue: queue, device: device)
        self.barrierTracker = BarrierTracker()
    }

    // MARK: - Buffer Management

    public func acquireBuffer(size: Int) -> MTLBuffer {
        let buffer = bufferCache.acquire(size: size)
        residencyManager.addBuffer(buffer)
        return buffer
    }

    public func recycleBuffer(_ buffer: MTLBuffer) {
        bufferCache.recycle(buffer)
    }

    // MARK: - Kernel Dispatch

    public func pipeline(for name: String) throws -> MTLComputePipelineState {
        try kernelRegistry.pipeline(for: name)
    }

    public func dispatch(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        let (_, encoder) = commandBatcher.encoder()

        for (buffer, _) in buffers {
            barrierTracker.insertBarrierIfNeeded(forReading: buffer, encoder: encoder)
        }

        encoder.setComputePipelineState(pipeline)
        for (buffer, index) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        if let (outBuffer, _) = buffers.last {
            barrierTracker.recordWrite(outBuffer)
        }
    }

    public func synchronize() {
        commandBatcher.flushAndWait()
        barrierTracker.reset()
    }

    // MARK: - Integration Helpers

    /// Runs elementwise_add_float on two float arrays and returns the result.
    /// All Metal objects stay within the actor boundary.
    public func elementwiseAddFloat(_ a: [Float], _ b: [Float]) throws -> [Float] {
        let count = a.count
        precondition(b.count == count, "Input arrays must have equal length")
        let byteCount = count * MemoryLayout<Float>.stride

        guard let bufA = device.makeBuffer(bytes: a, length: byteCount, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: b, length: byteCount, options: .storageModeShared),
              let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)
        else { throw MetalBackendError.allocationFailed }

        let pipe = try pipeline(for: "elementwise_add_float")

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { throw MetalBackendError.encoderCreationFailed }

        encoder.setComputePipelineState(pipe)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)
        var params = UInt32(count)
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.stride, index: 3)
        let tpg = MTLSize(width: count, height: 1, depth: 1)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tpg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - Testing Helpers

    /// Acquires a buffer, checks its length, recycles it, and returns the length.
    /// Keeps the non-Sendable MTLBuffer within the actor boundary.
    internal func acquireAndRecycleRoundTrip(size: Int) -> Int {
        let buffer = acquireBuffer(size: size)
        let length = buffer.length
        recycleBuffer(buffer)
        return length
    }

    /// Public API for integration tests: acquires a buffer of at least `size` bytes,
    /// records it with the residency manager, recycles it, and returns the actual length.
    public func acquireBufferSize(size: Int) -> Int {
        let buffer = bufferCache.acquire(size: size)
        residencyManager.addBuffer(buffer)
        let length = buffer.length
        bufferCache.recycle(buffer)
        return length
    }

    /// Returns the max threads per threadgroup for a named pipeline.
    internal func pipelineMaxThreads(for name: String) throws -> Int {
        let pipeline = try self.pipeline(for: name)
        return pipeline.maxTotalThreadsPerThreadgroup
    }
}
