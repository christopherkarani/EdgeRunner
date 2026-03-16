import Metal
import Synchronization

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

    // MARK: - Testing Helpers

    /// Acquires a buffer, checks its length, recycles it, and returns the length.
    /// Keeps the non-Sendable MTLBuffer within the actor boundary.
    internal func acquireAndRecycleRoundTrip(size: Int) -> Int {
        let buffer = acquireBuffer(size: size)
        let length = buffer.length
        recycleBuffer(buffer)
        return length
    }

    /// Returns the max threads per threadgroup for a named pipeline.
    internal func pipelineMaxThreads(for name: String) throws -> Int {
        let pipeline = try self.pipeline(for: name)
        return pipeline.maxTotalThreadsPerThreadgroup
    }
}
