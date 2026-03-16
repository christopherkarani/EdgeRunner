import Metal

public final class BarrierTracker: @unchecked Sendable {
    // @unchecked: access serialized through MetalBackend actor.
    private var writtenBuffers: Set<ObjectIdentifier> = []

    public init() {}

    public func needsBarrier(forReading buffer: MTLBuffer) -> Bool {
        writtenBuffers.contains(ObjectIdentifier(buffer))
    }

    public func recordWrite(_ buffer: MTLBuffer) {
        writtenBuffers.insert(ObjectIdentifier(buffer))
    }

    public func insertBarrierIfNeeded(forReading buffer: MTLBuffer, encoder: MTLComputeCommandEncoder) {
        if needsBarrier(forReading: buffer) {
            encoder.memoryBarrier(scope: .buffers)
            writtenBuffers.remove(ObjectIdentifier(buffer))
        }
    }

    public func reset() {
        writtenBuffers.removeAll()
    }
}
