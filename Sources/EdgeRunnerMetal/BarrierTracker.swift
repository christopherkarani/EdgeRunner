import Metal

final class BarrierTracker {
    private var writtenBuffers: Set<ObjectIdentifier> = []

    init() {}

    func needsBarrier(forReading buffer: MTLBuffer) -> Bool {
        writtenBuffers.contains(ObjectIdentifier(buffer))
    }

    func recordWrite(_ buffer: MTLBuffer) {
        writtenBuffers.insert(ObjectIdentifier(buffer))
    }

    func insertBarrierIfNeeded(forReading buffer: MTLBuffer, encoder: MTLComputeCommandEncoder) {
        if needsBarrier(forReading: buffer) {
            encoder.memoryBarrier(scope: .buffers)
            writtenBuffers.remove(ObjectIdentifier(buffer))
        }
    }

    func reset() {
        writtenBuffers.removeAll()
    }
}
