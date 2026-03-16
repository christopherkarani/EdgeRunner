import Metal

public final class ResidencyManager: @unchecked Sendable {
    // @unchecked: MTLResidencySet is Obj-C protocol, not Sendable.
    // Access serialized through MetalBackend actor.
    private let residencySet: MTLResidencySet?
    private let device: MTLDevice

    public init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 256
        if let set = try? device.makeResidencySet(descriptor: descriptor) {
            self.residencySet = set
            set.requestResidency()
            commandQueue.addResidencySet(set)
        } else {
            self.residencySet = nil
        }
    }

    public func addBuffer(_ buffer: MTLBuffer) {
        guard let set = residencySet else { return }
        set.addAllocation(buffer)
        set.commit()
    }

    public func addHeap(_ heap: MTLHeap) {
        guard let set = residencySet else { return }
        set.addAllocation(heap)
        set.commit()
    }
}
