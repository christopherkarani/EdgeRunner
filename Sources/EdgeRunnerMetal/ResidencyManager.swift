import Metal

final class ResidencyManager {
    private let residencySet: MTLResidencySet?

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
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

    func addBuffer(_ buffer: MTLBuffer) {
        guard let set = residencySet else { return }
        set.addAllocation(buffer)
        set.commit()
    }

    func addHeap(_ heap: MTLHeap) {
        guard let set = residencySet else { return }
        set.addAllocation(heap)
        set.commit()
    }
}
