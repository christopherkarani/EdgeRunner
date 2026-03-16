import Metal

public actor MetalBackend {
    public static let shared = MetalBackend()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue
    }
}
