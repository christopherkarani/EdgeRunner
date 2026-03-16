import Metal

final class CommandBatcher {
    private let commandQueue: MTLCommandQueue
    private let maxOpsPerBuffer: Int
    private var currentCommandBuffer: MTLCommandBuffer?
    private var currentEncoder: MTLComputeCommandEncoder?
    private var currentOpCount: Int = 0

    init(commandQueue: MTLCommandQueue, device: MTLDevice) {
        self.commandQueue = commandQueue
        if device.supportsFamily(.apple9) {
            self.maxOpsPerBuffer = 50
        } else if device.supportsFamily(.apple8) {
            self.maxOpsPerBuffer = 40
        } else {
            self.maxOpsPerBuffer = 30
        }
    }

    func encoder() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        if currentOpCount >= maxOpsPerBuffer { flush() }
        if currentCommandBuffer == nil {
            currentCommandBuffer = commandQueue.makeCommandBuffer()!
            currentEncoder = currentCommandBuffer!.makeComputeCommandEncoder(dispatchType: .concurrent)!
        }
        currentOpCount += 1
        return (currentCommandBuffer!, currentEncoder!)
    }

    func flush() {
        currentEncoder?.endEncoding()
        currentCommandBuffer?.commit()
        currentCommandBuffer = nil
        currentEncoder = nil
        currentOpCount = 0
    }

    func flushAndWait() {
        currentEncoder?.endEncoding()
        if let buffer = currentCommandBuffer {
            buffer.commit()
            buffer.waitUntilCompleted()
        }
        currentCommandBuffer = nil
        currentEncoder = nil
        currentOpCount = 0
    }
}
