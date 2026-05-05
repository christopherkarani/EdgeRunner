import Metal
import EdgeRunnerIO

public enum DequantDispatcher: Sendable {

    /// Validates that the tensor buffer has enough space for the requested read.
    private static func validateBufferBounds(
        tensor: TensorStorage,
        requiredBytes: Int
    ) throws {
        let totalRequired = tensor.byteOffset + requiredBytes
        guard totalRequired <= tensor.buffer.length else {
            throw EspressoError.bufferOutOfBounds(
                name: tensor.name,
                required: totalRequired,
                available: tensor.buffer.length
            )
        }
    }

    /// Validates that `elementCount` is evenly divisible by `weightsPerBlock`.
    private static func validateBlockAlignment(
        tensor: TensorStorage,
        weightsPerBlock: Int
    ) throws {
        guard tensor.elementCount % weightsPerBlock == 0 else {
            throw EspressoError.invalidTensorShape(
                "\(tensor.name): elementCount \(tensor.elementCount) not divisible by \(weightsPerBlock) for \(tensor.dataType)"
            )
        }
    }

    /// Dequantizes a `TensorStorage` to `[Float]` based on its `dataType`.
    public static func dequantize(
        tensor: TensorStorage,
        device: MTLDevice
    ) async throws -> [Float] {
        let pointer = tensor.buffer.contents().advanced(by: tensor.byteOffset)

        switch tensor.dataType {
        case .float32:
            let count = tensor.elementCount
            let requiredBytes = count * MemoryLayout<Float>.size
            try validateBufferBounds(tensor: tensor, requiredBytes: requiredBytes)
            let floatPtr = pointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: floatPtr, count: count))

        case .float16:
            let count = tensor.elementCount
            let requiredBytes = count * MemoryLayout<UInt16>.size
            try validateBufferBounds(tensor: tensor, requiredBytes: requiredBytes)
            let fp16Ptr = pointer.assumingMemoryBound(to: UInt16.self)
            var result = [Float](repeating: 0, count: count)
            for i in 0..<count {
                result[i] = Float(Float16(bitPattern: fp16Ptr[i]))
            }
            return result

        case .bfloat16:
            let count = tensor.elementCount
            let requiredBytes = count * MemoryLayout<UInt16>.size
            try validateBufferBounds(tensor: tensor, requiredBytes: requiredBytes)
            let bf16Ptr = pointer.assumingMemoryBound(to: UInt16.self)
            var result = [Float](repeating: 0, count: count)
            for i in 0..<count {
                result[i] = Float(bitPattern: UInt32(bf16Ptr[i]) << 16)
            }
            return result

        case .q4_0:
            let blockByteCount = 18
            let weightsPerBlock = 32
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let blockCount = tensor.elementCount / weightsPerBlock
            let byteCount = blockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ4_0Kernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, blockCount: blockCount, commandQueue: queue
            )

        case .q8_0:
            let blockByteCount = 34
            let weightsPerBlock = 32
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let blockCount = tensor.elementCount / weightsPerBlock
            let byteCount = blockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ8_0Kernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, blockCount: blockCount, commandQueue: queue
            )

        case .q4_K:
            let blockByteCount = 144
            let weightsPerBlock = 256
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let superBlockCount = tensor.elementCount / weightsPerBlock
            let byteCount = superBlockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ4KMKernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, superBlockCount: superBlockCount, commandQueue: queue
            )

        case .q5_K:
            let blockByteCount = 176
            let weightsPerBlock = 256
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let superBlockCount = tensor.elementCount / weightsPerBlock
            let byteCount = superBlockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ5KKernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, superBlockCount: superBlockCount, commandQueue: queue
            )

        case .q6_K:
            let blockByteCount = 210
            let weightsPerBlock = 256
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let superBlockCount = tensor.elementCount / weightsPerBlock
            let byteCount = superBlockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ6KKernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, superBlockCount: superBlockCount, commandQueue: queue
            )

        case .q3_K:
            let blockByteCount = 110
            let weightsPerBlock = 256
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let superBlockCount = tensor.elementCount / weightsPerBlock
            let byteCount = superBlockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ3KKernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, superBlockCount: superBlockCount, commandQueue: queue
            )

        case .q2_K:
            let blockByteCount = 84
            let weightsPerBlock = 256
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let superBlockCount = tensor.elementCount / weightsPerBlock
            let byteCount = superBlockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ2KKernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, superBlockCount: superBlockCount, commandQueue: queue
            )

        case .q5_0:
            let blockByteCount = 22
            let weightsPerBlock = 32
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let blockCount = tensor.elementCount / weightsPerBlock
            let byteCount = blockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ5_0Kernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, blockCount: blockCount, commandQueue: queue
            )

        case .q5_1:
            let blockByteCount = 24
            let weightsPerBlock = 32
            try validateBlockAlignment(tensor: tensor, weightsPerBlock: weightsPerBlock)
            let blockCount = tensor.elementCount / weightsPerBlock
            let byteCount = blockCount * blockByteCount
            try validateBufferBounds(tensor: tensor, requiredBytes: byteCount)
            let bytes = Array(UnsafeBufferPointer(
                start: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            guard let queue = device.makeCommandQueue() else {
                throw EspressoError.metalDeviceUnavailable
            }
            let kernel = try DequantQ5_1Kernel(device: device)
            return try await kernel.dequantise(
                blockData: bytes, blockCount: blockCount, commandQueue: queue
            )

        default:
            throw EspressoError.unsupportedDataType(String(describing: tensor.dataType))
        }
    }
}
