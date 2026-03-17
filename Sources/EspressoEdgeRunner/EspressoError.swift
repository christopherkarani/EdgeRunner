public enum EspressoError: Error, Sendable, Equatable {
    case unsupportedDataType(String)
    case tensorNotFound(String)
    case unmappedTensorName(String)
    case configMissingKey(String)
    case metalDeviceUnavailable
    case transposeDimensionMismatch(name: String, shape: [Int])
    case directoryCreationFailed(String)
    case invalidTensorShape(String)
    case bufferOutOfBounds(name: String, required: Int, available: Int)
    case payloadTooLarge(Int)
    case pathTraversal(String)
    case ioSurfaceLockFailed
}
