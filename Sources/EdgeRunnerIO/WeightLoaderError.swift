import Foundation

public enum WeightLoaderError: Error, Sendable, Equatable {
    case deviceNotAvailable
    case fileNotFound(URL)
    case invalidFormat(String)
    case unsupportedVersion(UInt32)
    case unsupportedDataType(UInt32)
    case allocationFailed(byteCount: Int)
    case mmapFailed(errno: Int32)
    case missingMetadata(String)
    case tensorNotFound(String)
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
    case checksumMismatch(name: String)
}
