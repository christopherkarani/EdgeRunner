import Foundation

struct DequantParams {
    var blockCount: UInt32
    var outputOffset: UInt32
}

struct DequantGEMVParams {
    var rows: UInt32
    var cols: UInt32
    var blocksPerRow: UInt32
}

public enum DequantKernelError: Error, Sendable {
    case encodingFailed
    case invalidBlockDataCount(expected: Int, actual: Int)
    case invalidMatrixShape
    case invalidVectorShape
    case allocationFailed(byteCount: Int)
}
