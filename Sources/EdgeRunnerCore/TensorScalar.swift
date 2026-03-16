import Metal
import EdgeRunnerSharedTypes

public protocol TensorScalar: Sendable, BitwiseCopyable {
    static var metalDataType: MTLDataType { get }
    static var byteSize: Int { get }
    static var erDType: ERDType { get }
    static var zero: Self { get }
}

extension Float: TensorScalar {
    public static let metalDataType: MTLDataType = .float
    public static let byteSize: Int = 4
    public static let erDType: ERDType = .float32
    public static let zero: Float = 0.0
}

extension Float16: TensorScalar {
    public static let metalDataType: MTLDataType = .half
    public static let byteSize: Int = 2
    public static let erDType: ERDType = .float16
    public static let zero: Float16 = 0.0
}

extension Int8: TensorScalar {
    public static let metalDataType: MTLDataType = .char
    public static let byteSize: Int = 1
    public static let erDType: ERDType = .int8
    public static let zero: Int8 = 0
}

extension UInt8: TensorScalar {
    public static let metalDataType: MTLDataType = .uchar
    public static let byteSize: Int = 1
    public static let erDType: ERDType = .uInt8
    public static let zero: UInt8 = 0
}
