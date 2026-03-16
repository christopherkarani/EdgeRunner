import Testing
import Metal
import EdgeRunnerSharedTypes
@testable import EdgeRunnerCore

@Suite("TensorScalar")
struct TensorScalarTests {

    @Test func float32Properties() {
        #expect(Float.erDType == .float32)
        #expect(Float.byteSize == 4)
    }

    @Test func float16Properties() {
        #expect(Float16.erDType == .float16)
        #expect(Float16.byteSize == 2)
    }

    @Test func int8Properties() {
        #expect(Int8.erDType == .int8)
        #expect(Int8.byteSize == 1)
    }

    @Test func uint8Properties() {
        #expect(UInt8.erDType == .uInt8)
        #expect(UInt8.byteSize == 1)
    }

    @Test func metalDataTypes() {
        #expect(Float.metalDataType == .float)
        #expect(Float16.metalDataType == .half)
        #expect(Int8.metalDataType == .char)
        #expect(UInt8.metalDataType == .uchar)
    }
}
