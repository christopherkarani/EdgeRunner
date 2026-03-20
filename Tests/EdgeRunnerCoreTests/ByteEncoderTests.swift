import Testing
@testable import EdgeRunnerCore

@Suite("ByteEncoder")
struct ByteEncoderTests {
    @Test func roundTripAllBytes() {
        for byte in UInt8.min...UInt8.max {
            let encoded = ByteEncoder.encode(byte)
            let decoded = ByteEncoder.decode(encoded)
            #expect(decoded == byte, "Byte \(byte) failed round-trip")
        }
    }

    @Test func spaceMapsToDotAboveG() {
        let encoded = ByteEncoder.encode(0x20)
        #expect(encoded == Character("\u{0120}"))
    }

    @Test func newlineMapsToCorrectChar() {
        let encoded = ByteEncoder.encode(0x0A)
        #expect(encoded == Character("\u{010A}"))
    }

    @Test func tabMapsToCorrectChar() {
        let encoded = ByteEncoder.encode(0x09)
        #expect(encoded == Character("\u{0109}"))
    }

    @Test func printableASCIIMapsToItself() {
        for byte: UInt8 in 0x21...0x7E {
            let encoded = ByteEncoder.encode(byte)
            #expect(encoded == Character(UnicodeScalar(byte)), "Byte \(byte) should map to itself")
        }
    }

    @Test func encodeStringConvertsAllBytes() {
        let result = ByteEncoder.encodeString(" hi")
        #expect(result == "\u{0120}hi")
    }

    @Test func decodeStringReversesEncoding() {
        let encoded = ByteEncoder.encodeString("Hello world")
        let decoded = ByteEncoder.decodeString(encoded)
        #expect(decoded == "Hello world")
    }

    @Test func decodeStringHandlesMultibyteUTF8() {
        let original = "café"
        let encoded = ByteEncoder.encodeString(original)
        let decoded = ByteEncoder.decodeString(encoded)
        #expect(decoded == original)
    }
}
