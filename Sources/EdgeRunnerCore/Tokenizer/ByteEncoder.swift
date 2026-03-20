import Foundation

/// GPT-2 byte-to-unicode mapping table.
///
/// Maps all 256 byte values to visible Unicode characters so BPE merge tables
/// can be expressed as plain strings. 188 "nice" bytes (printable ASCII + Latin-1)
/// map to themselves; 68 "ugly" bytes (control chars, space, DEL) map to U+0100–U+0143.
public enum ByteEncoder: Sendable {
    private static let byteToChar: [Character] = {
        var table = [Character](repeating: "\0", count: 256)
        for b in 0x21...0x7E { table[b] = Character(UnicodeScalar(b)!) }
        for b in 0xA1...0xAC { table[b] = Character(UnicodeScalar(b)!) }
        for b in 0xAE...0xFF { table[b] = Character(UnicodeScalar(b)!) }
        var offset = 0
        for b in 0...255 {
            let isNice = (0x21...0x7E).contains(b)
                || (0xA1...0xAC).contains(b)
                || (0xAE...0xFF).contains(b)
            if !isNice {
                table[b] = Character(UnicodeScalar(256 + offset)!)
                offset += 1
            }
        }
        return table
    }()

    private static let charToByte: [Character: UInt8] = {
        var map = [Character: UInt8](minimumCapacity: 256)
        for b in 0..<256 {
            map[byteToChar[b]] = UInt8(b)
        }
        return map
    }()

    public static func encode(_ byte: UInt8) -> Character {
        byteToChar[Int(byte)]
    }

    public static func decode(_ char: Character) -> UInt8? {
        charToByte[char]
    }

    public static func encodeString(_ text: String) -> String {
        String(Array(text.utf8).map { byteToChar[Int($0)] })
    }

    public static func decodeString(_ encoded: String) -> String? {
        var bytes = [UInt8]()
        bytes.reserveCapacity(encoded.count)
        for char in encoded {
            guard let byte = charToByte[char] else { return nil }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8)
    }
}
