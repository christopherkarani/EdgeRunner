import Foundation

/// Protocol for all tokenizer implementations.
public protocol Tokenizer: Sendable {
    func encode(_ text: String, addBOS: Bool) -> [Int]
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String
    var vocabularySize: Int { get }
    var eosTokenID: Int { get }
    var bosTokenID: Int? { get }
    var padTokenID: Int? { get }
}

extension Tokenizer {
    public func encode(_ text: String) -> [Int] {
        encode(text, addBOS: false)
    }

    public func decode(_ ids: [Int]) -> String {
        decode(ids, skipSpecialTokens: false)
    }
}
