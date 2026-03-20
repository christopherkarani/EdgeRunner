import Foundation

/// Protocol for all tokenizer implementations.
public protocol Tokenizer: Sendable {
    func encode(_ text: String, addBOS: Bool) -> [Int]
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String
    var vocabularySize: Int { get }
    var eosTokenID: Int { get }
    var bosTokenID: Int? { get }
    var padTokenID: Int? { get }
    var shouldAddBOS: Bool { get }
    func applyChatTemplate(messages: [ChatMessage], addGenerationPrompt: Bool) throws -> String?
}

extension Tokenizer {
    public func encode(_ text: String) -> [Int] {
        encode(text, addBOS: false)
    }

    public func decode(_ ids: [Int]) -> String {
        decode(ids, skipSpecialTokens: false)
    }

    public var shouldAddBOS: Bool { false }

    public func applyChatTemplate(messages: [ChatMessage], addGenerationPrompt: Bool = true) throws -> String? { nil }
}
