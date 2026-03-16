import Foundation

/// Bidirectional mapping between token strings and their integer IDs.
public struct TokenizerVocabulary: Sendable {
    private let tokenToIDMap: [String: Int]
    private let idToTokenMap: [Int: String]
    public let count: Int
    public let offset: Int

    public init(tokens: [String], offset: Int = 0) {
        self.offset = offset
        self.count = tokens.count
        var t2i = [String: Int](minimumCapacity: tokens.count)
        var i2t = [Int: String](minimumCapacity: tokens.count)
        for (index, token) in tokens.enumerated() {
            let id = offset + index
            t2i[token] = id
            i2t[id] = token
        }
        self.tokenToIDMap = t2i
        self.idToTokenMap = i2t
    }

    public func tokenToID(_ token: String) -> Int? { tokenToIDMap[token] }
    public func idToToken(_ id: Int) -> String? { idToTokenMap[id] }
    public func contains(_ token: String) -> Bool { tokenToIDMap[token] != nil }
}
