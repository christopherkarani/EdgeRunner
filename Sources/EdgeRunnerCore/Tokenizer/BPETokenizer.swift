import Foundation

/// Byte-Pair Encoding tokenizer compatible with Llama/GPT tokenizer formats.
public struct BPETokenizer: Tokenizer, Sendable {
    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let mergeRanks: [String: Int]
    private let mergeResults: [String: String]

    public var vocabularySize: Int {
        vocabulary.count + specialTokens.specialTokenIDs.count
    }

    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        merges: [(String, String)]
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        var ranks = [String: Int](minimumCapacity: merges.count)
        var results = [String: String](minimumCapacity: merges.count)
        for (index, merge) in merges.enumerated() {
            let key = "\(merge.0)\t\(merge.1)"
            ranks[key] = index
            results[key] = merge.0 + merge.1
        }
        self.mergeRanks = ranks
        self.mergeResults = results
    }

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }
        var ids = [Int]()
        if addBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }
        var tokens = text.map { String($0) }
        tokens = applyMerges(tokens)
        for token in tokens {
            if let id = vocabulary.tokenToID(token) {
                ids.append(id)
            } else if let id = specialTokens.specialTokenMap[token] {
                ids.append(id)
            }
        }
        return ids
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var result = ""
        for id in ids {
            if skipSpecialTokens && specialTokens.specialTokenIDs.contains(id) {
                continue
            }
            if let token = vocabulary.idToToken(id) {
                result += token
            } else if let token = specialTokenStringForID(id) {
                if !skipSpecialTokens {
                    result += token
                }
            }
        }
        return result
    }

    private func applyMerges(_ initialTokens: [String]) -> [String] {
        var tokens = initialTokens
        while tokens.count >= 2 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(tokens.count - 1) {
                let key = "\(tokens[i])\t\(tokens[i + 1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let key = "\(tokens[bestIndex])\t\(tokens[bestIndex + 1])"
            let merged = mergeResults[key]!
            var newTokens = [String]()
            newTokens.reserveCapacity(tokens.count - 1)
            for i in 0..<tokens.count {
                if i == bestIndex {
                    newTokens.append(merged)
                } else if i == bestIndex + 1 {
                    continue
                } else {
                    newTokens.append(tokens[i])
                }
            }
            tokens = newTokens
        }
        return tokens
    }

    private func specialTokenStringForID(_ id: Int) -> String? {
        if id == specialTokens.bosTokenID { return specialTokens.bosTokenString }
        if id == specialTokens.eosTokenID { return specialTokens.eosTokenString }
        if id == specialTokens.padTokenID { return specialTokens.padTokenString }
        return nil
    }
}
