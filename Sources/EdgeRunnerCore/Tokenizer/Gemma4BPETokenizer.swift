import Foundation

/// Gemma 4 GGUF tokenizer.
///
/// Gemma 4 stores a BPE vocabulary, but it uses SentencePiece-style whitespace
/// escaping instead of GPT-2 byte encoding. Spaces are normalized to `▁`, text
/// is split only around newline runs, and byte fallback uses `<0xHH>` tokens.
public struct Gemma4BPETokenizer: Tokenizer, Sendable {
    private static let escapedSpace = "\u{2581}"

    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let mergeRanks: [String: Int]
    private let mergeResults: [String: String]
    private let byteFallbackTable: [UInt8: Int]

    public let shouldAddBOS: Bool
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int { vocabulary.count }
    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        merges: [(String, String)],
        shouldAddBOS: Bool,
        byteFallbackTable: [UInt8: Int]? = nil,
        chatTemplateEngine: ChatTemplateEngine? = nil
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.shouldAddBOS = shouldAddBOS
        self.chatTemplateEngine = chatTemplateEngine

        var ranks = [String: Int](minimumCapacity: merges.count)
        var results = [String: String](minimumCapacity: merges.count)
        for (rank, merge) in merges.enumerated() {
            let key = Self.mergeKey(merge.0, merge.1)
            ranks[key] = rank
            results[key] = merge.0 + merge.1
        }
        self.mergeRanks = ranks
        self.mergeResults = results

        if let byteFallbackTable {
            self.byteFallbackTable = byteFallbackTable
        } else {
            self.byteFallbackTable = Self.buildHexByteFallbackTable(vocabulary: vocabulary)
        }
    }

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool
    ) throws -> String? {
        guard let engine = chatTemplateEngine else { return nil }
        let bosToken = specialTokens.bosTokenID.flatMap { vocabulary.idToToken($0) }
        let eosToken = vocabulary.idToToken(eosTokenID)
        return try engine.apply(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            bosToken: bosToken,
            eosToken: eosToken
        )
    }

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        var ids = [Int]()
        if addBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        for segment in splitAroundSpecialTokens(text) {
            if let specialID = specialTokens.specialTokenMap[segment] {
                ids.append(specialID)
                continue
            }
            appendNormalSegment(segment, to: &ids)
        }
        return ids
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(ids.count * 4)

        for id in ids {
            if specialTokens.specialTokenIDs.contains(id) {
                if !skipSpecialTokens, let token = specialTokenStringForID(id) {
                    bytes.append(contentsOf: token.utf8)
                }
                continue
            }

            guard let token = vocabulary.idToToken(id) else {
                bytes.append(contentsOf: "\u{FFFD}".utf8)
                continue
            }
            appendDecodedToken(token, to: &bytes)
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Encoding

    private func appendNormalSegment(_ segment: String, to ids: inout [Int]) {
        let escaped = segment.replacingOccurrences(of: " ", with: Self.escapedSpace)
        for chunk in newlineAwareChunks(escaped) {
            if chunk.allSatisfy({ $0 == "\n" }), let id = vocabulary.tokenToID(chunk) {
                ids.append(id)
                continue
            }

            var pieces = utf8ScalarPieces(chunk)
            pieces = applyMerges(pieces)

            for piece in pieces {
                if let id = vocabulary.tokenToID(piece) {
                    ids.append(id)
                } else {
                    appendByteFallback(piece, to: &ids)
                }
            }
        }
    }

    private func newlineAwareChunks(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var chunks = [String]()
        var current = ""
        var currentIsNewline: Bool?

        for scalar in text.unicodeScalars {
            let isNewline = scalar == "\n"
            if let currentIsNewline, currentIsNewline != isNewline {
                chunks.append(current)
                current.removeAll(keepingCapacity: true)
            }
            currentIsNewline = isNewline
            current.unicodeScalars.append(scalar)
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks
    }

    private func utf8ScalarPieces(_ text: String) -> [String] {
        text.unicodeScalars.map { String($0) }
    }

    private func appendByteFallback(_ piece: String, to ids: inout [Int]) {
        for byte in piece.utf8 {
            if let id = byteFallbackTable[byte] {
                ids.append(id)
            }
        }
    }

    private func applyMerges(_ initialPieces: [String]) -> [String] {
        var pieces = initialPieces
        while pieces.count >= 2 {
            var bestRank = Int.max
            var bestIndex = -1

            for index in 0..<(pieces.count - 1) {
                let key = Self.mergeKey(pieces[index], pieces[index + 1])
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }

            guard bestIndex >= 0 else { break }
            let key = Self.mergeKey(pieces[bestIndex], pieces[bestIndex + 1])
            guard let merged = mergeResults[key] else { break }
            pieces.replaceSubrange(bestIndex...(bestIndex + 1), with: [merged])
        }
        return pieces
    }

    // MARK: - Decoding

    private func appendDecodedToken(_ token: String, to bytes: inout [UInt8]) {
        if let byte = Self.parseHexByteToken(token) {
            bytes.append(byte)
            return
        }
        let text = token.replacingOccurrences(of: Self.escapedSpace, with: " ")
        bytes.append(contentsOf: text.utf8)
    }

    private static func parseHexByteToken(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else {
            return nil
        }
        let start = token.index(token.startIndex, offsetBy: 3)
        let end = token.index(token.startIndex, offsetBy: 5)
        return UInt8(token[start..<end], radix: 16)
    }

    // MARK: - Special Tokens

    private func splitAroundSpecialTokens(_ text: String) -> [String] {
        guard !specialTokens.specialTokenMap.isEmpty else { return [text] }
        let sortedSpecials = specialTokens.specialTokenMap.keys.sorted { $0.count > $1.count }

        var segments = [String]()
        var remaining = text
        while !remaining.isEmpty {
            var earliestRange: Range<String.Index>?
            var earliestToken: String?

            for special in sortedSpecials {
                if let range = remaining.range(of: special),
                   earliestRange == nil || range.lowerBound < earliestRange!.lowerBound {
                    earliestRange = range
                    earliestToken = special
                }
            }

            guard let range = earliestRange, let token = earliestToken else {
                segments.append(remaining)
                break
            }
            let prefix = String(remaining[..<range.lowerBound])
            if !prefix.isEmpty {
                segments.append(prefix)
            }
            segments.append(token)
            remaining = String(remaining[range.upperBound...])
        }
        return segments
    }

    private func specialTokenStringForID(_ id: Int) -> String? {
        if id == specialTokens.bosTokenID { return specialTokens.bosTokenString }
        if id == specialTokens.eosTokenID { return specialTokens.eosTokenString }
        if id == specialTokens.padTokenID { return specialTokens.padTokenString }
        for (token, tokenID) in specialTokens.specialTokenMap where tokenID == id {
            return token
        }
        return nil
    }

    private static func buildHexByteFallbackTable(vocabulary: TokenizerVocabulary) -> [UInt8: Int] {
        var table = [UInt8: Int](minimumCapacity: 256)
        for byte in UInt8.min...UInt8.max {
            let token = String(format: "<0x%02X>", byte)
            if let id = vocabulary.tokenToID(token) {
                table[byte] = id
            }
        }
        return table
    }

    private static func mergeKey(_ left: String, _ right: String) -> String {
        "\(left)\t\(right)"
    }
}
