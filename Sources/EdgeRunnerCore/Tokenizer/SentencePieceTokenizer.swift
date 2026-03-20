import Foundation

/// SentencePiece tokenizer using greedy bigram merge by float score.
///
/// Compatible with llama.cpp's SPM implementation, used by models such as
/// Gemma and Phi-3. Unlike BPE (where lower rank = merge first), SentencePiece
/// merges the pair with the **highest** score first.
public struct SentencePieceTokenizer: Tokenizer, Sendable {

    /// The SentencePiece space character (U+2581) used to represent whitespace.
    private static let spaceChar: Character = "\u{2581}"
    private static let spaceString: String = "\u{2581}"

    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let tokenScores: [String: Float]
    private let unknownTokenID: Int
    private let addSpacePrefix: Bool

    /// Whether to prepend BOS token during encoding (from GGUF metadata).
    public let shouldAddBOS: Bool

    /// Optional chat template engine for formatting conversation messages.
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int { vocabulary.count }
    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        tokenScores: [String: Float],
        unknownTokenID: Int,
        addSpacePrefix: Bool = true,
        shouldAddBOS: Bool = false,
        chatTemplateEngine: ChatTemplateEngine? = nil
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.tokenScores = tokenScores
        self.unknownTokenID = unknownTokenID
        self.addSpacePrefix = addSpacePrefix
        self.shouldAddBOS = shouldAddBOS
        self.chatTemplateEngine = chatTemplateEngine
    }

    // MARK: - Chat Template

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool
    ) throws -> String? {
        try applyChatTemplate(messages: messages, addGenerationPrompt: addGenerationPrompt, tools: nil)
    }

    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        tools: [ToolDefinition]? = nil
    ) throws -> String? {
        guard let engine = chatTemplateEngine else { return nil }
        let bosToken = specialTokens.bosTokenID.flatMap { vocabulary.idToToken($0) }
        let eosToken = vocabulary.idToToken(eosTokenID)
        return try engine.apply(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            bosToken: bosToken,
            eosToken: eosToken,
            tools: tools
        )
    }

    // MARK: - Encode

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        var ids = [Int]()

        if addBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        let segments = splitAroundSpecialTokens(text)
        var isFirstTextSegment = true

        for segment in segments {
            // Check if this segment is a special token.
            if let specialID = specialTokens.specialTokenMap[segment] {
                ids.append(specialID)
                continue
            }

            // Prepare text: replace spaces with ▁, optionally prepend ▁.
            var prepared = segment.replacingOccurrences(of: " ", with: Self.spaceString)
            if isFirstTextSegment && addSpacePrefix {
                prepared = Self.spaceString + prepared
            }
            isFirstTextSegment = false

            // Split into individual characters.
            var tokens = prepared.map { String($0) }

            // Apply greedy bigram merges.
            tokens = applyMerges(tokens)

            // Convert merged token strings to IDs.
            for token in tokens {
                if let id = vocabulary.tokenToID(token) {
                    ids.append(id)
                } else {
                    // Byte fallback: encode each byte as <0xHH>.
                    appendByteFallback(token, to: &ids)
                }
            }
        }

        return ids
    }

    // MARK: - Decode

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var pieces = ""
        for id in ids {
            if specialTokens.specialTokenIDs.contains(id) {
                if !skipSpecialTokens {
                    if let token = specialTokenStringForID(id) {
                        pieces += token
                    }
                }
                continue
            }
            if let token = vocabulary.idToToken(id) {
                pieces += token
            } else {
                pieces += "\u{FFFD}"
            }
        }

        // Replace ▁ with space, then strip leading space if addSpacePrefix was used.
        var result = pieces.replacingOccurrences(of: Self.spaceString, with: " ")
        if addSpacePrefix && result.hasPrefix(" ") {
            result.removeFirst()
        }
        return result
    }

    // MARK: - Private Helpers

    /// Greedy bigram merge: scan all adjacent pairs, find the one whose
    /// concatenation has the highest score in the vocabulary, merge it,
    /// and repeat until no mergeable pairs remain.
    private func applyMerges(_ initialTokens: [String]) -> [String] {
        var tokens = initialTokens
        while tokens.count >= 2 {
            var bestScore: Float = -.infinity
            var bestIndex = -1
            for i in 0..<(tokens.count - 1) {
                let merged = tokens[i] + tokens[i + 1]
                if let score = tokenScores[merged], score > bestScore {
                    bestScore = score
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let merged = tokens[bestIndex] + tokens[bestIndex + 1]
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

    /// Byte fallback: encode each UTF-8 byte of a token as `<0xHH>` and look it up.
    /// If even the byte token is not in the vocabulary, emit the unknown token.
    private func appendByteFallback(_ token: String, to ids: inout [Int]) {
        for byte in Array(token.utf8) {
            let hexToken = String(format: "<0x%02X>", byte)
            if let id = vocabulary.tokenToID(hexToken) {
                ids.append(id)
            } else {
                ids.append(unknownTokenID)
            }
        }
    }

    /// Splits text around special tokens, preserving them as separate segments.
    private func splitAroundSpecialTokens(_ text: String) -> [String] {
        guard !specialTokens.specialTokenMap.isEmpty else { return [text] }

        let sortedSpecials = specialTokens.specialTokenMap.keys
            .sorted { $0.count > $1.count }

        var segments = [String]()
        var remaining = text

        while !remaining.isEmpty {
            var earliestRange: Range<String.Index>?
            var earliestToken: String?

            for special in sortedSpecials {
                if let range = remaining.range(of: special) {
                    if earliestRange == nil || range.lowerBound < earliestRange!.lowerBound {
                        earliestRange = range
                        earliestToken = special
                    }
                }
            }

            guard let range = earliestRange, let _ = earliestToken else {
                segments.append(remaining)
                break
            }

            let prefix = String(remaining[remaining.startIndex..<range.lowerBound])
            if !prefix.isEmpty {
                segments.append(prefix)
            }
            segments.append(String(remaining[range]))
            remaining = String(remaining[range.upperBound...])
        }

        return segments
    }

    private func specialTokenStringForID(_ id: Int) -> String? {
        if id == specialTokens.bosTokenID { return specialTokens.bosTokenString }
        if id == specialTokens.eosTokenID { return specialTokens.eosTokenString }
        if id == specialTokens.padTokenID { return specialTokens.padTokenString }
        for (token, tokenID) in specialTokens.specialTokenMap {
            if tokenID == id { return token }
        }
        return nil
    }
}
