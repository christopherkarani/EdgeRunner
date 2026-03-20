import Foundation

/// Byte-Pair Encoding tokenizer with full production pipeline:
/// special-token scan -> pre-tokenize -> byte-encode -> BPE merge -> vocab lookup.
public struct BPETokenizer: Tokenizer, Sendable {
    private let vocabulary: TokenizerVocabulary
    private let specialTokens: SpecialTokens
    private let mergeRanks: [String: Int]
    private let mergeResults: [String: String]
    private let preTokenizer: PreTokenizer
    private let byteFallbackTable: [UInt8: Int]

    /// Whether to prepend BOS token during encoding (from GGUF metadata).
    public let shouldAddBOS: Bool

    /// Optional chat template engine for formatting conversation messages.
    public let chatTemplateEngine: ChatTemplateEngine?

    public var vocabularySize: Int {
        vocabulary.count
    }

    public var eosTokenID: Int { specialTokens.eosTokenID! }
    public var bosTokenID: Int? { specialTokens.bosTokenID }
    public var padTokenID: Int? { specialTokens.padTokenID }

    public init(
        vocabulary: TokenizerVocabulary,
        specialTokens: SpecialTokens,
        merges: [(String, String)],
        preTokenizer: PreTokenizer,
        shouldAddBOS: Bool = false,
        byteFallbackTable: [UInt8: Int]? = nil,
        chatTemplateEngine: ChatTemplateEngine? = nil
    ) {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.preTokenizer = preTokenizer
        self.shouldAddBOS = shouldAddBOS
        self.chatTemplateEngine = chatTemplateEngine
        var ranks = [String: Int](minimumCapacity: merges.count)
        var results = [String: String](minimumCapacity: merges.count)
        for (index, merge) in merges.enumerated() {
            let key = "\(merge.0)\t\(merge.1)"
            ranks[key] = index
            results[key] = merge.0 + merge.1
        }
        self.mergeRanks = ranks
        self.mergeResults = results

        // Use provided byte fallback table, or build one from vocabulary.
        if let provided = byteFallbackTable {
            self.byteFallbackTable = provided
        } else {
            var fallback = [UInt8: Int]()
            for b in UInt8.min...UInt8.max {
                let encoded = String(ByteEncoder.encode(b))
                if let id = vocabulary.tokenToID(encoded) {
                    fallback[b] = id
                }
            }
            self.byteFallbackTable = fallback
        }
    }

    // MARK: - Chat Template

    /// Apply the chat template to format conversation messages.
    /// Returns nil if no chat template is available.
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

    public func encode(_ text: String, addBOS: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }
        var ids = [Int]()

        let useBOS = addBOS || shouldAddBOS
        if useBOS, let bosID = specialTokens.bosTokenID {
            ids.append(bosID)
        }

        let segments = splitAroundSpecialTokens(text)

        for segment in segments {
            if let specialID = specialTokens.specialTokenMap[segment] {
                ids.append(specialID)
                continue
            }

            // Pre-tokenize the segment into words/chunks.
            let chunks = preTokenizer.split(segment)

            for chunk in chunks {
                // Byte-encode the chunk using GPT-2 byte-to-unicode mapping.
                let byteEncoded = ByteEncoder.encodeString(chunk)

                // Split into individual characters, then apply BPE merges.
                var tokens = byteEncoded.map { String($0) }
                tokens = applyMerges(tokens)

                // Look up each merged token in the vocabulary.
                for token in tokens {
                    if let id = vocabulary.tokenToID(token) {
                        ids.append(id)
                    } else {
                        // Byte fallback: encode each UTF-8 byte of the token individually.
                        appendByteFallback(token, to: &ids)
                    }
                }
            }
        }

        return ids
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var encoded = ""
        for id in ids {
            if specialTokens.specialTokenIDs.contains(id) {
                if !skipSpecialTokens {
                    if let token = specialTokenStringForID(id) {
                        encoded += token
                    }
                }
                continue
            }
            if let token = vocabulary.idToToken(id) {
                encoded += token
            } else {
                // Unknown token ID: produce Unicode replacement character.
                encoded += "\u{FFFD}"
            }
        }
        // Decode byte-encoded string back to UTF-8.
        return ByteEncoder.decodeString(encoded) ?? encoded
    }

    // MARK: - Private

    /// Splits text around special tokens, preserving them as separate segments.
    /// Returns an array alternating between normal text and special token strings.
    private func splitAroundSpecialTokens(_ text: String) -> [String] {
        guard !specialTokens.specialTokenMap.isEmpty else { return [text] }

        // Sort special tokens by length (longest first) for greedy matching.
        let sortedSpecials = specialTokens.specialTokenMap.keys
            .sorted { $0.count > $1.count }

        var segments = [String]()
        var remaining = text

        while !remaining.isEmpty {
            // Find the earliest occurrence of any special token.
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

            guard let range = earliestRange, let token = earliestToken else {
                // No more special tokens found; emit the rest as normal text.
                segments.append(remaining)
                break
            }

            // Emit text before the special token.
            let prefix = String(remaining[remaining.startIndex..<range.lowerBound])
            if !prefix.isEmpty {
                segments.append(prefix)
            }

            // Emit the special token itself.
            segments.append(token)

            // Advance past the special token.
            remaining = String(remaining[range.upperBound...])
        }

        return segments
    }

    /// Byte fallback: for tokens not in the vocabulary, encode each byte
    /// of the token's UTF-8 representation and look them up individually.
    private func appendByteFallback(_ token: String, to ids: inout [Int]) {
        for byte in Array(token.utf8) {
            if let id = byteFallbackTable[byte] {
                ids.append(id)
            }
            // If even the byte is not in vocab, we silently skip it.
        }
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
        // Check additional special tokens.
        for (token, tokenID) in specialTokens.specialTokenMap {
            if tokenID == id { return token }
        }
        return nil
    }
}
