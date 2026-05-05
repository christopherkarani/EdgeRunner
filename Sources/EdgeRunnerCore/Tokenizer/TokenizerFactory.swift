import Foundation
import EdgeRunnerIO

/// Errors that can occur when creating a tokenizer from GGUF metadata.
public enum TokenizerFactoryError: Error, Sendable, Equatable {
    case unsupportedModel(String)
    case missingRequiredToken(String)
}

/// Factory that bridges `GGUFTokenizerMetadata` to a configured `Tokenizer`.
public enum TokenizerFactory: Sendable {

    /// Create a `Tokenizer` from parsed GGUF tokenizer metadata.
    ///
    /// - Parameter metadata: The tokenizer metadata extracted from a GGUF file.
    /// - Returns: A fully configured tokenizer (BPE or SentencePiece).
    /// - Throws: `TokenizerFactoryError` if the model type is unsupported or required tokens are missing.
    public static func create(from metadata: GGUFTokenizerMetadata) throws -> any Tokenizer {
        switch metadata.model {
        case .gpt2, .llamaBPE:
            return try createBPE(from: metadata)
        case .gemma4:
            return try createGemma4BPE(from: metadata)
        case .llama, .sentencePiece:
            return try createSentencePiece(from: metadata)
        default:
            throw TokenizerFactoryError.unsupportedModel(metadata.model.rawValue)
        }
    }

    // MARK: - BPE Factory

    private static func createBPE(from metadata: GGUFTokenizerMetadata) throws -> BPETokenizer {
        // 1. Validate EOS exists and is within bounds
        guard let eosID = metadata.eosTokenID, eosID >= 0, eosID < metadata.tokens.count else {
            throw TokenizerFactoryError.missingRequiredToken("EOS")
        }

        // 2. Build TokenizerVocabulary from metadata.tokens
        let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)

        // 3. Collect control/userDefined tokens from tokenTypes into additionalSpecialTokens
        let (specialTokens, _) = buildSpecialTokens(from: metadata, eosID: eosID)

        // 4. Build byte fallback table from tokens with GGUFTokenType.byte
        let byteFallbackTable = buildByteFallbackTable(from: metadata)

        // 5. Convert merges to [(String, String)]
        let merges: [(String, String)] = metadata.merges.map { ($0.left, $0.right) }

        // 6. Resolve PreTokenizer from metadata.preTokenizer
        let preTokenizer = PreTokenizerPattern.resolve(metadata.preTokenizer)

        // 7. Create ChatTemplateEngine from metadata.chatTemplate (try?)
        let chatTemplateEngine = buildChatTemplateEngine(from: metadata)

        // 8. Read shouldAddBOS from metadata (default false for BPE)
        let shouldAddBOS = metadata.shouldAddBOS ?? false

        // 9. Return BPETokenizer
        return BPETokenizer(
            vocabulary: vocabulary,
            specialTokens: specialTokens,
            merges: merges,
            preTokenizer: preTokenizer,
            shouldAddBOS: shouldAddBOS,
            byteFallbackTable: byteFallbackTable.isEmpty ? nil : byteFallbackTable,
            chatTemplateEngine: chatTemplateEngine
        )
    }

    private static func createGemma4BPE(from metadata: GGUFTokenizerMetadata) throws -> Gemma4BPETokenizer {
        guard let eosID = metadata.eosTokenID, eosID >= 0, eosID < metadata.tokens.count else {
            throw TokenizerFactoryError.missingRequiredToken("EOS")
        }

        let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)
        let (specialTokens, _) = buildSpecialTokens(from: metadata, eosID: eosID)
        let byteFallbackTable = buildByteFallbackTable(from: metadata)
        let chatTemplateEngine = buildChatTemplateEngine(from: metadata)
        let merges = metadata.merges.map { ($0.left, $0.right) }

        return Gemma4BPETokenizer(
            vocabulary: vocabulary,
            specialTokens: specialTokens,
            merges: merges,
            shouldAddBOS: metadata.shouldAddBOS ?? true,
            byteFallbackTable: byteFallbackTable.isEmpty ? nil : byteFallbackTable,
            chatTemplateEngine: chatTemplateEngine
        )
    }

    // MARK: - SentencePiece Factory

    private static func createSentencePiece(from metadata: GGUFTokenizerMetadata) throws -> SentencePieceTokenizer {
        // 1. Validate EOS exists and is within bounds
        guard let eosID = metadata.eosTokenID, eosID >= 0, eosID < metadata.tokens.count else {
            throw TokenizerFactoryError.missingRequiredToken("EOS")
        }

        // 2. Validate scores exist (required for SentencePiece merge scoring)
        guard let scores = metadata.scores else {
            throw TokenizerFactoryError.missingRequiredToken("scores")
        }

        // 3. Build TokenizerVocabulary from metadata.tokens
        let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)

        // 4. Build tokenScores dictionary from tokens + scores arrays
        var tokenScores = [String: Float]()
        tokenScores.reserveCapacity(metadata.tokens.count)
        for (index, token) in metadata.tokens.enumerated() {
            tokenScores[token] = scores[index]
        }

        // 5. Collect control/userDefined tokens as special tokens (same as BPE)
        let (specialTokens, _) = buildSpecialTokens(from: metadata, eosID: eosID)

        // 6. Resolve unknown token ID (default to 0 if not specified)
        let unknownTokenID = metadata.unknownTokenID ?? 0

        // 7. Create ChatTemplateEngine from metadata.chatTemplate
        let chatTemplateEngine = buildChatTemplateEngine(from: metadata)

        // 8. Read addSpacePrefix and shouldAddBOS (default true for SPM)
        let addSpacePrefix = metadata.addSpacePrefix ?? true
        let shouldAddBOS = metadata.shouldAddBOS ?? true

        // 9. Return SentencePieceTokenizer
        return SentencePieceTokenizer(
            vocabulary: vocabulary,
            specialTokens: specialTokens,
            tokenScores: tokenScores,
            unknownTokenID: unknownTokenID,
            addSpacePrefix: addSpacePrefix,
            shouldAddBOS: shouldAddBOS,
            chatTemplateEngine: chatTemplateEngine
        )
    }

    // MARK: - Shared Helpers

    /// Collects special tokens (BOS/EOS/PAD + control/userDefined) from metadata.
    ///
    /// - Returns: A tuple of `(SpecialTokens, additionalSpecialTokens)`.
    private static func buildSpecialTokens(
        from metadata: GGUFTokenizerMetadata,
        eosID: Int
    ) -> (SpecialTokens, [String: Int]) {
        var additionalSpecialTokens = [String: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                if tokenType == .control || tokenType == .userDefined {
                    let tokenString = metadata.tokens[index]
                    additionalSpecialTokens[tokenString] = index
                }
            }
        }

        let bosToken: (String, Int)?
        if let bosID = metadata.bosTokenID, bosID >= 0, bosID < metadata.tokens.count {
            let bosString = metadata.tokens[bosID]
            bosToken = (bosString, bosID)
            additionalSpecialTokens.removeValue(forKey: bosString)
        } else {
            bosToken = nil
        }

        let eosString = metadata.tokens[eosID]
        let eosToken = (eosString, eosID)
        additionalSpecialTokens.removeValue(forKey: eosString)

        let padToken: (String, Int)?
        if let padID = metadata.paddingTokenID, padID >= 0, padID < metadata.tokens.count {
            let padString = metadata.tokens[padID]
            padToken = (padString, padID)
            additionalSpecialTokens.removeValue(forKey: padString)
        } else {
            padToken = nil
        }

        let specialTokens = SpecialTokens(
            bosToken: bosToken,
            eosToken: eosToken,
            padToken: padToken,
            additionalSpecialTokens: additionalSpecialTokens
        )

        return (specialTokens, additionalSpecialTokens)
    }

    /// Builds a byte fallback table from type-6 tokens.
    private static func buildByteFallbackTable(from metadata: GGUFTokenizerMetadata) -> [UInt8: Int] {
        var table = [UInt8: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                guard tokenType == .byte else { continue }
                let tokenString = metadata.tokens[index]
                if let byte = parseByteToken(tokenString) {
                    table[byte] = index
                }
            }
        }
        return table
    }

    /// Creates a ChatTemplateEngine from metadata, returning nil if not present or invalid.
    private static func buildChatTemplateEngine(from metadata: GGUFTokenizerMetadata) -> ChatTemplateEngine? {
        guard let template = metadata.chatTemplate else { return nil }
        return try? ChatTemplateEngine(template: template)
    }

    // MARK: - Private

    /// Parse a byte token string in `<0xHH>` format to a `UInt8`.
    private static func parseByteToken(_ token: String) -> UInt8? {
        // Match <0xHH> format
        if token.hasPrefix("<0x") && token.hasSuffix(">") {
            let hexStart = token.index(token.startIndex, offsetBy: 3)
            let hexEnd = token.index(before: token.endIndex)
            guard hexStart < hexEnd else { return nil }
            let hexString = String(token[hexStart..<hexEnd])
            guard let value = UInt8(hexString, radix: 16) else { return nil }
            return value
        }
        // Single byte-encoded character fallback
        if token.unicodeScalars.count == 1 {
            let scalar = token.unicodeScalars.first!
            if scalar.value < 256 {
                return UInt8(scalar.value)
            }
        }
        return nil
    }
}
