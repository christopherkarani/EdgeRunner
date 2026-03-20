import Foundation
import EdgeRunnerIO

/// Errors that can occur when creating a tokenizer from GGUF metadata.
public enum TokenizerFactoryError: Error, Sendable, Equatable {
    case unsupportedModel(String)
    case missingRequiredToken(String)
}

/// Factory that bridges `GGUFTokenizerMetadata` to a configured `BPETokenizer`.
public enum TokenizerFactory: Sendable {

    /// Create a `BPETokenizer` from parsed GGUF tokenizer metadata.
    ///
    /// - Parameter metadata: The tokenizer metadata extracted from a GGUF file.
    /// - Returns: A fully configured `BPETokenizer`.
    /// - Throws: `TokenizerFactoryError` if the model type is unsupported or required tokens are missing.
    public static func create(from metadata: GGUFTokenizerMetadata) throws -> BPETokenizer {
        // 1. Validate model type (.gpt2 or .llamaBPE only)
        switch metadata.model {
        case .gpt2, .llamaBPE:
            break
        default:
            throw TokenizerFactoryError.unsupportedModel(metadata.model.rawValue)
        }

        // 2. Validate EOS exists and is within bounds
        guard let eosID = metadata.eosTokenID, eosID >= 0, eosID < metadata.tokens.count else {
            throw TokenizerFactoryError.missingRequiredToken("EOS")
        }

        // 3. Build TokenizerVocabulary from metadata.tokens
        let vocabulary = TokenizerVocabulary(tokens: metadata.tokens)

        // 4. Collect control/userDefined tokens from tokenTypes into additionalSpecialTokens
        var additionalSpecialTokens = [String: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                if tokenType == .control || tokenType == .userDefined {
                    let tokenString = metadata.tokens[index]
                    additionalSpecialTokens[tokenString] = index
                }
            }
        }

        // 5. Build SpecialTokens with BOS/EOS/PAD + additional
        //    Look up the token STRING from metadata.tokens[id] for each.
        let bosToken: (String, Int)?
        if let bosID = metadata.bosTokenID, bosID >= 0, bosID < metadata.tokens.count {
            let bosString = metadata.tokens[bosID]
            bosToken = (bosString, bosID)
            // Remove from additional if present (avoid duplicates)
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

        // 6. Build byte fallback table from tokens with GGUFTokenType.byte
        var byteFallbackTable = [UInt8: Int]()
        if let tokenTypes = metadata.tokenTypes {
            for (index, tokenType) in tokenTypes.enumerated() {
                guard tokenType == .byte else { continue }
                let tokenString = metadata.tokens[index]
                if let byte = parseByteToken(tokenString) {
                    byteFallbackTable[byte] = index
                }
            }
        }

        // 7. Convert merges to [(String, String)]
        let merges: [(String, String)] = metadata.merges.map { ($0.left, $0.right) }

        // 8. Resolve PreTokenizer from metadata.preTokenizer
        let preTokenizer = PreTokenizerPattern.resolve(metadata.preTokenizer)

        // 9. Create ChatTemplateEngine from metadata.chatTemplate (try?)
        let chatTemplateEngine: ChatTemplateEngine?
        if let template = metadata.chatTemplate {
            chatTemplateEngine = try? ChatTemplateEngine(template: template)
        } else {
            chatTemplateEngine = nil
        }

        // 10. Read shouldAddBOS from metadata
        let shouldAddBOS = metadata.shouldAddBOS ?? false

        // 11. Return BPETokenizer
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
