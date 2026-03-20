import Foundation

public enum GGUFTokenizerMetadataError: Error, Sendable, Equatable {
    case missingKey(String)
    case invalidValue(key: String, description: String)
}

public enum GGUFTokenizerModel: Sendable, Equatable {
    case gpt2
    case llama
    case llamaBPE
    case sentencePiece
    case wordPiece
    case unknown(String)

    public init(rawValue: String) {
        switch rawValue.lowercased() {
        case "gpt2":
            self = .gpt2
        case "llama":
            self = .llama
        case "llama-bpe":
            self = .llamaBPE
        case "sentencepiece":
            self = .sentencePiece
        case "wordpiece":
            self = .wordPiece
        default:
            self = .unknown(rawValue)
        }
    }

    public var rawValue: String {
        switch self {
        case .gpt2:
            return "gpt2"
        case .llama:
            return "llama"
        case .llamaBPE:
            return "llama-bpe"
        case .sentencePiece:
            return "sentencepiece"
        case .wordPiece:
            return "wordpiece"
        case .unknown(let value):
            return value
        }
    }
}

public struct GGUFTokenType: RawRepresentable, Sendable, Equatable, Hashable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let undefined = Self(rawValue: 0)
    public static let normal = Self(rawValue: 1)
    public static let unknown = Self(rawValue: 2)
    public static let control = Self(rawValue: 3)
    public static let userDefined = Self(rawValue: 4)
    public static let unused = Self(rawValue: 5)
    public static let byte = Self(rawValue: 6)
}

public struct GGUFTokenizerMerge: Sendable, Equatable {
    public let left: String
    public let right: String
    public let rawValue: String

    public init(left: String, right: String, rawValue: String) {
        self.left = left
        self.right = right
        self.rawValue = rawValue
    }
}

public struct GGUFTokenizerMetadata: Sendable, Equatable {
    public let model: GGUFTokenizerModel
    public let preTokenizer: String?
    public let tokens: [String]
    public let merges: [GGUFTokenizerMerge]
    public let tokenTypes: [GGUFTokenType]?
    public let scores: [Float]?
    public let bosTokenID: Int?
    public let eosTokenID: Int?
    public let paddingTokenID: Int?
    public let unknownTokenID: Int?
    public let shouldAddBOS: Bool?
    public let addSpacePrefix: Bool?
    public let chatTemplate: String?

    public var vocabularySize: Int { tokens.count }

    public init(ggufMetadata metadata: [String: GGUFMetadataValue]) throws {
        try self.init(
            stringValueForKey: { metadata[$0]?.stringValue },
            intValueForKey: { metadata[$0]?.intValue },
            boolValueForKey: { metadata[$0]?.boolValue },
            stringArrayForKey: { metadata[$0]?.stringArrayValue },
            intArrayForKey: { metadata[$0]?.intArrayValue },
            floatArrayForKey: { metadata[$0]?.floatArrayValue }
        )
    }

    public init(metadata: [String: MetadataValue]) throws {
        try self.init(
            stringValueForKey: { metadata[$0]?.stringValue },
            intValueForKey: { metadata[$0]?.intValue },
            boolValueForKey: { metadata[$0]?.boolValue },
            stringArrayForKey: { metadata[$0]?.stringArrayValue },
            intArrayForKey: { metadata[$0]?.intArrayValue },
            floatArrayForKey: { metadata[$0]?.floatArrayValue }
        )
    }

    private init(
        stringValueForKey: (String) -> String?,
        intValueForKey: (String) -> Int?,
        boolValueForKey: (String) -> Bool?,
        stringArrayForKey: (String) -> [String]?,
        intArrayForKey: (String) -> [Int]?,
        floatArrayForKey: (String) -> [Float]?
    ) throws {
        let modelKey = "tokenizer.ggml.model"
        guard let rawModel = stringValueForKey(modelKey), !rawModel.isEmpty else {
            throw GGUFTokenizerMetadataError.missingKey(modelKey)
        }

        let tokensKey = "tokenizer.ggml.tokens"
        guard let tokens = stringArrayForKey(tokensKey), !tokens.isEmpty else {
            throw GGUFTokenizerMetadataError.missingKey(tokensKey)
        }

        let merges = try Self.parseMerges(stringArrayForKey("tokenizer.ggml.merges") ?? [])

        let tokenTypes: [GGUFTokenType]?
        if let rawTokenTypes = intArrayForKey("tokenizer.ggml.token_type") {
            guard rawTokenTypes.count == tokens.count else {
                throw GGUFTokenizerMetadataError.invalidValue(
                    key: "tokenizer.ggml.token_type",
                    description: "Expected \(tokens.count) token types, found \(rawTokenTypes.count)"
                )
            }
            tokenTypes = rawTokenTypes.map(GGUFTokenType.init(rawValue:))
        } else {
            tokenTypes = nil
        }

        let scores: [Float]?
        if let rawScores = floatArrayForKey("tokenizer.ggml.scores") {
            guard rawScores.count == tokens.count else {
                throw GGUFTokenizerMetadataError.invalidValue(
                    key: "tokenizer.ggml.scores",
                    description: "Expected \(tokens.count) scores, found \(rawScores.count)"
                )
            }
            scores = rawScores
        } else {
            scores = nil
        }

        self.model = GGUFTokenizerModel(rawValue: rawModel)
        self.preTokenizer = stringValueForKey("tokenizer.ggml.pre")
        self.tokens = tokens
        self.merges = merges
        self.tokenTypes = tokenTypes
        self.scores = scores
        self.bosTokenID = intValueForKey("tokenizer.ggml.bos_token_id")
        self.eosTokenID = intValueForKey("tokenizer.ggml.eos_token_id")
        self.paddingTokenID = intValueForKey("tokenizer.ggml.padding_token_id")
        self.unknownTokenID = intValueForKey("tokenizer.ggml.unknown_token_id")
        self.shouldAddBOS = boolValueForKey("tokenizer.ggml.add_bos_token")
        self.addSpacePrefix = boolValueForKey("tokenizer.ggml.add_space_prefix")
        self.chatTemplate = stringValueForKey("tokenizer.chat_template")
    }

    private static func parseMerges(_ rawMerges: [String]) throws -> [GGUFTokenizerMerge] {
        try rawMerges.map { rawMerge in
            guard let separatorIndex = rawMerge.firstIndex(where: \.isWhitespace) else {
                throw GGUFTokenizerMetadataError.invalidValue(
                    key: "tokenizer.ggml.merges",
                    description: "Invalid merge entry '\(rawMerge)'"
                )
            }

            let left = String(rawMerge[..<separatorIndex])
            let rightStart = rawMerge[separatorIndex...].drop { $0.isWhitespace }
            let right = String(rightStart)
            guard !left.isEmpty, !right.isEmpty else {
                throw GGUFTokenizerMetadataError.invalidValue(
                    key: "tokenizer.ggml.merges",
                    description: "Invalid merge entry '\(rawMerge)'"
                )
            }

            return GGUFTokenizerMerge(left: left, right: right, rawValue: rawMerge)
        }
    }
}

public extension ModelConfig {
    func tokenizerMetadata() throws -> GGUFTokenizerMetadata {
        try GGUFTokenizerMetadata(metadata: metadata)
    }
}

public extension GGUFMetadataValue {
    var stringArrayValue: [String]? {
        guard let values = arrayValue else { return nil }
        let strings = values.compactMap(\.stringValue)
        return strings.count == values.count ? strings : nil
    }

    var intArrayValue: [Int]? {
        guard let values = arrayValue else { return nil }
        let integers = values.compactMap(\.intValue)
        return integers.count == values.count ? integers : nil
    }

    var floatArrayValue: [Float]? {
        guard let values = arrayValue else { return nil }
        let floats = values.compactMap(\.floatValue)
        return floats.count == values.count ? floats : nil
    }
}

public extension MetadataValue {
    var stringArrayValue: [String]? {
        guard let values = arrayValue else { return nil }
        let strings = values.compactMap(\.stringValue)
        return strings.count == values.count ? strings : nil
    }

    var intArrayValue: [Int]? {
        guard let values = arrayValue else { return nil }
        let integers = values.compactMap(\.intValue)
        return integers.count == values.count ? integers : nil
    }
}
