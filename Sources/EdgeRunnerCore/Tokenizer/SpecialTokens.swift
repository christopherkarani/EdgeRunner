import Foundation

/// Container for special token definitions (BOS, EOS, PAD, etc.).
public struct SpecialTokens: Sendable {
    private let _bosString: String?
    private let _bosID: Int?
    private let _eosString: String?
    private let _eosID: Int?
    private let _padString: String?
    private let _padID: Int?

    public var bosTokenID: Int? { _bosID }
    public var eosTokenID: Int? { _eosID }
    public var padTokenID: Int? { _padID }
    public var bosTokenString: String? { _bosString }
    public var eosTokenString: String? { _eosString }
    public var padTokenString: String? { _padString }

    /// Set of all special token IDs for quick lookup.
    public let specialTokenIDs: Set<Int>
    /// Mapping from special token string to ID.
    public let specialTokenMap: [String: Int]

    public init(
        bosToken: (String, Int)?,
        eosToken: (String, Int)?,
        padToken: (String, Int)?,
        additionalSpecialTokens: [String: Int] = [:]
    ) {
        self._bosString = bosToken?.0
        self._bosID = bosToken?.1
        self._eosString = eosToken?.0
        self._eosID = eosToken?.1
        self._padString = padToken?.0
        self._padID = padToken?.1

        var ids = Set<Int>()
        var map = [String: Int]()
        if let bos = bosToken { ids.insert(bos.1); map[bos.0] = bos.1 }
        if let eos = eosToken { ids.insert(eos.1); map[eos.0] = eos.1 }
        if let pad = padToken { ids.insert(pad.1); map[pad.0] = pad.1 }
        for (token, id) in additionalSpecialTokens {
            ids.insert(id)
            map[token] = id
        }
        self.specialTokenIDs = ids
        self.specialTokenMap = map
    }
}
