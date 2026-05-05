import Foundation

enum Gemma4DecodeMode: Equatable, Sendable {
    case fullPrefill(tokens: [Int], startPosition: Int)
    case prefixReuse(tokens: [Int], startPosition: Int)
    case decode(token: Int, position: Int)
}

struct Gemma4DecodeState: Sendable {
    private var processedTokenIDs: [Int] = []

    var cachedPosition: Int {
        processedTokenIDs.count
    }

    mutating func prepare(tokenIDs: [Int]) -> Gemma4DecodeMode {
        guard !tokenIDs.isEmpty else {
            return .fullPrefill(tokens: [], startPosition: 0)
        }
        guard !processedTokenIDs.isEmpty else {
            return .fullPrefill(tokens: tokenIDs, startPosition: 0)
        }
        guard tokenIDs.count >= processedTokenIDs.count else {
            return .fullPrefill(tokens: tokenIDs, startPosition: 0)
        }
        guard tokenIDs.starts(with: processedTokenIDs) else {
            return .fullPrefill(tokens: tokenIDs, startPosition: 0)
        }

        let suffix = Array(tokenIDs.dropFirst(processedTokenIDs.count))
        if suffix.count == 1 {
            return .decode(token: suffix[0], position: processedTokenIDs.count)
        }
        if suffix.isEmpty {
            return .prefixReuse(tokens: [], startPosition: processedTokenIDs.count)
        }
        return .prefixReuse(tokens: suffix, startPosition: processedTokenIDs.count)
    }

    mutating func markProcessed(tokenIDs: [Int]) {
        processedTokenIDs = tokenIDs
    }

    mutating func reset() {
        processedTokenIDs.removeAll(keepingCapacity: true)
    }
}
