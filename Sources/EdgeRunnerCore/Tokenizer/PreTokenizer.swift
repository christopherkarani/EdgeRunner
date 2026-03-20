import Foundation

public protocol PreTokenizer: Sendable {
    func split(_ text: String) -> [String]
}

public struct RegexPreTokenizer: PreTokenizer, Sendable {
    private nonisolated(unsafe) let patterns: [Regex<Substring>]

    public init(pattern: Regex<Substring>) {
        self.patterns = [pattern]
    }

    public init(patterns: [Regex<Substring>]) {
        self.patterns = patterns
    }

    public func split(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var results = [String]()
        for pattern in patterns {
            let source = results.isEmpty ? [text] : results
            if !results.isEmpty {
                results = []
            }
            for chunk in source {
                results.append(contentsOf: chunk.matches(of: pattern).map { String($0.output) })
            }
        }
        return results
    }
}
