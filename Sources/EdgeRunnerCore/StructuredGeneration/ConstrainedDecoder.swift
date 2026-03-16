import Foundation

/// Applies grammar constraints to logits during generation,
/// ensuring the model only produces tokens that are valid JSON
/// at the current position.
public struct ConstrainedDecoder: Sendable {
    private let vocabulary: [String]

    public init(vocabulary: [String]) {
        self.vocabulary = vocabulary
    }

    /// Compute a boolean mask indicating which tokens are valid
    /// given the current grammar state.
    public func computeMask(for state: GrammarState) -> [Bool] {
        let allowed = state.allowedNextCharacters()
        return vocabulary.map { token in
            guard let firstChar = token.first else { return false }
            return allowed.contains(String(firstChar))
        }
    }

    /// Apply a boolean mask to logits: set disallowed tokens to -infinity.
    public func applyMask(_ mask: [Bool], to logits: [Float]) -> [Float] {
        precondition(mask.count == logits.count, "Mask and logits must have same length")
        var result = logits
        for i in 0..<result.count {
            if !mask[i] {
                result[i] = -.infinity
            }
        }
        return result
    }
}
