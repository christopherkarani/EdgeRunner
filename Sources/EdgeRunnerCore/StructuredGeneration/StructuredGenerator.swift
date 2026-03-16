import Foundation

/// Utilities for structured (typed) generation from model output.
public enum StructuredGenerator {

    /// Parse a JSON string into a Decodable type.
    public static func parse<T: Decodable>(json: String) throws -> T {
        guard let data = json.data(using: .utf8) else {
            throw GenerationError.structuredOutputFailed(reason: "Invalid UTF-8 in JSON string")
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw GenerationError.structuredOutputFailed(
                reason: "JSON decode failed: \(error.localizedDescription)"
            )
        }
    }

    /// Extract a JSON object or array from model output text.
    ///
    /// Handles common patterns:
    /// - JSON in a ```json code block
    /// - Raw JSON starting with { or [
    /// - JSON embedded in surrounding text
    public static func extractJSON(from text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Try code block extraction first
        if let codeBlockJSON = extractFromCodeBlock(trimmed) {
            return codeBlockJSON
        }

        // Try bracket matching
        if let bracketJSON = extractByBracketMatching(trimmed) {
            return bracketJSON
        }

        throw GenerationError.structuredOutputFailed(reason: "No valid JSON found")
    }

    private static func extractFromCodeBlock(_ text: String) -> String? {
        let patterns = ["```json\\s*\\n([\\s\\S]*?)\\n\\s*```", "```\\s*\\n([\\s\\S]*?)\\n\\s*```"]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
               let range = Range(match.range(at: 1), in: text) {
                return String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return nil
    }

    private static func extractByBracketMatching(_ text: String) -> String? {
        guard let startIndex = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else { return nil }
        let openBracket: Character = text[startIndex]
        let closeBracket: Character = openBracket == "{" ? "}" : "]"
        var depth = 0; var inString = false; var escaped = false
        for index in text[startIndex...].indices {
            let char = text[index]
            if escaped { escaped = false; continue }
            if char == "\\" && inString { escaped = true; continue }
            if char == "\"" { inString.toggle(); continue }
            if !inString {
                if char == openBracket { depth += 1 }
                else if char == closeBracket { depth -= 1; if depth == 0 { return String(text[startIndex...index]) } }
            }
        }
        return nil
    }
}
