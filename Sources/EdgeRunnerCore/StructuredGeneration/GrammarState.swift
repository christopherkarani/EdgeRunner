import Foundation

/// Tracks the state of JSON generation for grammar-guided constrained decoding.
///
/// Implements a simple state machine that determines which characters/tokens
/// are valid at each position in a JSON document conforming to a schema.
public struct GrammarState: Sendable {
    public enum ParseState: Sendable {
        case expectObjectOpen
        case expectKeyOrClose
        case expectColon
        case expectValue
        case expectCommaOrClose
        case expectArrayOpen
        case expectArrayValueOrClose
        case expectArrayCommaOrClose
        case complete
    }

    public let schema: JSONSchema
    public private(set) var state: ParseState
    public private(set) var buffer: String
    public private(set) var depth: Int

    public init(schema: JSONSchema) {
        self.schema = schema
        self.state = schema.type == .array ? .expectArrayOpen : .expectObjectOpen
        self.buffer = ""
        self.depth = 0
    }

    /// Returns the set of characters that are valid at the current position.
    public func allowedNextCharacters() -> Set<String> {
        switch state {
        case .expectObjectOpen:
            return ["{"]
        case .expectKeyOrClose:
            return ["\"", "}"]
        case .expectColon:
            return [":"]
        case .expectValue:
            // Allow any value start: string, number, bool, null, object, array
            return ["\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "t", "f", "n", "{", "["]
        case .expectCommaOrClose:
            return [",", "}"]
        case .expectArrayOpen:
            return ["["]
        case .expectArrayValueOrClose:
            return ["\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "t", "f", "n", "{", "[", "]"]
        case .expectArrayCommaOrClose:
            return [",", "]"]
        case .complete:
            return []
        }
    }

    /// Advance the state machine by one character.
    public mutating func advance(with character: String) {
        buffer += character

        switch (state, character) {
        case (.expectObjectOpen, "{"):
            depth += 1
            state = .expectKeyOrClose
        case (.expectKeyOrClose, "\""):
            state = .expectColon // simplified: skip key content
        case (.expectKeyOrClose, "}"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectColon, ":"):
            state = .expectValue
        case (.expectValue, "\""):
            state = .expectCommaOrClose // simplified: skip value content
        case (.expectValue, _) where character.first?.isNumber == true
             || character == "-" || character == "t" || character == "f" || character == "n":
            state = .expectCommaOrClose
        case (.expectValue, "{"):
            depth += 1
            state = .expectKeyOrClose
        case (.expectValue, "["):
            depth += 1
            state = .expectArrayValueOrClose
        case (.expectCommaOrClose, ","):
            state = .expectKeyOrClose
        case (.expectCommaOrClose, "}"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectArrayOpen, "["):
            depth += 1
            state = .expectArrayValueOrClose
        case (.expectArrayValueOrClose, "]"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        case (.expectArrayValueOrClose, _):
            state = .expectArrayCommaOrClose
        case (.expectArrayCommaOrClose, ","):
            state = .expectArrayValueOrClose
        case (.expectArrayCommaOrClose, "]"):
            depth -= 1
            state = depth > 0 ? .expectCommaOrClose : .complete
        default:
            break // stay in current state for content characters
        }
    }

    /// Validate that a complete JSON string conforms to the schema.
    public static func validate(json: String, against schema: JSONSchema) -> Bool {
        guard let data = json.data(using: .utf8) else { return false }
        guard let parsed = try? JSONSerialization.jsonObject(with: data) else { return false }
        return validateValue(parsed, against: schema)
    }

    private static func validateValue(_ value: Any, against schema: JSONSchema) -> Bool {
        switch schema.type {
        case .object:
            guard let dict = value as? [String: Any] else { return false }
            // Check required fields
            if let required = schema.required {
                for key in required {
                    guard dict[key] != nil else { return false }
                }
            }
            // Check property types
            if let properties = schema.properties {
                for (key, propSchema) in properties {
                    if let propValue = dict[key] {
                        if !validateValue(propValue, against: propSchema) {
                            return false
                        }
                    }
                }
            }
            return true

        case .array:
            guard let array = value as? [Any] else { return false }
            if let itemSchema = schema.items {
                for item in array {
                    if !validateValue(item, against: itemSchema) {
                        return false
                    }
                }
            }
            return true

        case .string:
            return value is String

        case .integer:
            if value is Int { return true }
            if let num = value as? NSNumber {
                return CFNumberIsFloatType(num as CFNumber) == false
            }
            return false

        case .number:
            return value is Double || value is Float || value is Int || value is NSNumber

        case .boolean:
            if let num = value as? NSNumber {
                return CFGetTypeID(num) == CFBooleanGetTypeID()
            }
            return false

        case .null:
            return value is NSNull
        }
    }
}
