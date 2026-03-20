import Foundation

/// A message in a chat conversation for template formatting.
public struct ChatMessage: Sendable, Equatable {
    /// The role of the message sender (e.g., "system", "user", "assistant", "tool").
    public let role: String
    /// The text content of the message.
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// A tool definition for function-calling chat templates.
public struct ToolDefinition: Sendable, Equatable {
    /// The tool function name.
    public let name: String
    /// A description of what the tool does.
    public let description: String
    /// JSON string describing the tool's parameters schema.
    public let parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}
