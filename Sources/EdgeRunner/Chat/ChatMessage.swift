import Foundation

/// A single message in a chat conversation.
public struct ChatMessage: Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public var content: String
    public let timestamp: Date

    public enum MessageRole: String, Sendable {
        case user, assistant, system
    }

    public init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}
