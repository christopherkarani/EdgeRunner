import Foundation

/// Lightweight message history manager for multi-turn conversations.
///
/// Tracks chat messages and provides convenience methods for building
/// conversation history. Use with `applyChatTemplate()` to format
/// the full history for the model.
///
/// ```swift
/// var convo = Conversation(systemPrompt: "You are helpful.")
/// convo.addUser("What is 2+2?")
/// let prompt = model.applyChatTemplate(
///     messages: convo.messages,
///     addGenerationPrompt: true
/// )
/// // ... generate response ...
/// convo.addAssistant(response)
/// convo.addUser("And 3+3?")
/// // KV cache prefix reuse happens automatically
/// ```
public struct Conversation: Sendable {
    public private(set) var messages: [ChatMessage]

    /// Create a new conversation, optionally with a system prompt.
    public init(systemPrompt: String? = nil) {
        if let systemPrompt {
            self.messages = [ChatMessage(role: .system, content: systemPrompt)]
        } else {
            self.messages = []
        }
    }

    /// Add a user message to the conversation.
    public mutating func addUser(_ content: String) {
        messages.append(ChatMessage(role: .user, content: content))
    }

    /// Add an assistant response to the conversation.
    public mutating func addAssistant(_ content: String) {
        messages.append(ChatMessage(role: .assistant, content: content))
    }

    /// Add a system message to the conversation.
    public mutating func addSystem(_ content: String) {
        messages.append(ChatMessage(role: .system, content: content))
    }

    /// The number of messages in the conversation.
    public var messageCount: Int { messages.count }

    /// Whether the conversation has any messages.
    public var isEmpty: Bool { messages.isEmpty }

    /// Reset the conversation. If keepSystem is true, preserves the initial system prompt.
    public mutating func reset(keepSystem: Bool = true) {
        if keepSystem, let first = messages.first, first.role == .system {
            messages = [first]
        } else {
            messages = []
        }
    }
}
