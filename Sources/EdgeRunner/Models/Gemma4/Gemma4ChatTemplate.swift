import Foundation

/// Conversation role for a Gemma 4 chat turn.
///
/// Gemma 4 renders `assistant` turns as `model` to match the HuggingFace
/// `google/gemma-4-E4B-it` chat template. The `tool` role is reserved for
/// future tool-calling support and is rejected by the v1 renderer.
public enum Gemma4ChatRole: String, Sendable, Equatable {
    case system
    case user
    case assistant
    case model
    case tool
}

/// A single message in a Gemma 4 conversation.
public struct Gemma4ChatMessage: Sendable, Equatable {
    public let role: Gemma4ChatRole
    public let content: String

    public init(role: Gemma4ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// Errors surfaced by the Gemma 4 chat template renderer.
public enum Gemma4ChatTemplateError: Error, Equatable {
    /// A `.tool` message was supplied. Tool sentinels are wired in the HF
    /// template but not supported in v1 of this renderer.
    case toolsUnsupportedInV1
}

/// Renders Gemma 4 chat messages using the custom sentinel format from
/// `chat_template.jinja` in `google/gemma-4-E4B-it`.
///
/// Format:
/// ```text
/// <|turn>{role}
/// {content}
/// <turn|>
/// ```
///
/// The `assistant` role is rewritten to `model`. When `addGenerationPrompt`
/// is `true`, a trailing `<|turn>model\n` is appended so the model can
/// complete the next turn.
public enum Gemma4ChatTemplate: Sendable {
    /// Renders messages into the Gemma 4 prompt string.
    ///
    /// Returns an empty string if any message uses the unsupported `.tool`
    /// role. Use ``renderThrowing(messages:addGenerationPrompt:)`` to surface
    /// that condition as an error.
    public static func render(
        messages: [Gemma4ChatMessage],
        addGenerationPrompt: Bool
    ) -> String {
        (try? renderThrowing(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt
        )) ?? ""
    }

    /// Renders messages into the Gemma 4 prompt string, throwing on
    /// unsupported inputs.
    public static func renderThrowing(
        messages: [Gemma4ChatMessage],
        addGenerationPrompt: Bool
    ) throws -> String {
        var output = ""
        var loopMessages = messages

        output.append("<|turn>system\n<|think|>\n")
        if let first = messages.first, first.role == .system {
            output.append(first.content.trimmingCharacters(in: .whitespacesAndNewlines))
            loopMessages.removeFirst()
        }
        output.append("<turn|>\n")

        for message in loopMessages {
            let role = try renderedRole(for: message.role)
            output.append("<|turn>\(role)\n")
            output.append(message.content.trimmingCharacters(in: .whitespacesAndNewlines))
            output.append("<turn|>\n")
        }
        if addGenerationPrompt {
            output.append("<|turn>model\n")
        }
        return output
    }

    private static func renderedRole(for role: Gemma4ChatRole) throws -> String {
        switch role {
        case .system, .user, .model:
            return role.rawValue
        case .assistant:
            return Gemma4ChatRole.model.rawValue
        case .tool:
            throw Gemma4ChatTemplateError.toolsUnsupportedInV1
        }
    }
}
