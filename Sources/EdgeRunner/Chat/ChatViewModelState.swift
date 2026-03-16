import Foundation

/// Pure state container for the chat view model (testable without SwiftUI).
public struct ChatViewModelState: Sendable {
    public var messages: [ChatMessage] = []
    public var isGenerating: Bool = false
    public var currentInput: String = ""
    public var selectedModel: ModelInfo? = nil
    public var memoryUsedMB: Double = 0
    public var memoryTotalMB: Double = 0
    public var tokensPerSecond: Double = 0
    public var error: String? = nil

    public init() {}

    public var memoryUsagePercent: Double {
        guard memoryTotalMB > 0 else { return 0 }
        return (memoryUsedMB / memoryTotalMB) * 100
    }

    public mutating func addUserMessage(_ content: String) {
        messages.append(ChatMessage(role: .user, content: content))
    }

    public mutating func addAssistantMessage(_ content: String) {
        messages.append(ChatMessage(role: .assistant, content: content))
    }

    public mutating func appendToLastMessage(_ text: String) {
        guard !messages.isEmpty else { return }
        messages[messages.count - 1].content += text
    }

    public mutating func clearMessages() {
        messages.removeAll()
    }

    public mutating func updateMemoryUsage(usedMB: Double, totalMB: Double) {
        self.memoryUsedMB = usedMB
        self.memoryTotalMB = totalMB
    }
}
