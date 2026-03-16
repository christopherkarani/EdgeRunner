import SwiftUI
import EdgeRunner

@Observable
final class ChatViewModel {
    var state = ChatViewModelState()

    func send() async {
        let input = state.currentInput.trimmingCharacters(in: .whitespaces)
        guard !input.isEmpty else { return }

        state.currentInput = ""
        state.addUserMessage(input)
        state.isGenerating = true
        state.error = nil

        state.addAssistantMessage("")

        // TODO: Integrate with GenerationSession for actual streaming inference.
        // For now, this is a placeholder that simulates a response.
        do {
            try await Task.sleep(for: .milliseconds(500))
            state.appendToLastMessage("This is a placeholder response. Connect a model to enable inference.")
        } catch {
            state.error = error.localizedDescription
        }

        state.isGenerating = false
    }

    func clearConversation() {
        state.clearMessages()
        state.error = nil
    }
}
