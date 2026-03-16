import Testing
import Foundation
@testable import EdgeRunner

@Suite("ChatMessage")
struct ChatMessageTests {
    @Test func messageCreation() {
        let msg = ChatMessage(role: .user, content: "Hello")
        #expect(msg.role == .user)
        #expect(msg.content == "Hello")
    }

    @Test func messageRoles() {
        let user = ChatMessage(role: .user, content: "")
        let assistant = ChatMessage(role: .assistant, content: "")
        let system = ChatMessage(role: .system, content: "")
        #expect(user.role == .user)
        #expect(assistant.role == .assistant)
        #expect(system.role == .system)
    }
}

@Suite("ModelInfo")
struct ModelInfoTests {
    @Test func modelInfoProperties() {
        let info = ModelInfo(
            name: "Llama 3 8B Q4_0",
            path: URL(fileURLWithPath: "/models/llama-3-8b-q4_0.gguf"),
            format: "gguf", parameterCount: "8B",
            quantization: "Q4_0", fileSizeBytes: 4_500_000_000
        )
        #expect(info.name == "Llama 3 8B Q4_0")
        #expect(info.format == "gguf")
        #expect(info.quantization == "Q4_0")
    }

    @Test func fileSizeFormatted() {
        let info = ModelInfo(
            name: "Test", path: URL(fileURLWithPath: "/test"),
            format: "gguf", parameterCount: "1B",
            quantization: "Q8_0", fileSizeBytes: 1_073_741_824
        )
        #expect(info.fileSizeFormatted == "1.0 GB")
    }

    @Test func smallFileSizeFormatted() {
        let info = ModelInfo(
            name: "Test", path: URL(fileURLWithPath: "/test"),
            format: "gguf", parameterCount: "100M",
            quantization: "Q4_0", fileSizeBytes: 52_428_800
        )
        #expect(info.fileSizeFormatted == "50.0 MB")
    }
}

@Suite("ChatViewModel Logic")
struct ChatViewModelLogicTests {
    @Test func initialState() {
        let vm = ChatViewModelState()
        #expect(vm.messages.isEmpty)
        #expect(vm.isGenerating == false)
        #expect(vm.currentInput == "")
    }

    @Test func addUserMessage() {
        var vm = ChatViewModelState()
        vm.addUserMessage("Hello there")
        #expect(vm.messages.count == 1)
        #expect(vm.messages[0].role == .user)
        #expect(vm.messages[0].content == "Hello there")
    }

    @Test func addAssistantMessage() {
        var vm = ChatViewModelState()
        vm.addAssistantMessage("")
        #expect(vm.messages.count == 1)
        #expect(vm.messages[0].role == .assistant)
    }

    @Test func appendToLastAssistantMessage() {
        var vm = ChatViewModelState()
        vm.addAssistantMessage("")
        vm.appendToLastMessage("Hello")
        vm.appendToLastMessage(" world")
        #expect(vm.messages.last?.content == "Hello world")
    }

    @Test func clearMessages() {
        var vm = ChatViewModelState()
        vm.addUserMessage("test")
        vm.addAssistantMessage("response")
        vm.clearMessages()
        #expect(vm.messages.isEmpty)
    }

    @Test func generatingState() {
        var vm = ChatViewModelState()
        vm.isGenerating = true
        #expect(vm.isGenerating)
        vm.isGenerating = false
        #expect(!vm.isGenerating)
    }

    @Test func memoryUsageTracking() {
        var vm = ChatViewModelState()
        vm.updateMemoryUsage(usedMB: 1024, totalMB: 8192)
        #expect(vm.memoryUsedMB == 1024)
        #expect(vm.memoryTotalMB == 8192)
        #expect(abs(vm.memoryUsagePercent - 12.5) < 0.1)
    }
}
