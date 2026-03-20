import Testing
@testable import EdgeRunner

@Suite("Conversation")
struct ConversationTests {
    @Test func emptyConversation() {
        let convo = Conversation()
        #expect(convo.isEmpty)
        #expect(convo.messageCount == 0)
        #expect(convo.messages.isEmpty)
    }

    @Test func conversationWithSystemPrompt() {
        let convo = Conversation(systemPrompt: "You are helpful.")
        #expect(convo.messageCount == 1)
        #expect(convo.messages[0].role == .system)
        #expect(convo.messages[0].content == "You are helpful.")
    }

    @Test func addMessages() {
        var convo = Conversation()
        convo.addUser("Hello")
        convo.addAssistant("Hi there!")
        convo.addUser("How are you?")

        #expect(convo.messageCount == 3)
        #expect(convo.messages[0].role == .user)
        #expect(convo.messages[1].role == .assistant)
        #expect(convo.messages[2].role == .user)
    }

    @Test func resetKeepsSystemPrompt() {
        var convo = Conversation(systemPrompt: "Be brief.")
        convo.addUser("Hello")
        convo.addAssistant("Hi")

        #expect(convo.messageCount == 3)
        convo.reset(keepSystem: true)
        #expect(convo.messageCount == 1)
        #expect(convo.messages[0].role == .system)
        #expect(convo.messages[0].content == "Be brief.")
    }

    @Test func resetClearsEverything() {
        var convo = Conversation(systemPrompt: "Be brief.")
        convo.addUser("Hello")

        convo.reset(keepSystem: false)
        #expect(convo.isEmpty)
    }

    @Test func resetWithoutSystemPrompt() {
        var convo = Conversation()
        convo.addUser("Hello")

        convo.reset(keepSystem: true)
        #expect(convo.isEmpty)
    }

    @Test func conversationIsSendable() {
        let convo = Conversation(systemPrompt: "Test")
        let ref: any Sendable = convo
        #expect(ref is Conversation)
    }
}
