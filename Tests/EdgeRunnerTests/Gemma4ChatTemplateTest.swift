import Testing
@testable import EdgeRunner

@Suite("Gemma4 chat template")
struct Gemma4ChatTemplateTests {
    @Test("Renders system + user turn with trailing model open")
    func rendersSystemUserThenModelOpen() {
        let rendered = Gemma4ChatTemplate.render(messages: [
            .init(role: .system, content: "You are helpful."),
            .init(role: .user, content: "Hi")
        ], addGenerationPrompt: true)

        #expect(rendered == """
        <|turn>system
        <|think|>
        You are helpful.<turn|>
        <|turn>user
        Hi<turn|>
        <|turn>model

        """)
    }

    @Test("Rewrites assistant role to model")
    func rewritesAssistantToModel() {
        let rendered = Gemma4ChatTemplate.render(messages: [
            .init(role: .user, content: "Hi"),
            .init(role: .assistant, content: "Hello!")
        ], addGenerationPrompt: false)

        #expect(rendered.contains("<|turn>model\nHello!<turn|>"))
        #expect(!rendered.contains("assistant"))
    }

    @Test("Injects thinking system block before user-only prompts")
    func injectsThinkingSystemBlock() {
        let rendered = Gemma4ChatTemplate.render(messages: [
            .init(role: .user, content: "Write one short sentence.")
        ], addGenerationPrompt: true)

        #expect(rendered == """
        <|turn>system
        <|think|>
        <turn|>
        <|turn>user
        Write one short sentence.<turn|>
        <|turn>model

        """)
    }

    @Test("Throws on tool messages (unsupported in v1)")
    func throwsOnToolMessages() {
        #expect(throws: Gemma4ChatTemplateError.toolsUnsupportedInV1) {
            _ = try Gemma4ChatTemplate.renderThrowing(messages: [
                .init(role: .tool, content: "{\"result\": 42}")
            ], addGenerationPrompt: false)
        }
    }
}
