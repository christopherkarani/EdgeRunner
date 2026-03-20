import Testing
@testable import EdgeRunnerCore

@Suite("ChatTemplateEngine")
struct ChatTemplateEngineTests {
    @Test func chatMLBasicFormat() throws {
        let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [ChatMessage(role: "user", content: "Hello")],
            addGenerationPrompt: true
        )
        #expect(result.contains("<|im_start|>user"))
        #expect(result.contains("Hello"))
        #expect(result.contains("<|im_end|>"))
        #expect(result.contains("<|im_start|>assistant"))
    }

    @Test func chatMLMultipleMessages() throws {
        let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "system", content: "You are helpful."),
                ChatMessage(role: "user", content: "Hi"),
            ],
            addGenerationPrompt: false
        )
        #expect(result.contains("system"))
        #expect(result.contains("You are helpful."))
        #expect(result.contains("user"))
        #expect(result.contains("Hi"))
    }

    @Test func ifElseConditional() throws {
        let template = "{% if add_generation_prompt %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let yes = try engine.apply(messages: [], addGenerationPrompt: true)
        #expect(yes == "YES")
        let no = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(no == "NO")
    }

    @Test func loopIndexVariables() throws {
        let template = "{% for m in messages %}{{ loop.index }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
                ChatMessage(role: "c", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "123")
    }

    @Test func loopFirstLast() throws {
        let template = "{% for m in messages %}{% if loop.first %}F{% endif %}{% if loop.last %}L{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "FL")
    }

    @Test func whitespaceStripping() throws {
        let template = "A {%- if true %} B {%- endif %} C"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "A B C")
    }

    @Test func setVariable() throws {
        let template = "{% set name = 'world' %}Hello {{ name }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "Hello world")
    }

    @Test func trimFilter() throws {
        let template = "{{ '  hello  ' | trim }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "hello")
    }

    @Test func unsupportedFeatureThrows() throws {
        let template = "{% macro test() %}{% endmacro %}"
        #expect(throws: ChatTemplateError.self) {
            _ = try ChatTemplateEngine(template: template)
        }
    }

    @Test func equalityComparison() throws {
        let template = "{% for m in messages %}{% if m['role'] == 'user' %}U{% else %}O{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "system", content: ""),
                ChatMessage(role: "user", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "OU")
    }

    @Test func stringConcatenation() throws {
        let template = "{{ 'Hello' + ' ' + 'World' }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "Hello World")
    }

    @Test func engineIsSendable() throws {
        let engine = try ChatTemplateEngine(template: "{{ 'test' }}")
        let sendableRef: any Sendable = engine
        #expect(sendableRef is ChatTemplateEngine)
    }
}
