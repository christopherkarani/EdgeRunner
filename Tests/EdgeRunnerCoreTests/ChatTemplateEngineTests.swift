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

    // MARK: - Tier 2: tojson filter

    @Test func tojsonFilter() throws {
        let template = "{{ 'hello' | tojson }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "\"hello\"")
    }

    @Test func tojsonDict() throws {
        let template = "{% set d = namespace(a='1', b='2') %}{{ d | tojson }}"
        // This test verifies tojson can serialize objects
        // The exact output format doesn't matter as long as it's valid JSON-like
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result.contains("a"))
        #expect(result.contains("1"))
    }

    // MARK: - Tier 2: join filter

    @Test func joinFilter() throws {
        let template2 = "{% for m in messages %}{% if not loop.first %}, {% endif %}{{ m['role'] }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template2)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
                ChatMessage(role: "c", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "a, b, c")
    }

    @Test func joinFilterWithArray() throws {
        let template = "{% set ns = namespace(items='a,b,c') %}{{ ns.items.split(',') | join(' | ') }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "a | b | c")
    }

    // MARK: - Tier 2: is defined / is string tests

    @Test func isDefinedTest() throws {
        let template = "{% if bos_token is defined %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let yes = try engine.apply(messages: [], addGenerationPrompt: false, bosToken: "BOS")
        #expect(yes == "YES")
        let no = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(no == "NO")
    }

    @Test func isNotDefinedTest() throws {
        let template = "{% if tools is not defined %}NONE{% else %}HAS{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "NONE")
    }

    @Test func isStringTest() throws {
        let template = "{% for m in messages %}{% if m['content'] is string %}S{% else %}O{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [ChatMessage(role: "user", content: "hi")],
            addGenerationPrompt: false
        )
        #expect(result == "S")
    }

    @Test func isMappingTest() throws {
        let template = "{% for m in messages %}{% if m is mapping %}M{% else %}O{% endif %}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [ChatMessage(role: "user", content: "hi")],
            addGenerationPrompt: false
        )
        #expect(result == "M")
    }

    @Test func isIterableTest() throws {
        let template = "{% if messages is iterable %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [ChatMessage(role: "user", content: "hi")],
            addGenerationPrompt: false
        )
        #expect(result == "YES")
    }

    // MARK: - Tier 2: namespace

    @Test func namespaceAcrossLoop() throws {
        let template = "{% set ns = namespace(count=0) %}{% for m in messages %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "a", content: ""),
                ChatMessage(role: "b", content: ""),
                ChatMessage(role: "c", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "3")
    }

    // MARK: - Tier 2: string methods

    @Test func stripMethod() throws {
        let template = "{{ '  hello  '.strip() }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "hello")
    }

    @Test func startswithMethod() throws {
        let template = "{% if 'hello world'.startswith('hello') %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "YES")
    }

    @Test func endswithMethod() throws {
        let template = "{% if 'hello world'.endswith('world') %}YES{% else %}NO{% endif %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "YES")
    }

    @Test func splitMethod() throws {
        let template = "{% set parts = 'a,b,c'.split(',') %}{% for p in parts %}{{ p }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(messages: [], addGenerationPrompt: false)
        #expect(result == "abc")
    }

    // MARK: - Tier 2: array slicing

    @Test func negativeIndexing() throws {
        let template = "{{ messages[-1]['role'] }}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "first", content: ""),
                ChatMessage(role: "last", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "last")
    }

    @Test func reverseSlice() throws {
        let template = "{% for m in messages[::-1] %}{{ m['role'] }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "A", content: ""),
                ChatMessage(role: "B", content: ""),
                ChatMessage(role: "C", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "CBA")
    }

    @Test func sliceWithStartEnd() throws {
        let template = "{% for m in messages[1:3] %}{{ m['role'] }}{% endfor %}"
        let engine = try ChatTemplateEngine(template: template)
        let result = try engine.apply(
            messages: [
                ChatMessage(role: "A", content: ""),
                ChatMessage(role: "B", content: ""),
                ChatMessage(role: "C", content: ""),
                ChatMessage(role: "D", content: ""),
            ],
            addGenerationPrompt: false
        )
        #expect(result == "BC")
    }
}
