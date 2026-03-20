import Testing
@testable import EdgeRunnerCore

@Suite("PreTokenizer")
struct PreTokenizerTests {
    @Test func gpt2SplitsWordsWithLeadingSpaces() {
        let pre = PreTokenizerPattern.resolve(nil)
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }

    @Test func gpt2SplitsContractions() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("I'm don't")
        #expect(result.contains("'m"))
        #expect(result.contains("'t"))
    }

    @Test func gpt2SplitsDigits() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("test 123 hello")
        #expect(result.contains(" 123"))
    }

    @Test func gpt2SplitsPunctuation() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("hello, world!")
        #expect(result.contains(","))
    }

    @Test func emptyStringReturnsEmpty() {
        let pre = PreTokenizerPattern.resolve("gpt-2")
        let result = pre.split("")
        #expect(result.isEmpty)
    }

    @Test func qwen2SplitsDigitsIndividually() {
        let pre = PreTokenizerPattern.resolve("qwen2")
        let result = pre.split("abc123")
        #expect(result.contains("1"))
        #expect(result.contains("2"))
        #expect(result.contains("3"))
    }

    @Test func qwen2CaseInsensitiveContractions() {
        let pre = PreTokenizerPattern.resolve("qwen2")
        let result = pre.split("I'M DON'T")
        #expect(result.contains("'M"))
        #expect(result.contains("'T"))
    }

    @Test func llama3GroupsDigitsUpToThree() {
        let pre = PreTokenizerPattern.resolve("llama3")
        let result = pre.split("price 123456")
        #expect(result.contains("123"))
        #expect(result.contains("456"))
    }

    @Test func unknownPreValueFallsBackToGPT2() {
        let pre = PreTokenizerPattern.resolve("some-unknown-model")
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }

    @Test func defaultPatternUsedWhenNil() {
        let pre = PreTokenizerPattern.resolve(nil)
        let result = pre.split("Hello world")
        #expect(result == ["Hello", " world"])
    }
}
