import Foundation
import Testing
@testable import EdgeRunner
@testable import EdgeRunnerCore

// MARK: - Mock Tools

struct WeatherTool: EdgeRunnerTool {
    static let name = "get_weather"
    static let description = "Get weather for a location"
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "location", type: .string, description: "City name", required: true),
        ToolParameter(name: "units", type: .string, description: "Temperature units", required: false)
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let location = arguments["location"] as? String ?? "unknown"
        return "Weather in \(location): 72F, sunny"
    }
}

struct CalculatorTool: EdgeRunnerTool {
    static let name = "calculate"
    static let description = "Perform arithmetic"
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "expression", type: .string, description: "Math expression", required: true)
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let expr = arguments["expression"] as? String ?? ""
        return "Result: \(expr) = 42"
    }
}

// MARK: - EdgeRunnerTool Protocol Tests

@Suite("EdgeRunnerTool Protocol")
struct EdgeRunnerToolProtocolTests {

    @Test func toolParameterInit() {
        let param = ToolParameter(name: "query", type: .string, description: "Search query", required: true)
        #expect(param.name == "query")
        #expect(param.type == .string)
        #expect(param.description == "Search query")
        #expect(param.required == true)
    }

    @Test func toolParameterTypeRawValues() {
        #expect(ToolParameterType.string.rawValue == "string")
        #expect(ToolParameterType.integer.rawValue == "integer")
        #expect(ToolParameterType.number.rawValue == "number")
        #expect(ToolParameterType.boolean.rawValue == "boolean")
        #expect(ToolParameterType.array.rawValue == "array")
        #expect(ToolParameterType.object.rawValue == "object")
    }

    @Test func toolCallInit() {
        let call = ToolCall(name: "get_weather", arguments: ["location": "NYC" as any Sendable])
        #expect(call.name == "get_weather")
        #expect(call.arguments["location"] as? String == "NYC")
    }

    @Test func weatherToolStaticProperties() {
        #expect(WeatherTool.name == "get_weather")
        #expect(WeatherTool.description == "Get weather for a location")
        #expect(WeatherTool.parameters.count == 2)
    }

    @Test func calculatorToolStaticProperties() {
        #expect(CalculatorTool.name == "calculate")
        #expect(CalculatorTool.description == "Perform arithmetic")
        #expect(CalculatorTool.parameters.count == 1)
    }

    @Test func jsonSchemaGeneration() {
        let schema = WeatherTool.jsonSchema
        #expect(schema.contains("get_weather"))
        #expect(schema.contains("location"))
        #expect(schema.contains("units"))
        #expect(schema.contains("required"))
    }

    @Test func jsonSchemaContainsDescription() {
        let schema = CalculatorTool.jsonSchema
        #expect(schema.contains("Perform arithmetic"))
        #expect(schema.contains("expression"))
    }

    @Test func toolInvocation() async throws {
        let tool = WeatherTool()
        let result = try await tool.invoke(arguments: ["location": "San Francisco"])
        #expect(result.contains("San Francisco"))
        #expect(result.contains("72F"))
    }

    @Test func calculatorInvocation() async throws {
        let tool = CalculatorTool()
        let result = try await tool.invoke(arguments: ["expression": "2+2"])
        #expect(result.contains("2+2"))
        #expect(result.contains("42"))
    }

    @Test func toolInvocationWithMissingArgs() async throws {
        let tool = WeatherTool()
        let result = try await tool.invoke(arguments: [:])
        #expect(result.contains("unknown"))
    }
}

// MARK: - ToolChoice Tests

@Suite("ToolChoice")
struct ToolChoiceTests {

    @Test func toolChoiceEquality() {
        #expect(ToolChoice.auto == ToolChoice.auto)
        #expect(ToolChoice.required == ToolChoice.required)
        #expect(ToolChoice.none == ToolChoice.none)
        #expect(ToolChoice.specific("get_weather") == ToolChoice.specific("get_weather"))
    }

    @Test func toolChoiceInequality() {
        #expect(ToolChoice.auto != ToolChoice.required)
        #expect(ToolChoice.none != ToolChoice.auto)
        #expect(ToolChoice.specific("a") != ToolChoice.specific("b"))
    }
}

// MARK: - ToolCallParser Tests

@Suite("ToolCallParser")
struct ToolCallParserTests {

    @Test func parseXMLStyleToolCall() throws {
        let output = """
        I'll check the weather.
        <tool_call>
        {"name": "get_weather", "arguments": {"location": "NYC"}}
        </tool_call>
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["location"] as? String == "NYC")
    }

    @Test func parseMultipleXMLToolCalls() throws {
        let output = """
        <tool_call>
        {"name": "get_weather", "arguments": {"location": "NYC"}}
        </tool_call>
        <tool_call>
        {"name": "calculate", "arguments": {"expression": "2+2"}}
        </tool_call>
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 2)
        #expect(calls[0].name == "get_weather")
        #expect(calls[1].name == "calculate")
    }

    @Test func parseFunctionCallStyle() throws {
        let output = """
        {"function_call": {"name": "get_weather", "arguments": "{\\"location\\": \\"NYC\\"}"}}
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["location"] as? String == "NYC")
    }

    @Test func parseFunctionCallWithDictArguments() throws {
        let output = """
        {"function_call": {"name": "calculate", "arguments": {"expression": "1+1"}}}
        """
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "calculate")
        #expect(calls[0].arguments["expression"] as? String == "1+1")
    }

    @Test func parseEmptyOutput() throws {
        let calls = try ToolCallParser.parse(modelOutput: "Hello, how can I help?")
        #expect(calls.isEmpty)
    }

    @Test func parseInvalidJSON() throws {
        let output = "<tool_call>not valid json</tool_call>"
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.isEmpty)
    }

    @Test func parseMissingName() throws {
        let output = "<tool_call>{\"arguments\": {}}</tool_call>"
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.isEmpty)
    }

    @Test func containsToolCallXML() {
        #expect(ToolCallParser.containsToolCall(in: "Some text <tool_call>...</tool_call>") == true)
        #expect(ToolCallParser.containsToolCall(in: "No tool calls here") == false)
    }

    @Test func containsToolCallFunctionCall() {
        #expect(ToolCallParser.containsToolCall(in: "{\"function_call\": {}}") == true)
    }

    @Test func parseXMLWithNoArguments() throws {
        let output = "<tool_call>{\"name\": \"simple_tool\"}</tool_call>"
        let calls = try ToolCallParser.parse(modelOutput: output)
        #expect(calls.count == 1)
        #expect(calls[0].name == "simple_tool")
        #expect(calls[0].arguments.isEmpty)
    }
}

// MARK: - ToolExecutor Tests

@Suite("ToolExecutor")
struct ToolExecutorTests {

    @Test func executeSingleTool() async throws {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let call = ToolCall(name: "get_weather", arguments: ["location": "London" as any Sendable])
        let result = try await executor.execute(call)
        #expect(result.contains("London"))
    }

    @Test func executeAllTools() async throws {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let calls = [
            ToolCall(name: "get_weather", arguments: ["location": "Tokyo" as any Sendable]),
            ToolCall(name: "calculate", arguments: ["expression": "3*7" as any Sendable])
        ]
        let results = try await executor.executeAll(calls)
        #expect(results.count == 2)
        #expect(results[0].contains("Tokyo"))
        #expect(results[1].contains("3*7"))
    }

    @Test func executeUnknownToolThrows() async throws {
        let executor = ToolExecutor(tools: [WeatherTool()])
        let call = ToolCall(name: "unknown_tool", arguments: [:])
        await #expect(throws: GenerationError.self) {
            try await executor.execute(call)
        }
    }

    @Test func toolDescriptionsContainsAllTools() {
        let executor = ToolExecutor(tools: [WeatherTool(), CalculatorTool()])
        let descriptions = executor.toolDescriptions()
        #expect(descriptions.contains("get_weather"))
        #expect(descriptions.contains("calculate"))
    }

    @Test func shouldAttemptToolCallNone() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .none, modelOutput: "<tool_call>...</tool_call>") == false)
    }

    @Test func shouldAttemptToolCallRequired() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .required, modelOutput: "no tools") == true)
    }

    @Test func shouldAttemptToolCallAutoWithToolCall() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .auto, modelOutput: "<tool_call>test</tool_call>") == true)
    }

    @Test func shouldAttemptToolCallAutoWithoutToolCall() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .auto, modelOutput: "Just text") == false)
    }

    @Test func shouldAttemptToolCallSpecificKnown() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .specific("get_weather"), modelOutput: "") == true)
    }

    @Test func shouldAttemptToolCallSpecificUnknown() {
        let executor = ToolExecutor(tools: [WeatherTool()])
        #expect(executor.shouldAttemptToolCall(choice: .specific("nonexistent"), modelOutput: "") == false)
    }

    @Test func emptyExecutor() {
        let executor = ToolExecutor(tools: [])
        #expect(executor.toolDescriptions().isEmpty)
        #expect(executor.shouldAttemptToolCall(choice: .auto, modelOutput: "<tool_call>x</tool_call>") == true)
    }
}
