import Foundation
import EdgeRunnerCore

public struct ToolExecutor: Sendable {
    private let tools: [String: any EdgeRunnerTool]

    public init(tools: [any EdgeRunnerTool]) {
        var registry = [String: any EdgeRunnerTool]()
        for tool in tools { registry[type(of: tool).name] = tool }
        self.tools = registry
    }

    public func execute(_ call: ToolCall) async throws -> String {
        guard let tool = tools[call.name] else {
            throw GenerationError.toolCallFailed(name: call.name, reason: "Tool '\(call.name)' not found")
        }
        return try await tool.invoke(arguments: call.arguments)
    }

    public func executeAll(_ calls: [ToolCall]) async throws -> [String] {
        var results = [String]()
        for call in calls { results.append(try await execute(call)) }
        return results
    }

    public func toolDescriptions() -> String {
        tools.values.map { type(of: $0).jsonSchema }.joined(separator: "\n")
    }

    public func shouldAttemptToolCall(choice: ToolChoice, modelOutput: String) -> Bool {
        switch choice {
        case .none: return false
        case .required: return true
        case .auto: return ToolCallParser.containsToolCall(in: modelOutput)
        case .specific(let name): return tools[name] != nil
        }
    }
}
