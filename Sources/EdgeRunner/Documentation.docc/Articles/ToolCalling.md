# Tool Calling

Enable models to call Swift functions with typed parameters.

## Overview

Define tools with the ``EdgeRunnerTool`` protocol, register them with a ``ToolExecutor``, and the system handles parsing, validation, and execution.

### Define a Tool

```swift
struct SearchTool: EdgeRunnerTool {
    static let name = "web_search"
    static let description = "Search the web."
    static let parameters: [ToolParameter] = [
        ToolParameter(name: "query", type: .string, description: "Search query", required: true),
    ]

    func invoke(arguments: [String: Any]) async throws -> String {
        let query = arguments["query"] as! String
        return "{\"results\": [\"Result for: \(query)\"]}"
    }
}
```

### Execute Tool Calls

```swift
let executor = ToolExecutor(tools: [SearchTool()])
let calls = try ToolCallParser.parse(modelOutput: modelResponse)
let results = try await executor.executeAll(calls)
```
