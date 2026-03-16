import Foundation
import EdgeRunnerCore

public enum ToolParameterType: String, Sendable {
    case string, integer, number, boolean, array, object
}

public struct ToolParameter: Sendable {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let required: Bool
    public init(name: String, type: ToolParameterType, description: String, required: Bool) {
        self.name = name; self.type = type; self.description = description; self.required = required
    }
}

public struct ToolCall: Sendable {
    public let name: String
    public let arguments: [String: any Sendable]
    public init(name: String, arguments: [String: any Sendable]) {
        self.name = name; self.arguments = arguments
    }
}

public protocol EdgeRunnerTool: Sendable {
    static var name: String { get }
    static var description: String { get }
    static var parameters: [ToolParameter] { get }
    func invoke(arguments: [String: Any]) async throws -> String
}

extension EdgeRunnerTool {
    public static var jsonSchema: String {
        var properties = [String: [String: String]]()
        var required = [String]()
        for param in parameters {
            properties[param.name] = ["type": param.type.rawValue, "description": param.description]
            if param.required { required.append(param.name) }
        }
        let schema: [String: Any] = [
            "name": name, "description": description,
            "parameters": ["type": "object", "properties": properties, "required": required] as [String: Any]
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys]),
              let str = String(data: data, encoding: .utf8) else { return "{}" }
        return str
    }
}
