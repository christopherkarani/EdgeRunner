import Foundation

public enum ToolCallParser {
    public static func parse(modelOutput: String) throws -> [ToolCall] {
        var calls = [ToolCall]()
        calls.append(contentsOf: parseXMLStyle(modelOutput))
        if calls.isEmpty { calls.append(contentsOf: parseFunctionCallStyle(modelOutput)) }
        return calls
    }

    public static func containsToolCall(in text: String) -> Bool {
        text.contains("<tool_call>") || text.contains("\"function_call\"")
    }

    private static func toSendable(_ dict: [String: Any]) -> [String: any Sendable] {
        var result = [String: any Sendable]()
        for (key, value) in dict {
            switch value {
            case let s as String: result[key] = s
            case let n as NSNumber: result[key] = n.doubleValue
            case let b as Bool: result[key] = b
            case let a as [Any]: result[key] = a.map { "\($0)" }
            case let d as [String: Any]: result[key] = toSendable(d)
            default: result[key] = "\(value)"
            }
        }
        return result
    }

    private static func parseXMLStyle(_ text: String) -> [ToolCall] {
        var calls = [ToolCall]()
        let pattern = "<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches {
            guard let range = Range(match.range(at: 1), in: text) else { continue }
            let jsonStr = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = jsonStr.data(using: .utf8),
                  let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = dict["name"] as? String else { continue }
            let rawArgs = dict["arguments"] as? [String: Any] ?? [:]
            calls.append(ToolCall(name: name, arguments: toSendable(rawArgs)))
        }
        return calls
    }

    private static func parseFunctionCallStyle(_ text: String) -> [ToolCall] {
        guard let data = text.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let functionCall = dict["function_call"] as? [String: Any],
              let name = functionCall["name"] as? String else { return [] }
        var rawArgs = [String: Any]()
        if let argsString = functionCall["arguments"] as? String,
           let argsData = argsString.data(using: .utf8),
           let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
            rawArgs = argsDict
        } else if let argsDict = functionCall["arguments"] as? [String: Any] {
            rawArgs = argsDict
        }
        return [ToolCall(name: name, arguments: toSendable(rawArgs))]
    }
}
