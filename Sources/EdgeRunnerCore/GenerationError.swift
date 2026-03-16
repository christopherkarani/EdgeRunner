import Foundation

/// Errors that can occur during model loading and text generation.
public enum GenerationError: Error, Sendable, CustomStringConvertible {
    case modelLoadFailed(reason: String)
    case contextWindowExceeded(requested: Int, maximum: Int)
    case invalidTokenID(Int)
    case decodingFailed(String)
    case cancelled
    case samplingFailed(String)
    case toolCallFailed(name: String, reason: String)
    case structuredOutputFailed(reason: String)

    public var description: String {
        switch self {
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .contextWindowExceeded(let requested, let maximum):
            return "Context window exceeded: requested \(requested) tokens, maximum \(maximum)"
        case .invalidTokenID(let id):
            return "Invalid token ID: \(id)"
        case .decodingFailed(let reason):
            return "Decoding failed: \(reason)"
        case .cancelled:
            return "Generation cancelled"
        case .samplingFailed(let reason):
            return "Sampling failed: \(reason)"
        case .toolCallFailed(let name, let reason):
            return "Tool call '\(name)' failed: \(reason)"
        case .structuredOutputFailed(let reason):
            return "Structured output failed: \(reason)"
        }
    }
}
