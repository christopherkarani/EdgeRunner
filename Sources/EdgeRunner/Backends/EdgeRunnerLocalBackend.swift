import Foundation

public protocol LocalModelBackend: EdgeRunnerLanguageModel {
    static var supportedFormat: String { get }
    func estimatedMemoryUsage() -> Int
}
