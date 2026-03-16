import Foundation

public enum FoundationModelsAvailability {
    public static var isAvailable: Bool {
        #if canImport(FoundationModels)
        return true
        #else
        return false
        #endif
    }
}

public protocol SystemModelBackend: EdgeRunnerLanguageModel {
    var supportsGuidedGeneration: Bool { get }
    func generateStream(prompt: String) -> AsyncThrowingStream<String, Error>
}
