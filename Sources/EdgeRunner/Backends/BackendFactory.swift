import Foundation
import Synchronization

public final class BackendRegistry: Sendable {
    private let state: Mutex<[String: any EdgeRunnerLanguageModel.Type]>

    public init() { self.state = Mutex([:]) }

    public func register<T: EdgeRunnerLanguageModel>(_ type: T.Type, for format: String) {
        state.withLock { $0[format] = type }
    }

    public func backend(for format: String) -> (any EdgeRunnerLanguageModel.Type)? {
        state.withLock { $0[format] }
    }

    public var availableBackends: Set<String> {
        state.withLock { Set($0.keys) }
    }

    public func load(from url: URL, format: String, configuration: ModelConfiguration = ModelConfiguration()) async throws -> any EdgeRunnerLanguageModel {
        guard let backendType = backend(for: format) else {
            throw GenerationError.modelLoadFailed(reason: "No backend for format '\(format)'")
        }
        return try await backendType.load(from: url, configuration: configuration)
    }
}
