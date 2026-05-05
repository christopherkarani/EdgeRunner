import Foundation
import Synchronization

public enum ModelFormat: String, Sendable, Equatable {
    case gguf
    case safetensors
    case npz

    public static func detect(from url: URL) -> ModelFormat? {
        switch url.pathExtension.lowercased() {
        case "gguf":
            return .gguf
        case "safetensors":
            return .safetensors
        case "npz":
            return .npz
        default:
            return nil
        }
    }
}

public protocol ArchitectureFactory: Sendable {
    var architectureName: String { get }
    func create(config: ModelConfig) throws -> any LoadableModel
}

public struct LlamaArchitectureFactory: ArchitectureFactory, Sendable {
    public let architectureName = "llama"

    public init() {}

    public func create(config: ModelConfig) throws -> any LoadableModel {
        let llamaConfig = try LlamaConfig(fromGGUFMetadata: config.metadata)
        return LlamaModel(config: llamaConfig)
    }
}

public final class ModelRegistry: Sendable {
    private let factories: Mutex<[String: any ArchitectureFactory]>

    public init() {
        self.factories = Mutex([:])
    }

    public static let `default`: ModelRegistry = {
        let registry = ModelRegistry()
        registry.register(LlamaArchitectureFactory())
        registry.register(Gemma4ArchitectureFactory())
        return registry
    }()

    public func register(_ factory: any ArchitectureFactory) {
        factories.withLock { state in
            state[factory.architectureName] = factory
        }
    }

    public func factory(for name: String) -> (any ArchitectureFactory)? {
        factories.withLock { state in
            state[name]
        }
    }

    var registeredArchitectureNames: [String] {
        factories.withLock { state in
            state.keys.sorted()
        }
    }
}
