import Foundation

public enum EdgeRunnerModel {
    public static func load(
        from url: URL,
        registry: ModelRegistry = .default
    ) async throws -> any LoadableModel {
        guard let format = ModelFormat.detect(from: url) else {
            throw ModelLoadError.unsupportedFormat(url.pathExtension)
        }

        let loader: any EdgeRunnerWeightLoader
        switch format {
        case .gguf:
            loader = try GGUFLoader(url: url)
        case .safetensors:
            loader = try SafeTensorLoader(url: url)
        case .npz:
            loader = try NPZLoader(url: url)
        }

        let config = loader.modelConfig
        let architectureName = try resolveArchitectureName(
            from: config,
            loader: loader,
            registry: registry
        )

        guard let factory = registry.factory(for: architectureName) else {
            throw ModelLoadError.unknownArchitecture(architectureName)
        }

        var model = try factory.create(
            config: ModelConfig(
                architectureName: architectureName,
                metadata: config.metadata
            )
        )
        let weightMap = try await loader.load(from: url)
        try model.loadWeights(from: weightMap)
        return model
    }

    private static func resolveArchitectureName(
        from config: ModelConfig,
        loader: any EdgeRunnerWeightLoader,
        registry: ModelRegistry
    ) throws -> String {
        if !config.architectureName.isEmpty {
            return config.architectureName
        }

        let inferredFromNames: String? = if let loader = loader as? SafeTensorLoader {
            inferArchitecture(from: loader.tensorNames)
        } else if let loader = loader as? NPZLoader {
            inferArchitecture(from: loader.tensorNames)
        } else {
            nil
        }

        if let inferredFromNames {
            return inferredFromNames
        }

        let registered = registry.registeredArchitectureNames
        if registered.count == 1, let onlyArchitecture = registered.first {
            return onlyArchitecture
        }

        throw ModelLoadError.unknownArchitecture(
            "Unable to determine architecture for \(String(describing: type(of: loader)))"
        )
    }

    private static func inferArchitecture(from tensorNames: [String]) -> String? {
        if tensorNames.contains(where: { $0.hasPrefix("blk.") })
            || tensorNames.contains("token_embd.weight")
            || tensorNames.contains("output.weight")
            || tensorNames.contains(where: { $0.hasPrefix("layers.0.") || $0.hasPrefix("model.layers.0.") })
        {
            return "llama"
        }

        return nil
    }
}
