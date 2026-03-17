import Foundation

public struct LlamaModel: LoadableModel, Sendable {
    public let config: LlamaConfig
    public let layers: [LlamaBlock]

    public private(set) var loadedWeights: [String: TensorStorage] = [:]

    public init(config: LlamaConfig) {
        self.config = config
        self.layers = (0..<config.layerCount).map { LlamaBlock(config: config, layerIndex: $0) }
    }

    public var parameterNames: [String] {
        var names = ["embedding.weight"]
        for layer in layers {
            names.append(contentsOf: layer.parameterNames)
        }
        names.append("finalNorm.weight")
        names.append("lmHead.weight")
        return names
    }

    /// Names that are allowed to be absent (e.g., tied embeddings, QK-norm).
    public static let optionalWeightNames: Set<String> = [
        "lmHead.weight",
    ]

    public mutating func loadWeights(from map: WeightMap) throws {
        var resolvedWeights: [String: TensorStorage] = [:]
        let expectedNames = Set(parameterNames)

        for sourceName in map.tensorNames {
            guard let storage = map[sourceName] else {
                continue
            }

            if expectedNames.contains(sourceName) {
                resolvedWeights[sourceName] = storage
                continue
            }

            let mappedName = LlamaWeightNameMapper.mapGGUFName(sourceName)
            if expectedNames.contains(mappedName) {
                resolvedWeights[mappedName] = storage
            } else {
                // Store unmapped weights too (e.g., QK-norm) for models that use them
                resolvedWeights[sourceName] = storage
            }
        }

        for requiredName in parameterNames {
            if resolvedWeights[requiredName] == nil
                && !Self.optionalWeightNames.contains(requiredName)
            {
                throw ModelLoadError.loadFailed(description: "Missing weight: \(requiredName)")
            }
        }

        loadedWeights = resolvedWeights
    }
}
