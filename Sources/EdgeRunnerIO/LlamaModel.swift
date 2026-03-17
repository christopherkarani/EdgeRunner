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
    /// Per-head Q/K RMSNorm weights are optional (only Qwen3 and similar models).
    public static let optionalWeightSuffixes: [String] = [
        "attention.qNorm.weight",
        "attention.kNorm.weight",
    ]

    public static let optionalWeightNames: Set<String> = [
        "lmHead.weight",
    ]

    /// Check if a weight name is optional (static names or per-layer suffix patterns).
    public static func isOptional(_ name: String) -> Bool {
        if optionalWeightNames.contains(name) { return true }
        return optionalWeightSuffixes.contains(where: { name.hasSuffix($0) })
    }

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
                && !Self.isOptional(requiredName)
            {
                throw ModelLoadError.loadFailed(description: "Missing weight: \(requiredName)")
            }
        }

        loadedWeights = resolvedWeights
    }
}
