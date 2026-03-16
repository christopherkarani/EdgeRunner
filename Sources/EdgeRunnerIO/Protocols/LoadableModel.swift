import Foundation

public enum ModelLoadError: Error, Sendable, Equatable {
    case loadFailed(description: String)
}

public protocol LoadableModel: Sendable {
    var parameterNames: [String] { get }
    mutating func loadWeights(from map: WeightMap) throws
}
