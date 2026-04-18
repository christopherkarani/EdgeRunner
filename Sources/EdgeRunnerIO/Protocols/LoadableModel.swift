import Foundation

public enum ModelLoadError: Error, Sendable, Equatable {
    case unsupportedFormat(String)
    case unknownArchitecture(String)
    case loadFailed(description: String)
    case notYetImplemented(architecture: String)

    public var description: String {
        switch self {
        case .unsupportedFormat(let format):
            return "Unsupported model format: \(format)"
        case .unknownArchitecture(let architecture):
            return "Unknown architecture: \(architecture)"
        case .loadFailed(let description):
            return "Model load failed: \(description)"
        case .notYetImplemented(let architecture):
            return "\(architecture) is not yet supported in this build"
        }
    }

    public var localizedDescription: String { description }
}

public protocol LoadableModel: Sendable {
    var parameterNames: [String] { get }
    mutating func loadWeights(from map: WeightMap) throws
}
