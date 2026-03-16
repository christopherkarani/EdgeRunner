import Foundation
import Metal

public protocol EdgeRunnerWeightLoader: Sendable {
    var modelConfig: ModelConfig { get }

    func canLoad(url: URL) -> Bool
    func load(from url: URL) async throws -> WeightMap
}
