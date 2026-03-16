import Foundation

/// Protocol for models that can participate in speculative decoding.
public protocol SpeculativeModel: Sendable {
    func logits(for tokenIDs: [Int]) async throws -> [Float]
    func batchLogits(for sequences: [[Int]]) async throws -> [[Float]]
}

extension SpeculativeModel {
    public func batchLogits(for sequences: [[Int]]) async throws -> [[Float]] {
        var results = [[Float]]()
        results.reserveCapacity(sequences.count)
        for seq in sequences {
            try await results.append(logits(for: seq))
        }
        return results
    }
}

/// Result of a single speculative decoding step.
public struct SpeculativeDecodingResult: Sendable {
    public let acceptedTokens: [Int]
    public let draftTokenCount: Int

    public var acceptanceRate: Float {
        guard draftTokenCount > 0 else { return 0.0 }
        let accepted = min(acceptedTokens.count - 1, draftTokenCount)
        return Float(max(accepted, 0)) / Float(draftTokenCount)
    }
}

/// Speculative decoding: uses a fast draft model to propose N token candidates,
/// then verifies them in parallel with the main model.
public struct SpeculativeDecoder: Sendable {
    private let draftModel: any SpeculativeModel
    private let verificationModel: any SpeculativeModel
    private let draftTokenCount: Int
    private let samplingPipeline: SamplingPipeline

    public init(
        draftModel: any SpeculativeModel,
        verificationModel: any SpeculativeModel,
        draftTokenCount: Int = 4,
        samplingPipeline: SamplingPipeline = .greedy
    ) {
        precondition(draftTokenCount >= 1, "Must draft at least 1 token")
        self.draftModel = draftModel
        self.verificationModel = verificationModel
        self.draftTokenCount = draftTokenCount
        self.samplingPipeline = samplingPipeline
    }

    public func decodeStep(inputTokens: [Int]) async throws -> SpeculativeDecodingResult {
        var draftTokens = [Int]()
        var currentSequence = inputTokens
        for _ in 0..<draftTokenCount {
            let logits = try await draftModel.logits(for: currentSequence)
            let token = samplingPipeline.sample(logits: logits)
            draftTokens.append(token)
            currentSequence.append(token)
        }

        var verificationSequences = [[Int]]()
        for i in 0...draftTokenCount {
            var seq = inputTokens
            seq.append(contentsOf: draftTokens.prefix(i))
            verificationSequences.append(seq)
        }

        let allVerifyLogits = try await verificationModel.batchLogits(for: verificationSequences)

        var acceptedTokens = [Int]()
        for i in 0..<draftTokenCount {
            let verifierLogits = allVerifyLogits[i]
            let verifierToken = samplingPipeline.sample(logits: verifierLogits)
            if verifierToken == draftTokens[i] {
                acceptedTokens.append(draftTokens[i])
            } else {
                acceptedTokens.append(verifierToken)
                return SpeculativeDecodingResult(
                    acceptedTokens: acceptedTokens,
                    draftTokenCount: draftTokenCount
                )
            }
        }

        if allVerifyLogits.count > draftTokenCount {
            let bonusLogits = allVerifyLogits[draftTokenCount]
            let bonusToken = samplingPipeline.sample(logits: bonusLogits)
            acceptedTokens.append(bonusToken)
        }

        return SpeculativeDecodingResult(
            acceptedTokens: acceptedTokens,
            draftTokenCount: draftTokenCount
        )
    }
}
