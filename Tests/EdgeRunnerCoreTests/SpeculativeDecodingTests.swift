import Testing
@testable import EdgeRunnerCore

private struct MockSpecModel: SpeculativeModel {
    let fixedLogitsSequence: [[Float]]
    let vocabSize: Int

    init(vocabSize: Int, fixedLogitsSequence: [[Float]]) {
        self.vocabSize = vocabSize
        self.fixedLogitsSequence = fixedLogitsSequence
    }

    func logits(for tokenIDs: [Int]) async throws -> [Float] {
        let step = tokenIDs.count - 1
        if step < fixedLogitsSequence.count {
            return fixedLogitsSequence[step]
        }
        return [Float](repeating: 0.0, count: vocabSize)
    }

    func batchLogits(for sequences: [[Int]]) async throws -> [[Float]] {
        try await sequences.asyncMap { try await logits(for: $0) }
    }
}

private func makeLogits(vocabSize: Int, peakAt index: Int, peakValue: Float = 10.0) -> [Float] {
    var logits = [Float](repeating: -10.0, count: vocabSize)
    logits[index] = peakValue
    return logits
}

@Suite("SpeculativeDecoder")
struct SpeculativeDecodingTests {

    @Test func allDraftTokensAccepted() async throws {
        let vocabSize = 5
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
            makeLogits(vocabSize: vocabSize, peakAt: 4),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)
        let decoder = SpeculativeDecoder(
            draftModel: draft, verificationModel: verifier,
            draftTokenCount: 3, samplingPipeline: .greedy
        )
        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.acceptedTokens == [1, 2, 3, 4])
        #expect(result.acceptanceRate == 1.0)
    }

    @Test func firstDraftTokenRejected() async throws {
        let vocabSize = 5
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 4),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)
        let decoder = SpeculativeDecoder(
            draftModel: draft, verificationModel: verifier,
            draftTokenCount: 3, samplingPipeline: .greedy
        )
        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.acceptedTokens == [4])
        #expect(result.acceptanceRate == 0.0)
    }

    @Test func partialAcceptance() async throws {
        let vocabSize = 5
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 3),
        ]
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 0),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)
        let decoder = SpeculativeDecoder(
            draftModel: draft, verificationModel: verifier,
            draftTokenCount: 3, samplingPipeline: .greedy
        )
        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.acceptedTokens == [1, 2, 0])
        #expect(abs(result.acceptanceRate - 2.0 / 3.0) < 1e-6)
    }

    @Test func draftTokenCountRespected() async throws {
        let vocabSize = 3
        let draftLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
        ]
        let verifyLogits: [[Float]] = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
            makeLogits(vocabSize: vocabSize, peakAt: 0),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)
        let decoder = SpeculativeDecoder(
            draftModel: draft, verificationModel: verifier,
            draftTokenCount: 2, samplingPipeline: .greedy
        )
        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.acceptedTokens.count <= 3)
    }

    @Test func decodingResultProperties() async throws {
        let vocabSize = 3
        let draftLogits = [makeLogits(vocabSize: vocabSize, peakAt: 1)]
        let verifyLogits = [
            makeLogits(vocabSize: vocabSize, peakAt: 1),
            makeLogits(vocabSize: vocabSize, peakAt: 2),
        ]
        let draft = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: draftLogits)
        let verifier = MockSpecModel(vocabSize: vocabSize, fixedLogitsSequence: verifyLogits)
        let decoder = SpeculativeDecoder(
            draftModel: draft, verificationModel: verifier,
            draftTokenCount: 1, samplingPipeline: .greedy
        )
        let result = try await decoder.decodeStep(inputTokens: [0])
        #expect(result.draftTokenCount == 1)
        #expect(result.acceptedTokens.count >= 1)
    }
}

extension Array {
    fileprivate func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var results = [T]()
        results.reserveCapacity(count)
        for element in self {
            try await results.append(transform(element))
        }
        return results
    }
}
