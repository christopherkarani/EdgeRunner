import Testing
@testable import EdgeRunner

@Suite("Gemma4DecodeState")
struct Gemma4DecodeStateTests {
    @Test("Classifies full prefill, prefix reuse, single-token decode, and reset")
    func modeDetection() {
        var state = Gemma4DecodeState()

        #expect(state.prepare(tokenIDs: [10, 11, 12]) == .fullPrefill(tokens: [10, 11, 12], startPosition: 0))
        state.markProcessed(tokenIDs: [10, 11, 12])

        #expect(state.prepare(tokenIDs: [10, 11, 12, 13]) == .decode(token: 13, position: 3))
        state.markProcessed(tokenIDs: [10, 11, 12, 13])

        #expect(state.prepare(tokenIDs: [10, 11, 12, 13, 14, 15]) == .prefixReuse(tokens: [14, 15], startPosition: 4))
        state.markProcessed(tokenIDs: [10, 11, 12, 13, 14, 15])

        #expect(state.prepare(tokenIDs: [99]) == .fullPrefill(tokens: [99], startPosition: 0))
        state.markProcessed(tokenIDs: [99])

        state.reset()
        #expect(state.prepare(tokenIDs: [1]) == .fullPrefill(tokens: [1], startPosition: 0))
    }

    @Test("Empty input is a reset-sized full prefill")
    func emptyInput() {
        var state = Gemma4DecodeState()
        #expect(state.prepare(tokenIDs: []) == .fullPrefill(tokens: [], startPosition: 0))
        state.markProcessed(tokenIDs: [1, 2, 3])
        #expect(state.prepare(tokenIDs: []) == .fullPrefill(tokens: [], startPosition: 0))
    }
}
