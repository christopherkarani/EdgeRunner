import Testing
@testable import EdgeRunnerCore

@Test func tensorInitialization() {
    let t = Tensor<Float>(shape: [2, 3])
    #expect(t.shape == [2, 3])
}
