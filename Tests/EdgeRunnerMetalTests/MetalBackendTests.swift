import Testing
import Foundation
import Metal
@testable import EdgeRunnerMetal

@Suite("MetalBackend")
struct MetalBackendTests {
    @Test func sharedInstanceExists() async {
        let backend = MetalBackend.shared
        let name = await backend.deviceName
        #expect(!name.isEmpty)
    }

    @Test func acquireAndRecycleBuffer() async throws {
        let backend = MetalBackend.shared
        let length = await backend.acquireAndRecycleRoundTrip(size: 1024)
        #expect(length >= 1024)
    }

    @Test func kernelRegistryAccessible() async throws {
        let backend = MetalBackend.shared
        let maxThreads = try await backend.pipelineMaxThreads(for: "elementwise_add_float")
        #expect(maxThreads > 0)
    }

}
