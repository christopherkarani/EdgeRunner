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
        let length = try await backend.acquireAndRecycleRoundTrip(size: 1024)
        #expect(length >= 1024)
    }

    @Test func kernelRegistryAccessible() async throws {
        let backend = MetalBackend.shared
        let maxThreads = try await backend.pipelineMaxThreads(for: "elementwise_add_float")
        #expect(maxThreads > 0)
    }

    @Test func elementwiseAddLargeArray() async throws {
        let backend = MetalBackend.shared
        let maxThreads = try await backend.pipelineMaxThreads(for: "elementwise_add_float")
        let count = maxThreads + 17
        let a = (0..<count).map { Float($0) }
        let b = (0..<count).map { Float($0) * 0.5 }

        let result = try await backend.elementwiseAddFloat(a, b)

        #expect(result.count == count)
        for index in 0..<count {
            #expect(abs(result[index] - (a[index] + b[index])) < 1e-5)
        }
    }
    
    @Test func publicActorBoundaryKeepsMetalTypesInternal() throws {
        let source = try loadRepoSource("Sources/EdgeRunnerMetal/MetalBackend.swift")

        #expect(source.contains("public actor MetalBackend"))
        #expect(source.contains("public var deviceName: String"))

        #expect(!source.contains("public let device: MTLDevice"))
        #expect(!source.contains("public let commandQueue: MTLCommandQueue"))
        #expect(!source.contains("public func acquireBuffer(size: Int) -> MTLBuffer"))
        #expect(!source.contains("public func recycleBuffer(_ buffer: MTLBuffer)"))
        #expect(!source.contains("public func pipeline(for name: String) throws -> MTLComputePipelineState"))
        #expect(!source.contains("public func dispatch("))
    }

    @Test func uncheckedSendableIsRestrictedToMetalWrappers() throws {
        let sources = try [
            "Sources/EdgeRunnerMetal/BufferCache.swift",
            "Sources/EdgeRunnerMetal/KernelRegistry.swift",
            "Sources/EdgeRunnerMetal/CommandBatcher.swift",
            "Sources/EdgeRunnerMetal/ResidencyManager.swift",
            "Sources/EdgeRunnerMetal/BarrierTracker.swift",
            "Sources/EdgeRunnerCore/TensorStorage.swift",
        ].map(loadRepoSource)

        for source in sources {
            #expect(!source.contains("final class CommandBatcher: @unchecked Sendable"))
            #expect(!source.contains("final class ResidencyManager: @unchecked Sendable"))
            #expect(!source.contains("final class BarrierTracker: @unchecked Sendable"))
            #expect(!source.contains("final class TensorStorage: @unchecked Sendable"))
            #expect(!source.contains("struct CacheState: ~Copyable, @unchecked Sendable"))
            #expect(!source.contains("struct PipelineCache: @unchecked Sendable"))
        }

        let bufferCacheSource = try loadRepoSource("Sources/EdgeRunnerMetal/BufferCache.swift")
        let kernelRegistrySource = try loadRepoSource("Sources/EdgeRunnerMetal/KernelRegistry.swift")
        let tensorStorageSource = try loadRepoSource("Sources/EdgeRunnerCore/TensorStorage.swift")

        #expect(bufferCacheSource.contains("MetalBufferHandle: @unchecked Sendable"))
        #expect(kernelRegistrySource.contains("MetalLibraryHandle: @unchecked Sendable"))
        #expect(kernelRegistrySource.contains("MetalPipelineHandle: @unchecked Sendable"))
        #expect(tensorStorageSource.contains("let buffer: MetalBufferHandle"))
    }
}

private func loadRepoSource(_ relativePath: String) throws -> String {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let fileURL = repoRoot.appendingPathComponent(relativePath)
    return try String(contentsOf: fileURL, encoding: .utf8)
}
