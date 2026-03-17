import Foundation
import Metal
import Synchronization

package struct MetalLibraryHandle: @unchecked Sendable {
    // @unchecked Sendable is limited to the Metal protocol wrapper.
    // The wrapped MTLLibrary stays inside package-internal APIs.
    package let rawValue: MTLLibrary
}

package struct MetalPipelineHandle: @unchecked Sendable {
    // @unchecked Sendable is limited to the Metal protocol wrapper.
    // The wrapped MTLComputePipelineState is cached under Mutex protection.
    package let rawValue: MTLComputePipelineState
}

package final class KernelRegistry {
    private let library: MetalLibraryHandle
    private let cache: Mutex<PipelineCache>
    private let device: MTLDevice

    /// The compiled Metal library. Exposed for callers that need to create
    /// functions with `MTLFunctionConstantValues` (e.g. fused-pattern kernels).
    package var metalLibrary: MTLLibrary { library.rawValue }

    private struct PipelineCache: Sendable {
        var entries: [String: MetalPipelineHandle] = [:]
    }

    package init(device: MTLDevice) throws {
        self.device = device
        self.library = MetalLibraryHandle(rawValue: try Self.loadLibrary(device: device))
        self.cache = Mutex(PipelineCache())
    }

    package func pipeline(for name: String) throws -> MTLComputePipelineState {
        if let cached = cache.withLock({ $0.entries[name] }) {
            return cached.rawValue
        }
        guard let function = library.rawValue.makeFunction(name: name) else {
            throw KernelRegistryError.functionNotFound(name)
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.supportIndirectCommandBuffers = true
        let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        cache.withLock { $0.entries[name] = MetalPipelineHandle(rawValue: pipeline) }
        return pipeline
    }

    /// Loads the Metal library from the bundle resource.
    /// SwiftPM copies .metal files as resources rather than compiling them,
    /// so we compile the shader source at runtime by concatenating all .metal files.
    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        // First try a pre-compiled metallib (e.g. when built with Xcode)
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        // Fall back to runtime compilation — gather all .metal source files in the bundle
        let bundle = Bundle.module
        let metalURLs = bundle.urls(forResourcesWithExtension: "metal", subdirectory: nil) ?? []
        guard !metalURLs.isEmpty else {
            throw KernelRegistryError.shaderSourceNotFound
        }
        // Concatenate all shader sources; each file gets a separator comment for clarity
        let combinedSource = try metalURLs
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { url -> String in
                let src = try String(contentsOf: url, encoding: .utf8)
                return "// --- \(url.lastPathComponent) ---\n" + src
            }
            .joined(separator: "\n\n")
        let options = MTLCompileOptions()
        return try device.makeLibrary(source: combinedSource, options: options)
    }
}

package enum KernelRegistryError: Error, Sendable {
    case functionNotFound(String)
    case shaderSourceNotFound
}
