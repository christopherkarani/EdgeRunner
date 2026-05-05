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
    private let primaryLibrary: MetalLibraryHandle
    private let fallbackSourceLibrary: Mutex<MetalLibraryHandle?>
    private let cache: Mutex<PipelineCache>
    private let device: MTLDevice

    /// The compiled Metal library. Exposed for callers that need to create
    /// functions with `MTLFunctionConstantValues` (e.g. fused-pattern kernels).
    package var metalLibrary: MTLLibrary { primaryLibrary.rawValue }

    private struct PipelineCache: Sendable {
        var entries: [String: MetalPipelineHandle] = [:]
    }

    package init(device: MTLDevice) throws {
        self.device = device
        self.primaryLibrary = MetalLibraryHandle(rawValue: try Self.loadPrimaryLibrary(device: device))
        self.fallbackSourceLibrary = Mutex(nil)
        self.cache = Mutex(PipelineCache())
    }

    package func pipeline(for name: String) throws -> MTLComputePipelineState {
        if let cached = cache.withLock({ $0.entries[name] }) {
            return cached.rawValue
        }
        let function = try resolveFunction(named: name)
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.supportIndirectCommandBuffers = true
        let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        cache.withLock { $0.entries[name] = MetalPipelineHandle(rawValue: pipeline) }
        return pipeline
    }

    /// Loads the primary Metal library from the bundle when available.
    private static func loadPrimaryLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        return try compileSourceLibrary(device: device)
    }

    /// SwiftPM copies .metal files as resources rather than compiling them,
    /// so we can compile the shader source at runtime by concatenating all
    /// bundled .metal files when a requested function is absent from the
    /// primary library.
    private static func compileSourceLibrary(device: MTLDevice) throws -> MTLLibrary {
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

    private func resolveFunction(named name: String) throws -> MTLFunction {
        if let function = primaryLibrary.rawValue.makeFunction(name: name) {
            return function
        }

        let fallbackLibrary = try fallbackSourceLibrary.withLock { cachedLibrary -> MetalLibraryHandle in
            if let cachedLibrary {
                return cachedLibrary
            }
            let compiledLibrary = MetalLibraryHandle(rawValue: try Self.compileSourceLibrary(device: device))
            cachedLibrary = compiledLibrary
            return compiledLibrary
        }

        if let function = fallbackLibrary.rawValue.makeFunction(name: name) {
            return function
        }

        throw KernelRegistryError.functionNotFound(name)
    }
}

package enum KernelRegistryError: Error, Sendable {
    case functionNotFound(String)
    case shaderSourceNotFound
}
