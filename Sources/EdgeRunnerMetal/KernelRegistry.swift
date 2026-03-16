import Foundation
import Metal
import Synchronization

public final class KernelRegistry: Sendable {
    private let library: MTLLibrary
    private let cache: Mutex<PipelineCache>
    private let device: MTLDevice

    /// The compiled Metal library. Exposed for callers that need to create
    /// functions with `MTLFunctionConstantValues` (e.g. fused-pattern kernels).
    public var metalLibrary: MTLLibrary { library }

    // MTLComputePipelineState is not Sendable but Mutex ensures exclusive access.
    struct PipelineCache: @unchecked Sendable {
        var entries: [String: MTLComputePipelineState] = [:]
    }

    public init(device: MTLDevice) throws {
        self.device = device
        self.library = try Self.loadLibrary(device: device)
        self.cache = Mutex(PipelineCache())
    }

    public func pipeline(for name: String) throws -> MTLComputePipelineState {
        if let cached = cache.withLock({ $0.entries[name] }) {
            return cached
        }
        guard let function = library.makeFunction(name: name) else {
            throw KernelRegistryError.functionNotFound(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        cache.withLock { $0.entries[name] = pipeline }
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

public enum KernelRegistryError: Error, Sendable {
    case functionNotFound(String)
    case shaderSourceNotFound
}
