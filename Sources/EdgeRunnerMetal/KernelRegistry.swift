import Foundation
import Metal
import Synchronization

public final class KernelRegistry: Sendable {
    private let library: MTLLibrary
    private let cache: Mutex<PipelineCache>
    private let device: MTLDevice

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
    /// so we compile the shader source at runtime.
    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        // First try a pre-compiled metallib (e.g. when built with Xcode)
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        // Fall back to runtime compilation from .metal source files
        let bundle = Bundle.module
        guard let metalURL = bundle.url(forResource: "Elementwise", withExtension: "metal") else {
            throw KernelRegistryError.shaderSourceNotFound
        }
        let source = try String(contentsOf: metalURL, encoding: .utf8)
        let options = MTLCompileOptions()
        return try device.makeLibrary(source: source, options: options)
    }
}

public enum KernelRegistryError: Error, Sendable {
    case functionNotFound(String)
    case shaderSourceNotFound
}
