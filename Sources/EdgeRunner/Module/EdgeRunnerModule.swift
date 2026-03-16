/// Core protocol for all neural network modules in EdgeRunner.
public protocol EdgeRunnerModule: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable

    /// Perform the forward computation.
    func forward(_ input: Input) async throws -> Output

    /// All learnable parameters, keyed by name.
    var parameters: [String: any TensorBox] { get }
}

/// Type-erased module wrapper for composing heterogeneous modules with the same
/// input and output type.
public struct AnyModule<Value: Sendable>: EdgeRunnerModule, Sendable {
    public typealias Input = Value
    public typealias Output = Value

    private let _forward: @Sendable (Value) async throws -> Value
    private let _parameters: @Sendable () -> [String: any TensorBox]

    public init<M: EdgeRunnerModule>(_ module: M) where M.Input == Value, M.Output == Value {
        self._forward = { input in
            try await module.forward(input)
        }
        self._parameters = {
            module.parameters
        }
    }

    public func forward(_ input: Value) async throws -> Value {
        try await _forward(input)
    }

    public var parameters: [String: any TensorBox] {
        _parameters()
    }
}
