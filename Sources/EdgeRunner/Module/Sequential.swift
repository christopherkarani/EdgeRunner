/// A container that chains modules in sequence, feeding the output of each
/// into the input of the next.
public struct Sequential<M: EdgeRunnerModule>: EdgeRunnerModule where M.Input == M.Output {
    public typealias Input = M.Input
    public typealias Output = M.Output

    private let modules: [M]

    public init(_ modules: M...) {
        self.modules = modules
    }

    public init(_ modules: [M]) {
        self.modules = modules
    }

    public init<M1: EdgeRunnerModule, M2: EdgeRunnerModule>(_ first: M1, _ second: M2)
    where
        M == AnyModule<M1.Input>,
        M1.Input == M1.Output,
        M2.Input == M2.Output,
        M1.Input == M2.Input
    {
        self.modules = [AnyModule(first), AnyModule(second)]
    }

    public func forward(_ input: Input) async throws -> Output {
        var current = input
        for module in modules {
            current = try await module.forward(current)
        }
        return current
    }

    public var parameters: [String: any TensorBox] {
        var result: [String: any TensorBox] = [:]
        for (index, module) in modules.enumerated() {
            for (key, value) in module.parameters {
                result["\(index).\(key)"] = value
            }
        }
        return result
    }
}
