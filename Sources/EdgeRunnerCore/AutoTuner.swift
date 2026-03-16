import Metal

public struct ThreadgroupConfig: Sendable {
    public let width: Int
    public let height: Int
    public let depth: Int

    public init(width: Int, height: Int, depth: Int) {
        self.width = width; self.height = height; self.depth = depth
    }

    public static let `default` = ThreadgroupConfig(width: 256, height: 1, depth: 1)

    public var metalSize: MTLSize { MTLSize(width: width, height: height, depth: depth) }

    public func threadgroups(for elementCount: Int) -> MTLSize {
        MTLSize(width: (elementCount + width - 1) / width, height: 1, depth: 1)
    }

    public func threadgroups2D(rows: Int, cols: Int) -> MTLSize {
        MTLSize(width: (cols + width - 1) / width, height: (rows + height - 1) / height, depth: 1)
    }
}

public enum KernelCategory: Sendable { case elementwise, reduction, transpose }

public enum AutoTuner {
    public static func config(for category: KernelCategory, elementCount: Int) -> ThreadgroupConfig {
        switch category {
        case .elementwise: return ThreadgroupConfig(width: min(256, elementCount), height: 1, depth: 1)
        case .reduction: return ThreadgroupConfig(width: min(256, nextPowerOf2(elementCount)), height: 1, depth: 1)
        case .transpose: return ThreadgroupConfig(width: 16, height: 16, depth: 1)
        }
    }

    private static func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; return v + 1
    }
}
