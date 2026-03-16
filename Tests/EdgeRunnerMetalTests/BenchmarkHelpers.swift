import Foundation
import Testing
import Metal

/// Result from a single benchmark run.
public struct BenchmarkResult: Sendable {
    public let name: String
    public let value: Double
    public let unit: String
    public let iterations: Int
    public let totalSeconds: Double

    public var perIterationMs: Double { (totalSeconds / Double(iterations)) * 1000.0 }

    public func print() {
        Swift.print("BENCHMARK: \(name) \(String(format: "%.1f", value)) \(unit) (\(String(format: "%.2f", perIterationMs)) ms/op, \(iterations) iters)")
    }
}

/// Time a block over multiple iterations with warmup.
public func benchmark(
    name: String,
    warmup: Int = 3,
    iterations: Int = 10,
    body: () async throws -> Void
) async throws -> BenchmarkResult {
    // Warmup
    for _ in 0..<warmup {
        try await body()
    }

    // Timed runs
    let clock = ContinuousClock()
    let start = clock.now
    for _ in 0..<iterations {
        try await body()
    }
    let elapsed = start.duration(to: clock.now)
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18

    return BenchmarkResult(name: name, value: 0, unit: "", iterations: iterations, totalSeconds: seconds)
}

/// Get device info string for reports.
public func deviceInfoString() -> String {
    guard let device = MTLCreateSystemDefaultDevice() else { return "No Metal device" }
    let memGB = String(format: "%.0f", Double(device.recommendedMaxWorkingSetSize) / 1_073_741_824.0)
    return "\(device.name) (\(memGB) GB, unified=\(device.hasUnifiedMemory))"
}
