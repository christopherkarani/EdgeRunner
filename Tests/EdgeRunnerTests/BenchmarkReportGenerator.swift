import Testing
import Foundation
import Metal

@Suite("Benchmark Report")
struct BenchmarkReport {

    @Test func printDeviceInfo() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No Metal device available")
            return
        }

        let maxMemGB = String(format: "%.0f", Double(device.recommendedMaxWorkingSetSize) / 1_073_741_824.0)

        print("""

        ============================================
        EDGERUNNER BENCHMARK REPORT
        ============================================

        ## System Info

        | Property | Value |
        |----------|-------|
        | Device | \(device.name) |
        | Unified Memory | \(device.hasUnifiedMemory) |
        | Max Working Set | \(maxMemGB) GB |
        | Date | \(ISO8601DateFormatter().string(from: Date())) |
        | Swift | 6.2 |

        To collect full results, run:
          swift test --filter "Benchmark" 2>&1 | grep "BENCHMARK:"

        ============================================

        """)
    }
}
