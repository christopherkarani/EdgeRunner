import Testing
import Foundation
@testable import EdgeRunner

struct BonsaiBenchmark {
    static let modelPath = "/tmp/edgerunner-models/Bonsai-1.7B.gguf"
    
    @Test("Bonsai 1.7B Q1_0_g128 decode benchmark")
    func bonsaiDecodeBenchmark() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }
        
        let fileSize = try FileManager.default.attributesOfItem(atPath: Self.modelPath)[.size] as? Int64 ?? 0
        print("\n=== Bonsai-1.7B Q1_0_g128 Benchmark ===")
        let fileSizeMB = Double(fileSize) / 1024.0 / 1024.0
        print("File size: \(String(format: "%.1f", fileSizeMB)) MB")
        
        var timings: [Double] = []
        var ttfts: [Double] = []
        
        for run in 0..<5 {
            let runner = try await EdgeRunner(modelPath: Self.modelPath)
            
            let prompt = "Explain quantum computing in simple terms."
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await runner.generate(prompt, maxTokens: 128)
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            
            let tokenCount = result.split(separator: " ").count
            let tokPerSec = Double(tokenCount) / totalTime
            
            timings.append(tokPerSec)
            ttfts.append(0)
            
            print("Run \(run): e2e=\(String(format: "%.1f", tokPerSec)) tok/s  tokens=\(tokenCount)")
            print("  Output: \(result.prefix(100))...")
        }
        
        let median = timings.sorted()[2]
        let mean = timings.reduce(0, +) / Double(timings.count)
        let maxTok = timings.max() ?? 0
        
        print("\n=== Results ===")
        print("Median e2e: \(String(format: "%.1f", median)) tok/s")
        print("Mean e2e: \(String(format: "%.1f", mean)) tok/s")
        print("Max e2e: \(String(format: "%.1f", maxTok)) tok/s")
        
        #expect(Bool(true), "Bonsai-1.7B Q1_0_g128 benchmark completed: \(String(format: "%.1f", median)) tok/s median")
    }
}
