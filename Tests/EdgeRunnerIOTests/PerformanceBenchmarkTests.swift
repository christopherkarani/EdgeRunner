import Foundation
import Testing
@testable import EdgeRunnerIO

@Suite("Performance Benchmark Tests")
struct PerformanceBenchmarkTests: Sendable {
    @Test("SafeTensor loader performance: 1000 tensors under 1 second")
    func safetensorLoadPerformance() throws {
        let url = performanceFileURL(ext: "safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let blob = SyntheticSafeTensorBenchmark.build(tensorCount: 1_000)
        try blob.write(to: url)

        let clock = ContinuousClock()
        let start = clock.now
        let loader = try SafeTensorLoader(url: url)
        let elapsed = start.duration(to: clock.now)

        #expect(loader.tensorNames.count == 1_000)
        #expect(elapsed < .seconds(1))
    }

    @Test("LlamaModel construction performance: 32 layers under 100ms")
    func modelConstructionPerformance() {
        let config = LlamaConfig(
            embeddingDim: 4_096,
            layerCount: 32,
            headCount: 32,
            kvHeadCount: 8,
            vocabSize: 128_256,
            intermediateDim: 14_336,
            ropeFreqBase: 500_000.0,
            rmsNormEpsilon: 1e-5
        )

        let clock = ContinuousClock()
        let start = clock.now
        let model = LlamaModel(config: config)
        let elapsed = start.duration(to: clock.now)

        #expect(model.layers.count == 32)
        #expect(elapsed < .milliseconds(100))
    }

    @Test("Memory estimate for Llama 3 8B at different quantisation levels")
    func memoryEstimation() {
        let paramCount = 8_000_000_000.0

        let q8Memory = paramCount * QuantisationLevel.q8_0.bitsPerWeight / 8.0
        let q4KM = paramCount * QuantisationLevel.q4_k_m.bitsPerWeight / 8.0
        let q4_0 = paramCount * QuantisationLevel.q4_0.bitsPerWeight / 8.0

        #expect(q8Memory > 7_500_000_000 && q8Memory < 8_500_000_000)
        #expect(q4KM > 4_000_000_000 && q4KM < 5_000_000_000)
        #expect(q4_0 > 3_500_000_000 && q4_0 < 4_500_000_000)
    }
}

private enum SyntheticSafeTensorBenchmark {
    static func build(tensorCount: Int) -> Data {
        var entries: [String] = []
        var dataSection = Data()

        for index in 0..<tensorCount {
            let begin = dataSection.count
            dataSection.append(Data(repeating: 0, count: 64 * MemoryLayout<Float>.stride))
            let end = dataSection.count
            entries.append(
                """
                "tensor_\(index)":{"dtype":"F32","shape":[8,8],"data_offsets":[\(begin),\(end)]}
                """
            )
        }

        let headerData = Data("{\(entries.joined(separator: ","))}".utf8)
        var headerSize = UInt64(headerData.count).littleEndian

        var blob = Data()
        withUnsafeBytes(of: &headerSize) { blob.append(contentsOf: $0) }
        blob.append(headerData)
        blob.append(dataSection)
        return blob
    }
}

private func performanceFileURL(ext: String) -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension(ext)
}
