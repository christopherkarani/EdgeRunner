import Testing
import Foundation
@testable import EdgeRunnerIO
import Metal

@Suite("Loading Benchmarks")
struct LoadingBenchmarks {
    @Test func safetensorParse1000() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let blob = buildSyntheticSafeTensors(tensorCount: 1000)
        try blob.write(to: url)

        let clock = ContinuousClock()
        let iterations = 10
        let start = clock.now
        for _ in 0..<iterations {
            let loader = try SafeTensorLoader(url: url)
            #expect(loader.tensorNames.count == 1000)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: safetensor_parse_1000 \(String(format: "%.2f", msPerOp)) ms/op (\(iterations) iters)")
        #expect(msPerOp < 1000)
    }

    @Test func safetensorParse5000() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let blob = buildSyntheticSafeTensors(tensorCount: 5000)
        try blob.write(to: url)

        let clock = ContinuousClock()
        let iterations = 5
        let start = clock.now
        for _ in 0..<iterations {
            let loader = try SafeTensorLoader(url: url)
            #expect(loader.tensorNames.count == 5000)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let msPerOp = seconds / Double(iterations) * 1000

        print("BENCHMARK: safetensor_parse_5000 \(String(format: "%.2f", msPerOp)) ms/op (\(iterations) iters)")
        #expect(msPerOp < 5000)
    }

    @Test func metalBufferCreation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LoadingBenchmarkError.noMetal
        }

        let bufferSize = 1_048_576 // 1 MB
        let bufferCount = 100
        let data = Data(repeating: 0, count: bufferSize)

        let clock = ContinuousClock()
        let start = clock.now
        var buffers: [MTLBuffer] = []
        buffers.reserveCapacity(bufferCount)
        for _ in 0..<bufferCount {
            let buf = data.withUnsafeBytes { ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: bufferSize, options: .storageModeShared)
            }
            buffers.append(buf!)
        }
        let elapsed = start.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let totalMB = Double(bufferSize * bufferCount) / 1_048_576.0
        let gbPerSec = totalMB / 1024.0 / seconds

        print("BENCHMARK: metal_buffer_create_100x1MB \(String(format: "%.1f", gbPerSec)) GB/s (\(String(format: "%.0f", totalMB)) MB total)")
        buffers.removeAll()
        #expect(gbPerSec > 0.1)
    }
}

private enum LoadingBenchmarkError: Error {
    case noMetal
}

private func buildSyntheticSafeTensors(tensorCount: Int) -> Data {
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
