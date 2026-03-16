import Foundation
import Metal
import Testing
@testable import EdgeRunnerIO

private struct SyntheticNPZ: Sendable {
    static func buildNPY(dtype: NPYDtype, shape: [Int], data: Data) -> Data {
        let descr: String
        switch dtype {
        case .float32:
            descr = "<f4"
        case .float16:
            descr = "<f2"
        case .int8:
            descr = "|i1"
        }

        let shapeString = shape.map(String.init).joined(separator: ", ")
        let tupleSuffix = shape.count == 1 ? "," : ""
        let headerString =
            "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(shapeString)\(tupleSuffix)), }"

        let preambleLength = 10
        var paddedHeader = headerString
        let totalLength = preambleLength + paddedHeader.utf8.count + 1
        let padding = (64 - (totalLength % 64)) % 64
        paddedHeader += String(repeating: " ", count: padding) + "\n"

        var result = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        append(UInt16(paddedHeader.utf8.count), to: &result)
        result.append(Data(paddedHeader.utf8))
        result.append(data)
        return result
    }

    static func buildNPZ(entries: [(name: String, data: Data)]) -> Data {
        var archive = Data()
        var centralDirectory = Data()

        for entry in entries {
            let localHeaderOffset = UInt32(archive.count)
            let fileName = Data(entry.name.utf8)
            let uncompressedSize = UInt32(entry.data.count)

            append(UInt32(0x04034b50), to: &archive)
            append(UInt16(20), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt16(0), to: &archive)
            append(UInt32(0), to: &archive)
            append(uncompressedSize, to: &archive)
            append(uncompressedSize, to: &archive)
            append(UInt16(fileName.count), to: &archive)
            append(UInt16(0), to: &archive)
            archive.append(fileName)
            archive.append(entry.data)

            append(UInt32(0x02014b50), to: &centralDirectory)
            append(UInt16(20), to: &centralDirectory)
            append(UInt16(20), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt32(0), to: &centralDirectory)
            append(uncompressedSize, to: &centralDirectory)
            append(uncompressedSize, to: &centralDirectory)
            append(UInt16(fileName.count), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt16(0), to: &centralDirectory)
            append(UInt32(0), to: &centralDirectory)
            append(localHeaderOffset, to: &centralDirectory)
            centralDirectory.append(fileName)
        }

        let centralDirectoryOffset = UInt32(archive.count)
        archive.append(centralDirectory)

        append(UInt32(0x06054b50), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(entries.count), to: &archive)
        append(UInt16(entries.count), to: &archive)
        append(UInt32(centralDirectory.count), to: &archive)
        append(centralDirectoryOffset, to: &archive)
        append(UInt16(0), to: &archive)

        return archive
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { pointer in
            data.append(contentsOf: pointer)
        }
    }
}

@Suite("NPZ Loader Tests")
struct NPZLoaderTests: Sendable {
    @Test("Parse .npy header: magic, version, dtype, shape")
    func parseNPYHeader() throws {
        let floats: [Float] = [1, 2, 3]
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let npy = SyntheticNPZ.buildNPY(dtype: .float32, shape: [3], data: data)

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .float32)
        #expect(header.shape == [3])
        #expect(!header.isFortranOrder)
    }

    @Test("Load float32 tensor from .npy")
    func loadFloat32() throws {
        let floats: [Float] = [1, 2, 3, 4, 5, 6]
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let npy = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 3], data: data)

        let (header, dataOffset) = try NPYHeader.parseWithOffset(from: npy)
        #expect(header.shape == [2, 3])
        #expect(npy.count - dataOffset == 24)
    }

    @Test("Load float16 tensor from .npy")
    func loadFloat16() throws {
        let npy = SyntheticNPZ.buildNPY(
            dtype: .float16,
            shape: [4],
            data: Data(repeating: 0, count: 8)
        )

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .float16)
        #expect(header.shape == [4])
    }

    @Test("Load int8 tensor from .npy")
    func loadInt8() throws {
        let npy = SyntheticNPZ.buildNPY(
            dtype: .int8,
            shape: [8],
            data: Data([1, 2, 3, 4, 5, 6, 7, 8])
        )

        let header = try NPYHeader.parse(from: npy)
        #expect(header.dtype == .int8)
        #expect(header.shape == [8])
    }

    @Test("Load tensors from NPZ (zip of .npy)")
    func loadNPZ() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw WeightLoaderError.deviceNotAvailable
        }

        let w1 = [Float](repeating: 1, count: 4).withUnsafeBufferPointer { Data(buffer: $0) }
        let w2 = [Float](repeating: 2, count: 6).withUnsafeBufferPointer { Data(buffer: $0) }
        let npy1 = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 2], data: w1)
        let npy2 = SyntheticNPZ.buildNPY(dtype: .float32, shape: [2, 3], data: w2)
        let npz = SyntheticNPZ.buildNPZ(
            entries: [
                ("weight_a.npy", npy1),
                ("weight_b.npy", npy2),
            ]
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".npz")
        try npz.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loader = try NPZLoader(url: tmpURL)
        #expect(loader.tensorNames == ["weight_a", "weight_b"])

        let storage = try loader.loadTensor(named: "weight_a")
        #expect(storage.shape == [2, 2])
        #expect(storage.dataType == .float32)
        #expect(readFloat32(storage, count: 4) == [1, 1, 1, 1])

        let weightMap = try await loader.load(from: tmpURL)
        #expect(weightMap.count == 2)
        #expect(weightMap["weight_b"]?.shape == [2, 3])
    }

    @Test("Invalid .npy magic throws error")
    func invalidMagic() {
        let garbage = Data([0, 1, 2, 3, 4, 5, 6, 7])
        #expect(throws: NPYError.self) {
            _ = try NPYHeader.parse(from: garbage)
        }
    }
}

private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { pointer in
        data.append(contentsOf: pointer)
    }
}

private func readFloat32(_ storage: TensorStorage, count: Int) -> [Float] {
    let pointer = storage.buffer.contents()
        .advanced(by: storage.byteOffset)
        .bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}
