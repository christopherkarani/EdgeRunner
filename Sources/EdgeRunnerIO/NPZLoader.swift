import Compression
import Foundation
import Metal

public struct NPZLoader: EdgeRunnerWeightLoader, Sendable {
    private struct Entry: Sendable {
        let name: String
        let header: NPYHeader
        let tensorData: Data
    }

    private struct PreparedArchive: Sendable {
        let url: URL
        let entries: [String: Entry]
    }

    private let prepared: PreparedArchive?

    public init() {
        self.prepared = nil
    }

    public init(url: URL) throws {
        self.prepared = try Self.prepare(url: url)
    }

    public var tensorNames: [String] {
        prepared?.entries.keys.sorted() ?? []
    }

    public var modelConfig: ModelConfig {
        ModelConfig(architectureName: "", metadata: [:])
    }

    public func canLoad(url: URL) -> Bool {
        url.pathExtension.lowercased() == "npz"
    }

    public func load(from url: URL) async throws -> WeightMap {
        let preparedArchive = if let prepared, prepared.url == url {
            prepared
        } else {
            try Self.prepare(url: url)
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }

        var weightMap = WeightMap()
        for name in preparedArchive.entries.keys.sorted() {
            guard let tensor = try loadTensor(named: name, from: preparedArchive, device: device) else {
                throw WeightLoaderError.tensorNotFound(name)
            }
            weightMap[name] = tensor
        }
        return weightMap
    }

    public func loadTensor(named name: String) throws -> TensorStorage {
        guard let prepared else {
            throw WeightLoaderError.invalidFormat(
                "NPZLoader must be initialised with a file URL before named loads"
            )
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }
        guard let tensor = try loadTensor(named: name, from: prepared, device: device) else {
            throw WeightLoaderError.tensorNotFound(name)
        }
        return tensor
    }

    private static func prepare(url: URL) throws -> PreparedArchive {
        let archiveData = try Data(contentsOf: url)
        var entries: [String: Entry] = [:]
        var offset = 0

        while offset + 4 <= archiveData.count {
            let signature = readUInt32(from: archiveData, at: offset)
            guard signature == 0x04034B50 else {
                break
            }

            guard offset + 30 <= archiveData.count else {
                throw NPYError.fileTooSmall
            }

            let flags = readUInt16(from: archiveData, at: offset + 6)
            let compressionMethod = readUInt16(from: archiveData, at: offset + 8)
            let compressedSize = Int(readUInt32(from: archiveData, at: offset + 18))
            let uncompressedSize = Int(readUInt32(from: archiveData, at: offset + 22))
            let fileNameLength = Int(readUInt16(from: archiveData, at: offset + 26))
            let extraLength = Int(readUInt16(from: archiveData, at: offset + 28))

            if flags & 0x08 != 0 {
                throw NPYError.invalidHeader("ZIP data descriptors are not supported")
            }

            let nameStart = offset + 30
            let nameEnd = nameStart + fileNameLength
            guard nameEnd <= archiveData.count else {
                throw NPYError.fileTooSmall
            }
            let dataStart = nameEnd + extraLength
            let dataEnd = dataStart + compressedSize

            guard dataEnd <= archiveData.count else {
                throw NPYError.fileTooSmall
            }

            let fileNameData = archiveData.subdata(in: nameStart..<nameEnd)
            guard let fileName = String(data: fileNameData, encoding: .utf8) else {
                throw NPYError.invalidHeader("ZIP entry filename is not valid UTF-8")
            }

            let payload = archiveData.subdata(in: dataStart..<dataEnd)
            let fileData: Data
            switch compressionMethod {
            case 0:
                fileData = payload
            case 8:
                fileData = try inflate(payload, expectedSize: uncompressedSize)
            default:
                throw NPYError.unsupportedCompressionMethod(compressionMethod)
            }

            if fileName.hasSuffix(".npy") {
                let (header, tensorOffset) = try NPYHeader.parseWithOffset(from: fileData)
                let tensorBytes = fileData.subdata(in: tensorOffset..<fileData.count)
                let tensorName = URL(fileURLWithPath: fileName)
                    .deletingPathExtension()
                    .lastPathComponent
                entries[tensorName] = Entry(name: tensorName, header: header, tensorData: tensorBytes)
            }

            offset = dataEnd
        }

        return PreparedArchive(url: url, entries: entries)
    }

    private func loadTensor(
        named name: String,
        from archive: PreparedArchive,
        device: MTLDevice
    ) throws -> TensorStorage? {
        guard let entry = archive.entries[name] else {
            return nil
        }

        guard let buffer = device.makeBuffer(
            bytes: [UInt8](entry.tensorData),
            length: entry.tensorData.count,
            options: .storageModeShared
        ) else {
            throw WeightLoaderError.allocationFailed(byteCount: entry.tensorData.count)
        }

        return TensorStorage(
            buffer: buffer,
            dataType: entry.header.dtype.tensorDataType,
            shape: entry.header.shape,
            name: entry.name
        )
    }
}

private func inflate(_ data: Data, expectedSize: Int) throws -> Data {
    guard expectedSize > 0 else {
        return Data()
    }

    var output = Data(count: expectedSize)
    let decodedCount = output.withUnsafeMutableBytes { outputBuffer in
        data.withUnsafeBytes { inputBuffer in
            compression_decode_buffer(
                outputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                expectedSize,
                inputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                data.count,
                nil,
                COMPRESSION_ZLIB
            )
        }
    }

    guard decodedCount > 0 else {
        throw NPYError.invalidHeader("Failed to inflate ZIP entry")
    }

    output.count = decodedCount
    return output
}
