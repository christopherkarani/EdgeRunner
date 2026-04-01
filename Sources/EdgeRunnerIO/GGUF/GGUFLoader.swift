import Foundation
import Metal

public struct GGUFLoader: EdgeRunnerWeightLoader, Sendable {
    private struct PreparedFile: Sendable {
        let url: URL
        let mappedFile: MemoryMappedFile
        let tensorInfos: [GGUFTensorInfo]
        let dataSectionOffset: Int
        let modelConfig: ModelConfig
    }

    private let prepared: PreparedFile?

    public init() {
        self.prepared = nil
    }

    public init(url: URL) throws {
        self.prepared = try Self.prepare(url: url)
    }

    public var modelConfig: ModelConfig {
        prepared?.modelConfig ?? ModelConfig(architectureName: "", metadata: [:])
    }

    public func canLoad(url: URL) -> Bool {
        url.pathExtension.lowercased() == "gguf"
    }

    public func load(from url: URL) async throws -> WeightMap {
        let preparedFile = if let prepared, prepared.url == url {
            prepared
        } else {
            try Self.prepare(url: url)
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw WeightLoaderError.deviceNotAvailable
        }

        var weightMap = WeightMap()
        for tensorInfo in preparedFile.tensorInfos {
            let tensorOffset = preparedFile.dataSectionOffset + Int(tensorInfo.offset)
            let byteCount = try Self.tensorByteCount(for: tensorInfo)
            guard tensorOffset + byteCount <= preparedFile.mappedFile.size else {
                throw WeightLoaderError.invalidFormat(
                    "Tensor \(tensorInfo.name) exceeds GGUF data section bounds"
                )
            }

            guard let dataType = tensorInfo.type.tensorDataType else {
                throw WeightLoaderError.unsupportedDataType(tensorInfo.type.rawValue)
            }

            let region = try preparedFile.mappedFile.makeMetalBufferRegion(
                device: device,
                offset: tensorOffset,
                length: byteCount
            )

            weightMap[tensorInfo.name] = TensorStorage(
                buffer: region.buffer,
                byteOffset: region.offset,
                dataType: dataType,
                shape: tensorInfo.dimensions.map(Int.init),
                name: tensorInfo.name,
                owner: preparedFile.mappedFile
            )
        }

        return weightMap
    }

    private static func prepare(url: URL) throws -> PreparedFile {
        let mappedFile = try MemoryMappedFile(url: url)
        let reader = GGUFReader(data: mappedFile.mappedData)
        let header = try reader.readHeader()
        let metadata = try reader.readMetadata(count: Int(header.metadataKVCount))
        let tensorInfos = try reader.readTensorInfos(count: Int(header.tensorCount))
        let dataSectionOffset = align(reader.currentOffset, to: 32)
        let modelConfig = try ModelConfig.from(ggufMetadata: metadata)

        return PreparedFile(
            url: url,
            mappedFile: mappedFile,
            tensorInfos: tensorInfos,
            dataSectionOffset: dataSectionOffset,
            modelConfig: modelConfig
        )
    }

    private static func align(_ offset: Int, to alignment: Int) -> Int {
        let remainder = offset % alignment
        return remainder == 0 ? offset : offset + (alignment - remainder)
    }

    private static func tensorByteCount(for info: GGUFTensorInfo) throws -> Int {
        let elementCount = info.elementCount

        func blocks(of size: Int) -> Int {
            (elementCount + size - 1) / size
        }

        switch info.type {
        case .f32:
            return elementCount * MemoryLayout<Float>.stride
        case .f16:
            return elementCount * MemoryLayout<UInt16>.stride
        case .q4_0:
            return blocks(of: 32) * 18
        case .q4_1:
            return blocks(of: 32) * 20
        case .q5_0:
            return blocks(of: 32) * 22
        case .q5_1:
            return blocks(of: 32) * 24
        case .q8_0:
            return blocks(of: 32) * 34
        case .q8_1:
            return blocks(of: 32) * 36
        case .q2_K:
            return blocks(of: 256) * 84
        case .q3_K:
            return blocks(of: 256) * 110
        case .q4_K:
            return blocks(of: 256) * 144
        case .q5_K:
            return blocks(of: 256) * 176
        case .q6_K:
            return blocks(of: 256) * 210
        case .q8_K:
            return blocks(of: 256) * 292
        case .i8:
            return elementCount * MemoryLayout<Int8>.stride
        case .i16:
            return elementCount * MemoryLayout<Int16>.stride
        case .i32:
            return elementCount * MemoryLayout<Int32>.stride
        case .i64:
            return elementCount * MemoryLayout<Int64>.stride
        case .f64:
            return elementCount * MemoryLayout<Double>.stride
        case .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs:
            throw WeightLoaderError.unsupportedDataType(info.type.rawValue)
        case .q1_0_g128:
            return blocks(of: 128) * 18
        }
    }
}
