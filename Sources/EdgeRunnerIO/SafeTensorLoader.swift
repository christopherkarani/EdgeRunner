import Foundation
import Metal

public struct SafeTensorLoader: EdgeRunnerWeightLoader, Sendable {
    private struct PreparedFile: Sendable {
        let url: URL
        let mappedFile: MemoryMappedFile
        let header: SafeTensorHeader
        let modelConfig: ModelConfig
    }

    private let prepared: PreparedFile?

    public init() {
        self.prepared = nil
    }

    public init(url: URL) throws {
        self.prepared = try Self.prepare(url: url)
    }

    public var tensorNames: [String] {
        prepared?.header.tensors.keys.sorted() ?? []
    }

    public var modelConfig: ModelConfig {
        prepared?.modelConfig ?? ModelConfig(architectureName: "", metadata: [:])
    }

    public func canLoad(url: URL) -> Bool {
        url.pathExtension.lowercased() == "safetensors"
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
        for name in preparedFile.header.tensors.keys.sorted() {
            guard let tensor = try loadTensor(
                named: name,
                from: preparedFile,
                device: device
            ) else {
                throw WeightLoaderError.tensorNotFound(name)
            }
            weightMap[name] = tensor
        }
        return weightMap
    }

    public func loadTensor(named name: String) throws -> TensorStorage {
        guard let prepared else {
            throw WeightLoaderError.invalidFormat(
                "SafeTensorLoader must be initialised with a file URL before named loads"
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

    private static func prepare(url: URL) throws -> PreparedFile {
        let mappedFile = try MemoryMappedFile(url: url)
        let header = try SafeTensorHeader.parse(from: mappedFile.mappedData)
        let architectureName = header.metadata["architecture"]?.stringValue
            ?? header.metadata["general.architecture"]?.stringValue
            ?? ""

        return PreparedFile(
            url: url,
            mappedFile: mappedFile,
            header: header,
            modelConfig: ModelConfig(
                architectureName: architectureName,
                metadata: header.metadata
            )
        )
    }

    private func loadTensor(
        named name: String,
        from prepared: PreparedFile,
        device: MTLDevice
    ) throws -> TensorStorage? {
        guard let meta = prepared.header.tensors[name] else {
            return nil
        }

        let absoluteOffset = prepared.header.dataOffset + meta.dataOffsets.begin
        let byteCount = meta.dataOffsets.end - meta.dataOffsets.begin
        guard absoluteOffset + byteCount <= prepared.mappedFile.size else {
            throw WeightLoaderError.invalidFormat(
                "Tensor \(name) exceeds SafeTensor data section bounds"
            )
        }

        let region = try prepared.mappedFile.makeMetalBufferRegion(
            device: device,
            offset: absoluteOffset,
            length: byteCount
        )

        return TensorStorage(
            buffer: region.buffer,
            byteOffset: region.offset,
            dataType: meta.dtype.tensorDataType,
            shape: meta.shape,
            name: name,
            owner: prepared.mappedFile
        )
    }
}
