import Foundation
import Metal
import EdgeRunnerIO

public struct WeightConverter: Sendable {
    private let device: MTLDevice

    public init(device: MTLDevice) throws {
        self.device = device
    }

    /// Converts all tensors in a `WeightMap` to BLOBFILE format under `outputDirectory`.
    /// Returns the number of tensors successfully converted.
    @discardableResult
    public func convert(
        weightMap: WeightMap,
        architecture: String,
        outputDirectory: URL
    ) async throws -> Int {
        var converted = 0
        for name in weightMap.tensorNames {
            guard let tensor = weightMap[name] else { continue }
            guard EspressoTensorNameMapper.espressoPath(for: name, architecture: architecture) != nil else { continue }
            try await convertTensor(
                tensor, ggufName: name,
                architecture: architecture,
                outputDirectory: outputDirectory
            )
            converted += 1
        }
        return converted
    }

    /// Converts a single tensor to a BLOBFILE at its Espresso path.
    public func convertTensor(
        _ tensor: TensorStorage,
        ggufName: String,
        architecture: String,
        outputDirectory: URL
    ) async throws {
        guard let relativePath = EspressoTensorNameMapper.espressoPath(for: ggufName, architecture: architecture) else {
            throw EspressoError.unmappedTensorName(ggufName)
        }

        // 1. Dequantize
        var floats = try await DequantDispatcher.dequantize(tensor: tensor, device: device)

        // 2. Transpose if needed (GPT-2 matrix weights)
        if EspressoTensorNameMapper.requiresTranspose(ggufName: ggufName, architecture: architecture) {
            guard tensor.shape.count == 2 else {
                throw EspressoError.transposeDimensionMismatch(name: ggufName, shape: tensor.shape)
            }
            let rows = tensor.shape[0]
            let cols = tensor.shape[1]
            guard floats.count == rows * cols else {
                throw EspressoError.transposeDimensionMismatch(
                    name: ggufName, shape: tensor.shape
                )
            }
            var transposed = [Float](repeating: 0, count: rows * cols)
            for i in 0..<rows {
                for j in 0..<cols {
                    transposed[j * rows + i] = floats[i * cols + j]
                }
            }
            floats = transposed
        }

        // 3. Validate output path stays within outputDirectory (path traversal protection)
        let outputURL = outputDirectory.appendingPathComponent(relativePath)
        let canonicalOutput = outputURL.standardizedFileURL.path
        let canonicalBase = outputDirectory.standardizedFileURL.path
        guard canonicalOutput.hasPrefix(canonicalBase + "/") || canonicalOutput == canonicalBase else {
            throw EspressoError.pathTraversal(relativePath)
        }

        let directory = outputURL.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw EspressoError.directoryCreationFailed(directory.path)
        }
        try BlobfileWriter.write(floats: floats, to: outputURL)
    }
}
