import Foundation
import Metal
import EdgeRunnerIO

public struct WeightConverter: Sendable {
    static let forceLlamaMatrixTransposeEnvKey = "ESPRESSO_FORCE_LLAMA_MATRIX_TRANSPOSE"
    static let skipLlamaMatrixTransposeEnvKey = "ESPRESSO_SKIP_LLAMA_MATRIX_TRANSPOSE"
    static let forceQKInverseInterleaveEnvKey = "ESPRESSO_FORCE_QK_INVERSE_INTERLEAVE"
    static let dimMajorToHeadMajorVWeightEnvKey = "ESPRESSO_DIM_MAJOR_TO_HEAD_MAJOR_V_WEIGHT"
    static let inverseInterleaveVWeightEnvKey = "ESPRESSO_INVERSE_INTERLEAVE_V_WEIGHT"
    static let forwardInterleaveVWeightEnvKey = "ESPRESSO_FORWARD_INTERLEAVE_V_WEIGHT"
    private static let exactFloat32TopLevelTensorNames: Set<String> = [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    ]

    /// Minimum contiguous-from-zero layer range that preserves Qwen 0.6B
    /// correctness under FP32 sidecar narrowing.  Binary search over
    /// `0..K` ranges found K=11: layers 0-10 FAIL, layers 0-11 PASS.
    /// Early-layer FP16 rounding errors compound forward; late-layer
    /// sidecars alone cannot compensate.
    static let qwenMinimumSidecarLayerCount = 12

    public enum ExactFloat32SidecarPolicy: Sendable, Equatable {
        case automatic
        case none
        case essential
        case selected(Set<String>)
        case full
    }

    enum ProjectionPermutation: Sendable, Equatable {
        case dimMajorToHeadMajor
        case inverseInterleaving
        case forwardInterleaving
    }

    public struct LlamaProjectionLayout: Sendable, Equatable {
        public let qHeadCount: Int
        public let kvHeadCount: Int
        public let headDim: Int

        public init(qHeadCount: Int, kvHeadCount: Int, headDim: Int) {
            self.qHeadCount = qHeadCount
            self.kvHeadCount = kvHeadCount
            self.headDim = headDim
        }
    }

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
        outputDirectory: URL,
        llamaProjectionLayout: LlamaProjectionLayout? = nil,
        exactFloat32SidecarPolicy: ExactFloat32SidecarPolicy = .automatic
    ) async throws -> Int {
        var converted = 0
        for name in weightMap.tensorNames {
            guard let tensor = weightMap[name] else { continue }
            guard EspressoTensorNameMapper.espressoPath(for: name, architecture: architecture) != nil else { continue }
            try await convertTensor(
                tensor, ggufName: name,
                architecture: architecture,
                outputDirectory: outputDirectory,
                llamaProjectionLayout: llamaProjectionLayout,
                exactFloat32SidecarPolicy: exactFloat32SidecarPolicy
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
        outputDirectory: URL,
        llamaProjectionLayout: LlamaProjectionLayout? = nil,
        exactFloat32SidecarPolicy: ExactFloat32SidecarPolicy = .automatic
    ) async throws {
        guard let relativePath = EspressoTensorNameMapper.espressoPath(for: ggufName, architecture: architecture) else {
            throw EspressoError.unmappedTensorName(ggufName)
        }

        // 1. Dequantize
        var floats = try await DequantDispatcher.dequantize(tensor: tensor, device: device)

        // 2. Re-order matrix storage when the source tensor still uses GGML's [in, out] convention.
        if shouldTransposeMatrix(ggufName: ggufName, architecture: architecture) {
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

        if let permutation = projectionPermutation(
            ggufName: ggufName,
            architecture: architecture,
            llamaProjectionLayout: llamaProjectionLayout
        ) {
            let headCount: Int
            switch ggufName.split(separator: ".").dropFirst(2).joined(separator: ".") {
            case "attn_q.weight":
                headCount = llamaProjectionLayout!.qHeadCount
            default:
                headCount = llamaProjectionLayout!.kvHeadCount
            }
            switch permutation {
            case .dimMajorToHeadMajor:
                floats = WeightPermutation.dimMajorToHeadMajor(
                    weights: floats,
                    numHeads: headCount,
                    headDim: llamaProjectionLayout!.headDim
                )
            case .inverseInterleaving:
                floats = WeightPermutation.inverseInterleaving(
                    weights: floats,
                    numHeads: headCount,
                    headDim: llamaProjectionLayout!.headDim
                )
            case .forwardInterleaving:
                floats = WeightPermutation.forwardInterleaving(
                    weights: floats,
                    numHeads: headCount,
                    headDim: llamaProjectionLayout!.headDim
                )
            }
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

        if Self.shouldWriteExactFloat32Sidecar(
            ggufName: ggufName,
            architecture: architecture,
            policy: exactFloat32SidecarPolicy
        ) {
            let exactSidecarURL = outputDirectory.appendingPathComponent(
                Self.exactFloat32SidecarRelativePath(for: relativePath)
            )
            try FileManager.default.createDirectory(
                at: exactSidecarURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Self.writeExactFloat32Sidecar(floats: floats, to: exactSidecarURL)
        }
    }

    private func projectionPermutation(
        ggufName: String,
        architecture: String,
        llamaProjectionLayout: LlamaProjectionLayout?
    ) -> ProjectionPermutation? {
        guard llamaProjectionLayout != nil else { return nil }
        guard architecture.lowercased() != "gpt2" else { return nil }

        let parts = ggufName.split(separator: ".")
        guard parts.count >= 3, parts[0] == "blk" else { return nil }

        let suffix = parts.dropFirst(2).joined(separator: ".")
        switch suffix {
        case "attn_q.weight" where Self.isEnabled(ProcessInfo.processInfo.environment[Self.forceQKInverseInterleaveEnvKey]):
            return .inverseInterleaving
        case "attn_k.weight" where Self.isEnabled(ProcessInfo.processInfo.environment[Self.forceQKInverseInterleaveEnvKey]):
            return .inverseInterleaving
        case "attn_v.weight" where Self.isEnabled(ProcessInfo.processInfo.environment[Self.dimMajorToHeadMajorVWeightEnvKey]):
            return .dimMajorToHeadMajor
        case "attn_v.weight" where Self.isEnabled(ProcessInfo.processInfo.environment[Self.forwardInterleaveVWeightEnvKey]):
            return .forwardInterleaving
        case "attn_v.weight" where Self.isEnabled(ProcessInfo.processInfo.environment[Self.inverseInterleaveVWeightEnvKey]):
            return .inverseInterleaving
        default:
            return nil
        }
    }

    static func isEnabled(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private func shouldTransposeMatrix(ggufName: String, architecture: String) -> Bool {
        guard EspressoTensorNameMapper.requiresTranspose(ggufName: ggufName, architecture: architecture) else {
            return false
        }
        guard architecture.lowercased() != "gpt2" else {
            return true
        }
        let environment = ProcessInfo.processInfo.environment
        if Self.isEnabled(environment[Self.forceLlamaMatrixTransposeEnvKey]) {
            return true
        }
        if Self.isEnabled(environment[Self.skipLlamaMatrixTransposeEnvKey]) {
            return false
        }
        return false
    }

    public static func shouldWriteExactFloat32Sidecar(
        ggufName: String,
        architecture: String,
        policy: ExactFloat32SidecarPolicy
    ) -> Bool {
        let normalizedArchitecture = architecture.lowercased()
        switch policy {
        case .automatic:
            guard normalizedArchitecture != "gpt2" else {
                return false
            }
            if normalizedArchitecture.contains("qwen") {
                return Self.exactFloat32TopLevelTensorNames.contains(ggufName)
                    || Self.isQwenEarlyLayerTensor(ggufName)
            }
            return Self.exactFloat32TopLevelTensorNames.contains(ggufName)
        case .none:
            return false
        case .essential:
            return Self.exactFloat32TopLevelTensorNames.contains(ggufName)
        case .selected(let tensorNames):
            return Self.exactFloat32TopLevelTensorNames.contains(ggufName) || tensorNames.contains(ggufName)
        case .full:
            return true
        }
    }

    /// Returns `true` when `ggufName` belongs to a layer in the early
    /// range `0 ..< qwenMinimumSidecarLayerCount` (currently layers 0-11).
    static func isQwenEarlyLayerTensor(_ ggufName: String) -> Bool {
        guard ggufName.hasPrefix("blk.") else { return false }
        let afterPrefix = ggufName.dropFirst(4)    // drop "blk."
        guard let dotIndex = afterPrefix.firstIndex(of: ".") else { return false }
        guard let layerIndex = Int(afterPrefix[afterPrefix.startIndex..<dotIndex]) else {
            return false
        }
        return layerIndex < qwenMinimumSidecarLayerCount
    }

    static func exactFloat32SidecarRelativePath(for relativePath: String) -> String {
        if relativePath.hasSuffix(".bin") {
            return String(relativePath.dropLast(4)) + ".float32.bin"
        }
        return relativePath + ".float32"
    }

    private static func writeExactFloat32Sidecar(
        floats: [Float],
        to url: URL
    ) throws {
        var data = Data(capacity: floats.count * MemoryLayout<UInt32>.stride)
        for value in floats {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        try data.write(to: url, options: .atomic)
    }
}
