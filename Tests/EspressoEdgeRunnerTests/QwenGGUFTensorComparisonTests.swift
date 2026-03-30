import Foundation
import Metal
import Testing
@testable import EspressoEdgeRunner
import EdgeRunnerIO

@Suite("Qwen GGUF Tensor Sidecar Comparison", .serialized)
struct QwenGGUFTensorComparisonTests {
    private static let runEnvKey = "ESPRESSO_DEBUG_COMPARE_GGUF_SIDECARS"
    private static let ggufEnvKey = "ESPRESSO_DEBUG_GGUF_MODEL"
    private static let weightDirEnvKey = "ESPRESSO_DEBUG_WEIGHT_DIR"
    private static let tensorsEnvKey = "ESPRESSO_DEBUG_COMPARE_TENSORS"
    private static let outputPathEnvKey = "ESPRESSO_DEBUG_COMPARE_OUTPUT_PATH"
    private static let diffThreshold: Float = 1e-6
    private static let maxDiffSamples = 8

    private struct DiffSample: Codable {
        let index: Int
        let rawValue: Float
        let sidecarValue: Float
        let absDiff: Float
    }

    private struct ComparisonStats: Codable {
        let count: Int
        let maxAbsDiff: Float
        let meanAbsDiff: Float
        let cosineSimilarity: Float
        let firstDiffSamples: [DiffSample]
    }

    private struct TensorReport: Codable {
        let ggufName: String
        let shape: [Int]
        let rawDataType: String
        let sidecarPath: String
        let direct: ComparisonStats
        let variants: [String: ComparisonStats]
        let bestVariant: String
    }

    @Test("Debug compare raw GGUF tensors against artifact float32 sidecars")
    func compareRawGGUFTensorsAgainstArtifactSidecars() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }

        let ggufPath = try #require(
            ProcessInfo.processInfo.environment[Self.ggufEnvKey],
            "Missing \(Self.ggufEnvKey)"
        )
        let weightDir = try #require(
            ProcessInfo.processInfo.environment[Self.weightDirEnvKey],
            "Missing \(Self.weightDirEnvKey)"
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EspressoError.metalDeviceUnavailable
        }

        let ggufURL = URL(fileURLWithPath: ggufPath)
        let weightRoot = URL(fileURLWithPath: weightDir, isDirectory: true)
        let loader = try GGUFLoader(url: ggufURL)
        let weightMap = try await loader.load(from: ggufURL)
        let espressoConfig = try EspressoModelConfig(from: loader.modelConfig)
        let layout = WeightConverter.LlamaProjectionLayout(
            qHeadCount: espressoConfig.headCount,
            kvHeadCount: espressoConfig.kvHeadCount,
            headDim: espressoConfig.headDim
        )
        let tensorNames: [String]
        if let rawTensorList = ProcessInfo.processInfo.environment[Self.tensorsEnvKey],
           !rawTensorList.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            tensorNames = rawTensorList
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        } else {
            tensorNames = weightMap.tensorNames.filter {
                EspressoTensorNameMapper.espressoPath(
                    for: $0,
                    architecture: espressoConfig.architectureName
                ) != nil
            }
        }
        #expect(!tensorNames.isEmpty)

        var reports: [TensorReport] = []
        reports.reserveCapacity(tensorNames.count)

        for ggufName in tensorNames {
            let tensor = try #require(weightMap[ggufName], "Missing raw GGUF tensor \(ggufName)")
            let relativePath = try #require(
                EspressoTensorNameMapper.espressoPath(
                    for: ggufName,
                    architecture: espressoConfig.architectureName
                ),
                "Unmapped tensor \(ggufName)"
            )
            let sidecarURL = weightRoot.appendingPathComponent(
                WeightConverter.exactFloat32SidecarRelativePath(for: relativePath)
            )
            #expect(
                FileManager.default.fileExists(atPath: sidecarURL.path),
                "Missing sidecar \(sidecarURL.path)"
            )

            let rawFloats = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
            let sidecarFloats = try Self.readFloat32Sidecar(at: sidecarURL)
            #expect(rawFloats.count == sidecarFloats.count, "Count mismatch for \(ggufName)")

            let direct = try Self.compare(raw: rawFloats, sidecar: sidecarFloats)
            var variants: [String: ComparisonStats] = [:]
            variants.reserveCapacity(6)

            for (label, candidate) in Self.candidateVariants(
                ggufName: ggufName,
                rawFloats: rawFloats,
                shape: tensor.shape,
                layout: layout
            ) {
                variants[label] = try Self.compare(raw: candidate, sidecar: sidecarFloats)
            }

            let bestVariant = ([("raw", direct)] + variants.map { ($0.key, $0.value) })
                .min { lhs, rhs in
                    if lhs.1.meanAbsDiff == rhs.1.meanAbsDiff {
                        return lhs.0 < rhs.0
                    }
                    return lhs.1.meanAbsDiff < rhs.1.meanAbsDiff
                }!

            reports.append(
                TensorReport(
                    ggufName: ggufName,
                    shape: tensor.shape,
                    rawDataType: String(describing: tensor.dataType),
                    sidecarPath: sidecarURL.path,
                    direct: direct,
                    variants: variants,
                    bestVariant: bestVariant.0
                )
            )
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(reports)
        if let outputPath = ProcessInfo.processInfo.environment[Self.outputPathEnvKey],
           !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        }
        print(String(decoding: data, as: UTF8.self))
    }

    private static func readFloat32Sidecar(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let scalarSize = MemoryLayout<UInt32>.stride
        precondition(data.count.isMultiple(of: scalarSize))
        return data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: scalarSize).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }

    private static func compare(raw: [Float], sidecar: [Float]) throws -> ComparisonStats {
        #expect(raw.count == sidecar.count)

        var maxAbs: Float = 0
        var sumAbs: Double = 0
        var dot: Double = 0
        var lhsNorm: Double = 0
        var rhsNorm: Double = 0
        var diffSamples: [DiffSample] = []
        diffSamples.reserveCapacity(Self.maxDiffSamples)

        for index in raw.indices {
            let lhs = raw[index]
            let rhs = sidecar[index]
            let absDiff = abs(lhs - rhs)
            maxAbs = max(maxAbs, absDiff)
            sumAbs += Double(absDiff)
            dot += Double(lhs) * Double(rhs)
            lhsNorm += Double(lhs) * Double(lhs)
            rhsNorm += Double(rhs) * Double(rhs)

            if absDiff > Self.diffThreshold && diffSamples.count < Self.maxDiffSamples {
                diffSamples.append(
                    DiffSample(index: index, rawValue: lhs, sidecarValue: rhs, absDiff: absDiff)
                )
            }
        }

        let denom = sqrt(lhsNorm) * sqrt(rhsNorm)
        let cosine = denom > 0 ? Float(dot / denom) : 1
        return ComparisonStats(
            count: raw.count,
            maxAbsDiff: maxAbs,
            meanAbsDiff: Float(sumAbs / Double(raw.count)),
            cosineSimilarity: cosine,
            firstDiffSamples: diffSamples
        )
    }

    private static func candidateVariants(
        ggufName: String,
        rawFloats: [Float],
        shape: [Int],
        layout: WeightConverter.LlamaProjectionLayout
    ) -> [(String, [Float])] {
        guard shape.count == 2 else {
            return []
        }

        let rows = shape[0]
        let cols = shape[1]
        guard rows * cols == rawFloats.count else {
            return []
        }

        var variants: [(String, [Float])] = []
        variants.reserveCapacity(6)
        variants.append(("transpose", transpose(rawFloats, rows: rows, cols: cols)))

        let suffix = ggufName.split(separator: ".").dropFirst(2).joined(separator: ".")
        let headCount: Int?
        switch suffix {
        case "attn_q.weight":
            headCount = layout.qHeadCount
        case "attn_k.weight", "attn_v.weight":
            headCount = layout.kvHeadCount
        default:
            headCount = nil
        }

        if let headCount, rows == headCount * layout.headDim {
            variants.append((
                "inverseInterleave",
                WeightPermutation.inverseInterleaving(
                    weights: rawFloats,
                    numHeads: headCount,
                    headDim: layout.headDim
                )
            ))
            variants.append((
                "forwardInterleave",
                WeightPermutation.forwardInterleaving(
                    weights: rawFloats,
                    numHeads: headCount,
                    headDim: layout.headDim
                )
            ))
            variants.append((
                "dimMajorToHeadMajor",
                WeightPermutation.dimMajorToHeadMajor(
                    weights: rawFloats,
                    numHeads: headCount,
                    headDim: layout.headDim
                )
            ))
        }

        return variants
    }

    private static func transpose(_ weights: [Float], rows: Int, cols: Int) -> [Float] {
        var result = [Float](repeating: 0, count: weights.count)
        for row in 0..<rows {
            for col in 0..<cols {
                result[col * rows + row] = weights[row * cols + col]
            }
        }
        return result
    }
}
