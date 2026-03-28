import Foundation
@testable import EdgeRunner

struct BenchmarkContract: Decodable {
    struct Model: Decodable {
        let name: String
        let hfRepo: String
        let hfFile: String
        let localPath: String
        let downloadURL: String
        let sizeBytes: Int64
        let sha256: String

        private enum CodingKeys: String, CodingKey {
            case name
            case hfRepo = "hf_repo"
            case hfFile = "hf_file"
            case localPath = "local_path"
            case downloadURL = "download_url"
            case sizeBytes = "size_bytes"
            case sha256
        }
    }

    struct Publishable: Decodable {
        let tokenCount: Int
        let runCount: Int
        let contextWindow: Int
        let expectedGreedyPrefix: [Int]
        let expectedTokenHash: String

        private enum CodingKeys: String, CodingKey {
            case tokenCount = "token_count"
            case runCount = "run_count"
            case contextWindow = "context_window"
            case expectedGreedyPrefix = "expected_greedy_prefix"
            case expectedTokenHash = "expected_token_hash"
        }
    }

    struct Smoke: Decodable {
        let generateCount: Int
        let expectedGreedyPrefix: [Int]

        private enum CodingKeys: String, CodingKey {
            case generateCount = "generate_count"
            case expectedGreedyPrefix = "expected_greedy_prefix"
        }
    }

    let model: Model
    let publishable: Publishable
    let smoke: Smoke

    static let pinned: BenchmarkContract = {
        do {
            return try load()
        } catch {
            fatalError("Failed to load benchmark contract: \(error)")
        }
    }()

    static func load() throws -> BenchmarkContract {
        let fileManager = FileManager.default
        let overridePath = ProcessInfo.processInfo.environment["EDGERUNNER_BENCHMARK_CONTRACT"]
        let projectDir = ProcessInfo.processInfo.environment["PROJECT_DIR"] ?? fileManager.currentDirectoryPath

        let contractURL: URL
        if let overridePath, !overridePath.isEmpty {
            if overridePath.hasPrefix("/") {
                contractURL = URL(fileURLWithPath: overridePath)
            } else {
                contractURL = URL(fileURLWithPath: projectDir).appendingPathComponent(overridePath)
            }
        } else {
            contractURL = URL(fileURLWithPath: projectDir)
                .appendingPathComponent("benchmarks/pinned_qwen3_0.6b_q8_0.json")
        }

        let data = try Data(contentsOf: contractURL)
        return try JSONDecoder().decode(BenchmarkContract.self, from: data)
    }
}

extension ModelConfiguration {
    static func pinnedBenchmarkConfiguration(contextWindow: Int) -> ModelConfiguration {
        pinnedBenchmarkConfiguration(
            contextWindow: contextWindow,
            decodeOverrides: LlamaDecodeOverrides(disableMegaKernel: false),
            prefillOverrides: nil
        )
    }

    static func pinnedBenchmarkConfiguration(
        contextWindow: Int,
        disableMegaKernel: Bool
    ) -> ModelConfiguration {
        pinnedBenchmarkConfiguration(
            contextWindow: contextWindow,
            decodeOverrides: LlamaDecodeOverrides(disableMegaKernel: disableMegaKernel),
            prefillOverrides: nil
        )
    }

    static func pinnedBenchmarkConfiguration(
        contextWindow: Int,
        decodeOverrides: LlamaDecodeOverrides?,
        prefillOverrides: LlamaPrefillOverrides?
    ) -> ModelConfiguration {
        var configuration = ModelConfiguration(contextWindowSize: contextWindow)
        // Benchmarks run on the canonical fast decode path by default. Tests can still
        // force the safe path explicitly when they need parity or fault-isolation probes.
        configuration.llamaDecodeOverrides = decodeOverrides
        configuration.llamaPrefillOverrides = prefillOverrides
        return configuration
    }
}
