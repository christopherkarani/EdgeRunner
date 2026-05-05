import Testing
import Foundation
import Metal
@testable import EdgeRunner
@testable import EdgeRunnerIO
@testable import EdgeRunnerCore
@testable import EdgeRunnerMetal

@Suite("Bonsai 1.7B Benchmark")
struct BonsaiBenchmark {
    static let modelPath: String = {
        if let override = ProcessInfo.processInfo.environment["EDGERUNNER_BONSAI_MODEL_PATH"], !override.isEmpty {
            return (override as NSString).expandingTildeInPath
        }
        return (NSHomeDirectory() as NSString).appendingPathComponent("edgerunner-models/Bonsai-1.7B.gguf")
    }()

    @Test("Bonsai 1.7B Q1_0_g128 coherence check")
    func bonsaiCoherenceCheck() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        // Load weight map to check tensor details
        let loader = try GGUFLoader(url: URL(fileURLWithPath: Self.modelPath))
        let weightMap = try await loader.load(from: URL(fileURLWithPath: Self.modelPath))

        // Check embedding table
        let embdName = weightMap.tensorNames.first { $0.contains("token_embd") } ?? ""
        if let embdStorage = weightMap[embdName] {
            print("=== Embedding table ===")
            print("  name=\(embdName), shape=\(embdStorage.shape), dtype=\(embdStorage.dataType)")
            print("  elementCount=\(embdStorage.elementCount)")
            print("  byteCount=\(embdStorage.byteCount)")

            // Check BOS token (151643) embedding
            let dim = 2048
            let blocksPerRow = dim / 128
            let bytesPerRow = blocksPerRow * 18
            let bosOffset = 151643 * bytesPerRow
            let basePtr = embdStorage.buffer.contents() + embdStorage.byteOffset
            let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: embdStorage.byteCount)

            if bosOffset + 18 <= embdStorage.byteCount {
                let scaleBits = UInt16(rawPtr[bosOffset]) | (UInt16(rawPtr[bosOffset + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                let bits = rawPtr[bosOffset + 2]
                print("  BOS token embedding: scale=\(String(format: "%.4f", scale)), bits=0x\(String(bits, radix: 16))")
            }

            // Also check token 1 embedding
            let token1Offset = 1 * bytesPerRow
            if token1Offset + 18 <= embdStorage.byteCount {
                let scaleBits = UInt16(rawPtr[token1Offset]) | (UInt16(rawPtr[token1Offset + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                let bits = rawPtr[token1Offset + 2]
                print("  Token 1 embedding: scale=\(String(format: "%.4f", scale)), bits=0x\(String(bits, radix: 16))")
            }
        }

        // Load model and test
        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        print("\n=== Test: Seed [1] ===")
        var seedIDs = [1]
        for i in 0..<8 {
            let result = try await model.greedyToken(for: seedIDs)
            seedIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))', hasNaN=\(result.hasNonFinite)")
            if result.token == 151645 || result.token == 2 || result.token == 0 { break }
        }

        print("\n=== Test: 'Hello' ===")
        var helloIDs = model.tokenize("Hello")
        print("  Prompt IDs: \(helloIDs)")
        for i in 0..<4 {
            let result = try await model.greedyToken(for: helloIDs)
            helloIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
        }

        print("\n=== Test: realistic prompt ===")
        var promptIDs = model.tokenize("The capital of France is")
        print("  Prompt IDs: \(promptIDs)")
        for i in 0..<6 {
            let result = try await model.greedyToken(for: promptIDs)
            promptIDs.append(result.token)
            print("  Token \(i): ID=\(result.token), text='\(model.detokenize([result.token]))'")
            if result.token == model.eosTokenID { break }
        }
    }

    @Test("Bonsai 1.7B Q1_0_g128 end-to-end benchmark")
    func bonsaiEndToEndBenchmark() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let prompt = "Explain quantum computing in simple terms:"
        let tokenIDs = model.tokenize(prompt)
        let promptTokenIDs = tokenIDs

        // Warmup
        var warmupTokens = Array(tokenIDs)
        for _ in 0..<4 {
            let result = try await model.greedyToken(for: warmupTokens)
            warmupTokens.append(result.token)
        }
        model.resetGenerationState(keepDecodeWarmup: true)

        var timings: [Double] = []
        for run in 0..<5 {
            var tokenIDs = Array(promptTokenIDs)
            let clock = ContinuousClock()
            let startTime = clock.now
            // Prefill via async path (initializes KV cache + decode warmup)
            let firstResult = try await model.greedyToken(for: tokenIDs)
            let prefillDuration = startTime.duration(to: clock.now)
            let ttftMs = (Double(prefillDuration.components.seconds) + Double(prefillDuration.components.attoseconds) * 1e-18) * 1000.0
            tokenIDs.append(firstResult.token)
            var generatedCount = 1

            // Decode via sync path (eliminates async scheduling overhead)
            let decodeStart = clock.now
            for _ in 1..<128 {
                let result = try model.greedyTokenSync(for: tokenIDs)
                tokenIDs.append(result.token)
                generatedCount += 1
                if result.token == 151645 || result.token == 2 || result.token == 0 { break }
            }

            let decodeDuration = decodeStart.duration(to: clock.now)
            let decodeSeconds = Double(decodeDuration.components.seconds) + Double(decodeDuration.components.attoseconds) * 1e-18
            let decodeTokens = generatedCount - 1
            let decodeTokPerSec = decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0
            timings.append(decodeTokPerSec)
            print("Run \(run): decode=\(String(format: "%.1f", decodeTokPerSec)) tok/s  tokens=\(generatedCount)  ttft=\(String(format: "%.1f", ttftMs))ms")
            model.resetGenerationState(keepDecodeWarmup: true)
        }

        let sortedTimings = timings.sorted()
        let median = sortedTimings[sortedTimings.count / 2]
        print("\nMedian decode: \(String(format: "%.1f", median)) tok/s")
        #expect(Bool(true), "Bonsai benchmark completed: \(String(format: "%.1f", median)) tok/s")
    }

    @Test("Bonsai 1.7B LM head profile")
    func bonsaiLMHeadProfile() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        var warmupTokens = model.tokenize("Explain quantum computing in simple terms:")
        let warmup = try await model.greedyToken(for: warmupTokens)
        warmupTokens.append(warmup.token)

        let lmHeadMs = try await model.measureLMHeadLatency(samples: 5)
        print("BONSAI_PROFILE: lm_head_ms \(String(format: "%.3f", lmHeadMs))")
        #expect(Bool(true), "Bonsai LM head profile completed")
    }

    @Test("Bonsai 1.7B Q1 projection profile")
    func bonsaiQ1ProjectionProfile() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let model = try await BonsaiLanguageModel.load(
            from: URL(fileURLWithPath: Self.modelPath),
            configuration: ModelConfiguration(contextWindowSize: 2048)
        )

        let warmupPrompt = model.tokenize("Explain quantum computing in simple terms:")
        _ = try await model.greedyToken(for: warmupPrompt)

        let results = try await model.measureQ1ProjectionLatencies(samples: 5)
        for (name, milliseconds) in results {
            print("BONSAI_PROFILE: q1_\(name)_ms \(String(format: "%.3f", milliseconds))")
        }
        #expect(!results.isEmpty, "Bonsai Q1 projection profile completed")
    }

    @Test("Q1 GEMV v1 vs v2 kernel comparison")
    func q1GemvKernelComparison() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            #expect(Bool(false), "Bonsai model not found at \(Self.modelPath)")
            return
        }

        let loader = try GGUFLoader(url: URL(fileURLWithPath: Self.modelPath))
        let weightMap = try await loader.load(from: URL(fileURLWithPath: Self.modelPath))

        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let q1Kernel = try DequantQ1_0_g128Kernel(device: device)

        // Pick a representative Q1 weight tensor (wq: 2048×2048)
        let wqName = "blk.0.attn_q.weight"
        let altName = "layers.0.attention.wq.weight"
        let tensorName = weightMap.tensorNames.first { $0.contains("attn_q") || $0.contains("wq") } ?? wqName
        guard let storage = weightMap[tensorName] else {
            print("SKIP: Could not find Q1 weight tensor (tried \(wqName), \(altName))")
            return
        }

        let elementCount = storage.elementCount
        let rows = 2048
        let cols = elementCount / rows
        let nb = cols / 128
        let q1ByteCount = nb * rows * 18

        guard let q1Buf = device.makeBuffer(
            bytesNoCopy: storage.buffer.contents() + storage.byteOffset,
            length: q1ByteCount,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            print("SKIP: Failed to create Q1 buffer")
            return
        }

        // Create input x and output buffers
        let inputBuf = device.makeBuffer(length: cols * 4, options: .storageModeShared)!
        let outputV1 = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
        let outputV2 = device.makeBuffer(length: rows * 4, options: .storageModeShared)!

        // Fill x with random values
        let xPtr = inputBuf.contents().bindMemory(to: Float.self, capacity: cols)
        for i in 0..<cols { xPtr[i] = Float.random(in: -1...1) }

        struct GEMVParams {
            var rows: UInt32
            var cols: UInt32
            var blocksPerRow: UInt32
        }

        let warmup = 3
        let runs = 10
        var params = GEMVParams(rows: UInt32(rows), cols: UInt32(cols), blocksPerRow: UInt32(nb))

        // Warmup v1
        for _ in 0..<warmup {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(q1Kernel.gemvPSO)
            enc.setBuffer(q1Buf, offset: 0, index: 0)
            enc.setBuffer(inputBuf, offset: 0, index: 1)
            enc.setBuffer(outputV1, offset: 0, index: 2)
            enc.setBytes(&params, length: 12, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            await cmd.completed()
        }

        // Benchmark v1
        var v1Times: [Double] = []
        for _ in 0..<runs {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(q1Kernel.gemvPSO)
            enc.setBuffer(q1Buf, offset: 0, index: 0)
            enc.setBuffer(inputBuf, offset: 0, index: 1)
            enc.setBuffer(outputV1, offset: 0, index: 2)
            enc.setBytes(&params, length: 12, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            await cmd.completed()
            let gpuTime = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0
            v1Times.append(gpuTime)
        }

        // Warmup v2
        for _ in 0..<warmup {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(q1Kernel.gemvV2PSO)
            enc.setBuffer(q1Buf, offset: 0, index: 0)
            enc.setBuffer(inputBuf, offset: 0, index: 1)
            enc.setBuffer(outputV2, offset: 0, index: 2)
            enc.setBytes(&params, length: 12, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            await cmd.completed()
        }

        // Benchmark v2
        var v2Times: [Double] = []
        for _ in 0..<runs {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(q1Kernel.gemvV2PSO)
            enc.setBuffer(q1Buf, offset: 0, index: 0)
            enc.setBuffer(inputBuf, offset: 0, index: 1)
            enc.setBuffer(outputV2, offset: 0, index: 2)
            enc.setBytes(&params, length: 12, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 1) / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            await cmd.completed()
            let gpuTime = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0
            v2Times.append(gpuTime)
        }

        // Compare outputs
        let v1Ptr = outputV1.contents().bindMemory(to: Float.self, capacity: rows)
        let v2Ptr = outputV2.contents().bindMemory(to: Float.self, capacity: rows)
        var maxDiff: Float = 0
        for i in 0..<rows {
            maxDiff = max(maxDiff, abs(v1Ptr[i] - v2Ptr[i]))
        }

        let v1Median = v1Times.sorted()[runs / 2]
        let v2Median = v2Times.sorted()[runs / 2]
        let dataBytes = Double(nb * rows * 18)
        let v1BW = dataBytes / (v1Median / 1000.0) / 1e9
        let v2BW = dataBytes / (v2Median / 1000.0) / 1e9

        print("Q1 GEMV [\(rows)×\(cols)]  v1: \(String(format: "%.3f", v1Median))ms (\(String(format: "%.1f", v1BW)) GB/s)  v2: \(String(format: "%.3f", v2Median))ms (\(String(format: "%.1f", v2BW)) GB/s)  speedup: \(String(format: "%.2f", v1Median/v2Median))x  maxDiff: \(maxDiff)")
        #expect(maxDiff < 1.0, "v1 vs v2 output divergence too large: \(maxDiff)")
    }
}
