import Foundation
import Metal
import EdgeRunnerCore
import EdgeRunnerIO
import EdgeRunnerMetal

/// Public Gemma 4 E4B runtime entry point.
///
/// This type owns Gemma 4-specific loading, tokenizer, and chat-template routing.
/// The forward pass is intentionally kept out of the Llama runtime because Gemma 4
/// uses PLE, GeGLU, dual head dimensions, KV sharing, sliding/global attention,
/// partial RoPE, and final logit softcapping.
public struct Gemma4LanguageModel: LogitsModel, @unchecked Sendable {
    public static let modelIdentifier = "gemma4"

    private struct PreludeState {
        let hidden: [Float]
        let hiddenBuffer: MTLBuffer?
        let perLayerInputs: [Float]?
        let perLayerInputsBuffer: MTLBuffer?
        let perLayerInputCount: Int
        let tokenCount: Int

        func perLayerInputSlice(layer: Int, perLayerDim: Int) throws -> [Float] {
            try perLayerInputSlice(
                tokenOffset: 0,
                layer: layer,
                perLayerDim: perLayerDim,
                numLayers: layerCount(perLayerDim: perLayerDim)
            )
        }

        func perLayerInputSlice(
            tokenOffset: Int,
            layer: Int,
            perLayerDim: Int,
            numLayers: Int
        ) throws -> [Float] {
            let start = Gemma4PrefillChunkPlan.pleInputElementOffset(
                tokenOffset: tokenOffset,
                layer: layer,
                numLayers: numLayers,
                perLayerDim: perLayerDim
            )
            let end = start + perLayerDim
            guard start >= 0, end <= perLayerInputCount else {
                throw GenerationError.modelLoadFailed(reason: "Gemma PLE input slice is out of bounds")
            }
            if let perLayerInputs {
                return Array(perLayerInputs[start..<end])
            }
            guard let perLayerInputsBuffer else {
                throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
            }
            let pointer = perLayerInputsBuffer.contents()
                .advanced(by: start * MemoryLayout<Float>.stride)
                .bindMemory(to: Float.self, capacity: perLayerDim)
            return Array(UnsafeBufferPointer(start: pointer, count: perLayerDim))
        }

        func perLayerInputByteOffset(layer: Int, perLayerDim: Int) throws -> Int {
            try perLayerInputByteOffset(
                tokenOffset: 0,
                layer: layer,
                perLayerDim: perLayerDim,
                numLayers: layerCount(perLayerDim: perLayerDim)
            )
        }

        func perLayerInputByteOffset(
            tokenOffset: Int,
            layer: Int,
            perLayerDim: Int,
            numLayers: Int
        ) throws -> Int {
            let start = Gemma4PrefillChunkPlan.pleInputElementOffset(
                tokenOffset: tokenOffset,
                layer: layer,
                numLayers: numLayers,
                perLayerDim: perLayerDim
            )
            let end = start + perLayerDim
            guard start >= 0, end <= perLayerInputCount else {
                throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer slice is out of bounds")
            }
            guard perLayerInputsBuffer != nil else {
                throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
            }
            return start * MemoryLayout<Float>.stride
        }

        private func layerCount(perLayerDim: Int) -> Int {
            guard tokenCount > 0, perLayerDim > 0 else { return 0 }
            return perLayerInputCount / (tokenCount * perLayerDim)
        }
    }

    private struct ProjectionRequest {
        let tensor: TensorStorage
        let rows: Int
        let cols: Int
    }

    private struct ProjectionResource {
        let request: ProjectionRequest
        let weightBuffer: MTLBuffer
    }

    private struct LayerRuntimeResources {
        let inputNormWeightBuffer: MTLBuffer
        let attnQ: ProjectionResource
        let attnK: ProjectionResource?
        let attnV: ProjectionResource?
        let attnO: ProjectionResource
        let attnQNormWeightBuffer: MTLBuffer
        let attnKNormWeightBuffer: MTLBuffer?
        let unitHeadWeightBuffer: MTLBuffer
        let postAttentionNormWeightBuffer: MTLBuffer
        let ffnNormWeightBuffer: MTLBuffer
        let ffnGate: ProjectionResource
        let ffnUp: ProjectionResource
        let ffnDown: ProjectionResource
        let postFFNNormWeightBuffer: MTLBuffer
        let pleGate: ProjectionResource
        let pleProjection: ProjectionResource
        let plePostNormWeightBuffer: MTLBuffer
        let layerOutputScale: Float
    }

    private struct LMHeadResources {
        let outputNormWeightBuffer: MTLBuffer
        let tokenEmbedding: ProjectionResource
    }

    private final class RuntimeProfiler: @unchecked Sendable {
        private let enabled: Bool
        private let lock = NSLock()
        private var totals: [String: Double] = [:]
        private var counts: [String: Int] = [:]
        private var tokenCount = 0

        init(enabled: Bool) {
            self.enabled = enabled
        }

        var isEnabled: Bool { enabled }

        func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
            guard enabled else { return try body() }
            let start = ProcessInfo.processInfo.systemUptime
            do {
                let value = try body()
                record(name, duration: ProcessInfo.processInfo.systemUptime - start)
                return value
            } catch {
                record(name, duration: ProcessInfo.processInfo.systemUptime - start)
                throw error
            }
        }

        func measureAsync<T>(_ name: String, _ body: () async throws -> T) async throws -> T {
            guard enabled else { return try await body() }
            let start = ProcessInfo.processInfo.systemUptime
            do {
                let value = try await body()
                record(name, duration: ProcessInfo.processInfo.systemUptime - start)
                return value
            } catch {
                record(name, duration: ProcessInfo.processInfo.systemUptime - start)
                throw error
            }
        }

        func recordDuration(_ name: String, duration: Double) {
            guard enabled, duration.isFinite, duration >= 0 else { return }
            record(name, duration: duration)
        }

        func markTokenComplete() {
            guard enabled else { return }
            lock.lock()
            tokenCount += 1
            let shouldPrint = tokenCount == 1 || tokenCount % 8 == 0
            let snapshotTotals = totals
            let snapshotCounts = counts
            let snapshotTokens = tokenCount
            lock.unlock()

            guard shouldPrint else { return }
            let total = snapshotTotals.values.reduce(0, +)
            let rowLimit = ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_PHASES"] == "1"
                ? 40
                : (ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_STACK"] == "1" ? 24 : 12)
            let rows = snapshotTotals
                .sorted { $0.value > $1.value }
                .prefix(rowLimit)
                .map { key, seconds in
                    let ms = seconds * 1000
                    let pct = total > 0 ? (seconds / total) * 100 : 0
                    let count = snapshotCounts[key] ?? 0
                    return "\(key)=\(String(format: "%.1f", ms))ms/\(count)x/\(String(format: "%.1f", pct))%"
                }
                .joined(separator: " ")
            print("GEMMA4_PROFILE tokens=\(snapshotTokens) \(rows)")
        }

        private func record(_ name: String, duration: Double) {
            lock.lock()
            totals[name, default: 0] += duration
            counts[name, default: 0] += 1
            lock.unlock()
        }
    }

    private final class RuntimeCache: @unchecked Sendable {
        private let lock = NSLock()
        private var rawTensorBuffers: [String: MTLBuffer] = [:]
        private var floatVectors: [String: [Float]] = [:]
        private var floatBuffers: [String: MTLBuffer] = [:]
        private var preludeStates: [Int: PreludeState] = [:]
        private var preludeLRU: [Int] = []
        private let maxPreludeStates = 128

        func rawTensorBuffer(
            storage: TensorStorage,
            requiredBytes: Int,
            device: MTLDevice
        ) throws -> MTLBuffer {
            let key = "\(storage.name)|\(storage.byteOffset)|\(requiredBytes)|\(storage.buffer.length)"
            lock.lock()
            if let cached = rawTensorBuffers[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()

            guard storage.byteOffset >= 0,
                  storage.byteOffset + requiredBytes <= storage.buffer.length else {
                throw GenerationError.modelLoadFailed(reason: "\(storage.name) buffer is smaller than expected")
            }
            guard let buffer = device.makeBuffer(
                bytesNoCopy: storage.buffer.contents() + storage.byteOffset,
                length: requiredBytes,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                throw GenerationError.modelLoadFailed(reason: "Failed to create raw buffer for \(storage.name)")
            }

            lock.lock()
            rawTensorBuffers[key] = buffer
            lock.unlock()
            return buffer
        }

        func floatVector(key: String, load: () throws -> [Float]) throws -> [Float] {
            lock.lock()
            if let cached = floatVectors[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()

            let values = try load()
            lock.lock()
            floatVectors[key] = values
            lock.unlock()
            return values
        }

        func floatBuffer(key: String, load: () throws -> MTLBuffer) throws -> MTLBuffer {
            lock.lock()
            if let cached = floatBuffers[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()

            let buffer = try load()
            lock.lock()
            floatBuffers[key] = buffer
            lock.unlock()
            return buffer
        }

        func preludeState(tokenID: Int) -> PreludeState? {
            lock.lock()
            defer { lock.unlock() }
            guard let state = preludeStates[tokenID] else {
                return nil
            }
            preludeLRU.removeAll { $0 == tokenID }
            preludeLRU.append(tokenID)
            return state
        }

        func storePreludeState(_ state: PreludeState, tokenID: Int) {
            lock.lock()
            preludeStates[tokenID] = state
            preludeLRU.removeAll { $0 == tokenID }
            preludeLRU.append(tokenID)
            while preludeLRU.count > maxPreludeStates {
                let evicted = preludeLRU.removeFirst()
                preludeStates.removeValue(forKey: evicted)
            }
            lock.unlock()
        }
    }

    private final class RuntimeDecodeState: @unchecked Sendable {
        private let lock = NSLock()
        private var state = Gemma4DecodeState()

        func prepare(tokenIDs: [Int]) -> Gemma4DecodeMode {
            lock.lock()
            defer { lock.unlock() }
            return state.prepare(tokenIDs: tokenIDs)
        }

        func markProcessed(tokenIDs: [Int]) {
            lock.lock()
            state.markProcessed(tokenIDs: tokenIDs)
            lock.unlock()
        }

        func reset() {
            lock.lock()
            state.reset()
            lock.unlock()
        }
    }

    private actor AsyncGate {
        private var isLocked = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func acquire() async {
            if !isLocked {
                isLocked = true
                return
            }
            await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }

        func release() {
            if waiters.isEmpty {
                isLocked = false
            } else {
                waiters.removeFirst().resume()
            }
        }
    }

    private let config: Gemma4ModelConfig
    private let weights: Gemma4Weights
    private let tokenizer: (any Tokenizer)?
    private let maxSeqLen: Int
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let gemvKernel: GEMVKernel
    private let ropeKernel: RoPEKernel
    private let gqaKernel: GQAKernel
    private let pleGatherKernel: PLEGatherKernel
    private let pleInputsKernel: PLEInputsKernel
    private let gegluKernel: GeGLUKernel
    private let pleGateKernel: PLEGateKernel
    private let pleSideChannelKernel: PLESideChannelKernel
    private let gemmaDecodeKernels: Gemma4DecodeKernels
    private let gpuKVCache: Gemma4GPUKVCache
    private let scratch: Gemma4Scratch
    private let layerRuntimePlans: [Gemma4LayerRuntimePlan]
    private let layerRuntimeResources: [LayerRuntimeResources]
    private let lmHeadResources: LMHeadResources
    private let scratchGate = AsyncGate()
    private let useCacheBackedAttention: Bool
    private let validateRuntimeTensors: Bool
    private let traceFastGQA: Bool
    private let runtimeOptions: Gemma4RuntimeOptions
    private let profiler: RuntimeProfiler
    private let runtimeCache = RuntimeCache()
    private let decodeState = RuntimeDecodeState()

    private init(
        config: Gemma4ModelConfig,
        weights: Gemma4Weights,
        tokenizer: (any Tokenizer)?,
        maxSeqLen: Int,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        gemvKernel: GEMVKernel,
        ropeKernel: RoPEKernel,
        gqaKernel: GQAKernel,
        pleGatherKernel: PLEGatherKernel,
        pleInputsKernel: PLEInputsKernel,
        gegluKernel: GeGLUKernel,
        pleGateKernel: PLEGateKernel,
        pleSideChannelKernel: PLESideChannelKernel,
        gemmaDecodeKernels: Gemma4DecodeKernels,
        gpuKVCache: Gemma4GPUKVCache,
        scratch: Gemma4Scratch,
        layerRuntimePlans: [Gemma4LayerRuntimePlan],
        layerRuntimeResources: [LayerRuntimeResources],
        lmHeadResources: LMHeadResources,
        useCacheBackedAttention: Bool,
        validateRuntimeTensors: Bool,
        traceFastGQA: Bool,
        runtimeOptions: Gemma4RuntimeOptions,
        profiler: RuntimeProfiler
    ) {
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
        self.device = device
        self.commandQueue = commandQueue
        self.gemvKernel = gemvKernel
        self.ropeKernel = ropeKernel
        self.gqaKernel = gqaKernel
        self.pleGatherKernel = pleGatherKernel
        self.pleInputsKernel = pleInputsKernel
        self.gegluKernel = gegluKernel
        self.pleGateKernel = pleGateKernel
        self.pleSideChannelKernel = pleSideChannelKernel
        self.gemmaDecodeKernels = gemmaDecodeKernels
        self.gpuKVCache = gpuKVCache
        self.scratch = scratch
        self.layerRuntimePlans = layerRuntimePlans
        self.layerRuntimeResources = layerRuntimeResources
        self.lmHeadResources = lmHeadResources
        self.useCacheBackedAttention = useCacheBackedAttention
        self.validateRuntimeTensors = validateRuntimeTensors
        self.traceFastGQA = traceFastGQA
        self.runtimeOptions = runtimeOptions
        self.profiler = profiler
    }

    public static func load(
        from url: URL,
        configuration: ModelConfiguration
    ) async throws -> Gemma4LanguageModel {
        let loader = try GGUFLoader(url: url)
        guard supports(modelConfig: loader.modelConfig) else {
            throw GenerationError.modelLoadFailed(
                reason: "Requested Gemma 4 backend for non-Gemma 4 model at \(url.path)"
            )
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GenerationError.modelLoadFailed(reason: "Metal device unavailable")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw GenerationError.modelLoadFailed(reason: "Metal command queue unavailable")
        }

        let config = try Gemma4ModelConfig(modelConfigMetadata: loader.modelConfig.metadata)
        let weightMap = try await loader.load(from: url)
        let weights = try Gemma4Weights(weightMap: weightMap, config: config, device: device)
        let gemvKernel = try GEMVKernel(device: device)
        let ropeKernel = try RoPEKernel(device: device)
        let gqaKernel = try GQAKernel(device: device)
        let pleGatherKernel = try PLEGatherKernel(device: device)
        let pleInputsKernel = try PLEInputsKernel(device: device)
        let gegluKernel = try GeGLUKernel(device: device)
        let pleGateKernel = try PLEGateKernel(device: device)
        let pleSideChannelKernel = try PLESideChannelKernel(device: device)
        let gemmaDecodeKernels = try Gemma4DecodeKernels(device: device)
        let gpuKVCache = try Gemma4GPUKVCache(
            device: device,
            config: config,
            maxSeqLen: min(configuration.contextWindowSize, config.maxPositionEmbeddings)
        )
        let scratch = try Gemma4Scratch(device: device, config: config)
        let layerRuntimePlans = try Self.makeLayerRuntimePlans(config: config, weights: weights)
        let layerRuntimeResources = try Self.makeLayerRuntimeResources(
            config: config,
            weights: weights,
            plans: layerRuntimePlans,
            device: device
        )
        let lmHeadResources = try Self.makeLMHeadResources(
            config: config,
            weights: weights,
            device: device
        )
        let useCacheBackedAttention = ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_SHORTCUT_ATTENTION"] != "1"
        let validateRuntimeTensors = ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_VALIDATE_FINITE"] == "1"
        let traceFastGQA = ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_TRACE_FAST_GQA"] == "1"
        let runtimeOptions = Gemma4RuntimeOptions(environment: ProcessInfo.processInfo.environment)
        let profiler = RuntimeProfiler(
            enabled: ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE"] == "1"
        )

        let loadedTokenizer: (any Tokenizer)?
        do {
            let tokenizerMetadata = try loader.modelConfig.tokenizerMetadata()
            loadedTokenizer = try TokenizerFactory.create(from: tokenizerMetadata)
        } catch {
            loadedTokenizer = nil
        }

        return Gemma4LanguageModel(
            config: config,
            weights: weights,
            tokenizer: loadedTokenizer,
            maxSeqLen: min(configuration.contextWindowSize, config.maxPositionEmbeddings),
            device: device,
            commandQueue: commandQueue,
            gemvKernel: gemvKernel,
            ropeKernel: ropeKernel,
            gqaKernel: gqaKernel,
            pleGatherKernel: pleGatherKernel,
            pleInputsKernel: pleInputsKernel,
            gegluKernel: gegluKernel,
            pleGateKernel: pleGateKernel,
            pleSideChannelKernel: pleSideChannelKernel,
            gemmaDecodeKernels: gemmaDecodeKernels,
            gpuKVCache: gpuKVCache,
            scratch: scratch,
            layerRuntimePlans: layerRuntimePlans,
            layerRuntimeResources: layerRuntimeResources,
            lmHeadResources: lmHeadResources,
            useCacheBackedAttention: useCacheBackedAttention,
            validateRuntimeTensors: validateRuntimeTensors,
            traceFastGQA: traceFastGQA,
            runtimeOptions: runtimeOptions,
            profiler: profiler
        )
    }

    static func supports(modelConfig: ModelConfig) -> Bool {
        modelConfig.architectureName.lowercased() == "gemma4"
    }

    private static func makeLayerRuntimePlans(
        config: Gemma4ModelConfig,
        weights: Gemma4Weights
    ) throws -> [Gemma4LayerRuntimePlan] {
        let globalRotaryFactor: Float?
        if let ropeFreqs = weights.ropeFreqs {
            globalRotaryFactor = try globalRotaryFactorFromRopeFreqs(
                ropeFreqs,
                expectedElementCount: config.globalHeadDim / 2
            )
        } else {
            globalRotaryFactor = nil
        }
        return Gemma4LayerRuntimePlan.makePlans(
            config: config,
            globalRotaryFactor: globalRotaryFactor
        )
    }

    private static func globalRotaryFactorFromRopeFreqs(
        _ storage: TensorStorage,
        expectedElementCount: Int
    ) throws -> Float {
        guard storage.elementCount == expectedElementCount else {
            throw GenerationError.modelLoadFailed(
                reason: "\(storage.name) element count \(storage.elementCount) must equal \(expectedElementCount)"
            )
        }
        let base = storage.buffer.contents() + storage.byteOffset
        let rotatedPairs: Int
        switch storage.dataType {
        case .float32:
            let ptr = base.bindMemory(to: Float.self, capacity: expectedElementCount)
            rotatedPairs = (0..<expectedElementCount).reduce(into: 0) { count, index in
                if ptr[index] < 1.0e20 { count += 1 }
            }
        case .float16:
            let ptr = base.bindMemory(to: Float16.self, capacity: expectedElementCount)
            rotatedPairs = (0..<expectedElementCount).reduce(into: 0) { count, index in
                if Float(ptr[index]) < 1.0e20 { count += 1 }
            }
        case .bfloat16:
            let ptr = base.bindMemory(to: UInt16.self, capacity: expectedElementCount)
            rotatedPairs = (0..<expectedElementCount).reduce(into: 0) { count, index in
                if Float(bitPattern: UInt32(ptr[index]) << 16) < 1.0e20 { count += 1 }
            }
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(storage.name) must be F32/F16/BF16 for Gemma rope freqs, got \(storage.dataType)"
            )
        }
        return Float(rotatedPairs) / Float(expectedElementCount)
    }

    private static func makeLayerRuntimeResources(
        config: Gemma4ModelConfig,
        weights: Gemma4Weights,
        plans: [Gemma4LayerRuntimePlan],
        device: MTLDevice
    ) throws -> [LayerRuntimeResources] {
        try zip(weights.blocks, plans).map { block, plan in
            let hiddenSize = config.hiddenSize
            let perLayerDim = config.perLayerDim
            let inputNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.inputNorm,
                expectedElementCount: hiddenSize,
                device: device
            )
            let attnQ = try makeStaticProjectionResource(
                tensor: block.attnQ,
                rows: plan.qRows,
                cols: hiddenSize,
                device: device
            )
            let attnK = try block.attnK.map {
                try makeStaticProjectionResource(tensor: $0, rows: plan.kvRows, cols: hiddenSize, device: device)
            }
            let attnV = try block.attnV.map {
                try makeStaticProjectionResource(tensor: $0, rows: plan.kvRows, cols: hiddenSize, device: device)
            }
            let attnO = try makeStaticProjectionResource(
                tensor: block.attnO,
                rows: hiddenSize,
                cols: plan.qRows,
                device: device
            )
            let attnQNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.attnQNorm,
                expectedElementCount: plan.headDim,
                device: device
            )
            let attnKNormWeightBuffer = try block.attnK.map { _ in
                try makeStaticFloatBuffer(
                    storage: block.attnKNorm,
                    expectedElementCount: plan.headDim,
                    device: device
                )
            }
            let unitHeadWeightBuffer = try makeStaticFloatBuffer(
                values: [Float](repeating: 1, count: plan.headDim),
                label: "unitHeadWeight.\(plan.headDim)",
                device: device
            )
            let postAttentionNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.postAttentionNorm,
                expectedElementCount: hiddenSize,
                device: device
            )
            let ffnNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.ffnNorm,
                expectedElementCount: hiddenSize,
                device: device
            )
            let ffnGate = try makeStaticProjectionResource(
                tensor: block.ffnGate,
                rows: config.intermediateSize,
                cols: hiddenSize,
                device: device
            )
            let ffnUp = try makeStaticProjectionResource(
                tensor: block.ffnUp,
                rows: config.intermediateSize,
                cols: hiddenSize,
                device: device
            )
            let ffnDown = try makeStaticProjectionResource(
                tensor: block.ffnDown,
                rows: hiddenSize,
                cols: config.intermediateSize,
                device: device
            )
            let postFFNNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.postFFNNorm,
                expectedElementCount: hiddenSize,
                device: device
            )
            let pleGate = try makeStaticProjectionResource(
                tensor: block.perLayerInputGate,
                rows: perLayerDim,
                cols: hiddenSize,
                device: device
            )
            let pleProjection = try makeStaticProjectionResource(
                tensor: block.perLayerProjection,
                rows: hiddenSize,
                cols: perLayerDim,
                device: device
            )
            let plePostNormWeightBuffer = try makeStaticFloatBuffer(
                storage: block.postPerLayerInputNorm,
                expectedElementCount: hiddenSize,
                device: device
            )
            let layerOutputScale = try readStaticFloatVector(
                storage: block.layerOutputScale,
                expectedElementCount: 1
            )[0]

            return LayerRuntimeResources(
                inputNormWeightBuffer: inputNormWeightBuffer,
                attnQ: attnQ,
                attnK: attnK,
                attnV: attnV,
                attnO: attnO,
                attnQNormWeightBuffer: attnQNormWeightBuffer,
                attnKNormWeightBuffer: attnKNormWeightBuffer,
                unitHeadWeightBuffer: unitHeadWeightBuffer,
                postAttentionNormWeightBuffer: postAttentionNormWeightBuffer,
                ffnNormWeightBuffer: ffnNormWeightBuffer,
                ffnGate: ffnGate,
                ffnUp: ffnUp,
                ffnDown: ffnDown,
                postFFNNormWeightBuffer: postFFNNormWeightBuffer,
                pleGate: pleGate,
                pleProjection: pleProjection,
                plePostNormWeightBuffer: plePostNormWeightBuffer,
                layerOutputScale: layerOutputScale
            )
        }
    }

    private static func makeLMHeadResources(
        config: Gemma4ModelConfig,
        weights: Gemma4Weights,
        device: MTLDevice
    ) throws -> LMHeadResources {
        let outputNormWeightBuffer = try makeStaticFloatBuffer(
            storage: weights.outputNorm,
            expectedElementCount: config.hiddenSize,
            device: device
        )
        let tokenEmbedding = try makeStaticProjectionResource(
            tensor: weights.tokenEmbedding,
            rows: config.vocabSize,
            cols: config.hiddenSize,
            device: device
        )
        return LMHeadResources(
            outputNormWeightBuffer: outputNormWeightBuffer,
            tokenEmbedding: tokenEmbedding
        )
    }

    private static func makeStaticProjectionResource(
        tensor: TensorStorage,
        rows: Int,
        cols: Int,
        device: MTLDevice
    ) throws -> ProjectionResource {
        let request = ProjectionRequest(tensor: tensor, rows: rows, cols: cols)
        try validateStaticMatrixShape(tensor: tensor, cols: cols, rows: rows)
        let requiredBytes = try staticProjectionRequiredBytes(request)
        let weightBuffer = try makeStaticRawTensorBuffer(
            storage: tensor,
            requiredBytes: requiredBytes,
            device: device
        )
        return ProjectionResource(request: request, weightBuffer: weightBuffer)
    }

    private static func validateStaticMatrixShape(
        tensor: TensorStorage,
        cols: Int,
        rows: Int
    ) throws {
        guard tensor.shape.count == 2,
              tensor.shape[0] == cols,
              tensor.shape[1] == rows else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensor.name) shape \(tensor.shape) must be [\(cols), \(rows)]"
            )
        }
    }

    private static func staticProjectionRequiredBytes(_ request: ProjectionRequest) throws -> Int {
        switch request.tensor.dataType {
        case .float32:
            return request.tensor.elementCount * MemoryLayout<Float>.stride
        case .bfloat16:
            return request.tensor.elementCount * MemoryLayout<UInt16>.stride
        case .q4_K:
            return try staticPackedByteCount(
                tensorName: request.tensor.name,
                rows: request.rows,
                cols: request.cols,
                weightsPerBlock: 256,
                blockByteCount: 144
            )
        case .q6_K:
            return try staticPackedByteCount(
                tensorName: request.tensor.name,
                rows: request.rows,
                cols: request.cols,
                weightsPerBlock: 256,
                blockByteCount: 210
            )
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(request.tensor.name) projection data type \(request.tensor.dataType) is not supported by Gemma decoder projection"
            )
        }
    }

    private static func staticPackedByteCount(
        tensorName: String,
        rows: Int,
        cols: Int,
        weightsPerBlock: Int,
        blockByteCount: Int
    ) throws -> Int {
        guard rows >= 0, cols >= 0, cols.isMultiple(of: weightsPerBlock) else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensorName) shape rows=\(rows) cols=\(cols) is invalid for packed \(weightsPerBlock)-wide blocks"
            )
        }
        return rows * (cols / weightsPerBlock) * blockByteCount
    }

    private static func makeStaticRawTensorBuffer(
        storage: TensorStorage,
        requiredBytes: Int,
        device: MTLDevice
    ) throws -> MTLBuffer {
        guard storage.byteOffset >= 0,
              storage.byteOffset + requiredBytes <= storage.buffer.length else {
            throw GenerationError.modelLoadFailed(reason: "\(storage.name) buffer is smaller than expected")
        }
        guard let buffer = device.makeBuffer(
            bytesNoCopy: storage.buffer.contents() + storage.byteOffset,
            length: requiredBytes,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create raw buffer for \(storage.name)")
        }
        return buffer
    }

    private static func makeStaticFloatBuffer(
        storage: TensorStorage,
        expectedElementCount: Int,
        device: MTLDevice
    ) throws -> MTLBuffer {
        let values = try readStaticFloatVector(
            storage: storage,
            expectedElementCount: expectedElementCount
        )
        return try makeStaticFloatBuffer(values: values, label: storage.name, device: device)
    }

    private static func makeStaticFloatBuffer(
        values: [Float],
        label: String,
        device: MTLDevice
    ) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create Float buffer for \(label)")
        }
        return buffer
    }

    private static func readStaticFloatVector(
        storage: TensorStorage,
        expectedElementCount: Int
    ) throws -> [Float] {
        guard storage.elementCount == expectedElementCount else {
            throw GenerationError.modelLoadFailed(
                reason: "\(storage.name) element count \(storage.elementCount) must equal \(expectedElementCount)"
            )
        }
        let base = storage.buffer.contents() + storage.byteOffset
        var values = [Float](repeating: 0, count: expectedElementCount)
        switch storage.dataType {
        case .float32:
            values.withUnsafeMutableBytes { rawBuffer in
                rawBuffer.copyMemory(from: UnsafeRawBufferPointer(
                    start: base,
                    count: expectedElementCount * MemoryLayout<Float>.stride
                ))
            }
        case .float16:
            let ptr = base.bindMemory(to: Float16.self, capacity: expectedElementCount)
            for index in 0..<expectedElementCount {
                values[index] = Float(ptr[index])
            }
        case .bfloat16:
            let ptr = base.bindMemory(to: UInt16.self, capacity: expectedElementCount)
            for index in 0..<expectedElementCount {
                values[index] = Float(bitPattern: UInt32(ptr[index]) << 16)
            }
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(storage.name) must be F32/F16/BF16 for Gemma norm, got \(storage.dataType)"
            )
        }
        return values
    }

    public func tokenize(_ text: String) -> [Int] {
        if let tokenizer {
            return tokenizer.encode(text, addBOS: tokenizer.shouldAddBOS)
        }
        return Array(text.utf8).map(Int.init)
    }

    public func detokenize(_ ids: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(ids, skipSpecialTokens: true)
        }
        return String(ids.compactMap { UInt8(exactly: $0) }.map { Character(UnicodeScalar($0)) })
    }

    public var eosTokenID: Int { tokenizer?.eosTokenID ?? 1 }
    public var bosTokenID: Int? { tokenizer?.bosTokenID }
    public var vocabularySize: Int { tokenizer?.vocabularySize ?? config.vocabSize }

    public func nextToken(for tokenIDs: [Int], sampling: SamplingConfiguration) async throws -> Int {
        let isPureGreedy = sampling.temperature <= 0 && sampling.repetitionPenalty <= 1.0
        guard isPureGreedy else {
            let logitsArray = try await logits(for: tokenIDs)
            let pipeline = sampling.toPipeline()
            return pipeline.sample(logits: logitsArray, previousTokens: tokenIDs)
        }

        guard tokenIDs.count <= maxSeqLen else {
            throw GenerationError.contextWindowExceeded(
                requested: tokenIDs.count,
                maximum: maxSeqLen
            )
        }

        let token = try await profiler.measureAsync("token_total") {
            let mode = decodeState.prepare(tokenIDs: tokenIDs)
            if useCacheBackedAttention,
               runtimeOptions.useGPULayerRunner,
               weights.tokenEmbedding.dataType == .q6_K {
                return try await runCacheBackedGreedyToken(
                    mode: mode,
                    fallbackTokenIDs: tokenIDs
                )
            }
            let finalHidden: [Float]
            if useCacheBackedAttention {
                finalHidden = try await runCacheBackedTokens(mode: mode, fallbackTokenIDs: tokenIDs)
            } else {
                let prelude = try await runPLEPrelude(tokenIDs: tokenIDs)
                finalHidden = try await runDecoderLayerStack(
                    prelude: prelude,
                    position: max(0, tokenIDs.count - 1),
                    useCacheBackedAttention: false
                )
            }
            return try await computeGreedyToken(hidden: finalHidden)
        }
        decodeState.markProcessed(tokenIDs: tokenIDs)
        profiler.markTokenComplete()
        return token
    }

    public func applyChatTemplate(
        messages: [EdgeRunnerCore.ChatMessage],
        addGenerationPrompt: Bool
    ) -> String? {
        let gemmaMessages = messages.map { message in
            Gemma4ChatMessage(
                role: Gemma4ChatRole(rawValue: message.role) ?? .user,
                content: message.content
            )
        }
        return try? Gemma4ChatTemplate.renderThrowing(
            messages: gemmaMessages,
            addGenerationPrompt: addGenerationPrompt
        )
    }

    public func logits(for tokenIDs: [Int]) async throws -> [Float] {
        guard tokenIDs.count <= maxSeqLen else {
            throw GenerationError.contextWindowExceeded(
                requested: tokenIDs.count,
                maximum: maxSeqLen
            )
        }

        let mode = decodeState.prepare(tokenIDs: tokenIDs)
        let finalHidden: [Float]
        if useCacheBackedAttention {
            finalHidden = try await runCacheBackedTokens(mode: mode, fallbackTokenIDs: tokenIDs)
        } else {
            let prelude = try await runPLEPrelude(tokenIDs: tokenIDs)
            finalHidden = try await runDecoderLayerStack(
                prelude: prelude,
                position: max(0, tokenIDs.count - 1),
                useCacheBackedAttention: false
            )
        }
        let logits = try await computeLogits(hidden: finalHidden)
        decodeState.markProcessed(tokenIDs: tokenIDs)
        return logits
    }

    private func runCacheBackedTokens(
        mode: Gemma4DecodeMode,
        fallbackTokenIDs: [Int]
    ) async throws -> [Float] {
        switch mode {
        case let .fullPrefill(tokens, startPosition):
            return try await runTokenSequence(tokens, startPosition: startPosition)
        case let .prefixReuse(tokens, startPosition):
            if tokens.isEmpty, let last = fallbackTokenIDs.last {
                return try await runSingleToken(last, position: max(0, fallbackTokenIDs.count - 1))
            }
            return try await runTokenSequence(tokens, startPosition: startPosition)
        case let .decode(token, position):
            return try await runSingleToken(token, position: position)
        }
    }

    private func runCacheBackedGreedyToken(
        mode: Gemma4DecodeMode,
        fallbackTokenIDs: [Int]
    ) async throws -> Int {
        switch mode {
        case let .fullPrefill(tokens, startPosition):
            return try await runTokenSequenceForGreedy(tokens, startPosition: startPosition)
        case let .prefixReuse(tokens, startPosition):
            if tokens.isEmpty, let last = fallbackTokenIDs.last {
                return try await runSingleTokenGreedy(last, position: max(0, fallbackTokenIDs.count - 1))
            }
            return try await runTokenSequenceForGreedy(tokens, startPosition: startPosition)
        case let .decode(token, position):
            return try await runSingleTokenGreedy(token, position: position)
        }
    }

    private func runTokenSequenceForGreedy(_ tokens: [Int], startPosition: Int) async throws -> Int {
        guard let last = tokens.last else {
            throw GenerationError.decodingFailed("Gemma greedy decode requires at least one token")
        }
        if tokens.count > 1 {
            try await runPrefillOnlyTokens(Array(tokens.dropLast()), startPosition: startPosition)
        }
        return try await runSingleTokenGreedy(last, position: startPosition + tokens.count - 1)
    }

    private func runTokenSequence(_ tokens: [Int], startPosition: Int) async throws -> [Float] {
        guard !tokens.isEmpty else { return [] }
        var finalHidden: [Float] = []
        if tokens.count > 1 {
            try await runPrefillOnlyTokens(Array(tokens.dropLast()), startPosition: startPosition)
        }
        if let last = tokens.last {
            finalHidden = try await runSingleToken(last, position: startPosition + tokens.count - 1)
        }
        return finalHidden
    }

    private func runPrefillOnlyTokens(_ tokens: [Int], startPosition: Int) async throws {
        guard !tokens.isEmpty else { return }
        guard runtimeOptions.useGPULayerRunner,
              weights.tokenEmbedding.dataType == .q6_K else {
            for (offset, tokenID) in tokens.enumerated() {
                try await runSingleTokenPrefillOnly(tokenID, position: startPosition + offset)
            }
            return
        }

        let chunks = try Gemma4PrefillChunkPlan.makeChunks(
            tokenCount: tokens.count,
            startPosition: startPosition,
            chunkSize: 64,
            numLayers: config.numHiddenLayers,
            perLayerDim: config.perLayerDim
        )
        for chunk in chunks {
            let range = chunk.tokenStartIndex..<(chunk.tokenStartIndex + chunk.tokenCount)
            let chunkTokens = Array(tokens[range])
            let prelude = try await runPLEPrelude(
                tokenIDs: chunkTokens,
                useLastTokenOnly: false,
                allowSingleTokenCache: false
            )
            guard prelude.tokenCount == chunk.tokenCount else {
                throw GenerationError.modelLoadFailed(reason: "Gemma prefill chunk PLE token count mismatch")
            }
            for tokenOffset in 0..<chunk.tokenCount {
                try await runSingleTokenPrefillOnly(
                    prelude: prelude,
                    position: chunk.position(forTokenOffset: tokenOffset),
                    tokenOffset: tokenOffset
                )
            }
        }
    }

    private func runSingleTokenPrefillOnly(_ tokenID: Int, position: Int) async throws {
        let prelude = try await runPLEPrelude(tokenIDs: [tokenID])
        try await runSingleTokenPrefillOnly(prelude: prelude, position: position, tokenOffset: 0)
    }

    private func runSingleTokenPrefillOnly(
        prelude: PreludeState,
        position: Int,
        tokenOffset: Int
    ) async throws {
        if runtimeOptions.useGPULayerRunner {
            _ = try await runDecoderLayerStackWithGPUCache(
                prelude: prelude,
                position: position,
                tokenOffset: tokenOffset,
                readOutput: false
            )
        } else {
            _ = try await runDecoderLayerStack(
                prelude: prelude,
                position: position,
                useCacheBackedAttention: true
            )
        }
    }

    private func runSingleTokenGreedy(_ tokenID: Int, position: Int) async throws -> Int {
        let prelude = try await runPLEPrelude(tokenIDs: [tokenID])
        return try await runDecoderLayerStackWithGPUCacheGreedy(
            prelude: prelude,
            position: position
        )
    }

    private func runSingleToken(_ tokenID: Int, position: Int) async throws -> [Float] {
        let prelude = try await runPLEPrelude(tokenIDs: [tokenID])
        return try await runDecoderLayerStack(
            prelude: prelude,
            position: position,
            useCacheBackedAttention: true
        )
    }

    private func runPLEPrelude(
        tokenIDs: [Int],
        useLastTokenOnly: Bool = true,
        allowSingleTokenCache: Bool = true
    ) async throws -> PreludeState {
        guard !tokenIDs.isEmpty else {
            return PreludeState(
                hidden: [],
                hiddenBuffer: nil,
                perLayerInputs: [],
                perLayerInputsBuffer: nil,
                perLayerInputCount: 0,
                tokenCount: 0
            )
        }

        for tokenID in tokenIDs {
            guard tokenID >= 0, tokenID < config.vocabSize, tokenID < config.perLayerVocabSize else {
                throw GenerationError.decodingFailed(
                    "Gemma 4 token id \(tokenID) is outside vocab \(config.vocabSize) or PLE vocab \(config.perLayerVocabSize)"
                )
            }
        }

        let hiddenSize = config.hiddenSize
        let numLayers = config.numHiddenLayers
        let perLayerDim = config.perLayerDim
        let projectionDim = numLayers * perLayerDim
        let activeTokenIDs = useLastTokenOnly ? [tokenIDs[tokenIDs.count - 1]] : tokenIDs
        if activeTokenIDs.count == 1,
           allowSingleTokenCache,
           let cached = runtimeCache.preludeState(tokenID: activeTokenIDs[0]) {
            return cached
        }
        let batchSeq = activeTokenIDs.count
        if batchSeq == 1,
           useCacheBackedAttention,
           runtimeOptions.useGPULayerRunner,
           runtimeOptions.useBufferNativePrelude,
           weights.tokenEmbedding.dataType == .q6_K {
            return try await runPLEPreludeBufferNative(
                tokenID: activeTokenIDs[0],
                hiddenSize: hiddenSize,
                numLayers: numLayers,
                perLayerDim: perLayerDim,
                projectionDim: projectionDim
            )
        }

        let hiddenScale = sqrt(Float(hiddenSize))
        let hidden: [Float] = try await profiler.measureAsync("ple_token_embedding") {
            if weights.tokenEmbedding.dataType == .q6_K {
                let rowStride = try rowStrideBytes(
                    rowWidth: hiddenSize,
                    weightsPerBlock: 256,
                    blockByteCount: 210,
                    tensorName: weights.tokenEmbedding.name
                )
                return try await gemmaDecodeKernels.runGatherQ6KTokenEmbedding(
                    tableBuffer: weights.tokenEmbedding.buffer,
                    tokenIDs: activeTokenIDs,
                    rowWidth: hiddenSize,
                    rowStrideBytes: rowStride,
                    tableByteOffset: weights.tokenEmbedding.byteOffset,
                    scale: hiddenScale,
                    commandQueue: commandQueue
                )
            } else {
                var cpuHidden = [Float](repeating: 0, count: batchSeq * hiddenSize)
                try fillEmbeddingRows(
                    storage: weights.tokenEmbedding,
                    tokenIDs: activeTokenIDs,
                    vocabSize: config.vocabSize,
                    rowWidth: hiddenSize,
                    into: &cpuHidden
                )
                for index in cpuHidden.indices {
                    cpuHidden[index] *= hiddenScale
                }
                return cpuHidden
            }
        }
        try validateFinite(hidden, label: "Gemma PLE token embedding")

        let pleRowsBuffer = try profiler.measure("ple_row_gather") {
            try makePLEGatherBuffer(
                tokenIDs: activeTokenIDs,
                perLayerDim: perLayerDim,
                numLayers: numLayers
            )
        }
        let projection = try await profiler.measureAsync("ple_model_projection") {
            try await projectPLEInputs(
                hidden: hidden,
                batchSeq: batchSeq,
                hiddenSize: hiddenSize,
                projectionDim: projectionDim
            )
        }
        let projectionBuffer = try makeFloatBuffer(projection, label: "Gemma PLE projection")
        let normWeightBuffer = try makeFloatBuffer(
            storage: weights.perLayerProjectionNorm,
            expectedElementCount: perLayerDim
        )
        guard let perLayerInputsBuffer = device.makeBuffer(
            length: batchSeq * projectionDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma PLE input buffers")
        }

        try await profiler.measureAsync("ple_inputs_build") {
            try pleInputsKernel.encode(
                commandBuffer: commandBuffer,
                projectionBuffer: projectionBuffer,
                normWeightBuffer: normWeightBuffer,
                pleRowsBuffer: pleRowsBuffer,
                outputBuffer: perLayerInputsBuffer,
                hiddenSize: hiddenSize,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                batchSeq: batchSeq,
                rmsEps: config.rmsNormEps
            )

            commandBuffer.commit()
            await commandBuffer.completed()
            if let error = commandBuffer.error {
                throw error
            }
        }

        let perLayerInputCount = batchSeq * projectionDim
        if validateRuntimeTensors {
            let perLayerInputPointer = perLayerInputsBuffer.contents().bindMemory(
                to: Float.self,
                capacity: perLayerInputCount
            )
            let perLayerInputs = Array(
                UnsafeBufferPointer(start: perLayerInputPointer, count: perLayerInputCount)
            )
            try validateFinite(perLayerInputs, label: "Gemma PLE inputs")
        }
        let prelude = PreludeState(
            hidden: hidden,
            hiddenBuffer: nil,
            perLayerInputs: nil,
            perLayerInputsBuffer: perLayerInputsBuffer,
            perLayerInputCount: perLayerInputCount,
            tokenCount: batchSeq
        )
        if batchSeq == 1, allowSingleTokenCache {
            runtimeCache.storePreludeState(prelude, tokenID: activeTokenIDs[0])
        }
        return prelude
    }

    private func runPLEPreludeBufferNative(
        tokenID: Int,
        hiddenSize: Int,
        numLayers: Int,
        perLayerDim: Int,
        projectionDim: Int
    ) async throws -> PreludeState {
        let hiddenScale = sqrt(Float(hiddenSize))
        let rowStride = try rowStrideBytes(
            rowWidth: hiddenSize,
            weightsPerBlock: 256,
            blockByteCount: 210,
            tensorName: weights.tokenEmbedding.name
        )
        let totalPLEElements = numLayers * perLayerDim
        guard let tokenID32 = Int32(exactly: tokenID) else {
            throw GenerationError.decodingFailed("Gemma token id \(tokenID) cannot fit in Int32")
        }
        guard let tokenBuffer = device.makeBuffer(
            bytes: [tokenID32],
            length: MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ),
        let hiddenBuffer = device.makeBuffer(
            length: hiddenSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let projectionBuffer = device.makeBuffer(
            length: projectionDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let pleRowsBuffer = device.makeBuffer(
            length: totalPLEElements * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let perLayerInputsBuffer = device.makeBuffer(
            length: projectionDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma buffer-native PLE buffers")
        }

        try await profiler.measureAsync("ple_token_embedding") {
            try gemmaDecodeKernels.encodeGatherQ6KTokenEmbedding(
                commandBuffer: commandBuffer,
                tableBuffer: weights.tokenEmbedding.buffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: hiddenBuffer,
                tokenCount: 1,
                rowWidth: hiddenSize,
                rowStrideBytes: rowStride,
                tableByteOffset: weights.tokenEmbedding.byteOffset,
                scale: hiddenScale
            )
        }

        try profiler.measure("ple_row_gather") {
            try encodePLEGatherRows(
                commandBuffer: commandBuffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: pleRowsBuffer,
                tokenCount: 1,
                perLayerDim: perLayerDim,
                numLayers: numLayers
            )
        }

        try await profiler.measureAsync("ple_model_projection") {
            let request = ProjectionRequest(
                tensor: weights.perLayerModelProjection,
                rows: projectionDim,
                cols: hiddenSize
            )
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
            let weightBuffer = try makeRawTensorBuffer(
                storage: request.tensor,
                requiredBytes: try projectionRequiredBytes(request)
            )
            try encodeProjection(
                request,
                weightBuffer: weightBuffer,
                inputBuffer: hiddenBuffer,
                outputBuffer: projectionBuffer,
                commandBuffer: commandBuffer
            )
        }

        let normWeightBuffer = try makeFloatBuffer(
            storage: weights.perLayerProjectionNorm,
            expectedElementCount: perLayerDim
        )
        try await profiler.measureAsync("ple_inputs_build") {
            try pleInputsKernel.encode(
                commandBuffer: commandBuffer,
                projectionBuffer: projectionBuffer,
                normWeightBuffer: normWeightBuffer,
                pleRowsBuffer: pleRowsBuffer,
                outputBuffer: perLayerInputsBuffer,
                hiddenSize: hiddenSize,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                batchSeq: 1,
                rmsEps: config.rmsNormEps
            )
        }

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let hidden: [Float]
        if validateRuntimeTensors {
            let hiddenPointer = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            hidden = Array(UnsafeBufferPointer(start: hiddenPointer, count: hiddenSize))
            try validateFinite(hidden, label: "Gemma PLE token embedding")
            let inputsPointer = perLayerInputsBuffer.contents().bindMemory(to: Float.self, capacity: projectionDim)
            let inputs = Array(UnsafeBufferPointer(start: inputsPointer, count: projectionDim))
            try validateFinite(inputs, label: "Gemma PLE inputs")
        } else {
            hidden = []
        }

        let prelude = PreludeState(
            hidden: hidden,
            hiddenBuffer: hiddenBuffer,
            perLayerInputs: nil,
            perLayerInputsBuffer: perLayerInputsBuffer,
            perLayerInputCount: projectionDim,
            tokenCount: 1
        )
        runtimeCache.storePreludeState(prelude, tokenID: tokenID)
        return prelude
    }

    private func runDecoderLayerStack(
        prelude: PreludeState,
        position: Int,
        useCacheBackedAttention: Bool
    ) async throws -> [Float] {
        guard position >= 0 else { return [] }
        let hidden = prelude.hidden
        guard hidden.count == config.hiddenSize || prelude.hiddenBuffer != nil else {
            throw GenerationError.modelLoadFailed(reason: "Gemma hidden buffer has invalid shape")
        }
        guard prelude.perLayerInputCount == config.numHiddenLayers * config.perLayerDim else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer has invalid shape")
        }

        if useCacheBackedAttention,
           runtimeOptions.useGPULayerRunner {
            return try await runDecoderLayerStackWithGPUCache(
                prelude: prelude,
                position: position
            )
        }

        let hiddenSize = config.hiddenSize
        var current = Array(hidden[0..<hiddenSize])
        var cachedValues: [Int: [Float]] = [:]

        for layer in 0..<config.numHiddenLayers {
            if useCacheBackedAttention {
                current = try await runDecoderLayerWithCache(
                    layer: layer,
                    position: position,
                    hidden: current,
                    prelude: prelude
                )
            } else {
                current = try await runDecoderLayerShortcut(
                    layer: layer,
                    hidden: current,
                    prelude: prelude,
                    cachedValues: &cachedValues
                )
            }
            try validateFinite(current, label: "Gemma decoder layer \(layer)")
        }
        return current
    }

    private func runDecoderLayerStackWithGPUCache(
        prelude: PreludeState,
        position: Int,
        tokenOffset: Int = 0,
        readOutput: Bool = true
    ) async throws -> [Float] {
        guard let pleInputBuffer = prelude.perLayerInputsBuffer else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
        }
        let expectedPLEInputCount = prelude.tokenCount * config.numHiddenLayers * config.perLayerDim
        guard prelude.perLayerInputCount == expectedPLEInputCount else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer has invalid shape")
        }

        return try await profiler.measureAsync("gpu_layer_stack") {
            await scratchGate.acquire()
            do {
                if let hiddenBuffer = prelude.hiddenBuffer {
                    try encodeCopyBuffer(
                        source: hiddenBuffer,
                        destination: scratch.currentHidden,
                        sourceOffset: tokenOffset * config.hiddenSize * MemoryLayout<Float>.stride,
                        byteCount: config.hiddenSize * MemoryLayout<Float>.stride
                    )
                } else if prelude.tokenCount > 1 {
                    try scratch.copyHiddenBatch(prelude.hidden, tokenOffset: tokenOffset)
                } else {
                    try scratch.copyHidden(prelude.hidden)
                }
                guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                    throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma GPU layer command buffer")
                }

                try profiler.measure("gpu_layer_stack_encode_layers") {
                    for plan in layerRuntimePlans {
                        try profiler.measure("gpu_layer_encode_total") {
                            try encodeDecoderLayerWithCache(
                                commandBuffer: commandBuffer,
                                plan: plan,
                                position: position,
                                pleInputBuffer: pleInputBuffer,
                                pleInputOffset: try prelude.perLayerInputByteOffset(
                                    tokenOffset: tokenOffset,
                                    layer: plan.layer,
                                    perLayerDim: config.perLayerDim,
                                    numLayers: config.numHiddenLayers
                                ),
                                scratch: scratch
                            )
                        }
                    }
                }

                try await profiler.measureAsync("gpu_layer_stack_wait") {
                    commandBuffer.commit()
                    await commandBuffer.completed()
                    if let error = commandBuffer.error {
                        throw error
                    }
                }
                recordMetalTimings(commandBuffer, prefix: "gpu_layer_stack")

                guard readOutput else {
                    await scratchGate.release()
                    return []
                }

                let output = try profiler.measure("gpu_layer_stack_read_hidden") {
                    try scratch.readHidden()
                }
                try validateFinite(output, label: "Gemma GPU decoder layer stack")
                await scratchGate.release()
                return output
            } catch {
                await scratchGate.release()
                throw error
            }
        }
    }

    private func runDecoderLayerStackWithGPUCacheGreedy(
        prelude: PreludeState,
        position: Int,
        tokenOffset: Int = 0
    ) async throws -> Int {
        guard let pleInputBuffer = prelude.perLayerInputsBuffer else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
        }
        let expectedPLEInputCount = prelude.tokenCount * config.numHiddenLayers * config.perLayerDim
        guard prelude.perLayerInputCount == expectedPLEInputCount else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer has invalid shape")
        }

        return try await profiler.measureAsync("gpu_layer_stack") {
            if profiler.isEnabled,
               ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_STACK"] == "1" {
                return try await runDecoderLayerStackWithGPUCacheGreedySplitProfile(
                    prelude: prelude,
                    position: position
                )
            }
            await scratchGate.acquire()
            do {
                if let hiddenBuffer = prelude.hiddenBuffer {
                    try encodeCopyBuffer(
                        source: hiddenBuffer,
                        destination: scratch.currentHidden,
                        sourceOffset: tokenOffset * config.hiddenSize * MemoryLayout<Float>.stride,
                        byteCount: config.hiddenSize * MemoryLayout<Float>.stride
                    )
                } else if prelude.tokenCount > 1 {
                    try scratch.copyHiddenBatch(prelude.hidden, tokenOffset: tokenOffset)
                } else {
                    try scratch.copyHidden(prelude.hidden)
                }
                guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                    throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma GPU greedy command buffer")
                }

                try profiler.measure("gpu_layer_stack_encode_layers") {
                    for plan in layerRuntimePlans {
                        try profiler.measure("gpu_layer_encode_total") {
                            try encodeDecoderLayerWithCache(
                                commandBuffer: commandBuffer,
                                plan: plan,
                                position: position,
                                pleInputBuffer: pleInputBuffer,
                                pleInputOffset: try prelude.perLayerInputByteOffset(
                                    tokenOffset: tokenOffset,
                                    layer: plan.layer,
                                    perLayerDim: config.perLayerDim,
                                    numLayers: config.numHiddenLayers
                                ),
                                scratch: scratch
                            )
                        }
                    }
                }
                try profiler.measure("gpu_layer_stack_encode_lm_head") {
                    try encodeGreedyQ6KLogitsFromScratch(commandBuffer: commandBuffer, scratch: scratch)
                }

                try await profiler.measureAsync("gpu_layer_stack_wait") {
                    commandBuffer.commit()
                    await commandBuffer.completed()
                    if let error = commandBuffer.error {
                        throw error
                    }
                }
                recordMetalTimings(commandBuffer, prefix: "gpu_layer_stack")

                let token = try profiler.measure("gpu_layer_stack_argmax") {
                    try readGreedyTokenFromScratch(scratch)
                }
                await scratchGate.release()
                return token
            } catch {
                await scratchGate.release()
                throw error
            }
        }
    }

    private func runDecoderLayerStackWithGPUCacheGreedySplitProfile(
        prelude: PreludeState,
        position: Int
    ) async throws -> Int {
        guard let pleInputBuffer = prelude.perLayerInputsBuffer else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
        }

        await scratchGate.acquire()
        do {
            if let hiddenBuffer = prelude.hiddenBuffer {
                try encodeCopyBuffer(
                    source: hiddenBuffer,
                    destination: scratch.currentHidden,
                    byteCount: config.hiddenSize * MemoryLayout<Float>.stride
                )
            } else {
                try scratch.copyHidden(prelude.hidden)
            }

            for layer in 0..<config.numHiddenLayers {
                let pleInputOffset = try prelude.perLayerInputByteOffset(
                    layer: layer,
                    perLayerDim: config.perLayerDim
                )
                if ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_PHASES"] == "1" {
                    try await runSplitProfilePhase(
                        aggregatePrefix: splitProfileAggregatePrefix("gpu_split_attention", layer: layer),
                        layerPrefix: "gpu_split_layer_\(layer)_attention",
                        encode: { commandBuffer in
                            try encodeDecoderLayerAttentionWithCache(
                                commandBuffer: commandBuffer,
                                layer: layer,
                                position: position,
                                scratch: scratch
                            )
                        }
                    )
                    if ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_FFN"] == "1" {
                        if ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_FFN_ACTIVATION"] == "1" {
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_norm", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_norm",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNInputNormWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_gate_up", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_gate_up",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNGateUpWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_geglu", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_geglu",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNActivationOnlyWithCache(
                                        commandBuffer: commandBuffer,
                                        scratch: scratch
                                    )
                                }
                            )
                        } else {
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_activation", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_activation",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNInputAndActivationWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                        }
                        if ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_FFN_DOWN"] == "1" {
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_down_projection", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_down_projection",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNDownProjectionWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_post_norm", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_post_norm",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNPostNormWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                        } else {
                            try await runSplitProfilePhase(
                                aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn_down", layer: layer),
                                layerPrefix: "gpu_split_layer_\(layer)_ffn_down",
                                encode: { commandBuffer in
                                    try encodeDecoderLayerFFNDownAndPostNormWithCache(
                                        commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch
                                    )
                                }
                            )
                        }
                    } else {
                        try await runSplitProfilePhase(
                            aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ffn", layer: layer),
                            layerPrefix: "gpu_split_layer_\(layer)_ffn",
                            encode: { commandBuffer in
                                try encodeDecoderLayerFFNWithCache(
                                    commandBuffer: commandBuffer,
                                    layer: layer,
                                    scratch: scratch
                                )
                            }
                        )
                    }
                    try await runSplitProfilePhase(
                        aggregatePrefix: splitProfileAggregatePrefix("gpu_split_ple", layer: layer),
                        layerPrefix: "gpu_split_layer_\(layer)_ple",
                        encode: { commandBuffer in
                            try encodeDecoderLayerPLEWithCache(
                                commandBuffer: commandBuffer,
                                layer: layer,
                                pleInputBuffer: pleInputBuffer,
                                pleInputOffset: pleInputOffset,
                                scratch: scratch
                            )
                        }
                    )
                } else {
                    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                        throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma split-profile layer command buffer")
                    }
                    try profiler.measure("gpu_split_layer_encode") {
                        try profiler.measure("gpu_split_layer_\(layer)_encode") {
                            try encodeDecoderLayerWithCache(
                                commandBuffer: commandBuffer,
                                layer: layer,
                                position: position,
                                pleInputBuffer: pleInputBuffer,
                                pleInputOffset: pleInputOffset,
                                scratch: scratch
                            )
                        }
                    }
                    try await profiler.measureAsync("gpu_split_layer_wait") {
                        try await profiler.measureAsync("gpu_split_layer_\(layer)_wait") {
                            commandBuffer.commit()
                            await commandBuffer.completed()
                            if let error = commandBuffer.error {
                                throw error
                            }
                        }
                    }
                    recordMetalTimings(commandBuffer, prefix: "gpu_split_layer")
                    recordMetalTimings(commandBuffer, prefix: "gpu_split_layer_\(layer)")
                }
            }

            guard let lmHeadCommandBuffer = commandQueue.makeCommandBuffer() else {
                throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma split-profile LM-head command buffer")
            }
            try profiler.measure("gpu_split_lm_head_encode") {
                try encodeGreedyQ6KLogitsFromScratch(commandBuffer: lmHeadCommandBuffer, scratch: scratch)
            }
            try await profiler.measureAsync("gpu_split_lm_head_wait") {
                lmHeadCommandBuffer.commit()
                await lmHeadCommandBuffer.completed()
                if let error = lmHeadCommandBuffer.error {
                    throw error
                }
            }
            recordMetalTimings(lmHeadCommandBuffer, prefix: "gpu_split_lm_head")

            let token = try profiler.measure("gpu_split_argmax") {
                try readGreedyTokenFromScratch(scratch)
            }
            await scratchGate.release()
            return token
        } catch {
            await scratchGate.release()
            throw error
        }
    }

    private func runSplitProfilePhase(
        aggregatePrefix: String,
        layerPrefix: String,
        encode: (MTLCommandBuffer) throws -> Void
    ) async throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma split-profile phase command buffer")
        }
        try profiler.measure("\(aggregatePrefix)_encode") {
            try profiler.measure("\(layerPrefix)_encode") {
                try encode(commandBuffer)
            }
        }
        try await profiler.measureAsync("\(aggregatePrefix)_wait") {
            try await profiler.measureAsync("\(layerPrefix)_wait") {
                commandBuffer.commit()
                await commandBuffer.completed()
                if let error = commandBuffer.error {
                    throw error
                }
            }
        }
        recordMetalTimings(commandBuffer, prefix: aggregatePrefix)
        recordMetalTimings(commandBuffer, prefix: layerPrefix)
    }

    private func splitProfileAggregatePrefix(_ base: String, layer: Int) -> String {
        guard ProcessInfo.processInfo.environment["EDGERUNNER_GEMMA4_PROFILE_SPLIT_LAYER_TYPES"] == "1" else {
            return base
        }

        let attentionType: String
        switch config.layerTypes[layer] {
        case .sliding:
            attentionType = "sliding"
        case .global:
            attentionType = "global"
        }

        let kvOwnership = config.kvSourceLayer(for: layer) == layer ? "ownkv" : "sharedkv"
        return "\(base)_\(attentionType)_\(kvOwnership)"
    }

    private func recordMetalTimings(_ commandBuffer: MTLCommandBuffer, prefix: String) {
        let gpuDuration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        profiler.recordDuration("\(prefix)_gpu_time", duration: gpuDuration)

        let kernelDuration = commandBuffer.kernelEndTime - commandBuffer.kernelStartTime
        profiler.recordDuration("\(prefix)_kernel_time", duration: kernelDuration)
    }

    private func encodeGreedyQ6KLogitsFromScratch(
        commandBuffer: MTLCommandBuffer,
        scratch: Gemma4Scratch
    ) throws {
        let tensor = lmHeadResources.tokenEmbedding.request.tensor
        guard tensor.dataType == .q6_K else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma GPU greedy path requires Q6_K tied token embeddings"
            )
        }
        try gemmaDecodeKernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: scratch.currentHidden,
            weightBuffer: lmHeadResources.outputNormWeightBuffer,
            outputBuffer: scratch.normed,
            rows: 1,
            cols: config.hiddenSize,
            eps: config.rmsNormEps
        )
        let weightBuffer = lmHeadResources.tokenEmbedding.weightBuffer
        if runtimeOptions.useQ6Top1 {
            try gemvKernel.encodeQ6KWeightsPackedTop1(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: scratch.normed,
                partialValuesBuffer: scratch.top1PartialValues,
                partialIndicesBuffer: scratch.top1PartialIndices,
                outputIndexBuffer: scratch.top1Token,
                M: config.vocabSize,
                K: config.hiddenSize
            )
        } else if runtimeOptions.useQ6Packed {
            try gemvKernel.encodeQ6KWeightsPacked(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: scratch.normed,
                outputBuffer: scratch.logits,
                M: config.vocabSize,
                K: config.hiddenSize
            )
        } else {
            try gemvKernel.encodeQ6KWeights(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: scratch.normed,
                outputBuffer: scratch.logits,
                M: config.vocabSize,
                K: config.hiddenSize
            )
        }
    }

    private func readGreedyTokenFromScratch(_ scratch: Gemma4Scratch) throws -> Int {
        if runtimeOptions.useQ6Top1 {
            let pointer = scratch.top1Token.contents().bindMemory(to: UInt32.self, capacity: 1)
            let token = Int(pointer.pointee)
            guard token >= 0, token < config.vocabSize else {
                throw GenerationError.decodingFailed("Gemma top-1 token \(token) is outside vocab \(config.vocabSize)")
            }
            return token
        }

        let pointer = scratch.logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize)
        return try greedyArgmax(pointer: pointer, count: config.vocabSize)
    }

    private func encodeCopyBuffer(
        source: MTLBuffer,
        destination: MTLBuffer,
        sourceOffset: Int = 0,
        byteCount: Int
    ) throws {
        guard sourceOffset >= 0,
              source.length >= sourceOffset + byteCount,
              destination.length >= byteCount,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to copy Gemma scratch buffer")
        }
        encoder.copy(
            from: source,
            sourceOffset: sourceOffset,
            to: destination,
            destinationOffset: 0,
            size: byteCount
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
    }

    private func encodeDecoderLayerWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        position: Int,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerWithCache(
            commandBuffer: commandBuffer,
            plan: layerRuntimePlans[layer],
            position: position,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerWithCache(
        commandBuffer: MTLCommandBuffer,
        plan: Gemma4LayerRuntimePlan,
        position: Int,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerAttentionWithCache(
            commandBuffer: commandBuffer,
            plan: plan,
            position: position,
            scratch: scratch
        )
        try encodeDecoderLayerFFNWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[plan.layer],
            scratch: scratch
        )
        try encodeDecoderLayerPLEWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[plan.layer],
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerAttentionWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        position: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerAttentionWithCache(
            commandBuffer: commandBuffer,
            plan: layerRuntimePlans[layer],
            position: position,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerAttentionWithCache(
        commandBuffer: MTLCommandBuffer,
        plan: Gemma4LayerRuntimePlan,
        position: Int,
        scratch: Gemma4Scratch
    ) throws {
        let layer = plan.layer
        let resources = layerRuntimeResources[layer]
        let hiddenSize = config.hiddenSize
        let layerHeadDim = plan.headDim
        let qRows = plan.qRows
        let kvRows = plan.kvRows

        try gemmaDecodeKernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: scratch.currentHidden,
            weightBuffer: resources.inputNormWeightBuffer,
            outputBuffer: scratch.normed,
            rows: 1,
            cols: hiddenSize,
            eps: config.rmsNormEps
        )

        let qRequest = resources.attnQ.request
        if let kResource = resources.attnK, let vResource = resources.attnV {
            let kRequest = kResource.request
            let vRequest = vResource.request
            if qRequest.tensor.dataType == .q4_K,
               kRequest.tensor.dataType == .q4_K,
               vRequest.tensor.dataType == .q4_K {
                if runtimeOptions.useQ4Packed {
                    try gemvKernel.encodeQ4KWeightsTriplePacked(
                        commandBuffer: commandBuffer,
                        weightBufferA: resources.attnQ.weightBuffer,
                        weightBufferB: kResource.weightBuffer,
                        weightBufferC: vResource.weightBuffer,
                        inputBuffer: scratch.normed,
                        outputBufferA: scratch.q,
                        outputBufferB: scratch.k,
                        outputBufferC: scratch.v,
                        rowsA: qRows,
                        rowsB: kvRows,
                        rowsC: kvRows,
                        K: hiddenSize
                    )
                } else {
                    try gemvKernel.encodeQ4KWeightsTriple(
                        commandBuffer: commandBuffer,
                        weightBufferA: resources.attnQ.weightBuffer,
                        weightBufferB: kResource.weightBuffer,
                        weightBufferC: vResource.weightBuffer,
                        inputBuffer: scratch.normed,
                        outputBufferA: scratch.q,
                        outputBufferB: scratch.k,
                        outputBufferC: scratch.v,
                        rowsA: qRows,
                        rowsB: kvRows,
                        rowsC: kvRows,
                        K: hiddenSize
                    )
                }
            } else {
                try encodeProjection(
                    qRequest,
                    weightBuffer: resources.attnQ.weightBuffer,
                    inputBuffer: scratch.normed,
                    outputBuffer: scratch.q,
                    commandBuffer: commandBuffer
                )
                try encodeProjection(
                    kRequest,
                    weightBuffer: kResource.weightBuffer,
                    inputBuffer: scratch.normed,
                    outputBuffer: scratch.k,
                    commandBuffer: commandBuffer
                )
                try encodeProjection(
                    vRequest,
                    weightBuffer: vResource.weightBuffer,
                    inputBuffer: scratch.normed,
                    outputBuffer: scratch.v,
                    commandBuffer: commandBuffer
                )
            }
        } else {
            try encodeProjection(
                qRequest,
                weightBuffer: resources.attnQ.weightBuffer,
                inputBuffer: scratch.normed,
                outputBuffer: scratch.q,
                commandBuffer: commandBuffer
            )
        }

        try gemmaDecodeKernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: scratch.q,
            weightBuffer: resources.attnQNormWeightBuffer,
            outputBuffer: scratch.qRotated,
            rows: config.numAttentionHeads,
            cols: layerHeadDim,
            eps: config.rmsNormEps
        )
        try ropeKernel.encodeNeoX(
            commandBuffer: commandBuffer,
            inputBuffer: scratch.qRotated,
            outputBuffer: scratch.q,
            seqLen: 1,
            numHeads: config.numAttentionHeads,
            headDim: layerHeadDim,
            startPos: position,
            theta: plan.ropeTheta,
            partialRotaryFactor: plan.rotaryFactor
        )

        if resources.attnK != nil, resources.attnV != nil, let kNormWeightBuffer = resources.attnKNormWeightBuffer {
            try gemmaDecodeKernels.encodeRMSNorm(
                commandBuffer: commandBuffer,
                inputBuffer: scratch.k,
                weightBuffer: kNormWeightBuffer,
                outputBuffer: scratch.kRotated,
                rows: config.numKeyValueHeads,
                cols: layerHeadDim,
                eps: config.rmsNormEps
            )
            try ropeKernel.encodeNeoX(
                commandBuffer: commandBuffer,
                inputBuffer: scratch.kRotated,
                outputBuffer: scratch.k,
                seqLen: 1,
                numHeads: config.numKeyValueHeads,
                headDim: layerHeadDim,
                startPos: position,
                theta: plan.ropeTheta,
                partialRotaryFactor: plan.rotaryFactor
            )
            try gemmaDecodeKernels.encodeStoreF32ToF16(
                commandBuffer: commandBuffer,
                inputBuffer: scratch.k,
                outputBuffer: gpuKVCache.keyBuffer(forLayer: layer),
                outputOffset: gpuKVCache.writeOffset(layer: layer, position: position),
                count: kvRows
            )

            try gemmaDecodeKernels.encodeRMSNorm(
                commandBuffer: commandBuffer,
                inputBuffer: scratch.v,
                weightBuffer: resources.unitHeadWeightBuffer,
                outputBuffer: scratch.kRotated,
                rows: config.numKeyValueHeads,
                cols: layerHeadDim,
                eps: config.rmsNormEps
            )
            try gemmaDecodeKernels.encodeStoreF32ToF16(
                commandBuffer: commandBuffer,
                inputBuffer: scratch.kRotated,
                outputBuffer: gpuKVCache.valueBuffer(forLayer: layer),
                outputOffset: gpuKVCache.writeOffset(layer: layer, position: position),
                count: kvRows
            )
        }

        let kvSource = plan.kvSourceLayer
        let attentionRange = gpuKVCache.attentionRange(layer: kvSource, currentPosition: position)
        try gemmaDecodeKernels.encodeDecodeGQAF16KVWindowedBestAvailable(
            commandBuffer: commandBuffer,
            qBuffer: scratch.q,
            keyCacheBuffer: gpuKVCache.keyBuffer(forLayer: kvSource),
            valueCacheBuffer: gpuKVCache.valueBuffer(forLayer: kvSource),
            outputBuffer: scratch.attention,
            numHeads: config.numAttentionHeads,
            numKVHeads: config.numKeyValueHeads,
            headDim: layerHeadDim,
            kvStart: gpuKVCache.physicalPosition(layer: kvSource, logicalPosition: attentionRange.start),
            kvCount: attentionRange.count,
            kvCapacity: gpuKVCache.capacity(forLayer: kvSource),
            attentionScale: 1.0
        )

        try encodeProjection(
            resources.attnO.request,
            weightBuffer: resources.attnO.weightBuffer,
            inputBuffer: scratch.attention,
            outputBuffer: scratch.nextHidden,
            commandBuffer: commandBuffer
        )
        try gemmaDecodeKernels.encodeResidualRMSNormAdd(
            commandBuffer: commandBuffer,
            residualBuffer: scratch.currentHidden,
            inputBuffer: scratch.nextHidden,
            weightBuffer: resources.postAttentionNormWeightBuffer,
            outputBuffer: scratch.nextHidden,
            count: hiddenSize,
            eps: config.rmsNormEps
        )
    }

    private func encodeDecoderLayerFFNWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNInputAndActivationWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
        try encodeDecoderLayerFFNDownAndPostNormWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNInputAndActivationWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNInputAndActivationWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNInputAndActivationWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNInputNormWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
        try encodeDecoderLayerFFNGateUpWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
        try encodeDecoderLayerFFNActivationOnlyWithCache(
            commandBuffer: commandBuffer,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNInputNormWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNInputNormWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNInputNormWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        let hiddenSize = config.hiddenSize
        try gemmaDecodeKernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: scratch.nextHidden,
            weightBuffer: resources.ffnNormWeightBuffer,
            outputBuffer: scratch.ffnInput,
            rows: 1,
            cols: hiddenSize,
            eps: config.rmsNormEps
        )
    }

    private func encodeDecoderLayerFFNGateUpWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNGateUpWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNGateUpWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        try encodeGeGLUActivation(
            gate: resources.ffnGate,
            up: resources.ffnUp,
            inputBuffer: scratch.ffnInput,
            gateBuffer: scratch.ffnGate,
            upBuffer: scratch.ffnUp,
            activatedBuffer: scratch.ffnActivated,
            encodeActivation: false,
            commandBuffer: commandBuffer
        )
    }

    private func encodeDecoderLayerFFNActivationOnlyWithCache(
        commandBuffer: MTLCommandBuffer,
        scratch: Gemma4Scratch
    ) throws {
        try gegluKernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: scratch.ffnGate,
            upBuffer: scratch.ffnUp,
            outputBuffer: scratch.ffnActivated,
            count: config.intermediateSize
        )
    }

    private func encodeDecoderLayerFFNDownAndPostNormWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNDownAndPostNormWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNDownAndPostNormWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNDownProjectionWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
        try encodeDecoderLayerFFNPostNormWithCache(
            commandBuffer: commandBuffer,
            resources: resources,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNDownProjectionWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNDownProjectionWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNDownProjectionWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        try encodeGeGLUDownProjection(
            down: resources.ffnDown,
            activatedBuffer: scratch.ffnActivated,
            downBuffer: scratch.ffnDown,
            commandBuffer: commandBuffer
        )
    }

    private func encodeDecoderLayerFFNPostNormWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerFFNPostNormWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            scratch: scratch
        )
    }

    private func encodeDecoderLayerFFNPostNormWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        scratch: Gemma4Scratch
    ) throws {
        let hiddenSize = config.hiddenSize
        try gemmaDecodeKernels.encodeResidualRMSNormAdd(
            commandBuffer: commandBuffer,
            residualBuffer: scratch.nextHidden,
            inputBuffer: scratch.ffnDown,
            weightBuffer: resources.postFFNNormWeightBuffer,
            outputBuffer: scratch.currentHidden,
            count: hiddenSize,
            eps: config.rmsNormEps
        )
    }

    private func encodeDecoderLayerPLEWithCache(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodeDecoderLayerPLEWithCache(
            commandBuffer: commandBuffer,
            resources: layerRuntimeResources[layer],
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            scratch: scratch
        )
    }

    private func encodeDecoderLayerPLEWithCache(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodePLESideChannel(
            commandBuffer: commandBuffer,
            resources: resources,
            hiddenBuffer: scratch.currentHidden,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            scratch: scratch
        )
    }

    private func runDecoderLayerWithCache(
        layer: Int,
        position: Int,
        hidden tokenHidden: [Float],
        prelude: PreludeState
    ) async throws -> [Float] {
        let block = weights.blocks[layer]
        let hiddenSize = config.hiddenSize
        let layerHeadDim = headDim(forLayer: layer)
        let qRows = config.numAttentionHeads * layerHeadDim
        let kvRows = config.numKeyValueHeads * layerHeadDim
        let inputNormWeight = try readFloatVector(
            storage: block.inputNorm,
            expectedElementCount: hiddenSize
        )
        let normed = try profiler.measure("layer_input_norm") {
            try Gemma4Ops.rmsNorm(
                tokenHidden,
                weight: inputNormWeight,
                eps: config.rmsNormEps
            )
        }
        try validateFinite(normed, label: "Gemma layer \(layer) attention norm")

        let qProjected: [Float]
        let qRotated: [Float]
        if let attnK = block.attnK, let attnV = block.attnV {
            let projections = try await profiler.measureAsync("layer_qkv_projection") {
                try await projectTensors(
                    [
                        ProjectionRequest(tensor: block.attnQ, rows: qRows, cols: hiddenSize),
                        ProjectionRequest(tensor: attnK, rows: kvRows, cols: hiddenSize),
                        ProjectionRequest(tensor: attnV, rows: kvRows, cols: hiddenSize)
                    ],
                    input: normed
                )
            }
            qProjected = projections[0]
            let kProjected = projections[1]
            let vProjected = projections[2]
            try validateFinite(qProjected, label: "Gemma layer \(layer) Q projection")
            try validateFinite(kProjected, label: "Gemma layer \(layer) K projection")
            try validateFinite(vProjected, label: "Gemma layer \(layer) V projection")
            let qNormWeight = try readFloatVector(
                storage: block.attnQNorm,
                expectedElementCount: layerHeadDim
            )
            let kNormWeight = try readFloatVector(
                storage: block.attnKNorm,
                expectedElementCount: layerHeadDim
            )
            let qNorm = try profiler.measure("layer_q_norm") {
                try normalizeHeadRows(
                    qProjected,
                    headCount: config.numAttentionHeads,
                    headDim: layerHeadDim,
                    weight: qNormWeight
                )
            }
            let kNorm = try profiler.measure("layer_k_norm") {
                try normalizeHeadRows(
                    kProjected,
                    headCount: config.numKeyValueHeads,
                    headDim: layerHeadDim,
                    weight: kNormWeight
                )
            }
            let vNorm = profiler.measure("layer_v_norm") {
                normalizeHeadRowsUnscaled(
                    vProjected,
                    headCount: config.numKeyValueHeads,
                    headDim: layerHeadDim
                )
            }
            try validateFinite(qNorm, label: "Gemma layer \(layer) Q norm")
            try validateFinite(kNorm, label: "Gemma layer \(layer) K norm")
            try validateFinite(vNorm, label: "Gemma layer \(layer) V norm")
            let rotated = try await profiler.measureAsync("layer_rope") {
                try await ropeKernel.applyToQKNeoX(
                    q: qNorm,
                    k: kNorm,
                    seqLen: 1,
                    numHeads: config.numAttentionHeads,
                    numKVHeads: config.numKeyValueHeads,
                    headDim: layerHeadDim,
                    startPos: position,
                    theta: ropeTheta(forLayer: layer),
                    partialRotaryFactor: try rotaryFactor(forLayer: layer),
                    commandQueue: commandQueue
                )
            }
            qRotated = rotated.0
            try validateFinite(qRotated, label: "Gemma layer \(layer) Q RoPE")
            try validateFinite(rotated.1, label: "Gemma layer \(layer) K RoPE")
            try profiler.measure("layer_kv_store") {
                try gpuKVCache.store(
                    layer: layer,
                    position: position,
                    keys: rotated.1,
                    values: vNorm
                )
            }
        } else {
            qProjected = try await profiler.measureAsync("layer_q_projection_shared") {
                try await projectTensor(
                    block.attnQ,
                    input: normed,
                    cols: hiddenSize,
                    rows: qRows
                )
            }
            try validateFinite(qProjected, label: "Gemma layer \(layer) shared-Q projection")
            let qNormWeight = try readFloatVector(
                storage: block.attnQNorm,
                expectedElementCount: layerHeadDim
            )
            let qNorm = try profiler.measure("layer_q_norm") {
                try normalizeHeadRows(
                    qProjected,
                    headCount: config.numAttentionHeads,
                    headDim: layerHeadDim,
                    weight: qNormWeight
                )
            }
            try validateFinite(qNorm, label: "Gemma layer \(layer) shared-Q norm")
            let dummyK = [Float](repeating: 0, count: kvRows)
            qRotated = try await profiler.measureAsync("layer_rope_shared") {
                try await ropeKernel.applyToQKNeoX(
                    q: qNorm,
                    k: dummyK,
                    seqLen: 1,
                    numHeads: config.numAttentionHeads,
                    numKVHeads: config.numKeyValueHeads,
                    headDim: layerHeadDim,
                    startPos: position,
                    theta: ropeTheta(forLayer: layer),
                    partialRotaryFactor: try rotaryFactor(forLayer: layer),
                    commandQueue: commandQueue
                ).0
            }
            try validateFinite(qRotated, label: "Gemma layer \(layer) shared-Q RoPE")
        }

        let kvSource = config.kvSourceLayer(for: layer)
        let attentionRange = gpuKVCache.attentionRange(layer: kvSource, currentPosition: position)
        if traceFastGQA {
            print(
                "GEMMA4_FAST_GQA_TRACE layer=\(layer) kvSource=\(kvSource) position=\(position) " +
                "headDim=\(layerHeadDim) kvStart=\(gpuKVCache.physicalPosition(layer: kvSource, logicalPosition: attentionRange.start)) " +
                "kvCount=\(attentionRange.count) kvCapacity=\(gpuKVCache.capacity(forLayer: kvSource)) " +
                "qCount=\(qRotated.count)"
            )
        }
        let attention = try await profiler.measureAsync("layer_attention") {
            try await gemmaDecodeKernels.runDecodeGQAF16KVWindowed(
                q: qRotated,
                keyCacheBuffer: gpuKVCache.keyBuffer(forLayer: kvSource),
                valueCacheBuffer: gpuKVCache.valueBuffer(forLayer: kvSource),
                numHeads: config.numAttentionHeads,
                numKVHeads: config.numKeyValueHeads,
                headDim: layerHeadDim,
                kvStart: gpuKVCache.physicalPosition(layer: kvSource, logicalPosition: attentionRange.start),
                kvCount: attentionRange.count,
                kvCapacity: gpuKVCache.capacity(forLayer: kvSource),
                attentionScale: 1.0,
                commandQueue: commandQueue
            )
        }
        try validateFinite(attention, label: "Gemma layer \(layer) attention")
        let attentionProjection = try await profiler.measureAsync("layer_o_projection") {
            try await projectTensor(
                block.attnO,
                input: attention,
                cols: qRows,
                rows: hiddenSize
            )
        }
        try validateFinite(attentionProjection, label: "Gemma layer \(layer) attention output projection")
        let postAttentionNormWeight = try readFloatVector(
            storage: block.postAttentionNorm,
            expectedElementCount: hiddenSize
        )
        let postAttention = try profiler.measure("layer_post_attention_norm") {
            try Gemma4Ops.rmsNorm(
                attentionProjection,
                weight: postAttentionNormWeight,
                eps: config.rmsNormEps
            )
        }
        try validateFinite(postAttention, label: "Gemma layer \(layer) post-attention norm")
        let afterAttention = zip(tokenHidden, postAttention).map(+)
        try validateFinite(afterAttention, label: "Gemma layer \(layer) attention residual")
        let ffnNormWeight = try readFloatVector(
            storage: block.ffnNorm,
            expectedElementCount: hiddenSize
        )
        let ffnInput = try profiler.measure("layer_ffn_norm") {
            try Gemma4Ops.rmsNorm(
                afterAttention,
                weight: ffnNormWeight,
                eps: config.rmsNormEps
            )
        }
        try validateFinite(ffnInput, label: "Gemma layer \(layer) FFN norm")
        let down = try await profiler.measureAsync("layer_ffn_fused_gate_up_down") {
            try await projectGeGLUDown(
                gateTensor: block.ffnGate,
                upTensor: block.ffnUp,
                downTensor: block.ffnDown,
                input: ffnInput,
                scratch: scratch
            )
        }
        try validateFinite(down, label: "Gemma layer \(layer) FFN down")
        let postFFNNormWeight = try readFloatVector(
            storage: block.postFFNNorm,
            expectedElementCount: hiddenSize
        )
        let postFFN = try profiler.measure("layer_post_ffn_norm") {
            try Gemma4Ops.rmsNorm(
                down,
                weight: postFFNNormWeight,
                eps: config.rmsNormEps
            )
        }
        try validateFinite(postFFN, label: "Gemma layer \(layer) post-FFN norm")
        let afterFFN = zip(afterAttention, postFFN).map(+)
        try validateFinite(afterFFN, label: "Gemma layer \(layer) FFN residual")

        guard let pleInputBuffer = prelude.perLayerInputsBuffer else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE input buffer is unavailable")
        }
        let pleInputOffset = try prelude.perLayerInputByteOffset(layer: layer, perLayerDim: config.perLayerDim)
        return try await profiler.measureAsync("layer_ple_side_channel") {
            try await applyPLESideChannel(
                block: block,
                hidden: afterFFN,
                pleInputBuffer: pleInputBuffer,
                pleInputOffset: pleInputOffset,
                scratch: scratch
            )
        }
    }

    private func runDecoderLayerShortcut(
        layer: Int,
        hidden tokenHidden: [Float],
        prelude: PreludeState,
        cachedValues: inout [Int: [Float]]
    ) async throws -> [Float] {
        let block = weights.blocks[layer]
        let hiddenSize = config.hiddenSize
        let layerHeadDim = headDim(forLayer: layer)
        let qRows = config.numAttentionHeads * layerHeadDim
        let kvRows = config.numKeyValueHeads * layerHeadDim
        let inputNormWeight = try readFloatVector(
            storage: block.inputNorm,
            expectedElementCount: hiddenSize
        )
        let normed = try Gemma4Ops.rmsNorm(
            tokenHidden,
            weight: inputNormWeight,
            eps: config.rmsNormEps
        )

        let v: [Float]
        let kvSource = config.kvSourceLayer(for: layer)
        if let attnV = block.attnV {
            var projectedV = try await projectTensor(
                attnV,
                input: normed,
                cols: hiddenSize,
                rows: kvRows
            )
            projectedV = normalizeHeadRowsUnscaled(
                projectedV,
                headCount: config.numKeyValueHeads,
                headDim: layerHeadDim
            )
            cachedValues[layer] = projectedV
            v = projectedV
        } else if let sharedV = cachedValues[kvSource] {
            v = sharedV
        } else {
            throw GenerationError.decodingFailed(
                "Gemma 4 layer \(layer) requires shared V from layer \(kvSource), but it has not been computed"
            )
        }

        let attention = try Gemma4Ops.expandSingleTokenGQAValue(
            v,
            headDim: layerHeadDim,
            numHeads: config.numAttentionHeads,
            numKVHeads: config.numKeyValueHeads
        )
        let attentionProjection = try await projectTensor(
            block.attnO,
            input: attention,
            cols: qRows,
            rows: hiddenSize
        )
        let postAttentionNormWeight = try readFloatVector(
            storage: block.postAttentionNorm,
            expectedElementCount: hiddenSize
        )
        let postAttention = try Gemma4Ops.rmsNorm(
            attentionProjection,
            weight: postAttentionNormWeight,
            eps: config.rmsNormEps
        )
        let afterAttention = zip(tokenHidden, postAttention).map(+)
        let ffnNormWeight = try readFloatVector(
            storage: block.ffnNorm,
            expectedElementCount: hiddenSize
        )
        let ffnInput = try Gemma4Ops.rmsNorm(
            afterAttention,
            weight: ffnNormWeight,
            eps: config.rmsNormEps
        )
        let down = try await projectGeGLUDown(
            gateTensor: block.ffnGate,
            upTensor: block.ffnUp,
            downTensor: block.ffnDown,
            input: ffnInput
        )
        let postFFNNormWeight = try readFloatVector(
            storage: block.postFFNNorm,
            expectedElementCount: hiddenSize
        )
        let postFFN = try Gemma4Ops.rmsNorm(
            down,
            weight: postFFNNormWeight,
            eps: config.rmsNormEps
        )
        let afterFFN = zip(afterAttention, postFFN).map(+)

        let pleInput = try prelude.perLayerInputSlice(layer: layer, perLayerDim: config.perLayerDim)
        return try await applyPLESideChannel(
            block: block,
            hidden: afterFFN,
            pleInput: pleInput
        )
    }

    private func computeLogits(hidden: [Float]) async throws -> [Float] {
        let outputNormWeight = try readFloatVector(
            storage: weights.outputNorm,
            expectedElementCount: config.hiddenSize
        )
        let normed = try Gemma4Ops.rmsNorm(
            hidden,
            weight: outputNormWeight,
            eps: config.rmsNormEps
        )
        var logits = try await projectTensor(
            weights.tokenEmbedding,
            input: normed,
            cols: config.hiddenSize,
            rows: config.vocabSize
        )
        let softcap = config.finalLogitSoftcapping
        if softcap > 0 {
            for index in logits.indices {
                logits[index] = tanh(logits[index] / softcap) * softcap
            }
        }
        try validateFinite(logits, label: "Gemma logits")
        return logits
    }

    private func computeGreedyToken(hidden: [Float]) async throws -> Int {
        let outputNormWeight = try readFloatVector(
            storage: weights.outputNorm,
            expectedElementCount: config.hiddenSize
        )
        let normed = try Gemma4Ops.rmsNorm(
            hidden,
            weight: outputNormWeight,
            eps: config.rmsNormEps
        )
        return try await greedyTokenFromTiedEmbedding(input: normed)
    }

    private func greedyTokenFromTiedEmbedding(input: [Float]) async throws -> Int {
        let tensor = weights.tokenEmbedding
        guard tensor.dataType == .q6_K else {
            let logits = try await projectTensor(
                tensor,
                input: input,
                cols: config.hiddenSize,
                rows: config.vocabSize
            )
            return greedyArgmax(logits)
        }
        try validateMatrixShape(tensor: tensor, cols: config.hiddenSize, rows: config.vocabSize)
        let requiredBytes = try packedByteCount(
            tensorName: tensor.name,
            rows: config.vocabSize,
            cols: config.hiddenSize,
            weightsPerBlock: 256,
            blockByteCount: 210
        )
        let weightBuffer = try makeRawTensorBuffer(storage: tensor, requiredBytes: requiredBytes)
        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma greedy LM-head buffers")
        }

        if runtimeOptions.useQ6Packed {
            try gemvKernel.encodeQ6KWeightsPacked(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                M: config.vocabSize,
                K: config.hiddenSize
            )
        } else {
            try gemvKernel.encodeQ6KWeights(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                M: config.vocabSize,
                K: config.hiddenSize
            )
        }
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: config.vocabSize)
        return try greedyArgmax(pointer: pointer, count: config.vocabSize)
    }

    private func greedyArgmax(_ logits: [Float]) -> Int {
        var maxValue: Float = -.infinity
        var maxIndex = 0
        for (index, value) in logits.enumerated() where value > maxValue {
            maxValue = value
            maxIndex = index
        }
        return maxIndex
    }

    private func greedyArgmax(pointer: UnsafePointer<Float>, count: Int) throws -> Int {
        var maxValue: Float = -.infinity
        var maxIndex = 0
        for index in 0..<count {
            let value = pointer[index]
            guard value.isFinite else {
                throw GenerationError.decodingFailed("Gemma greedy logits contain non-finite value at index \(index): \(value)")
            }
            if value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }
        return maxIndex
    }

    private func validateFinite(_ values: [Float], label: String) throws {
        guard validateRuntimeTensors else { return }
        for (index, value) in values.enumerated() where !value.isFinite {
            throw GenerationError.decodingFailed("\(label) produced non-finite value at index \(index): \(value)")
        }
    }

    private func normalizeHeadRows(
        _ values: [Float],
        headCount: Int,
        headDim: Int,
        weight: [Float]
    ) throws -> [Float] {
        guard values.count == headCount * headDim else {
            throw GenerationError.modelLoadFailed(reason: "Gemma head tensor has invalid shape")
        }
        guard weight.count == headDim else {
            throw GenerationError.modelLoadFailed(reason: "Gemma head norm has invalid shape")
        }
        var output = [Float](repeating: 0, count: values.count)
        for head in 0..<headCount {
            let start = head * headDim
            var meanSquare: Float = 0
            for col in 0..<headDim {
                let value = values[start + col]
                meanSquare += value * value
            }
            meanSquare /= Float(headDim)
            let scale = 1.0 / sqrt(meanSquare + config.rmsNormEps)
            for col in 0..<headDim {
                output[start + col] = values[start + col] * scale * weight[col]
            }
        }
        return output
    }

    private func normalizeHeadRowsUnscaled(
        _ values: [Float],
        headCount: Int,
        headDim: Int
    ) -> [Float] {
        guard values.count == headCount * headDim else {
            return values
        }
        var output = [Float](repeating: 0, count: values.count)
        for head in 0..<headCount {
            let start = head * headDim
            var meanSquare: Float = 0
            for col in 0..<headDim {
                let value = values[start + col]
                meanSquare += value * value
            }
            meanSquare /= Float(headDim)
            let scale = 1.0 / sqrt(meanSquare + config.rmsNormEps)
            for col in 0..<headDim {
                output[start + col] = values[start + col] * scale
            }
        }
        return output
    }

    private func headDim(forLayer layer: Int) -> Int {
        switch config.layerTypes[layer] {
        case .sliding:
            return config.headDim
        case .global:
            return config.globalHeadDim
        }
    }

    private func ropeTheta(forLayer layer: Int) -> Float {
        switch config.layerTypes[layer] {
        case .sliding:
            return config.ropeThetaLocal
        case .global:
            return config.ropeThetaGlobal
        }
    }

    private func rotaryFactor(forLayer layer: Int) throws -> Float {
        let rotaryDimension: Int
        let headDimension: Int
        switch config.layerTypes[layer] {
        case .sliding:
            rotaryDimension = config.localRotaryDimension
            headDimension = config.headDim
        case .global:
            if let ropeFreqs = weights.ropeFreqs {
                let values = try readFloatVector(
                    storage: ropeFreqs,
                    expectedElementCount: config.globalHeadDim / 2
                )
                let rotatedPairs = values.filter { $0 < 1.0e20 }.count
                return Float(rotatedPairs) / Float(config.globalHeadDim / 2)
            }
            rotaryDimension = config.globalRotaryDimension
            headDimension = config.globalHeadDim
        }
        return Float(rotaryDimension) / Float(headDimension)
    }

    private func projectTensor(
        _ tensor: TensorStorage,
        input: [Float],
        cols: Int,
        rows: Int
    ) async throws -> [Float] {
        try validateMatrixShape(tensor: tensor, cols: cols, rows: rows)
        guard input.count == cols else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensor.name) projection input has \(input.count) values; expected \(cols)"
            )
        }

        switch tensor.dataType {
        case .float32:
            let weightBuffer = try makeRawTensorBuffer(
                storage: tensor,
                requiredBytes: tensor.elementCount * MemoryLayout<Float>.stride
            )
            return try await gemvKernel.executeWithWeightBuffer(
                weightBuffer: weightBuffer,
                x: input,
                M: rows,
                K: cols,
                commandQueue: commandQueue
            )
        case .bfloat16:
            let weightBuffer = try makeRawTensorBuffer(
                storage: tensor,
                requiredBytes: tensor.elementCount * MemoryLayout<UInt16>.stride
            )
            return try await gemvKernel.executeBF16WeightsWithWeightBuffer(
                weightBuffer: weightBuffer,
                x: input,
                M: rows,
                K: cols,
                commandQueue: commandQueue
            )
        case .q4_K:
            let weightBuffer = try makeRawTensorBuffer(
                storage: tensor,
                requiredBytes: try packedByteCount(
                    tensorName: tensor.name,
                    rows: rows,
                    cols: cols,
                    weightsPerBlock: 256,
                    blockByteCount: 144
                )
            )
            return try await gemvKernel.executeQ4KWeightsWithWeightBuffer(
                weightBuffer: weightBuffer,
                x: input,
                M: rows,
                K: cols,
                commandQueue: commandQueue
            )
        case .q6_K:
            let weightBuffer = try makeRawTensorBuffer(
                storage: tensor,
                requiredBytes: try packedByteCount(
                    tensorName: tensor.name,
                    rows: rows,
                    cols: cols,
                    weightsPerBlock: 256,
                    blockByteCount: 210
                )
            )
            return try await gemvKernel.executeQ6KWeightsWithWeightBuffer(
                weightBuffer: weightBuffer,
                x: input,
                M: rows,
                K: cols,
                commandQueue: commandQueue
            )
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(tensor.name) projection data type \(tensor.dataType) is not supported by Gemma decoder projection"
            )
        }
    }

    private func projectTensors(
        _ requests: [ProjectionRequest],
        input: [Float]
    ) async throws -> [[Float]] {
        guard !requests.isEmpty else { return [] }
        let cols = requests[0].cols
        guard input.count == cols else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma projection input has \(input.count) values; expected \(cols)"
            )
        }
        for request in requests {
            guard request.cols == cols else {
                throw GenerationError.modelLoadFailed(reason: "Gemma batched projections must share an input width")
            }
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
        }

        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma batched projection buffers")
        }

        var outputBuffers: [MTLBuffer] = []
        outputBuffers.reserveCapacity(requests.count)
        for request in requests {
            guard let outputBuffer = device.makeBuffer(
                length: request.rows * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma projection output buffer")
            }
            outputBuffers.append(outputBuffer)
            let weightBuffer = try makeRawTensorBuffer(
                storage: request.tensor,
                requiredBytes: try projectionRequiredBytes(request)
            )
            try encodeProjection(
                request,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                commandBuffer: commandBuffer
            )
        }

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        return zip(requests, outputBuffers).map { request, outputBuffer in
            let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: request.rows)
            return Array(UnsafeBufferPointer(start: pointer, count: request.rows))
        }
    }

    private func projectGeGLUInputs(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        input: [Float]
    ) async throws -> [Float] {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let requests = [
            ProjectionRequest(tensor: gateTensor, rows: intermediateSize, cols: hiddenSize),
            ProjectionRequest(tensor: upTensor, rows: intermediateSize, cols: hiddenSize)
        ]
        guard input.count == hiddenSize else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma FFN input has \(input.count) values; expected \(hiddenSize)"
            )
        }
        for request in requests {
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
        }

        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let gateBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let upBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let activatedBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma GeGLU projection buffers")
        }

        let outputBuffers = [gateBuffer, upBuffer]
        for (request, outputBuffer) in zip(requests, outputBuffers) {
            let weightBuffer = try makeRawTensorBuffer(
                storage: request.tensor,
                requiredBytes: try projectionRequiredBytes(request)
            )
            try encodeProjection(
                request,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                commandBuffer: commandBuffer
            )
        }

        try gegluKernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            outputBuffer: activatedBuffer,
            count: intermediateSize
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = activatedBuffer.contents().bindMemory(to: Float.self, capacity: intermediateSize)
        return Array(UnsafeBufferPointer(start: pointer, count: intermediateSize))
    }

    private func projectGeGLUDown(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        downTensor: TensorStorage,
        input: [Float]
    ) async throws -> [Float] {
        let downBuffer = try await projectGeGLUDownBuffer(
            gateTensor: gateTensor,
            upTensor: upTensor,
            downTensor: downTensor,
            input: input
        )
        let pointer = downBuffer.contents().bindMemory(to: Float.self, capacity: config.hiddenSize)
        return Array(UnsafeBufferPointer(start: pointer, count: config.hiddenSize))
    }

    private func projectGeGLUDown(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        downTensor: TensorStorage,
        input: [Float],
        scratch: Gemma4Scratch
    ) async throws -> [Float] {
        await scratchGate.acquire()
        do {
            try scratch.copyFFNInput(input)
            let downBuffer = try await projectGeGLUDownBuffer(
                gateTensor: gateTensor,
                upTensor: upTensor,
                downTensor: downTensor,
                inputBuffer: scratch.ffnInput,
                gateBuffer: scratch.ffnGate,
                upBuffer: scratch.ffnUp,
                activatedBuffer: scratch.ffnActivated,
                downBuffer: scratch.ffnDown
            )
            let pointer = downBuffer.contents().bindMemory(to: Float.self, capacity: config.hiddenSize)
            let output = Array(UnsafeBufferPointer(start: pointer, count: config.hiddenSize))
            await scratchGate.release()
            return output
        } catch {
            await scratchGate.release()
            throw error
        }
    }

    private func projectGeGLUDownBuffer(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        downTensor: TensorStorage,
        input: [Float]
    ) async throws -> MTLBuffer {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let gateRequest = ProjectionRequest(tensor: gateTensor, rows: intermediateSize, cols: hiddenSize)
        let upRequest = ProjectionRequest(tensor: upTensor, rows: intermediateSize, cols: hiddenSize)
        let downRequest = ProjectionRequest(tensor: downTensor, rows: hiddenSize, cols: intermediateSize)
        guard input.count == hiddenSize else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma FFN input has \(input.count) values; expected \(hiddenSize)"
            )
        }
        for request in [gateRequest, upRequest, downRequest] {
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
        }

        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let gateBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let upBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let activatedBuffer = device.makeBuffer(
            length: intermediateSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let downBuffer = device.makeBuffer(
            length: hiddenSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma fused FFN buffers")
        }

        return try await projectGeGLUDownBuffer(
            gateTensor: gateTensor,
            upTensor: upTensor,
            downTensor: downTensor,
            inputBuffer: inputBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            activatedBuffer: activatedBuffer,
            downBuffer: downBuffer
        )
    }

    private func projectGeGLUDownBuffer(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        downTensor: TensorStorage,
        inputBuffer: MTLBuffer,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        downBuffer: MTLBuffer
    ) async throws -> MTLBuffer {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let gateRequest = ProjectionRequest(tensor: gateTensor, rows: intermediateSize, cols: hiddenSize)
        let upRequest = ProjectionRequest(tensor: upTensor, rows: intermediateSize, cols: hiddenSize)
        let downRequest = ProjectionRequest(tensor: downTensor, rows: hiddenSize, cols: intermediateSize)
        for request in [gateRequest, upRequest, downRequest] {
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
        }
        let f32 = MemoryLayout<Float>.stride
        guard inputBuffer.length >= hiddenSize * f32,
              gateBuffer.length >= intermediateSize * f32,
              upBuffer.length >= intermediateSize * f32,
              activatedBuffer.length >= intermediateSize * f32,
              downBuffer.length >= hiddenSize * f32,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma fused FFN buffers")
        }

        try encodeGeGLUDown(
            gateTensor: gateTensor,
            upTensor: upTensor,
            downTensor: downTensor,
            inputBuffer: inputBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            activatedBuffer: activatedBuffer,
            downBuffer: downBuffer,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        return downBuffer
    }

    private func encodeGeGLUDown(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        downTensor: TensorStorage,
        inputBuffer: MTLBuffer,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        downBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        try encodeGeGLUActivation(
            gateTensor: gateTensor,
            upTensor: upTensor,
            inputBuffer: inputBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            activatedBuffer: activatedBuffer,
            commandBuffer: commandBuffer
        )
        try encodeGeGLUDownProjection(
            downTensor: downTensor,
            activatedBuffer: activatedBuffer,
            downBuffer: downBuffer,
            commandBuffer: commandBuffer
        )
    }

    private func encodeGeGLUActivation(
        gateTensor: TensorStorage,
        upTensor: TensorStorage,
        inputBuffer: MTLBuffer,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        encodeActivation: Bool = true,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let gateRequest = ProjectionRequest(tensor: gateTensor, rows: intermediateSize, cols: hiddenSize)
        let upRequest = ProjectionRequest(tensor: upTensor, rows: intermediateSize, cols: hiddenSize)
        for request in [gateRequest, upRequest] {
            try validateMatrixShape(tensor: request.tensor, cols: request.cols, rows: request.rows)
        }

        if gateRequest.tensor.dataType == .q4_K,
           upRequest.tensor.dataType == .q4_K {
            let gateWeightBuffer = try makeRawTensorBuffer(
                storage: gateRequest.tensor,
                requiredBytes: try projectionRequiredBytes(gateRequest)
            )
            let upWeightBuffer = try makeRawTensorBuffer(
                storage: upRequest.tensor,
                requiredBytes: try projectionRequiredBytes(upRequest)
            )
            if runtimeOptions.useQ4Packed {
                if runtimeOptions.useQ4LlamaStyleDual {
                    try gemvKernel.encodeQ4KWeightsLlamaStyleDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: gateWeightBuffer,
                        weightBufferB: upWeightBuffer,
                        inputBuffer: inputBuffer,
                        outputBufferA: gateBuffer,
                        outputBufferB: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    if encodeActivation {
                        try gegluKernel.encode(
                            commandBuffer: commandBuffer,
                            gateBuffer: gateBuffer,
                            upBuffer: upBuffer,
                            outputBuffer: activatedBuffer,
                            count: intermediateSize
                        )
                    }
                    return
                }
                if runtimeOptions.useQ4Tiled {
                    try gemvKernel.encodeQ4KWeightsPackedFourRows(
                        commandBuffer: commandBuffer,
                        weightBuffer: gateWeightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: gateBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    try gemvKernel.encodeQ4KWeightsPackedFourRows(
                        commandBuffer: commandBuffer,
                        weightBuffer: upWeightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                } else {
                    try gemvKernel.encodeQ4KWeightsPacked(
                        commandBuffer: commandBuffer,
                        weightBuffer: gateWeightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: gateBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    try gemvKernel.encodeQ4KWeightsPacked(
                        commandBuffer: commandBuffer,
                        weightBuffer: upWeightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                }
                if encodeActivation {
                    try gegluKernel.encode(
                        commandBuffer: commandBuffer,
                        gateBuffer: gateBuffer,
                        upBuffer: upBuffer,
                        outputBuffer: activatedBuffer,
                        count: intermediateSize
                    )
                }
            } else if encodeActivation,
                      runtimeOptions.useQ4FusedGeGLU {
                try gemvKernel.encodeQ4KWeightsDualGeGLU(
                    commandBuffer: commandBuffer,
                    gateWeightBuffer: gateWeightBuffer,
                    upWeightBuffer: upWeightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: activatedBuffer,
                    M: intermediateSize,
                    K: hiddenSize
                )
            } else {
                try gemvKernel.encodeQ4KWeightsDual(
                    commandBuffer: commandBuffer,
                    weightBufferA: gateWeightBuffer,
                    weightBufferB: upWeightBuffer,
                    inputBuffer: inputBuffer,
                    outputBufferA: gateBuffer,
                    outputBufferB: upBuffer,
                    M: intermediateSize,
                    K: hiddenSize
                )

                if encodeActivation {
                    try gegluKernel.encode(
                        commandBuffer: commandBuffer,
                        gateBuffer: gateBuffer,
                        upBuffer: upBuffer,
                        outputBuffer: activatedBuffer,
                        count: intermediateSize
                    )
                }
            }
        } else {
            let gateWeightBuffer = try makeRawTensorBuffer(
                storage: gateRequest.tensor,
                requiredBytes: try projectionRequiredBytes(gateRequest)
            )
            try encodeProjection(
                gateRequest,
                weightBuffer: gateWeightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: gateBuffer,
                commandBuffer: commandBuffer
            )

            let upWeightBuffer = try makeRawTensorBuffer(
                storage: upRequest.tensor,
                requiredBytes: try projectionRequiredBytes(upRequest)
            )
            try encodeProjection(
                upRequest,
                weightBuffer: upWeightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: upBuffer,
                commandBuffer: commandBuffer
            )

            if encodeActivation {
                try gegluKernel.encode(
                    commandBuffer: commandBuffer,
                    gateBuffer: gateBuffer,
                    upBuffer: upBuffer,
                    outputBuffer: activatedBuffer,
                    count: intermediateSize
                )
            }
        }
    }

    private func encodeGeGLUActivation(
        gate: ProjectionResource,
        up: ProjectionResource,
        inputBuffer: MTLBuffer,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        encodeActivation: Bool = true,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = gate.request.cols
        let intermediateSize = gate.request.rows

        if gate.request.tensor.dataType == .q4_K,
           up.request.tensor.dataType == .q4_K {
            if runtimeOptions.useQ4Packed {
                if runtimeOptions.useQ4LlamaStyleDual {
                    try gemvKernel.encodeQ4KWeightsLlamaStyleDual(
                        commandBuffer: commandBuffer,
                        weightBufferA: gate.weightBuffer,
                        weightBufferB: up.weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBufferA: gateBuffer,
                        outputBufferB: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    if encodeActivation {
                        try gegluKernel.encode(
                            commandBuffer: commandBuffer,
                            gateBuffer: gateBuffer,
                            upBuffer: upBuffer,
                            outputBuffer: activatedBuffer,
                            count: intermediateSize
                        )
                    }
                    return
                }
                if runtimeOptions.useQ4Tiled {
                    try gemvKernel.encodeQ4KWeightsPackedFourRows(
                        commandBuffer: commandBuffer,
                        weightBuffer: gate.weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: gateBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    try gemvKernel.encodeQ4KWeightsPackedFourRows(
                        commandBuffer: commandBuffer,
                        weightBuffer: up.weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                } else {
                    try gemvKernel.encodeQ4KWeightsPacked(
                        commandBuffer: commandBuffer,
                        weightBuffer: gate.weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: gateBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                    try gemvKernel.encodeQ4KWeightsPacked(
                        commandBuffer: commandBuffer,
                        weightBuffer: up.weightBuffer,
                        inputBuffer: inputBuffer,
                        outputBuffer: upBuffer,
                        M: intermediateSize,
                        K: hiddenSize
                    )
                }
                if encodeActivation {
                    try gegluKernel.encode(
                        commandBuffer: commandBuffer,
                        gateBuffer: gateBuffer,
                        upBuffer: upBuffer,
                        outputBuffer: activatedBuffer,
                        count: intermediateSize
                    )
                }
            } else if encodeActivation,
                      runtimeOptions.useQ4FusedGeGLU {
                try gemvKernel.encodeQ4KWeightsDualGeGLU(
                    commandBuffer: commandBuffer,
                    gateWeightBuffer: gate.weightBuffer,
                    upWeightBuffer: up.weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: activatedBuffer,
                    M: intermediateSize,
                    K: hiddenSize
                )
            } else {
                try gemvKernel.encodeQ4KWeightsDual(
                    commandBuffer: commandBuffer,
                    weightBufferA: gate.weightBuffer,
                    weightBufferB: up.weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBufferA: gateBuffer,
                    outputBufferB: upBuffer,
                    M: intermediateSize,
                    K: hiddenSize
                )

                if encodeActivation {
                    try gegluKernel.encode(
                        commandBuffer: commandBuffer,
                        gateBuffer: gateBuffer,
                        upBuffer: upBuffer,
                        outputBuffer: activatedBuffer,
                        count: intermediateSize
                    )
                }
            }
        } else {
            try encodeProjection(
                gate.request,
                weightBuffer: gate.weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: gateBuffer,
                commandBuffer: commandBuffer
            )
            try encodeProjection(
                up.request,
                weightBuffer: up.weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: upBuffer,
                commandBuffer: commandBuffer
            )

            if encodeActivation {
                try gegluKernel.encode(
                    commandBuffer: commandBuffer,
                    gateBuffer: gateBuffer,
                    upBuffer: upBuffer,
                    outputBuffer: activatedBuffer,
                    count: intermediateSize
                )
            }
        }
    }

    private func encodeGeGLUDownProjection(
        downTensor: TensorStorage,
        activatedBuffer: MTLBuffer,
        downBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let downRequest = ProjectionRequest(tensor: downTensor, rows: hiddenSize, cols: intermediateSize)
        try validateMatrixShape(tensor: downRequest.tensor, cols: downRequest.cols, rows: downRequest.rows)
        let downWeightBuffer = try makeRawTensorBuffer(
            storage: downRequest.tensor,
            requiredBytes: try projectionRequiredBytes(downRequest)
        )
        try encodeProjection(
            downRequest,
            weightBuffer: downWeightBuffer,
            inputBuffer: activatedBuffer,
            outputBuffer: downBuffer,
            commandBuffer: commandBuffer
        )
    }

    private func encodeGeGLUDownProjection(
        down: ProjectionResource,
        activatedBuffer: MTLBuffer,
        downBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        try encodeProjection(
            down.request,
            weightBuffer: down.weightBuffer,
            inputBuffer: activatedBuffer,
            outputBuffer: downBuffer,
            commandBuffer: commandBuffer
        )
    }

    private func applyPLESideChannel(
        block: Gemma4BlockWeights,
        hidden: [Float],
        pleInput: [Float]
    ) async throws -> [Float] {
        guard let hiddenBuffer = device.makeBuffer(
            bytes: hidden,
            length: hidden.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let pleInputBuffer = device.makeBuffer(
            bytes: pleInput,
            length: pleInput.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma PLE input buffers")
        }
        return try await applyPLESideChannel(
            block: block,
            hiddenBuffer: hiddenBuffer,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: 0,
            scratch: nil
        )
    }

    private func applyPLESideChannel(
        block: Gemma4BlockWeights,
        hidden: [Float],
        pleInput: [Float],
        scratch: Gemma4Scratch
    ) async throws -> [Float] {
        await scratchGate.acquire()
        do {
            try scratch.copyHidden(hidden)
            try scratch.copyPLEInput(pleInput)
            let output = try await applyPLESideChannel(
                block: block,
                hiddenBuffer: scratch.currentHidden,
                pleInputBuffer: scratch.pleInput,
                pleInputOffset: 0,
                scratch: scratch
            )
            await scratchGate.release()
            return output
        } catch {
            await scratchGate.release()
            throw error
        }
    }

    private func applyPLESideChannel(
        block: Gemma4BlockWeights,
        hidden: [Float],
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) async throws -> [Float] {
        await scratchGate.acquire()
        do {
            try scratch.copyHidden(hidden)
            let output = try await applyPLESideChannel(
                block: block,
                hiddenBuffer: scratch.currentHidden,
                pleInputBuffer: pleInputBuffer,
                pleInputOffset: pleInputOffset,
                scratch: scratch
            )
            await scratchGate.release()
            return output
        } catch {
            await scratchGate.release()
            throw error
        }
    }

    private func applyPLESideChannel(
        block: Gemma4BlockWeights,
        hiddenBuffer: MTLBuffer,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch?
    ) async throws -> [Float] {
        let hiddenSize = config.hiddenSize
        let perLayerDim = config.perLayerDim
        let f32 = MemoryLayout<Float>.stride
        guard hiddenBuffer.length >= hiddenSize * MemoryLayout<Float>.stride else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma PLE hidden input buffer is too small; expected \(hiddenSize) Float values"
            )
        }
        guard pleInputOffset >= 0,
              pleInputBuffer.length >= pleInputOffset + perLayerDim * f32 else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma PLE input buffer is too small for layer slice"
            )
        }

        let gateRequest = ProjectionRequest(
            tensor: block.perLayerInputGate,
            rows: perLayerDim,
            cols: hiddenSize
        )
        let projectionRequest = ProjectionRequest(
            tensor: block.perLayerProjection,
            rows: hiddenSize,
            cols: perLayerDim
        )
        try validateMatrixShape(tensor: gateRequest.tensor, cols: gateRequest.cols, rows: gateRequest.rows)
        try validateMatrixShape(
            tensor: projectionRequest.tensor,
            cols: projectionRequest.cols,
            rows: projectionRequest.rows
        )
        let gateBuffer: MTLBuffer
        let activatedBuffer: MTLBuffer
        let projectionBuffer: MTLBuffer
        if let scratch {
            gateBuffer = scratch.pleGate
            activatedBuffer = scratch.pleActivated
            projectionBuffer = scratch.pleProjection
        } else {
            guard let allocatedGateBuffer = device.makeBuffer(
                length: perLayerDim * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
            let allocatedActivatedBuffer = device.makeBuffer(
                length: perLayerDim * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
            let allocatedProjectionBuffer = device.makeBuffer(
                length: hiddenSize * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma PLE side-channel buffers")
            }
            gateBuffer = allocatedGateBuffer
            activatedBuffer = allocatedActivatedBuffer
            projectionBuffer = allocatedProjectionBuffer
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma PLE side-channel buffers")
        }

        try encodePLESideChannel(
            commandBuffer: commandBuffer,
            block: block,
            hiddenBuffer: hiddenBuffer,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            gateBuffer: gateBuffer,
            activatedBuffer: activatedBuffer,
            projectionBuffer: projectionBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: hiddenSize)
        let output = Array(UnsafeBufferPointer(start: pointer, count: hiddenSize))
        try validateFinite(output, label: "Gemma PLE side-channel")
        return output
    }

    private func encodePLESideChannel(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        hiddenBuffer: MTLBuffer,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodePLESideChannel(
            commandBuffer: commandBuffer,
            resources: resources,
            hiddenBuffer: hiddenBuffer,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            gateBuffer: scratch.pleGate,
            activatedBuffer: scratch.pleActivated,
            projectionBuffer: scratch.pleProjection
        )
    }

    private func encodePLESideChannel(
        commandBuffer: MTLCommandBuffer,
        resources: LayerRuntimeResources,
        hiddenBuffer: MTLBuffer,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        gateBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        projectionBuffer: MTLBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let perLayerDim = config.perLayerDim
        let f32 = MemoryLayout<Float>.stride
        guard hiddenBuffer.length >= hiddenSize * f32,
              pleInputOffset >= 0,
              pleInputBuffer.length >= pleInputOffset + perLayerDim * f32,
              gateBuffer.length >= perLayerDim * f32,
              activatedBuffer.length >= perLayerDim * f32,
              projectionBuffer.length >= hiddenSize * f32 else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE side-channel buffer is too small")
        }

        try encodeProjection(
            resources.pleGate.request,
            weightBuffer: resources.pleGate.weightBuffer,
            inputBuffer: hiddenBuffer,
            outputBuffer: gateBuffer,
            commandBuffer: commandBuffer
        )
        try pleGateKernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            pleBuffer: pleInputBuffer,
            outputBuffer: activatedBuffer,
            count: perLayerDim,
            pleBufferOffset: pleInputOffset
        )

        try encodeProjection(
            resources.pleProjection.request,
            weightBuffer: resources.pleProjection.weightBuffer,
            inputBuffer: activatedBuffer,
            outputBuffer: projectionBuffer,
            commandBuffer: commandBuffer
        )

        try pleSideChannelKernel.encode(
            commandBuffer: commandBuffer,
            hiddenBuffer: hiddenBuffer,
            projectionBuffer: projectionBuffer,
            postNormWeightBuffer: resources.plePostNormWeightBuffer,
            hiddenSize: hiddenSize,
            batchSeq: 1,
            rmsEps: config.rmsNormEps
        )
        try gemmaDecodeKernels.encodeMulScalar(
            commandBuffer: commandBuffer,
            valuesBuffer: hiddenBuffer,
            scale: resources.layerOutputScale,
            count: hiddenSize
        )
    }

    private func encodePLESideChannel(
        commandBuffer: MTLCommandBuffer,
        block: Gemma4BlockWeights,
        hiddenBuffer: MTLBuffer,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        scratch: Gemma4Scratch
    ) throws {
        try encodePLESideChannel(
            commandBuffer: commandBuffer,
            block: block,
            hiddenBuffer: hiddenBuffer,
            pleInputBuffer: pleInputBuffer,
            pleInputOffset: pleInputOffset,
            gateBuffer: scratch.pleGate,
            activatedBuffer: scratch.pleActivated,
            projectionBuffer: scratch.pleProjection
        )
    }

    private func encodePLESideChannel(
        commandBuffer: MTLCommandBuffer,
        block: Gemma4BlockWeights,
        hiddenBuffer: MTLBuffer,
        pleInputBuffer: MTLBuffer,
        pleInputOffset: Int,
        gateBuffer: MTLBuffer,
        activatedBuffer: MTLBuffer,
        projectionBuffer: MTLBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let perLayerDim = config.perLayerDim
        let f32 = MemoryLayout<Float>.stride
        guard hiddenBuffer.length >= hiddenSize * f32,
              pleInputOffset >= 0,
              pleInputBuffer.length >= pleInputOffset + perLayerDim * f32,
              gateBuffer.length >= perLayerDim * f32,
              activatedBuffer.length >= perLayerDim * f32,
              projectionBuffer.length >= hiddenSize * f32 else {
            throw GenerationError.modelLoadFailed(reason: "Gemma PLE side-channel buffer is too small")
        }

        let gateRequest = ProjectionRequest(
            tensor: block.perLayerInputGate,
            rows: perLayerDim,
            cols: hiddenSize
        )
        let projectionRequest = ProjectionRequest(
            tensor: block.perLayerProjection,
            rows: hiddenSize,
            cols: perLayerDim
        )
        try validateMatrixShape(tensor: gateRequest.tensor, cols: gateRequest.cols, rows: gateRequest.rows)
        try validateMatrixShape(
            tensor: projectionRequest.tensor,
            cols: projectionRequest.cols,
            rows: projectionRequest.rows
        )

        let gateWeightBuffer = try makeRawTensorBuffer(
            storage: gateRequest.tensor,
            requiredBytes: try projectionRequiredBytes(gateRequest)
        )
        try encodeProjection(
            gateRequest,
            weightBuffer: gateWeightBuffer,
            inputBuffer: hiddenBuffer,
            outputBuffer: gateBuffer,
            commandBuffer: commandBuffer
        )
        try pleGateKernel.encode(
            commandBuffer: commandBuffer,
            gateBuffer: gateBuffer,
            pleBuffer: pleInputBuffer,
            outputBuffer: activatedBuffer,
            count: perLayerDim,
            pleBufferOffset: pleInputOffset
        )

        let projectionWeightBuffer = try makeRawTensorBuffer(
            storage: projectionRequest.tensor,
            requiredBytes: try projectionRequiredBytes(projectionRequest)
        )
        try encodeProjection(
            projectionRequest,
            weightBuffer: projectionWeightBuffer,
            inputBuffer: activatedBuffer,
            outputBuffer: projectionBuffer,
            commandBuffer: commandBuffer
        )

        let postNormWeightBuffer = try makeFloatBuffer(
            storage: block.postPerLayerInputNorm,
            expectedElementCount: hiddenSize
        )
        try pleSideChannelKernel.encode(
            commandBuffer: commandBuffer,
            hiddenBuffer: hiddenBuffer,
            projectionBuffer: projectionBuffer,
            postNormWeightBuffer: postNormWeightBuffer,
            hiddenSize: hiddenSize,
            batchSeq: 1,
            rmsEps: config.rmsNormEps
        )
        let layerOutputScale = try readFloatVector(
            storage: block.layerOutputScale,
            expectedElementCount: 1
        )[0]
        try gemmaDecodeKernels.encodeMulScalar(
            commandBuffer: commandBuffer,
            valuesBuffer: hiddenBuffer,
            scale: layerOutputScale,
            count: hiddenSize
        )
    }

    private func projectionRequiredBytes(_ request: ProjectionRequest) throws -> Int {
        switch request.tensor.dataType {
        case .float32:
            return request.tensor.elementCount * MemoryLayout<Float>.stride
        case .bfloat16:
            return request.tensor.elementCount * MemoryLayout<UInt16>.stride
        case .q4_K:
            return try packedByteCount(
                tensorName: request.tensor.name,
                rows: request.rows,
                cols: request.cols,
                weightsPerBlock: 256,
                blockByteCount: 144
            )
        case .q6_K:
            return try packedByteCount(
                tensorName: request.tensor.name,
                rows: request.rows,
                cols: request.cols,
                weightsPerBlock: 256,
                blockByteCount: 210
            )
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(request.tensor.name) projection data type \(request.tensor.dataType) is not supported by Gemma decoder projection"
            )
        }
    }

    private func encodeProjection(
        _ request: ProjectionRequest,
        weightBuffer: MTLBuffer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        switch request.tensor.dataType {
        case .float32:
            try gemvKernel.encode(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                M: request.rows,
                K: request.cols
            )
        case .bfloat16:
            try gemvKernel.encodeBF16Weights(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                M: request.rows,
                K: request.cols
            )
        case .q4_K:
            if runtimeOptions.useQ4LlamaStyle {
                try gemvKernel.encodeQ4KWeightsLlamaStyle(
                    commandBuffer: commandBuffer,
                    weightBuffer: weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: outputBuffer,
                    M: request.rows,
                    K: request.cols
                )
            } else if runtimeOptions.useQ4Packed {
                try gemvKernel.encodeQ4KWeightsPacked(
                    commandBuffer: commandBuffer,
                    weightBuffer: weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: outputBuffer,
                    M: request.rows,
                    K: request.cols
                )
            } else if runtimeOptions.useQ4TwoRow,
               request.rows >= 512 {
                try gemvKernel.encodeQ4KWeightsTwoRows(
                    commandBuffer: commandBuffer,
                    weightBuffer: weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: outputBuffer,
                    M: request.rows,
                    K: request.cols
                )
            } else {
                try gemvKernel.encodeQ4KWeights(
                    commandBuffer: commandBuffer,
                    weightBuffer: weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: outputBuffer,
                    M: request.rows,
                    K: request.cols
                )
            }
        case .q6_K:
            try gemvKernel.encodeQ6KWeights(
                commandBuffer: commandBuffer,
                weightBuffer: weightBuffer,
                inputBuffer: inputBuffer,
                outputBuffer: outputBuffer,
                M: request.rows,
                K: request.cols
            )
        default:
            throw GenerationError.modelLoadFailed(
                reason: "\(request.tensor.name) projection data type \(request.tensor.dataType) is not supported by Gemma decoder projection"
            )
        }
    }

    private func makePLEGatherBuffer(
        tokenIDs: [Int],
        perLayerDim: Int,
        numLayers: Int
    ) throws -> MTLBuffer {
        let tokenIDs32 = try tokenIDs.map { tokenID -> Int32 in
            guard let value = Int32(exactly: tokenID) else {
                throw GenerationError.decodingFailed("Gemma 4 token id \(tokenID) cannot fit in Int32")
            }
            return value
        }
        let outputCount = tokenIDs.count * perLayerDim * numLayers
        guard let tokenBuffer = device.makeBuffer(
            bytes: tokenIDs32,
            length: tokenIDs32.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma PLE gather buffers")
        }

        let table = weights.perLayerTokenEmbed
        let totalElemsPerRow = perLayerDim * numLayers
        switch table.dataType {
        case .q8_0:
            let rowStrideBytes = try rowStrideBytes(
                rowWidth: totalElemsPerRow,
                weightsPerBlock: 32,
                blockByteCount: 34,
                tensorName: table.name
            )
            try pleGatherKernel.encode(
                commandBuffer: commandBuffer,
                q8TableBuffer: table.buffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: outputBuffer,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                numTokens: tokenIDs.count,
                rowStrideBytes: rowStrideBytes,
                tableByteOffset: table.byteOffset
            )
        case .q6_K:
            let rowStrideBytes = try rowStrideBytes(
                rowWidth: totalElemsPerRow,
                weightsPerBlock: 256,
                blockByteCount: 210,
                tensorName: table.name
            )
            try pleGatherKernel.encodeQ6K(
                commandBuffer: commandBuffer,
                q6KTableBuffer: table.buffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: outputBuffer,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                numTokens: tokenIDs.count,
                rowStrideBytes: rowStrideBytes,
                tableByteOffset: table.byteOffset
            )
        default:
            throw GenerationError.decodingFailed(
                "Gemma 4 PLE gather supports Q8_0 and Q6_K, got \(table.dataType) for \(table.name)"
            )
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        return outputBuffer
    }

    private func encodePLEGatherRows(
        commandBuffer: MTLCommandBuffer,
        tokenBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        tokenCount: Int,
        perLayerDim: Int,
        numLayers: Int
    ) throws {
        let table = weights.perLayerTokenEmbed
        let totalElemsPerRow = perLayerDim * numLayers
        switch table.dataType {
        case .q8_0:
            let rowStrideBytes = try rowStrideBytes(
                rowWidth: totalElemsPerRow,
                weightsPerBlock: 32,
                blockByteCount: 34,
                tensorName: table.name
            )
            try pleGatherKernel.encode(
                commandBuffer: commandBuffer,
                q8TableBuffer: table.buffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: outputBuffer,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                numTokens: tokenCount,
                rowStrideBytes: rowStrideBytes,
                tableByteOffset: table.byteOffset
            )
        case .q6_K:
            let rowStrideBytes = try rowStrideBytes(
                rowWidth: totalElemsPerRow,
                weightsPerBlock: 256,
                blockByteCount: 210,
                tensorName: table.name
            )
            try pleGatherKernel.encodeQ6K(
                commandBuffer: commandBuffer,
                q6KTableBuffer: table.buffer,
                tokenBuffer: tokenBuffer,
                outputBuffer: outputBuffer,
                perLayerDim: perLayerDim,
                numLayers: numLayers,
                numTokens: tokenCount,
                rowStrideBytes: rowStrideBytes,
                tableByteOffset: table.byteOffset
            )
        default:
            throw GenerationError.decodingFailed(
                "Gemma 4 PLE gather supports Q8_0 and Q6_K, got \(table.dataType) for \(table.name)"
            )
        }
    }

    private func projectPLEInputs(
        hidden: [Float],
        batchSeq: Int,
        hiddenSize: Int,
        projectionDim: Int
    ) async throws -> [Float] {
        let projection = weights.perLayerModelProjection
        try validateMatrixShape(
            tensor: projection,
            cols: hiddenSize,
            rows: projectionDim
        )

        var output = [Float](repeating: 0, count: batchSeq * projectionDim)
        let outputScale = 1.0 / sqrt(Float(hiddenSize))
        switch projection.dataType {
        case .bfloat16:
            let weightBuffer = try makeRawTensorBuffer(
                storage: projection,
                requiredBytes: projection.elementCount * MemoryLayout<UInt16>.stride
            )
            if batchSeq > 1 {
                guard let inputBuffer = device.makeBuffer(
                    bytes: hidden,
                    length: hidden.count * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                ),
                let outputBuffer = device.makeBuffer(
                    length: batchSeq * projectionDim * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                ),
                let commandBuffer = commandQueue.makeCommandBuffer() else {
                    throw GenerationError.modelLoadFailed(reason: "Failed to allocate Gemma batched PLE projection buffers")
                }
                try gemvKernel.encodeBatchedBF16Weights(
                    commandBuffer: commandBuffer,
                    weightBuffer: weightBuffer,
                    inputBuffer: inputBuffer,
                    outputBuffer: outputBuffer,
                    batchSeq: batchSeq,
                    M: projectionDim,
                    K: hiddenSize
                )
                commandBuffer.commit()
                await commandBuffer.completed()
                if let error = commandBuffer.error {
                    throw error
                }
                let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
                output = Array(UnsafeBufferPointer(start: pointer, count: output.count))
                for index in output.indices {
                    output[index] *= outputScale
                }
            } else {
                let projected = try await gemvKernel.executeBF16WeightsWithWeightBuffer(
                    weightBuffer: weightBuffer,
                    x: hidden,
                    M: projectionDim,
                    K: hiddenSize,
                    commandQueue: commandQueue
                )
                for index in 0..<projectionDim {
                    output[index] = projected[index] * outputScale
                }
            }
        case .float32:
            let weightBuffer = try makeRawTensorBuffer(
                storage: projection,
                requiredBytes: projection.elementCount * MemoryLayout<Float>.stride
            )
            for tokenIndex in 0..<batchSeq {
                let start = tokenIndex * hiddenSize
                let tokenHidden = Array(hidden[start..<(start + hiddenSize)])
                let projected = try await gemvKernel.executeWithWeightBuffer(
                    weightBuffer: weightBuffer,
                    x: tokenHidden,
                    M: projectionDim,
                    K: hiddenSize,
                    commandQueue: commandQueue
                )
                let outputStart = tokenIndex * projectionDim
                for index in 0..<projectionDim {
                    output[outputStart + index] = projected[index] * outputScale
                }
            }
        default:
            throw GenerationError.decodingFailed(
                "Gemma 4 PLE projection supports BF16/F32 weights, got \(projection.dataType) for \(projection.name)"
            )
        }
        return output
    }

    private func validateMatrixShape(tensor: TensorStorage, cols: Int, rows: Int) throws {
        guard tensor.shape.count == 2,
              tensor.shape[0] == cols,
              tensor.shape[1] == rows else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensor.name) shape \(tensor.shape) must be [\(cols), \(rows)]"
            )
        }
    }

    private func rowStrideBytes(
        rowWidth: Int,
        weightsPerBlock: Int,
        blockByteCount: Int,
        tensorName: String
    ) throws -> Int {
        guard rowWidth % weightsPerBlock == 0 else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensorName) row width \(rowWidth) must be divisible by \(weightsPerBlock)"
            )
        }
        return (rowWidth / weightsPerBlock) * blockByteCount
    }

    private func packedByteCount(
        tensorName: String,
        rows: Int,
        cols: Int,
        weightsPerBlock: Int,
        blockByteCount: Int
    ) throws -> Int {
        guard cols % weightsPerBlock == 0 else {
            throw GenerationError.modelLoadFailed(
                reason: "\(tensorName) column count \(cols) must be divisible by \(weightsPerBlock)"
            )
        }
        return rows * (cols / weightsPerBlock) * blockByteCount
    }

    private func makeRawTensorBuffer(storage: TensorStorage, requiredBytes: Int) throws -> MTLBuffer {
        try runtimeCache.rawTensorBuffer(
            storage: storage,
            requiredBytes: requiredBytes,
            device: device
        )
    }

    private func fillEmbeddingRows(
        storage: TensorStorage,
        tokenIDs: [Int],
        vocabSize: Int,
        rowWidth: Int,
        into destination: inout [Float]
    ) throws {
        guard destination.count == tokenIDs.count * rowWidth else {
            throw GenerationError.modelLoadFailed(
                reason: "Gemma embedding destination has \(destination.count) values; expected \(tokenIDs.count * rowWidth)"
            )
        }
        guard storage.elementCount >= vocabSize * rowWidth else {
            throw GenerationError.modelLoadFailed(
                reason: "\(storage.name) has \(storage.elementCount) elements; expected at least \(vocabSize * rowWidth)"
            )
        }

        let basePtr = storage.buffer.contents() + storage.byteOffset
        try destination.withUnsafeMutableBufferPointer { dstBuffer in
            guard let destinationBase = dstBuffer.baseAddress else { return }
            switch storage.dataType {
            case .float32:
                let ptr = basePtr.bindMemory(to: Float.self, capacity: storage.elementCount)
                for (index, tokenID) in tokenIDs.enumerated() {
                    memcpy(
                        destinationBase + index * rowWidth,
                        ptr + tokenID * rowWidth,
                        rowWidth * MemoryLayout<Float>.stride
                    )
                }
            case .float16:
                let ptr = basePtr.bindMemory(to: Float16.self, capacity: storage.elementCount)
                for (index, tokenID) in tokenIDs.enumerated() {
                    let srcOffset = tokenID * rowWidth
                    let dstOffset = index * rowWidth
                    for col in 0..<rowWidth {
                        destinationBase[dstOffset + col] = Float(ptr[srcOffset + col])
                    }
                }
            case .bfloat16:
                let ptr = basePtr.bindMemory(to: UInt16.self, capacity: storage.elementCount)
                for (index, tokenID) in tokenIDs.enumerated() {
                    let srcOffset = tokenID * rowWidth
                    let dstOffset = index * rowWidth
                    for col in 0..<rowWidth {
                        destinationBase[dstOffset + col] = Float(bitPattern: UInt32(ptr[srcOffset + col]) << 16)
                    }
                }
            case .q8_0:
                try fillQ8EmbeddingRows(
                    storage: storage,
                    basePtr: basePtr,
                    tokenIDs: tokenIDs,
                    rowWidth: rowWidth,
                    destination: destinationBase
                )
            case .q4_0:
                try fillQ4_0EmbeddingRows(
                    storage: storage,
                    basePtr: basePtr,
                    tokenIDs: tokenIDs,
                    rowWidth: rowWidth,
                    destination: destinationBase
                )
            case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K:
                try fillKQuantEmbeddingRows(
                    storage: storage,
                    basePtr: basePtr,
                    tokenIDs: tokenIDs,
                    rowWidth: rowWidth,
                    destination: destinationBase
                )
            default:
                throw GenerationError.modelLoadFailed(
                    reason: "Unsupported Gemma token embedding data type \(storage.dataType)"
                )
            }
        }
    }

    private func fillQ8EmbeddingRows(
        storage: TensorStorage,
        basePtr: UnsafeMutableRawPointer,
        tokenIDs: [Int],
        rowWidth: Int,
        destination: UnsafeMutablePointer<Float>
    ) throws {
        let bytesPerBlock = 34
        let weightsPerBlock = 32
        let bytesPerRow = try rowStrideBytes(
            rowWidth: rowWidth,
            weightsPerBlock: weightsPerBlock,
            blockByteCount: bytesPerBlock,
            tensorName: storage.name
        )
        let requiredBytes = tokenIDs.isEmpty ? 0 : (tokenIDs.max()! + 1) * bytesPerRow
        guard storage.byteOffset + requiredBytes <= storage.buffer.length else {
            throw GenerationError.modelLoadFailed(reason: "\(storage.name) buffer is smaller than requested token rows")
        }
        let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: storage.buffer.length - storage.byteOffset)
        let blocksPerRow = rowWidth / weightsPerBlock
        for (index, tokenID) in tokenIDs.enumerated() {
            let rowStart = tokenID * bytesPerRow
            let dstOffset = index * rowWidth
            for block in 0..<blocksPerRow {
                let blockStart = rowStart + block * bytesPerBlock
                let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                for j in 0..<weightsPerBlock {
                    let qval = Int8(bitPattern: rawPtr[blockStart + 2 + j])
                    destination[dstOffset + block * weightsPerBlock + j] = scale * Float(qval)
                }
            }
        }
    }

    private func fillQ4_0EmbeddingRows(
        storage: TensorStorage,
        basePtr: UnsafeMutableRawPointer,
        tokenIDs: [Int],
        rowWidth: Int,
        destination: UnsafeMutablePointer<Float>
    ) throws {
        let bytesPerBlock = 18
        let weightsPerBlock = 32
        let bytesPerRow = try rowStrideBytes(
            rowWidth: rowWidth,
            weightsPerBlock: weightsPerBlock,
            blockByteCount: bytesPerBlock,
            tensorName: storage.name
        )
        let requiredBytes = tokenIDs.isEmpty ? 0 : (tokenIDs.max()! + 1) * bytesPerRow
        guard storage.byteOffset + requiredBytes <= storage.buffer.length else {
            throw GenerationError.modelLoadFailed(reason: "\(storage.name) buffer is smaller than requested token rows")
        }
        let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: storage.buffer.length - storage.byteOffset)
        let blocksPerRow = rowWidth / weightsPerBlock
        for (index, tokenID) in tokenIDs.enumerated() {
            let rowStart = tokenID * bytesPerRow
            let dstOffset = index * rowWidth
            for block in 0..<blocksPerRow {
                let blockStart = rowStart + block * bytesPerBlock
                let scaleBits = UInt16(rawPtr[blockStart]) | (UInt16(rawPtr[blockStart + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                for j in 0..<16 {
                    let byte = rawPtr[blockStart + 2 + j]
                    destination[dstOffset + block * weightsPerBlock + j * 2] =
                        scale * Float(Int(byte & 0x0F) - 8)
                    destination[dstOffset + block * weightsPerBlock + j * 2 + 1] =
                        scale * Float(Int(byte >> 4) - 8)
                }
            }
        }
    }

    private func fillKQuantEmbeddingRows(
        storage: TensorStorage,
        basePtr: UnsafeMutableRawPointer,
        tokenIDs: [Int],
        rowWidth: Int,
        destination: UnsafeMutablePointer<Float>
    ) throws {
        let weightsPerBlock = 256
        let blockByteCount = Self.kQuantBlockByteCount(for: storage.dataType)
        guard blockByteCount > 0 else {
            throw GenerationError.modelLoadFailed(reason: "Unsupported K-quant embedding data type \(storage.dataType)")
        }
        let bytesPerRow = try rowStrideBytes(
            rowWidth: rowWidth,
            weightsPerBlock: weightsPerBlock,
            blockByteCount: blockByteCount,
            tensorName: storage.name
        )
        let requiredBytes = tokenIDs.isEmpty ? 0 : (tokenIDs.max()! + 1) * bytesPerRow
        guard storage.byteOffset + requiredBytes <= storage.buffer.length else {
            throw GenerationError.modelLoadFailed(reason: "\(storage.name) buffer is smaller than requested token rows")
        }
        let rawPtr = basePtr.bindMemory(to: UInt8.self, capacity: storage.buffer.length - storage.byteOffset)
        let blocksPerRow = rowWidth / weightsPerBlock
        for (index, tokenID) in tokenIDs.enumerated() {
            try Self.dequantizeKQuantRow(
                dataType: storage.dataType,
                rawPtr: rawPtr,
                rowStart: tokenID * bytesPerRow,
                blockCount: blocksPerRow,
                destination: destination + index * rowWidth
            )
        }
    }

    private static func kQuantBlockByteCount(for dataType: TensorDataType) -> Int {
        switch dataType {
        case .q2_K: 84
        case .q3_K: 110
        case .q4_K: 144
        case .q5_K: 176
        case .q6_K: 210
        default: 0
        }
    }

    private static func dequantizeKQuantRow(
        dataType: TensorDataType,
        rawPtr: UnsafePointer<UInt8>,
        rowStart: Int,
        blockCount: Int,
        destination: UnsafeMutablePointer<Float>
    ) throws {
        let blockByteCount = kQuantBlockByteCount(for: dataType)
        guard blockByteCount > 0 else {
            throw GenerationError.modelLoadFailed(reason: "Unsupported K-quant embedding data type: \(dataType)")
        }

        for blockIndex in 0..<blockCount {
            let block = rawPtr + rowStart + blockIndex * blockByteCount
            let dst = destination + blockIndex * 256
            switch dataType {
            case .q2_K:
                dequantizeQ2KBlock(block, into: dst)
            case .q3_K:
                dequantizeQ3KBlock(block, into: dst)
            case .q4_K:
                dequantizeQ4KBlock(block, into: dst)
            case .q5_K:
                dequantizeQ5KBlock(block, into: dst)
            case .q6_K:
                dequantizeQ6KBlock(block, into: dst)
            default:
                throw GenerationError.modelLoadFailed(reason: "Unsupported K-quant embedding data type: \(dataType)")
            }
        }
    }

    private static func f16(at ptr: UnsafePointer<UInt8>, offset: Int) -> Float {
        let bits = UInt16(ptr[offset]) | (UInt16(ptr[offset + 1]) << 8)
        return Float(Float16(bitPattern: bits))
    }

    private static func dequantizeQ2KBlock(
        _ block: UnsafePointer<UInt8>,
        into destination: UnsafeMutablePointer<Float>
    ) {
        let d = f16(at: block, offset: 80)
        let dmin = f16(at: block, offset: 82)
        for i in 0..<256 {
            let sub = i / 16
            let scaleByte = block[sub]
            let sc = Float(scaleByte & 0x0F)
            let m = Float(scaleByte >> 4)
            let qsByte = block[16 + i / 4]
            let q2 = Float((qsByte >> ((i % 4) * 2)) & 0x03)
            destination[i] = d * sc * q2 - dmin * m
        }
    }

    private static func dequantizeQ3KBlock(
        _ block: UnsafePointer<UInt8>,
        into destination: UnsafeMutablePointer<Float>
    ) {
        let d = f16(at: block, offset: 108)
        var scales = [Float](repeating: 0, count: 16)
        for i in 0..<16 {
            let lower4 = (block[96 + i / 2] >> ((i % 2) * 4)) & 0x0F
            let upper2 = (block[96 + 8 + i / 4] >> ((i % 4) * 2)) & 0x03
            let raw6 = Int(lower4) | (Int(upper2) << 4)
            scales[i] = d * Float(raw6 - 32)
        }

        for i in 0..<256 {
            let lower2 = (block[32 + i / 4] >> ((i % 4) * 2)) & 0x03
            let highBit = (block[i / 8] >> (i % 8)) & 1
            let q3 = Int(lower2) | (Int(highBit) << 2)
            destination[i] = scales[i / 16] * Float(q3 - 4)
        }
    }

    private static func unpackKQuantScalesAndMins(
        _ block: UnsafePointer<UInt8>
    ) -> (scales: [UInt8], mins: [UInt8]) {
        var scales = [UInt8](repeating: 0, count: 8)
        var mins = [UInt8](repeating: 0, count: 8)
        for subBlock in 0..<4 {
            scales[subBlock] = block[4 + subBlock] & 0x3F
            scales[subBlock + 4] = (block[12 + subBlock] & 0x0F)
                | ((block[4 + subBlock] >> 6) << 4)
            mins[subBlock] = block[8 + subBlock] & 0x3F
            mins[subBlock + 4] = ((block[12 + subBlock] >> 4) & 0x0F)
                | ((block[8 + subBlock] >> 6) << 4)
        }
        return (scales, mins)
    }

    private static func dequantizeQ4KBlock(
        _ block: UnsafePointer<UInt8>,
        into destination: UnsafeMutablePointer<Float>
    ) {
        let d = f16(at: block, offset: 0)
        let dmin = f16(at: block, offset: 2)
        let (scales, mins) = unpackKQuantScalesAndMins(block)

        for subBlock in 0..<8 {
            let scale = d * Float(scales[subBlock])
            let minValue = dmin * Float(mins[subBlock])
            for index in 0..<32 {
                let byteIndex = 16 + (subBlock / 2) * 32 + index
                let nibble = subBlock.isMultiple(of: 2)
                    ? (block[byteIndex] & 0x0F)
                    : ((block[byteIndex] >> 4) & 0x0F)
                destination[subBlock * 32 + index] = scale * Float(nibble) - minValue
            }
        }
    }

    private static func dequantizeQ5KBlock(
        _ block: UnsafePointer<UInt8>,
        into destination: UnsafeMutablePointer<Float>
    ) {
        let d = f16(at: block, offset: 0)
        let dmin = f16(at: block, offset: 2)
        let (scales, mins) = unpackKQuantScalesAndMins(block)

        for subBlock in 0..<8 {
            let scale = d * Float(scales[subBlock])
            let minValue = dmin * Float(mins[subBlock])
            for index in 0..<32 {
                let globalIndex = subBlock * 32 + index
                let qsByteIndex = 48 + globalIndex / 2
                let lower4 = globalIndex.isMultiple(of: 2)
                    ? (block[qsByteIndex] & 0x0F)
                    : ((block[qsByteIndex] >> 4) & 0x0F)
                let bit5 = (block[16 + globalIndex / 8] >> (globalIndex % 8)) & 1
                let q5 = lower4 | (bit5 << 4)
                destination[globalIndex] = scale * Float(q5) - minValue
            }
        }
    }

    private static func dequantizeQ6KBlock(
        _ block: UnsafePointer<UInt8>,
        into destination: UnsafeMutablePointer<Float>
    ) {
        let d = f16(at: block, offset: 208)
        for halfBlock in 0..<2 {
            let qlBase = halfBlock * 64
            let qhBase = 128 + halfBlock * 32
            let scaleBase = 192 + halfBlock * 8
            let outBase = halfBlock * 128
            for lane in 0..<32 {
                let firstScale = lane / 16

                let q1 = Int((block[qlBase + lane] & 0x0F) | (((block[qhBase + lane] >> 0) & 0x03) << 4)) - 32
                let q2 = Int((block[qlBase + 32 + lane] & 0x0F) | (((block[qhBase + lane] >> 2) & 0x03) << 4)) - 32
                let q3 = Int((block[qlBase + lane] >> 4) | (((block[qhBase + lane] >> 4) & 0x03) << 4)) - 32
                let q4 = Int((block[qlBase + 32 + lane] >> 4) | (((block[qhBase + lane] >> 6) & 0x03) << 4)) - 32

                let s1 = Int8(bitPattern: block[scaleBase + firstScale + 0])
                let s2 = Int8(bitPattern: block[scaleBase + firstScale + 2])
                let s3 = Int8(bitPattern: block[scaleBase + firstScale + 4])
                let s4 = Int8(bitPattern: block[scaleBase + firstScale + 6])

                destination[outBase + lane] = d * Float(s1) * Float(q1)
                destination[outBase + 32 + lane] = d * Float(s2) * Float(q2)
                destination[outBase + 64 + lane] = d * Float(s3) * Float(q3)
                destination[outBase + 96 + lane] = d * Float(s4) * Float(q4)
            }
        }
    }

    private func readFloatVector(storage: TensorStorage, expectedElementCount: Int) throws -> [Float] {
        let key = "floatVector|\(storage.name)|\(storage.byteOffset)|\(expectedElementCount)|\(storage.dataType)"
        return try runtimeCache.floatVector(key: key) {
            guard storage.elementCount == expectedElementCount else {
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name) element count \(storage.elementCount) must equal \(expectedElementCount)"
                )
            }
            let base = storage.buffer.contents() + storage.byteOffset
            var values = [Float](repeating: 0, count: expectedElementCount)
            switch storage.dataType {
            case .float32:
                values.withUnsafeMutableBytes { rawBuffer in
                    rawBuffer.copyMemory(from: UnsafeRawBufferPointer(
                        start: base,
                        count: expectedElementCount * MemoryLayout<Float>.stride
                    ))
                }
            case .float16:
                let ptr = base.bindMemory(to: Float16.self, capacity: expectedElementCount)
                for index in 0..<expectedElementCount {
                    values[index] = Float(ptr[index])
                }
            case .bfloat16:
                let ptr = base.bindMemory(to: UInt16.self, capacity: expectedElementCount)
                for index in 0..<expectedElementCount {
                    values[index] = Float(bitPattern: UInt32(ptr[index]) << 16)
                }
            default:
                throw GenerationError.modelLoadFailed(
                    reason: "\(storage.name) must be F32/F16/BF16 for Gemma norm, got \(storage.dataType)"
                )
            }
            return values
        }
    }

    private func makeFloatBuffer(storage: TensorStorage, expectedElementCount: Int) throws -> MTLBuffer {
        let key = "floatBuffer|\(storage.name)|\(storage.byteOffset)|\(expectedElementCount)|\(storage.dataType)"
        return try runtimeCache.floatBuffer(key: key) {
            let values = try readFloatVector(
                storage: storage,
                expectedElementCount: expectedElementCount
            )
            return try makeFloatBuffer(values, label: storage.name)
        }
    }

    private func makeUnitFloatBuffer(count: Int) throws -> MTLBuffer {
        try runtimeCache.floatBuffer(key: "unitFloatBuffer|\(count)") {
            try makeFloatBuffer([Float](repeating: 1, count: count), label: "unitFloatBuffer.\(count)")
        }
    }

    private func makeFloatBuffer(_ values: [Float], label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create Float buffer for \(label)")
        }
        return buffer
    }
}
