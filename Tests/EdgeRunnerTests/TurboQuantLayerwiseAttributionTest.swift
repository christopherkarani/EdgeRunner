import Foundation
import Metal
import Testing
@testable import EdgeRunner
import EdgeRunnerMetal

@Suite("TurboQuant v2 Layerwise Attribution")
struct TurboQuantLayerwiseAttributionTest {
    private static let runEnvKey = "EDGERUNNER_RUN_TURBOQUANT_V2_LAYERWISE"
    private static let modelPath = BenchmarkContract.pinnedModelPath

    @Test
    func compareLayerwiseTraceAgainstQ8Baseline() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let guidedDecodeSteps = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_GUIDED_STEPS"] ?? "2") ?? 2
        let divergenceThreshold = Float(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_THRESHOLD"] ?? "0.25") ?? 0.25

        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let guidedTokens = try await greedyPrefix(
            modelURL: modelURL,
            prompt: prompt,
            steps: guidedDecodeSteps
        )
        let traceTokens = prompt + guidedTokens

        let q8Trace = try await loadTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .q8_0
        )
        let turboTrace = try await loadTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .turboquantV2,
            captureDenseValueReference: true
        )
        let turboDenseValueReference = try #require(turboTrace.denseValueReferenceAttentionOutputStates)
        let turboDenseAttentionReference = try #require(turboTrace.denseAttentionReferenceAttentionOutputStates)

        let attentionOutputDeltas = zip(q8Trace.attentionOutputStates, turboTrace.attentionOutputStates).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let denseValueReferenceDeltas = zip(q8Trace.attentionOutputStates, turboDenseValueReference).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let turboToDenseValueReferenceDeltas = zip(turboTrace.attentionOutputStates, turboDenseValueReference).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let denseAttentionReferenceDeltas = zip(q8Trace.attentionOutputStates, turboDenseAttentionReference).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let turboToDenseAttentionReferenceDeltas = zip(turboTrace.attentionOutputStates, turboDenseAttentionReference).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let denseValueToDenseAttentionReferenceDeltas = zip(turboDenseValueReference, turboDenseAttentionReference).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let attentionDeltas = zip(q8Trace.attentionResidualStates, turboTrace.attentionResidualStates).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let layerDeltas = zip(q8Trace.layerHiddenStates, turboTrace.layerHiddenStates).enumerated().map { index, pair in
            LayerDelta(
                layerIndex: index,
                maxAbsoluteDelta: zip(pair.0, pair.1).reduce(Float.zero) { partial, values in
                    max(partial, abs(values.0 - values.1))
                }
            )
        }
        let firstDivergentAttentionOutputLayer = attentionOutputDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestAttentionOutputLayerDelta = attentionOutputDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstDenseValueReferenceLayer = denseValueReferenceDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestDenseValueReferenceLayerDelta = denseValueReferenceDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstTurboToDenseValueReferenceLayer = turboToDenseValueReferenceDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestTurboToDenseValueReferenceLayerDelta = turboToDenseValueReferenceDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstDenseAttentionReferenceLayer = denseAttentionReferenceDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestDenseAttentionReferenceLayerDelta = denseAttentionReferenceDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstTurboToDenseAttentionReferenceLayer = turboToDenseAttentionReferenceDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestTurboToDenseAttentionReferenceLayerDelta = turboToDenseAttentionReferenceDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstDenseValueToDenseAttentionReferenceLayer = denseValueToDenseAttentionReferenceDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestDenseValueToDenseAttentionReferenceLayerDelta = denseValueToDenseAttentionReferenceDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstDivergentAttentionLayer = attentionDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestAttentionLayerDelta = attentionDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let firstDivergentLayer = layerDeltas.first { $0.maxAbsoluteDelta > divergenceThreshold }
        let largestLayerDelta = layerDeltas.max(by: { $0.maxAbsoluteDelta < $1.maxAbsoluteDelta })
        let q8Argmax = argmax(q8Trace.logits)
        let turboArgmax = argmax(turboTrace.logits)
        let maxLogitDelta = zip(q8Trace.logits, turboTrace.logits).reduce(Float.zero) { partial, values in
            max(partial, abs(values.0 - values.1))
        }

        print("""
        [turboquant-v2-layerwise]
          prompt_len=\(promptLength)
          guided_decode_steps=\(guidedDecodeSteps)
          guided_tokens=\(guidedTokens)
          trace_token_count=\(traceTokens.count)
          first_divergent_attention_output_layer=\(firstDivergentAttentionOutputLayer?.layerIndex.description ?? "none")
          first_divergent_attention_output_layer_max_abs_delta=\(firstDivergentAttentionOutputLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_attention_output_layer_delta_layer=\(largestAttentionOutputLayerDelta?.layerIndex.description ?? "none")
          largest_attention_output_layer_delta=\(largestAttentionOutputLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_dense_v_reference_attention_output_layer=\(firstDenseValueReferenceLayer?.layerIndex.description ?? "none")
          first_dense_v_reference_attention_output_layer_max_abs_delta=\(firstDenseValueReferenceLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_dense_v_reference_attention_output_layer_delta_layer=\(largestDenseValueReferenceLayerDelta?.layerIndex.description ?? "none")
          largest_dense_v_reference_attention_output_layer_delta=\(largestDenseValueReferenceLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_turbo_to_dense_v_reference_attention_output_layer=\(firstTurboToDenseValueReferenceLayer?.layerIndex.description ?? "none")
          first_turbo_to_dense_v_reference_attention_output_layer_max_abs_delta=\(firstTurboToDenseValueReferenceLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_turbo_to_dense_v_reference_attention_output_layer_delta_layer=\(largestTurboToDenseValueReferenceLayerDelta?.layerIndex.description ?? "none")
          largest_turbo_to_dense_v_reference_attention_output_layer_delta=\(largestTurboToDenseValueReferenceLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_dense_attention_reference_output_layer=\(firstDenseAttentionReferenceLayer?.layerIndex.description ?? "none")
          first_dense_attention_reference_output_layer_max_abs_delta=\(firstDenseAttentionReferenceLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_dense_attention_reference_output_layer_delta_layer=\(largestDenseAttentionReferenceLayerDelta?.layerIndex.description ?? "none")
          largest_dense_attention_reference_output_layer_delta=\(largestDenseAttentionReferenceLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_turbo_to_dense_attention_reference_output_layer=\(firstTurboToDenseAttentionReferenceLayer?.layerIndex.description ?? "none")
          first_turbo_to_dense_attention_reference_output_layer_max_abs_delta=\(firstTurboToDenseAttentionReferenceLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_turbo_to_dense_attention_reference_output_layer_delta_layer=\(largestTurboToDenseAttentionReferenceLayerDelta?.layerIndex.description ?? "none")
          largest_turbo_to_dense_attention_reference_output_layer_delta=\(largestTurboToDenseAttentionReferenceLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_dense_v_to_dense_attention_reference_output_layer=\(firstDenseValueToDenseAttentionReferenceLayer?.layerIndex.description ?? "none")
          first_dense_v_to_dense_attention_reference_output_layer_max_abs_delta=\(firstDenseValueToDenseAttentionReferenceLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_dense_v_to_dense_attention_reference_output_layer_delta_layer=\(largestDenseValueToDenseAttentionReferenceLayerDelta?.layerIndex.description ?? "none")
          largest_dense_v_to_dense_attention_reference_output_layer_delta=\(largestDenseValueToDenseAttentionReferenceLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_divergent_attention_layer=\(firstDivergentAttentionLayer?.layerIndex.description ?? "none")
          first_divergent_attention_layer_max_abs_delta=\(firstDivergentAttentionLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_attention_layer_delta_layer=\(largestAttentionLayerDelta?.layerIndex.description ?? "none")
          largest_attention_layer_delta=\(largestAttentionLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          first_divergent_layer=\(firstDivergentLayer?.layerIndex.description ?? "none")
          first_divergent_layer_max_abs_delta=\(firstDivergentLayer.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          largest_layer_delta_layer=\(largestLayerDelta?.layerIndex.description ?? "none")
          largest_layer_delta=\(largestLayerDelta.map { String(format: "%.6f", $0.maxAbsoluteDelta) } ?? "0")
          q8_argmax=\(q8Argmax)
          turboquant_v2_argmax=\(turboArgmax)
          max_abs_logit_delta=\(String(format: "%.6f", maxLogitDelta))
        """)

        #expect(q8Trace.layerHiddenStates.count == turboTrace.layerHiddenStates.count)
        #expect(q8Trace.attentionOutputStates.count == turboTrace.attentionOutputStates.count)
        #expect(q8Trace.attentionOutputStates.count == turboDenseValueReference.count)
        #expect(q8Trace.attentionOutputStates.count == turboDenseAttentionReference.count)
        #expect(q8Trace.attentionResidualStates.count == turboTrace.attentionResidualStates.count)
        #expect(q8Trace.attentionOutputStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(turboTrace.attentionOutputStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(turboDenseValueReference.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(turboDenseAttentionReference.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(q8Trace.attentionResidualStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(turboTrace.attentionResidualStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(q8Trace.layerHiddenStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(turboTrace.layerHiddenStates.allSatisfy { layer in layer.allSatisfy { value in value.isFinite } })
        #expect(q8Trace.logits.allSatisfy { value in value.isFinite })
        #expect(turboTrace.logits.allSatisfy { value in value.isFinite })
    }

    @Test
    func compareAttentionScoreTraceAgainstQ8Baseline() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)

        let q8Trace = try await loadAttentionScoreTrace(
            modelURL: modelURL,
            tokenIDs: prompt,
            compression: .q8_0
        )
        let turboTrace = try await loadAttentionScoreTrace(
            modelURL: modelURL,
            tokenIDs: prompt,
            compression: .turboquantV2
        )

        let scoreThreshold = Float(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_SCORE_THRESHOLD"] ?? "0.25") ?? 0.25
        let softmaxThreshold = Float(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_SOFTMAX_THRESHOLD"] ?? "0.05") ?? 0.05

        let firstQ8ScoreLayer = q8Trace.summaries.first { $0.exactVsStoredScoreMaxAbs > scoreThreshold }
        let firstTurboScoreLayer = turboTrace.summaries.first { $0.exactVsStoredScoreMaxAbs > scoreThreshold }
        let firstTurboDecodedScoreLayer = turboTrace.summaries.first { $0.exactVsDecodedScoreMaxAbs > scoreThreshold }
        let firstQ8SoftmaxLayer = q8Trace.summaries.first { $0.exactVsStoredSoftmaxMaxAbs > softmaxThreshold }
        let firstTurboSoftmaxLayer = turboTrace.summaries.first { $0.exactVsStoredSoftmaxMaxAbs > softmaxThreshold }
        let firstTurboDecodedSoftmaxLayer = turboTrace.summaries.first { $0.exactVsDecodedSoftmaxMaxAbs > softmaxThreshold }
        let worstQ8ScoreLayer = q8Trace.summaries.max(by: { $0.exactVsStoredScoreMaxAbs < $1.exactVsStoredScoreMaxAbs })
        let worstTurboScoreLayer = turboTrace.summaries.max(by: { $0.exactVsStoredScoreMaxAbs < $1.exactVsStoredScoreMaxAbs })
        let worstTurboDecodedScoreLayer = turboTrace.summaries.max(by: { $0.exactVsDecodedScoreMaxAbs < $1.exactVsDecodedScoreMaxAbs })
        let worstQ8SoftmaxLayer = q8Trace.summaries.max(by: { $0.exactVsStoredSoftmaxMaxAbs < $1.exactVsStoredSoftmaxMaxAbs })
        let worstTurboSoftmaxLayer = turboTrace.summaries.max(by: { $0.exactVsStoredSoftmaxMaxAbs < $1.exactVsStoredSoftmaxMaxAbs })
        let worstTurboDecodedSoftmaxLayer = turboTrace.summaries.max(by: { $0.exactVsDecodedSoftmaxMaxAbs < $1.exactVsDecodedSoftmaxMaxAbs })

        print("""
        [turboquant-v2-score-trace]
          prompt_len=\(promptLength)
          first_q8_score_divergent_layer=\(firstQ8ScoreLayer?.layerIndex.description ?? "none")
          first_q8_score_divergent_layer_max_abs_delta=\(firstQ8ScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMaxAbs) } ?? "0")
          first_turbo_score_divergent_layer=\(firstTurboScoreLayer?.layerIndex.description ?? "none")
          first_turbo_score_divergent_layer_max_abs_delta=\(firstTurboScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMaxAbs) } ?? "0")
          first_turbo_decoded_score_divergent_layer=\(firstTurboDecodedScoreLayer?.layerIndex.description ?? "none")
          first_turbo_decoded_score_divergent_layer_max_abs_delta=\(firstTurboDecodedScoreLayer.map { String(format: "%.6f", $0.exactVsDecodedScoreMaxAbs) } ?? "0")
          worst_q8_score_layer=\(worstQ8ScoreLayer?.layerIndex.description ?? "none")
          worst_q8_score_layer_max_abs_delta=\(worstQ8ScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMaxAbs) } ?? "0")
          worst_q8_score_layer_mse=\(worstQ8ScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMSE) } ?? "0")
          worst_turbo_score_layer=\(worstTurboScoreLayer?.layerIndex.description ?? "none")
          worst_turbo_score_layer_max_abs_delta=\(worstTurboScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMaxAbs) } ?? "0")
          worst_turbo_score_layer_mse=\(worstTurboScoreLayer.map { String(format: "%.6f", $0.exactVsStoredScoreMSE) } ?? "0")
          worst_turbo_decoded_score_layer=\(worstTurboDecodedScoreLayer?.layerIndex.description ?? "none")
          worst_turbo_decoded_score_layer_max_abs_delta=\(worstTurboDecodedScoreLayer.map { String(format: "%.6f", $0.exactVsDecodedScoreMaxAbs) } ?? "0")
          worst_turbo_decoded_score_layer_mse=\(worstTurboDecodedScoreLayer.map { String(format: "%.6f", $0.exactVsDecodedScoreMSE) } ?? "0")
          first_q8_softmax_divergent_layer=\(firstQ8SoftmaxLayer?.layerIndex.description ?? "none")
          first_q8_softmax_divergent_layer_max_abs_delta=\(firstQ8SoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMaxAbs) } ?? "0")
          first_turbo_softmax_divergent_layer=\(firstTurboSoftmaxLayer?.layerIndex.description ?? "none")
          first_turbo_softmax_divergent_layer_max_abs_delta=\(firstTurboSoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMaxAbs) } ?? "0")
          first_turbo_decoded_softmax_divergent_layer=\(firstTurboDecodedSoftmaxLayer?.layerIndex.description ?? "none")
          first_turbo_decoded_softmax_divergent_layer_max_abs_delta=\(firstTurboDecodedSoftmaxLayer.map { String(format: "%.6f", $0.exactVsDecodedSoftmaxMaxAbs) } ?? "0")
          worst_q8_softmax_layer=\(worstQ8SoftmaxLayer?.layerIndex.description ?? "none")
          worst_q8_softmax_layer_max_abs_delta=\(worstQ8SoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMaxAbs) } ?? "0")
          worst_q8_softmax_layer_mse=\(worstQ8SoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMSE) } ?? "0")
          worst_turbo_softmax_layer=\(worstTurboSoftmaxLayer?.layerIndex.description ?? "none")
          worst_turbo_softmax_layer_max_abs_delta=\(worstTurboSoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMaxAbs) } ?? "0")
          worst_turbo_softmax_layer_mse=\(worstTurboSoftmaxLayer.map { String(format: "%.6f", $0.exactVsStoredSoftmaxMSE) } ?? "0")
          worst_turbo_decoded_softmax_layer=\(worstTurboDecodedSoftmaxLayer?.layerIndex.description ?? "none")
          worst_turbo_decoded_softmax_layer_max_abs_delta=\(worstTurboDecodedSoftmaxLayer.map { String(format: "%.6f", $0.exactVsDecodedSoftmaxMaxAbs) } ?? "0")
          worst_turbo_decoded_softmax_layer_mse=\(worstTurboDecodedSoftmaxLayer.map { String(format: "%.6f", $0.exactVsDecodedSoftmaxMSE) } ?? "0")
        """)

        #expect(q8Trace.summaries.count == turboTrace.summaries.count)
        #expect(q8Trace.summaries.allSatisfy { $0.exactVsStoredScoreMaxAbs.isFinite && $0.exactVsStoredSoftmaxMaxAbs.isFinite && $0.exactVsDecodedScoreMaxAbs.isFinite && $0.exactVsDecodedSoftmaxMaxAbs.isFinite })
        #expect(turboTrace.summaries.allSatisfy { $0.exactVsStoredScoreMaxAbs.isFinite && $0.exactVsStoredSoftmaxMaxAbs.isFinite && $0.exactVsDecodedScoreMaxAbs.isFinite && $0.exactVsDecodedSoftmaxMaxAbs.isFinite })
    }

    @Test
    func compareExperimentalKeyPresetFidelity() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let inputs = try await loadAttentionInputsTrace(
            modelURL: modelURL,
            tokenIDs: prompt,
            compression: .disabled
        )

        let presets: [TurboQuantPreset] = [
            .aggressive,
            .aggressive64,
            .balanced,
            .balanced64,
            .balanced96,
            .fiveBit,
            .sixBit,
            .sevenBit,
        ]

        var results: [(preset: TurboQuantPreset, summaries: [LlamaAttentionScoreSummary])] = []
        results.reserveCapacity(presets.count)
        for preset in presets {
            let summaries = try cpuPresetFidelitySummaries(
                inputs: inputs,
                preset: preset,
                outlierSelection: .magnitude
            )
            results.append((preset: preset, summaries: summaries))
        }

        print("[turboquant-key-preset-fidelity]")
        print("  prompt_len=\(promptLength)")
        for result in results {
            let worstScore = result.summaries.max(by: { $0.exactVsDecodedScoreMaxAbs < $1.exactVsDecodedScoreMaxAbs })
            let worstSoftmax = result.summaries.max(by: { $0.exactVsDecodedSoftmaxMaxAbs < $1.exactVsDecodedSoftmaxMaxAbs })
            let effectiveBitsString = String(format: "%.2f", result.preset.descriptor.effectiveBits)
            print("  preset=\(result.preset.rawValue) effective_bits=\(effectiveBitsString)")
            print("    worst_score_layer=\(worstScore?.layerIndex.description ?? "none")")
            print("    worst_score_max_abs_delta=\(worstScore.map { String(format: "%.6f", $0.exactVsDecodedScoreMaxAbs) } ?? "0")")
            print("    worst_score_mse=\(worstScore.map { String(format: "%.6f", $0.exactVsDecodedScoreMSE) } ?? "0")")
            print("    worst_softmax_layer=\(worstSoftmax?.layerIndex.description ?? "none")")
            print("    worst_softmax_max_abs_delta=\(worstSoftmax.map { String(format: "%.6f", $0.exactVsDecodedSoftmaxMaxAbs) } ?? "0")")
            print("    worst_softmax_mse=\(worstSoftmax.map { String(format: "%.6f", $0.exactVsDecodedSoftmaxMSE) } ?? "0")")
        }

        #expect(results.allSatisfy { result in
            result.summaries.allSatisfy {
                $0.exactVsDecodedScoreMaxAbs.isFinite && $0.exactVsDecodedSoftmaxMaxAbs.isFinite
            }
        })
    }

    @Test
    func replayLayer0AttentionOnRealActivations() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            Issue.record("Metal device unavailable")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let layerIndex = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_REPLAY_LAYER"] ?? "0") ?? 0
        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let guidedDecodeSteps = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_GUIDED_STEPS"] ?? "2") ?? 2
        let guidedTokens = try await greedyPrefix(
            modelURL: modelURL,
            prompt: prompt,
            steps: guidedDecodeSteps
        )
        let traceTokens = prompt + guidedTokens
        let inputs = try await loadAttentionInputsTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .disabled
        )
        let turboTrace = try await loadTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .turboquantV2,
            captureDenseValueReference: true
        )

        let query = inputs.queries[layerIndex]
        let queryRows = inputs.queryRows[layerIndex]
        let keyRows = inputs.exactKeys[layerIndex]
        let valueRows = inputs.exactValues[layerIndex]
        let queryMax = query.reduce(Float.zero) { max($0, abs($1)) }
        let queryRowsMax = queryRows.flatMap { $0 }.reduce(Float.zero) { max($0, abs($1)) }
        let keyRowsMax = keyRows.flatMap { $0 }.reduce(Float.zero) { max($0, abs($1)) }
        let valueRowsMax = valueRows.flatMap { $0 }.reduce(Float.zero) { max($0, abs($1)) }

        let denseOutput = denseGroupedAttention(
            query: query,
            keyRows: keyRows,
            valueRows: valueRows
        )
        let decodedValueRows = try decodeTurboValueRows(
            valueRows: valueRows,
            valuePreset: TurboQuantV2Contract.valuePreset
        )
        let exactKDecodedVOutput = denseGroupedAttention(
            query: query,
            keyRows: keyRows,
            valueRows: decodedValueRows
        )
        let decodedKExactVOutput = try turboGroupedAttentionCPUExactValues(
            query: query,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: TurboQuantV2Contract.keyPreset
        )
        let cpuTurboOutput = try turboGroupedAttentionCPU(
            query: query,
            keyRows: keyRows,
            valueRows: valueRows,
            keyPreset: TurboQuantV2Contract.keyPreset,
            valuePreset: TurboQuantV2Contract.valuePreset
        )
        let gpuTurboOutput = try turboGroupedAttentionGPU(
            device: device,
            commandQueue: commandQueue,
            query: query,
            keyRows: keyRows,
            valueRows: valueRows
        )
        let gpuTurboPrefillOutput = try turboGroupedAttentionPrefillGPU(
            device: device,
            commandQueue: commandQueue,
            queryRows: queryRows,
            keyRows: keyRows,
            valueRows: valueRows
        )

        let cpuDenseMax = maxAbsoluteDelta(cpuTurboOutput, denseOutput)
        let gpuDenseMax = maxAbsoluteDelta(gpuTurboOutput, denseOutput)
        let gpuPrefillDenseMax = maxAbsoluteDelta(gpuTurboPrefillOutput, denseOutput)
        let cpuDenseMSE = meanSquaredError(cpuTurboOutput, denseOutput)
        let gpuDenseMSE = meanSquaredError(gpuTurboOutput, denseOutput)
        let gpuPrefillDenseMSE = meanSquaredError(gpuTurboPrefillOutput, denseOutput)
        let exactKDecodedVMax = maxAbsoluteDelta(exactKDecodedVOutput, denseOutput)
        let exactKDecodedVMSE = meanSquaredError(exactKDecodedVOutput, denseOutput)
        let decodedKExactVMax = maxAbsoluteDelta(decodedKExactVOutput, denseOutput)
        let decodedKExactVMSE = meanSquaredError(decodedKExactVOutput, denseOutput)
        let cpuGpuMax = maxAbsoluteDelta(cpuTurboOutput, gpuTurboOutput)
        let cpuGpuMSE = meanSquaredError(cpuTurboOutput, gpuTurboOutput)
        let cpuGpuPrefillMax = maxAbsoluteDelta(cpuTurboOutput, gpuTurboPrefillOutput)
        let cpuGpuPrefillMSE = meanSquaredError(cpuTurboOutput, gpuTurboPrefillOutput)
        let runtimeReplayMax = maxAbsoluteDelta(turboTrace.attentionOutputStates[layerIndex], gpuTurboPrefillOutput)
        let runtimeReplayMSE = meanSquaredError(turboTrace.attentionOutputStates[layerIndex], gpuTurboPrefillOutput)
        let runtimeDenseRef = try #require(turboTrace.denseAttentionReferenceAttentionOutputStates)
        let runtimeDenseRefMax = maxAbsoluteDelta(runtimeDenseRef[layerIndex], gpuTurboPrefillOutput)
        let runtimeDenseRefMSE = meanSquaredError(runtimeDenseRef[layerIndex], gpuTurboPrefillOutput)
        let denseOutputMax = denseOutput.reduce(Float.zero) { max($0, abs($1)) }

        print("""
        [turboquant-v2-real-activation-replay]
          prompt_len=\(promptLength)
          guided_decode_steps=\(guidedDecodeSteps)
          guided_tokens=\(guidedTokens)
          layer_index=\(layerIndex)
          token_count=\(inputs.tokenIDs.count)
          query_last_token_max_abs=\(String(format: "%.6f", queryMax))
          query_all_tokens_max_abs=\(String(format: "%.6f", queryRowsMax))
          key_all_tokens_max_abs=\(String(format: "%.6f", keyRowsMax))
          value_all_tokens_max_abs=\(String(format: "%.6f", valueRowsMax))
          key_preset=\(TurboQuantV2Contract.keyPreset.rawValue)
          value_preset=\(TurboQuantV2Contract.valuePreset.rawValue)
          cpu_vs_dense_max_abs=\(String(format: "%.6f", cpuDenseMax))
          cpu_vs_dense_mse=\(String(format: "%.6f", cpuDenseMSE))
          gpu_vs_dense_max_abs=\(String(format: "%.6f", gpuDenseMax))
          gpu_vs_dense_mse=\(String(format: "%.6f", gpuDenseMSE))
          gpu_prefill_vs_dense_max_abs=\(String(format: "%.6f", gpuPrefillDenseMax))
          gpu_prefill_vs_dense_mse=\(String(format: "%.6f", gpuPrefillDenseMSE))
          exact_k_decoded_v_max_abs=\(String(format: "%.6f", exactKDecodedVMax))
          exact_k_decoded_v_mse=\(String(format: "%.6f", exactKDecodedVMSE))
          decoded_k_exact_v_max_abs=\(String(format: "%.6f", decodedKExactVMax))
          decoded_k_exact_v_mse=\(String(format: "%.6f", decodedKExactVMSE))
          cpu_vs_gpu_max_abs=\(String(format: "%.6f", cpuGpuMax))
          cpu_vs_gpu_mse=\(String(format: "%.6f", cpuGpuMSE))
          cpu_vs_gpu_prefill_max_abs=\(String(format: "%.6f", cpuGpuPrefillMax))
          cpu_vs_gpu_prefill_mse=\(String(format: "%.6f", cpuGpuPrefillMSE))
          runtime_trace_vs_gpu_prefill_max_abs=\(String(format: "%.6f", runtimeReplayMax))
          runtime_trace_vs_gpu_prefill_mse=\(String(format: "%.6f", runtimeReplayMSE))
          runtime_dense_reference_vs_gpu_prefill_max_abs=\(String(format: "%.6f", runtimeDenseRefMax))
          runtime_dense_reference_vs_gpu_prefill_mse=\(String(format: "%.6f", runtimeDenseRefMSE))
          dense_output_max_abs=\(String(format: "%.6f", denseOutputMax))
        """)

        #expect(denseOutput.allSatisfy { $0.isFinite })
        #expect(cpuTurboOutput.allSatisfy { $0.isFinite })
        #expect(gpuTurboOutput.allSatisfy { $0.isFinite })
        #expect(gpuTurboPrefillOutput.allSatisfy { $0.isFinite })
        #expect(cpuGpuMax < 0.25)
    }

    @Test
    func compareAttentionInputsAgainstQ8Baseline() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let layerIndex = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_REPLAY_LAYER"] ?? "0") ?? 0
        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let guidedDecodeSteps = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_GUIDED_STEPS"] ?? "2") ?? 2
        let guidedTokens = try await greedyPrefix(
            modelURL: modelURL,
            prompt: prompt,
            steps: guidedDecodeSteps
        )
        let traceTokens = prompt + guidedTokens

        let q8Inputs = try await loadAttentionInputsTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .q8_0
        )
        let turboInputs = try await loadAttentionInputsTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .turboquantV2
        )

        let queryDelta = maxAbsoluteDelta(q8Inputs.queries[layerIndex], turboInputs.queries[layerIndex])
        let queryRowsDelta = maxAbsoluteDelta(
            q8Inputs.queryRows[layerIndex].flatMap { $0 },
            turboInputs.queryRows[layerIndex].flatMap { $0 }
        )
        let keyDelta = maxAbsoluteDelta(
            q8Inputs.exactKeys[layerIndex].flatMap { $0 },
            turboInputs.exactKeys[layerIndex].flatMap { $0 }
        )
        let valueDelta = maxAbsoluteDelta(
            q8Inputs.exactValues[layerIndex].flatMap { $0 },
            turboInputs.exactValues[layerIndex].flatMap { $0 }
        )

        print("""
        [turboquant-v2-attention-inputs]
          prompt_len=\(promptLength)
          guided_decode_steps=\(guidedDecodeSteps)
          guided_tokens=\(guidedTokens)
          layer_index=\(layerIndex)
          query_last_token_max_abs_delta=\(String(format: "%.6f", queryDelta))
          query_all_tokens_max_abs_delta=\(String(format: "%.6f", queryRowsDelta))
          key_all_tokens_max_abs_delta=\(String(format: "%.6f", keyDelta))
          value_all_tokens_max_abs_delta=\(String(format: "%.6f", valueDelta))
        """)

        #expect(q8Inputs.queries[layerIndex].allSatisfy { $0.isFinite })
        #expect(turboInputs.queries[layerIndex].allSatisfy { $0.isFinite })
        #expect(q8Inputs.exactKeys[layerIndex].flatMap { $0 }.allSatisfy { $0.isFinite })
        #expect(turboInputs.exactKeys[layerIndex].flatMap { $0 }.allSatisfy { $0.isFinite })
        #expect(q8Inputs.exactValues[layerIndex].flatMap { $0 }.allSatisfy { $0.isFinite })
        #expect(turboInputs.exactValues[layerIndex].flatMap { $0 }.allSatisfy { $0.isFinite })
    }

    @Test
    func compareAttentionApproximationAgainstQ8Baseline() async throws {
        guard ProcessInfo.processInfo.environment[Self.runEnvKey] == "1" else {
            return
        }
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            Issue.record("Missing model file at \(Self.modelPath)")
            return
        }

        let promptLength = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_PROMPT_LEN"] ?? "128") ?? 128
        let prompt = Array(repeating: 9707, count: promptLength)
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let guidedDecodeSteps = Int(ProcessInfo.processInfo.environment["EDGERUNNER_TURBOQUANT_V2_LAYERWISE_GUIDED_STEPS"] ?? "2") ?? 2
        let guidedTokens = try await greedyPrefix(
            modelURL: modelURL,
            prompt: prompt,
            steps: guidedDecodeSteps
        )
        let traceTokens = prompt + guidedTokens

        let q8Trace = try await loadAttentionApproximationTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .q8_0
        )
        let turboTrace = try await loadAttentionApproximationTrace(
            modelURL: modelURL,
            tokenIDs: traceTokens,
            compression: .turboquantV2
        )

        let worstQ8KeyLayer = q8Trace.summaries.max(by: { $0.keyMaxAbs < $1.keyMaxAbs })
        let worstTurboKeyLayer = turboTrace.summaries.max(by: { $0.keyMaxAbs < $1.keyMaxAbs })
        let worstQ8ValueLayer = q8Trace.summaries.max(by: { $0.valueMaxAbs < $1.valueMaxAbs })
        let worstTurboValueLayer = turboTrace.summaries.max(by: { $0.valueMaxAbs < $1.valueMaxAbs })
        let worstTurboLastKeyLayer = turboTrace.summaries.max(by: { $0.lastTokenKeyMaxAbs < $1.lastTokenKeyMaxAbs })
        let worstTurboLastValueLayer = turboTrace.summaries.max(by: { $0.lastTokenValueMaxAbs < $1.lastTokenValueMaxAbs })

        print("""
        [turboquant-v2-approximation]
          prompt_len=\(promptLength)
          guided_decode_steps=\(guidedDecodeSteps)
          guided_tokens=\(guidedTokens)
          trace_token_count=\(traceTokens.count)
          worst_q8_key_layer=\(worstQ8KeyLayer?.layerIndex.description ?? "none")
          worst_q8_key_max_abs=\(worstQ8KeyLayer.map { String(format: "%.6f", $0.keyMaxAbs) } ?? "0")
          worst_q8_key_mse=\(worstQ8KeyLayer.map { String(format: "%.6f", $0.keyMSE) } ?? "0")
          worst_turbo_key_layer=\(worstTurboKeyLayer?.layerIndex.description ?? "none")
          worst_turbo_key_max_abs=\(worstTurboKeyLayer.map { String(format: "%.6f", $0.keyMaxAbs) } ?? "0")
          worst_turbo_key_mse=\(worstTurboKeyLayer.map { String(format: "%.6f", $0.keyMSE) } ?? "0")
          worst_turbo_key_token_index=\(worstTurboKeyLayer?.worstKeyTokenIndex.description ?? "none")
          worst_turbo_key_token_max_abs=\(worstTurboKeyLayer.map { String(format: "%.6f", $0.worstKeyTokenMaxAbs) } ?? "0")
          worst_turbo_last_token_key_layer=\(worstTurboLastKeyLayer?.layerIndex.description ?? "none")
          worst_turbo_last_token_key_max_abs=\(worstTurboLastKeyLayer.map { String(format: "%.6f", $0.lastTokenKeyMaxAbs) } ?? "0")
          worst_turbo_last_token_key_mse=\(worstTurboLastKeyLayer.map { String(format: "%.6f", $0.lastTokenKeyMSE) } ?? "0")
          worst_q8_value_layer=\(worstQ8ValueLayer?.layerIndex.description ?? "none")
          worst_q8_value_max_abs=\(worstQ8ValueLayer.map { String(format: "%.6f", $0.valueMaxAbs) } ?? "0")
          worst_q8_value_mse=\(worstQ8ValueLayer.map { String(format: "%.6f", $0.valueMSE) } ?? "0")
          worst_turbo_value_layer=\(worstTurboValueLayer?.layerIndex.description ?? "none")
          worst_turbo_value_max_abs=\(worstTurboValueLayer.map { String(format: "%.6f", $0.valueMaxAbs) } ?? "0")
          worst_turbo_value_mse=\(worstTurboValueLayer.map { String(format: "%.6f", $0.valueMSE) } ?? "0")
          worst_turbo_value_token_index=\(worstTurboValueLayer?.worstValueTokenIndex.description ?? "none")
          worst_turbo_value_token_max_abs=\(worstTurboValueLayer.map { String(format: "%.6f", $0.worstValueTokenMaxAbs) } ?? "0")
          worst_turbo_last_token_value_layer=\(worstTurboLastValueLayer?.layerIndex.description ?? "none")
          worst_turbo_last_token_value_max_abs=\(worstTurboLastValueLayer.map { String(format: "%.6f", $0.lastTokenValueMaxAbs) } ?? "0")
          worst_turbo_last_token_value_mse=\(worstTurboLastValueLayer.map { String(format: "%.6f", $0.lastTokenValueMSE) } ?? "0")
        """)

        #expect(q8Trace.summaries.allSatisfy { $0.keyMaxAbs.isFinite && $0.valueMaxAbs.isFinite })
        #expect(turboTrace.summaries.allSatisfy { $0.keyMaxAbs.isFinite && $0.valueMaxAbs.isFinite })
    }

    private func greedyPrefix(
        modelURL: URL,
        prompt: [Int],
        steps: Int
    ) async throws -> [Int] {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(prompt.count + steps + 16, 16_384),
                kvCacheCompression: .q8_0
            )
        )
        var tokenIDs = prompt
        var generated: [Int] = []
        for _ in 0..<steps {
            let logits = try await model.logits(for: tokenIDs)
            let token = argmax(logits)
            generated.append(token)
            tokenIDs.append(token)
        }
        return generated
    }

    private func loadTrace(
        modelURL: URL,
        tokenIDs: [Int],
        compression: KVCacheCompression,
        captureDenseValueReference: Bool = false
    ) async throws -> LlamaLayerwiseTrace {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(tokenIDs.count + 16, 16_384),
                kvCacheCompression: compression
            )
        )
        return try await model.layerwiseTrace(
            for: tokenIDs,
            captureDenseValueReference: captureDenseValueReference
        )
    }

    private func loadAttentionScoreTrace(
        modelURL: URL,
        tokenIDs: [Int],
        compression: KVCacheCompression
    ) async throws -> LlamaAttentionScoreTrace {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(tokenIDs.count + 16, 16_384),
                kvCacheCompression: compression
            )
        )
        return try await model.attentionScoreTrace(for: tokenIDs)
    }

    private func loadAttentionInputsTrace(
        modelURL: URL,
        tokenIDs: [Int],
        compression: KVCacheCompression
    ) async throws -> LlamaAttentionInputsTrace {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(tokenIDs.count + 16, 16_384),
                kvCacheCompression: compression
            )
        )
        return try await model.attentionInputsTrace(for: tokenIDs)
    }

    private func loadAttentionApproximationTrace(
        modelURL: URL,
        tokenIDs: [Int],
        compression: KVCacheCompression
    ) async throws -> LlamaAttentionApproximationTrace {
        let model = try await LlamaLanguageModel.load(
            from: modelURL,
            configuration: ModelConfiguration(
                contextWindowSize: max(tokenIDs.count + 16, 16_384),
                kvCacheCompression: compression
            )
        )
        return try await model.attentionApproximationTrace(for: tokenIDs)
    }

    private func denseGroupedAttention(
        query: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]]
    ) -> [Float] {
        let headDim = 128
        let numHeads = 16
        let numKVHeads = 8
        let groupSize = numHeads / numKVHeads
        let scale = 1.0 / sqrt(Float(headDim))
        var output = [Float](repeating: 0, count: numHeads * headDim)

        for headIndex in 0..<numHeads {
            let kvHeadIndex = headIndex / groupSize
            let qBase = headIndex * headDim
            let qSlice = Array(query[qBase..<(qBase + headDim)])
            var scores: [Float] = []
            scores.reserveCapacity(keyRows.count)
            for tokenIndex in 0..<keyRows.count {
                let row = keyRows[tokenIndex]
                let keyBase = kvHeadIndex * headDim
                let keySlice = Array(row[keyBase..<(keyBase + headDim)])
                scores.append(dot(qSlice, keySlice) * scale)
            }

            let weights = softmax(scores, chunkSize: scores.count)
            let outputBase = headIndex * headDim
            for tokenIndex in 0..<valueRows.count {
                let row = valueRows[tokenIndex]
                let valueBase = kvHeadIndex * headDim
                let valueSlice = Array(row[valueBase..<(valueBase + headDim)])
                for dim in 0..<headDim {
                    output[outputBase + dim] += weights[tokenIndex] * valueSlice[dim]
                }
            }
        }

        return output
    }

    private func turboGroupedAttentionCPUExactValues(
        query: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]],
        keyPreset: TurboQuantPreset
    ) throws -> [Float] {
        let headDim = 128
        let numHeads = 16
        let numKVHeads = 8
        let groupSize = numHeads / numKVHeads
        let scale = 1.0 / sqrt(Float(headDim))
        var output = [Float](repeating: 0, count: numHeads * headDim)

        let encodedKeys: [[TurboQuantRuntimeRow]] = try keyRows.map { row in
            try (0..<numKVHeads).map { kvHeadIndex in
                let base = kvHeadIndex * headDim
                let keySlice = Array(row[base..<(base + headDim)])
                let encoded = try TurboQuantReferenceEncoder.encode(
                    keySlice,
                    preset: keyPreset,
                    outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                )
                return try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
            }
        }

        for headIndex in 0..<numHeads {
            let kvHeadIndex = headIndex / groupSize
            let qBase = headIndex * headDim
            let qSlice = Array(query[qBase..<(qBase + headDim)])
            var scores: [Float] = []
            scores.reserveCapacity(keyRows.count)
            for tokenIndex in 0..<keyRows.count {
                scores.append(try TurboQuantReferenceEncoder.approximateScore(
                    query: qSlice,
                    runtimeRow: encodedKeys[tokenIndex][kvHeadIndex],
                    residualWeight: TurboQuantV2Contract.keyResidualScale,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual,
                    scale: scale
                ))
            }

            let weights = softmax(scores, chunkSize: scores.count)
            let outputBase = headIndex * headDim
            for tokenIndex in 0..<valueRows.count {
                let row = valueRows[tokenIndex]
                let valueBase = kvHeadIndex * headDim
                let valueSlice = Array(row[valueBase..<(valueBase + headDim)])
                for dim in 0..<headDim {
                    output[outputBase + dim] += weights[tokenIndex] * valueSlice[dim]
                }
            }
        }

        return output
    }

    private func decodeTurboValueRows(
        valueRows: [[Float]],
        valuePreset: TurboQuantPreset
    ) throws -> [[Float]] {
        let headDim = 128
        let numKVHeads = 8
        return try valueRows.map { row in
            try (0..<numKVHeads).flatMap { kvHeadIndex in
                let base = kvHeadIndex * headDim
                let valueSlice = Array(row[base..<(base + headDim)])
                let encoded = try TurboQuantReferenceEncoder.encode(
                    valueSlice,
                    preset: valuePreset,
                    outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
                return try TurboQuantReferenceEncoder.approximateDecode(
                    encoded,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
            }
        }
    }

    private func turboGroupedAttentionCPU(
        query: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]],
        keyPreset: TurboQuantPreset,
        valuePreset: TurboQuantPreset
    ) throws -> [Float] {
        let headDim = 128
        let numHeads = 16
        let numKVHeads = 8
        let groupSize = numHeads / numKVHeads
        let scale = 1.0 / sqrt(Float(headDim))
        var output = [Float](repeating: 0, count: numHeads * headDim)

        var encodedKeys: [[TurboQuantRuntimeRow]] = Array(repeating: [], count: keyRows.count)
        var encodedValues: [[Float]] = Array(repeating: [], count: valueRows.count)
        for tokenIndex in 0..<keyRows.count {
            encodedKeys[tokenIndex] = try (0..<numKVHeads).map { kvHeadIndex in
                let base = kvHeadIndex * headDim
                let keySlice = Array(keyRows[tokenIndex][base..<(base + headDim)])
                let encoded = try TurboQuantReferenceEncoder.encode(
                    keySlice,
                    preset: keyPreset,
                    outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual
                )
                return try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
            }
            encodedValues[tokenIndex] = try (0..<numKVHeads).flatMap { kvHeadIndex in
                let base = kvHeadIndex * headDim
                let valueSlice = Array(valueRows[tokenIndex][base..<(base + headDim)])
                let encoded = try TurboQuantReferenceEncoder.encode(
                    valueSlice,
                    preset: valuePreset,
                    outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
                return try TurboQuantReferenceEncoder.approximateDecode(
                    encoded,
                    rotationSeed: TurboQuantSeeds.valueRotation,
                    residualSeed: TurboQuantSeeds.valueResidual
                )
            }
        }

        for headIndex in 0..<numHeads {
            let kvHeadIndex = headIndex / groupSize
            let qBase = headIndex * headDim
            let qSlice = Array(query[qBase..<(qBase + headDim)])
            var scores: [Float] = []
            scores.reserveCapacity(keyRows.count)
            for tokenIndex in 0..<keyRows.count {
                scores.append(try TurboQuantReferenceEncoder.approximateScore(
                    query: qSlice,
                    runtimeRow: encodedKeys[tokenIndex][kvHeadIndex],
                    residualWeight: TurboQuantV2Contract.keyResidualScale,
                    rotationSeed: TurboQuantSeeds.keyRotation,
                    residualSeed: TurboQuantSeeds.keyResidual,
                    scale: scale
                ))
            }

            let weights = softmax(scores, chunkSize: scores.count)
            let outputBase = headIndex * headDim
            for tokenIndex in 0..<encodedValues.count {
                let valueBase = kvHeadIndex * headDim
                for dim in 0..<headDim {
                    output[outputBase + dim] += weights[tokenIndex] * encodedValues[tokenIndex][valueBase + dim]
                }
            }
        }

        return output
    }

    private func turboGroupedAttentionGPU(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        query: [Float],
        keyRows: [[Float]],
        valueRows: [[Float]]
    ) throws -> [Float] {
        let numHeads = 16
        let numKVHeads = 8
        let kvSeqLen = keyRows.count
        let headDim = 128
        let kernel = try TurboQuantKernel(device: device)
        let keyBuffers = try quantizeTurboRowsGPU(
            device: device,
            commandQueue: commandQueue,
            kernel: kernel,
            rows: keyRows,
            preset: TurboQuantV2Contract.keyPreset,
            outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
            signs: kernel.keySigns
        )
        let valueBuffers = try quantizeTurboRowsGPU(
            device: device,
            commandQueue: commandQueue,
            kernel: kernel,
            rows: valueRows,
            preset: TurboQuantV2Contract.valuePreset,
            outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
            signs: kernel.valueSigns
        )
        let qBuffer = device.makeBuffer(
            bytes: query,
            length: query.count * MemoryLayout<Float>.stride
        )!
        let outputBuffer = device.makeBuffer(length: query.count * MemoryLayout<Float>.stride)!
        let keyLayout = try TurboQuantV2Contract.makeKeyLayout()
        let valueLayout = try TurboQuantV2Contract.makeValueLayout()
        let keyDescriptor = TurboQuantV2Contract.keyPreset.descriptor
        let valueDescriptor = TurboQuantV2Contract.valuePreset.descriptor
        var params = TurboQuantAttentionParams(
            seqLen: 1,
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(numHeads / numKVHeads),
            scale: 1.0 / sqrt(Float(headDim)),
            keyResidualScale: TurboQuantV2Contract.keyResidualScale,
            valueResidualScale: TurboQuantV2Contract.valueResidualScale(forLayer: 0, layerCount: 1),
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: UInt32(kvSeqLen),
            qOffset: UInt32(kvSeqLen - 1),
            codeWordsPerRow: UInt32(keyLayout.codeWordsPerRow),
            regularBits: UInt32(keyDescriptor.regularBits),
            highPrecisionBits: UInt32(keyDescriptor.highPrecisionBits),
            valueCodeWordsPerRow: UInt32(valueLayout.codeWordsPerRow),
            valueRegularBits: UInt32(valueDescriptor.regularBits),
            valueHighPrecisionBits: UInt32(valueDescriptor.highPrecisionBits),
            reserved: 0
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestIssue.commandEncoderUnavailable
        }

        encoder.setComputePipelineState(kernel.decodeAttentionPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffers.codes, offset: 0, index: 1)
        encoder.setBuffer(keyBuffers.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(keyBuffers.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(keyBuffers.metadata, offset: 0, index: 4)
        encoder.setBuffer(valueBuffers.codes, offset: 0, index: 5)
        encoder.setBuffer(valueBuffers.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(valueBuffers.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(valueBuffers.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keySigns.rotation, offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        return Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: query.count),
                count: query.count
            )
        )
    }

    private func turboGroupedAttentionPrefillGPU(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        queryRows: [[Float]],
        keyRows: [[Float]],
        valueRows: [[Float]]
    ) throws -> [Float] {
        let numHeads = 16
        let numKVHeads = 8
        let kvSeqLen = keyRows.count
        let headDim = 128
        let qDim = numHeads * headDim
        let kernel = try TurboQuantKernel(device: device)
        let keyBuffers = try quantizeTurboRowsGPU(
            device: device,
            commandQueue: commandQueue,
            kernel: kernel,
            rows: keyRows,
            preset: TurboQuantV2Contract.keyPreset,
            outlierSelection: TurboQuantV2Contract.keyOutlierSelection,
            signs: kernel.keySigns
        )
        let valueBuffers = try quantizeTurboRowsGPU(
            device: device,
            commandQueue: commandQueue,
            kernel: kernel,
            rows: valueRows,
            preset: TurboQuantV2Contract.valuePreset,
            outlierSelection: TurboQuantV2Contract.valueOutlierSelection,
            signs: kernel.valueSigns
        )
        let flattenedQueries = queryRows.flatMap { $0 }
        let qBuffer = device.makeBuffer(
            bytes: flattenedQueries,
            length: flattenedQueries.count * MemoryLayout<Float>.stride
        )!
        let outputBuffer = device.makeBuffer(length: flattenedQueries.count * MemoryLayout<Float>.stride)!
        let keyLayout = try TurboQuantV2Contract.makeKeyLayout()
        let valueLayout = try TurboQuantV2Contract.makeValueLayout()
        let keyDescriptor = TurboQuantV2Contract.keyPreset.descriptor
        let valueDescriptor = TurboQuantV2Contract.valuePreset.descriptor
        var params = TurboQuantAttentionParams(
            seqLen: UInt32(kvSeqLen),
            headDim: UInt32(headDim),
            numHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads),
            groupSize: UInt32(numHeads / numKVHeads),
            scale: 1.0 / sqrt(Float(headDim)),
            keyResidualScale: TurboQuantV2Contract.keyResidualScale,
            valueResidualScale: TurboQuantV2Contract.valueResidualScale(forLayer: 0, layerCount: 1),
            causal: 1,
            kvBlockSize: UInt32(GQAKernel.blockSize),
            qBlockSize: UInt32(GQAKernel.blockSize),
            kvSeqLen: UInt32(kvSeqLen),
            qOffset: 0,
            codeWordsPerRow: UInt32(keyLayout.codeWordsPerRow),
            regularBits: UInt32(keyDescriptor.regularBits),
            highPrecisionBits: UInt32(keyDescriptor.highPrecisionBits),
            valueCodeWordsPerRow: UInt32(valueLayout.codeWordsPerRow),
            valueRegularBits: UInt32(valueDescriptor.regularBits),
            valueHighPrecisionBits: UInt32(valueDescriptor.highPrecisionBits),
            reserved: 0
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestIssue.commandEncoderUnavailable
        }

        encoder.setComputePipelineState(kernel.attentionPipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffers.codes, offset: 0, index: 1)
        encoder.setBuffer(keyBuffers.residualSigns, offset: 0, index: 2)
        encoder.setBuffer(keyBuffers.outlierMask, offset: 0, index: 3)
        encoder.setBuffer(keyBuffers.metadata, offset: 0, index: 4)
        encoder.setBuffer(valueBuffers.codes, offset: 0, index: 5)
        encoder.setBuffer(valueBuffers.residualSigns, offset: 0, index: 6)
        encoder.setBuffer(valueBuffers.outlierMask, offset: 0, index: 7)
        encoder.setBuffer(valueBuffers.metadata, offset: 0, index: 8)
        encoder.setBuffer(outputBuffer, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantAttentionParams>.stride, index: 10)
        encoder.setBuffer(kernel.keySigns.rotation, offset: 0, index: 11)
        encoder.setBuffer(kernel.keySigns.residual, offset: 0, index: 12)
        encoder.setBuffer(kernel.valueSigns.rotation, offset: 0, index: 13)
        encoder.setBuffer(kernel.valueSigns.residual, offset: 0, index: 14)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: GQAKernel.blockSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let allOutputs = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: flattenedQueries.count),
                count: flattenedQueries.count
            )
        )
        let lastTokenBase = (kvSeqLen - 1) * qDim
        return Array(allOutputs[lastTokenBase..<(lastTokenBase + qDim)])
    }

    private func quantizeTurboRowsGPU(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        kernel: TurboQuantKernel,
        rows: [[Float]],
        preset: TurboQuantPreset,
        outlierSelection: TurboQuantOutlierSelection,
        signs: TurboQuantSignBuffers
    ) throws -> TestTurboQuantMetalBuffers {
        let rowCount = rows.count * 8
        let layout = try TurboQuantLayout(preset: preset)
        let flattened = rows.flatMap { row in
            stride(from: 0, to: row.count, by: 128).flatMap { base in
                Array(row[base..<(base + 128)])
            }
        }

        guard let sourceBuffer = device.makeBuffer(
            bytes: flattened,
            length: flattened.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let codesBuffer = device.makeBuffer(
            length: rowCount * layout.runtimeCodeWordsPerRow * MemoryLayout<UInt32>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ),
        let residualSignsBuffer = device.makeBuffer(
            length: rowCount * TurboQuantLayout.residualWordsPerRow * MemoryLayout<UInt32>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ),
        let outlierMaskBuffer = device.makeBuffer(
            length: rowCount * TurboQuantLayout.outlierMaskWordsPerRow * MemoryLayout<UInt32>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ),
        let metadataBuffer = device.makeBuffer(
            length: rowCount * TurboQuantLayout.metadataScalarsPerRow * MemoryLayout<Float>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw TestIssue.bufferAllocationFailed
        }

        var params = TurboQuantQuantizeParams(
            rowCount: UInt32(rowCount),
            sourceRowStride: UInt32(128),
            destinationRowBase: 0,
            codeWordsPerRow: UInt32(layout.codeWordsPerRow),
            regularBits: UInt32(preset.descriptor.regularBits),
            highPrecisionBits: UInt32(preset.descriptor.highPrecisionBits),
            highPrecisionChannelCount: UInt32(preset.descriptor.highPrecisionChannelCount),
            reserved: outlierSelection == .quantizationBenefit ? 1 : 0
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestIssue.commandEncoderUnavailable
        }

        encoder.setComputePipelineState(kernel.quantizePipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(codesBuffer, offset: 0, index: 1)
        encoder.setBuffer(residualSignsBuffer, offset: 0, index: 2)
        encoder.setBuffer(outlierMaskBuffer, offset: 0, index: 3)
        encoder.setBuffer(metadataBuffer, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<TurboQuantQuantizeParams>.stride, index: 5)
        encoder.setBuffer(signs.rotation, offset: 0, index: 6)
        encoder.setBuffer(signs.residual, offset: 0, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(rowCount, kernel.quantizePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        return TestTurboQuantMetalBuffers(
            codes: codesBuffer,
            residualSigns: residualSignsBuffer,
            outlierMask: outlierMaskBuffer,
            metadata: metadataBuffer
        )
    }

    private func cpuPresetFidelitySummaries(
        inputs: LlamaAttentionInputsTrace,
        preset: TurboQuantPreset,
        outlierSelection: TurboQuantOutlierSelection
    ) throws -> [LlamaAttentionScoreSummary] {
        let tokenCount = inputs.tokenIDs.count
        let headDim = 128
        let numHeads = 16
        let numKVHeads = 8
        let groupSize = numHeads / numKVHeads
        let scale = 1.0 / sqrt(Float(headDim))

        return try inputs.queries.enumerated().map { layerIndex, query in
            let layerExactKeys = inputs.exactKeys[layerIndex]
            var exactScores: [Float] = []
            var decodedScores: [Float] = []
            exactScores.reserveCapacity(numHeads * tokenCount)
            decodedScores.reserveCapacity(numHeads * tokenCount)

            for headIndex in 0..<numHeads {
                let kvHeadIndex = headIndex / groupSize
                let qBase = headIndex * headDim
                let qSlice = Array(query[qBase..<(qBase + headDim)])

                for tokenIndex in 0..<tokenCount {
                    let exactRow = layerExactKeys[tokenIndex]
                    let kBase = kvHeadIndex * headDim
                    let exactKeySlice = Array(exactRow[kBase..<(kBase + headDim)])
                    exactScores.append(dot(qSlice, exactKeySlice) * scale)

                    let encoded = try TurboQuantReferenceEncoder.encode(
                        exactKeySlice,
                        preset: preset,
                        outlierSelection: outlierSelection,
                        rotationSeed: TurboQuantSeeds.keyRotation,
                        residualSeed: TurboQuantSeeds.keyResidual
                    )
                    let decoded = try TurboQuantReferenceEncoder.approximateDecode(
                        encoded,
                        residualWeight: TurboQuantV2Contract.keyResidualScale,
                        rotationSeed: TurboQuantSeeds.keyRotation,
                        residualSeed: TurboQuantSeeds.keyResidual
                    )
                    decodedScores.append(dot(qSlice, decoded) * scale)
                }
            }

            let exactSoftmax = softmax(exactScores, chunkSize: tokenCount)
            let decodedSoftmax = softmax(decodedScores, chunkSize: tokenCount)
            return LlamaAttentionScoreSummary(
                layerIndex: layerIndex,
                exactVsStoredScoreMaxAbs: .zero,
                exactVsStoredScoreMSE: .zero,
                exactVsDecodedScoreMaxAbs: maxAbsoluteDelta(exactScores, decodedScores),
                exactVsDecodedScoreMSE: meanSquaredError(exactScores, decodedScores),
                exactVsStoredSoftmaxMaxAbs: .zero,
                exactVsStoredSoftmaxMSE: .zero,
                exactVsDecodedSoftmaxMaxAbs: maxAbsoluteDelta(exactSoftmax, decodedSoftmax),
                exactVsDecodedSoftmaxMSE: meanSquaredError(exactSoftmax, decodedSoftmax)
            )
        }
    }

    private func softmax(_ values: [Float], chunkSize: Int) -> [Float] {
        var result = values
        for offset in stride(from: 0, to: values.count, by: chunkSize) {
            let end = min(offset + chunkSize, values.count)
            let chunk = Array(values[offset..<end])
            let maxValue = chunk.max() ?? 0
            let exps = chunk.map { Foundation.exp($0 - maxValue) }
            let sum = exps.reduce(Float.zero, +)
            for index in 0..<chunk.count {
                result[offset + index] = sum > 0 ? exps[index] / sum : 0
            }
        }
        return result
    }

    private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, pair in partial + pair.0 * pair.1 }
    }

    private func meanSquaredError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta * delta
        } / Float(max(lhs.count, 1))
    }

    private func maxAbsoluteDelta(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
    }

    private func argmax(_ logits: [Float]) -> Int {
        var maxValue: Float = -.infinity
        var maxIndex = 0
        for (index, value) in logits.enumerated() where value > maxValue {
            maxValue = value
            maxIndex = index
        }
        return maxIndex
    }
}

private struct LayerDelta: Sendable {
    let layerIndex: Int
    let maxAbsoluteDelta: Float
}

private struct TurboQuantAttentionParams {
    var seqLen: UInt32
    var headDim: UInt32
    var numHeads: UInt32
    var numKVHeads: UInt32
    var groupSize: UInt32
    var scale: Float
    var keyResidualScale: Float
    var valueResidualScale: Float
    var causal: UInt32
    var kvBlockSize: UInt32
    var qBlockSize: UInt32
    var kvSeqLen: UInt32
    var qOffset: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var valueCodeWordsPerRow: UInt32
    var valueRegularBits: UInt32
    var valueHighPrecisionBits: UInt32
    var reserved: UInt32
}

private struct TestTurboQuantMetalBuffers {
    let codes: MTLBuffer
    let residualSigns: MTLBuffer
    let outlierMask: MTLBuffer
    let metadata: MTLBuffer
}

private enum TestIssue: Error {
    case commandEncoderUnavailable
    case bufferAllocationFailed
}

private struct TurboQuantQuantizeParams {
    var rowCount: UInt32
    var sourceRowStride: UInt32
    var destinationRowBase: UInt32
    var codeWordsPerRow: UInt32
    var regularBits: UInt32
    var highPrecisionBits: UInt32
    var highPrecisionChannelCount: UInt32
    var reserved: UInt32
}
