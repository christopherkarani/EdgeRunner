//
//  LlamaLanguageModel+ICB.swift
//  EdgeRunner
//
//  Isolated implementation of the Metal Indirect Command Buffer (ICB) optimization
//  for the synchronous decode fast path.
//

import Metal
import Foundation
import Accelerate

extension LlamaLanguageModel {
    
    /// Encodes all 142 dispatch commands (5 per layer + final norm + LM head) into an MTLIndirectCommandBuffer.
    /// This happens exactly once. During generation, we only execute the ICB and update the small CPU paramsBuffer.
    public func buildDecodeICB(hiddenBuf: MTLBuffer) -> MTLIndirectCommandBuffer {
        let dim = config.embeddingDim
        let headDim = config.headDim
        let qDim = config.headCount * headDim
        let kvDim = config.kvHeadCount * headDim
        let interDim = config.intermediateDim
        let numHeads = config.headCount
        let numKVHeads = config.kvHeadCount
        let layerCount = config.layerCount

        let icbDesc = MTLIndirectCommandBufferDescriptor()
        icbDesc.commandTypes = [.concurrentDispatchThreads, .concurrentDispatch]
        icbDesc.inheritPipelineState = false
        icbDesc.inheritBuffers = false
        icbDesc.maxVertexBufferBindCount = 0
        icbDesc.maxFragmentBufferBindCount = 0
        icbDesc.maxKernelBufferBindCount = 31

        var cmdCount = layerCount * 5
        let hasLMHeadRaw = preloadedWeights.lmHeadRaw != nil
        if hasLMHeadRaw && !decodeDebugOptions.disableFusedFinalNormLMHead {
            cmdCount += 1
        } else {
            cmdCount += 1
            cmdCount += hasLMHeadRaw ? 1 : 1
        }

        guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDesc, maxCommandCount: cmdCount, options: .storageModePrivate) else {
            fatalError("Failed to create MTLIndirectCommandBuffer for decode fast path")
        }

        let qkvGridWidth = (qDim + kvDim + kvDim + 1) / 2
        let megaThreadsPerHead = 32
        let megaTotalHeads = numHeads + numKVHeads
        let megaTGWidth = min(megaThreadsPerHead, fusedNormRoPEGQAPipeline.maxTotalThreadsPerThreadgroup)
        let dimGridWidth = (dim + 1) / 2
        let interDimGridWidth = (interDim + 1) / 2

        var currentHidden = hiddenBuf
        let afterAttnBuf = scratch.afterAttn
        let outputBufA = scratch.outputA
        let outputBufB = scratch.outputB
        let allQBuf = scratch.allQ
        let allKBuf = scratch.allK
        let attnOutBuf = scratch.attnOut
        let activBuf = scratch.activ

        var c = 0
        for layerIndex in 0..<layerCount {
            let lw = preloadedWeights.layers[layerIndex]
            let layerOutputBuf = (layerIndex % 2 == 0) ? outputBufA : outputBufB
            let layerKCache = layerKCaches[layerIndex]
            let layerVCache = layerVCaches[layerIndex]

            // DISPATCH 1: Fused QKV
            let cmd1 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd1.setComputePipelineState(fusedQKVPipeline)
            cmd1.setKernelBuffer(lw.wqRaw!, offset: 0, at: 0)
            cmd1.setKernelBuffer(lw.wkRaw!, offset: 0, at: 1)
            cmd1.setKernelBuffer(lw.wvRaw!, offset: 0, at: 2)
            cmd1.setKernelBuffer(currentHidden, offset: 0, at: 3)
            cmd1.setKernelBuffer(allQBuf, offset: 0, at: 4)
            cmd1.setKernelBuffer(allKBuf, offset: 0, at: 5)
            cmd1.setKernelBuffer(layerVCache, offset: 0, at: 6)
            cmd1.setKernelBuffer(decodeParamsBuffer!, offset: 0, at: 7)
            cmd1.setKernelBuffer(lw.attnNorm, offset: 0, at: 8)
            cmd1.concurrentDispatchThreadgroups(MTLSize(width: qkvGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // DISPATCH 2: Mega-kernel Q/K norm + RoPE + GQA
            let cmd2 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd2.setComputePipelineState(fusedNormRoPEGQAPipeline)
            cmd2.setKernelBuffer(allQBuf, offset: 0, at: 0)
            cmd2.setKernelBuffer(allKBuf, offset: 0, at: 1)
            cmd2.setKernelBuffer(lw.qNorm!, offset: 0, at: 2)
            cmd2.setKernelBuffer(lw.kNorm!, offset: 0, at: 3)
            cmd2.setKernelBuffer(attnOutBuf, offset: 0, at: 4)
            cmd2.setKernelBuffer(layerKCache, offset: 0, at: 5)
            cmd2.setKernelBuffer(layerVCache, offset: 0, at: 6)
            cmd2.setKernelBuffer(decodeParamsBuffer!, offset: 256, at: 7)
            cmd2.concurrentDispatchThreads(MTLSize(width: megaThreadsPerHead, height: megaTotalHeads, depth: 1), threadsPerThreadgroup: MTLSize(width: megaTGWidth, height: 1, depth: 1))

            // DISPATCH 3: Wo + residual add
            let cmd3 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd3.setComputePipelineState(gemvAddPipeline)
            cmd3.setKernelBuffer(lw.woRaw!, offset: 0, at: 0)
            cmd3.setKernelBuffer(attnOutBuf, offset: 0, at: 1)
            cmd3.setKernelBuffer(currentHidden, offset: 0, at: 2)
            cmd3.setKernelBuffer(afterAttnBuf, offset: 0, at: 3)
            cmd3.setKernelBuffer(decodeParamsBuffer!, offset: 512, at: 4)
            cmd3.concurrentDispatchThreadgroups(MTLSize(width: dimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // DISPATCH 4: Gate + Up + SiLU
            let cmd4 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd4.setComputePipelineState(fusedGateUpSiluPipeline)
            cmd4.setKernelBuffer(lw.gateRaw!, offset: 0, at: 0)
            cmd4.setKernelBuffer(lw.upRaw!, offset: 0, at: 1)
            cmd4.setKernelBuffer(afterAttnBuf, offset: 0, at: 2)
            cmd4.setKernelBuffer(activBuf, offset: 0, at: 3)
            cmd4.setKernelBuffer(decodeParamsBuffer!, offset: 768, at: 4)
            cmd4.setKernelBuffer(lw.ffnNorm, offset: 0, at: 5)
            cmd4.concurrentDispatchThreadgroups(MTLSize(width: interDimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            // DISPATCH 5: Down + residual add
            let cmd5 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd5.setComputePipelineState(gemvAddPipeline)
            cmd5.setKernelBuffer(lw.downRaw!, offset: 0, at: 0)
            cmd5.setKernelBuffer(activBuf, offset: 0, at: 1)
            cmd5.setKernelBuffer(afterAttnBuf, offset: 0, at: 2)
            cmd5.setKernelBuffer(layerOutputBuf, offset: 0, at: 3)
            cmd5.setKernelBuffer(decodeParamsBuffer!, offset: 1024, at: 4)
            cmd5.concurrentDispatchThreadgroups(MTLSize(width: dimGridWidth, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            currentHidden = layerOutputBuf
        }

        let logitsBuf = scratch.logits
        if let lmRaw = preloadedWeights.lmHeadRaw, !decodeDebugOptions.disableFusedFinalNormLMHead {
            let cmd = icb.indirectComputeCommand(at: c)!; c += 1
            cmd.setComputePipelineState(fusedFinalNormGemvPipeline)
            cmd.setKernelBuffer(lmRaw, offset: 0, at: 0)
            cmd.setKernelBuffer(currentHidden, offset: 0, at: 1)
            cmd.setKernelBuffer(logitsBuf, offset: 0, at: 2)
            cmd.setKernelBuffer(preloadedWeights.finalNorm!, offset: 0, at: 3)
            cmd.setKernelBuffer(decodeParamsBuffer!, offset: 1792, at: 4)
            cmd.concurrentDispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            let finalOutputBuf = scratch.finalOut
            let cmd1 = icb.indirectComputeCommand(at: c)!; c += 1
            cmd1.setComputePipelineState(rmsNormKernel.pipeline)
            cmd1.setKernelBuffer(currentHidden, offset: 0, at: 0)
            cmd1.setKernelBuffer(preloadedWeights.finalNorm!, offset: 0, at: 1)
            cmd1.setKernelBuffer(finalOutputBuf, offset: 0, at: 2)
            cmd1.setKernelBuffer(decodeParamsBuffer!, offset: 1280, at: 3)
            cmd1.concurrentDispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

            if let lmRaw = preloadedWeights.lmHeadRaw {
                let cmd2 = icb.indirectComputeCommand(at: c)!; c += 1
                cmd2.setComputePipelineState(fusedQ8GemvPipeline)
                cmd2.setKernelBuffer(lmRaw, offset: 0, at: 0)
                cmd2.setKernelBuffer(finalOutputBuf, offset: 0, at: 1)
                cmd2.setKernelBuffer(logitsBuf, offset: 0, at: 2)
                cmd2.setKernelBuffer(decodeParamsBuffer!, offset: 1536, at: 3)
                cmd2.concurrentDispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                let cmd2 = icb.indirectComputeCommand(at: c)!; c += 1
                cmd2.setComputePipelineState(gemvKernel.f32Pipeline)
                cmd2.setKernelBuffer(preloadedWeights.lmHead!, offset: 0, at: 0)
                cmd2.setKernelBuffer(finalOutputBuf, offset: 0, at: 1)
                cmd2.setKernelBuffer(logitsBuf, offset: 0, at: 2)
                cmd2.setKernelBuffer(decodeParamsBuffer!, offset: 1536, at: 3)
                cmd2.concurrentDispatchThreadgroups(MTLSize(width: (config.vocabSize + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        
        // Store command count onto the ICB via a side channel or simply return it
        // We will store it in decoder state in practice
        return icb
    }

    /// Optimized Metal 3 decode using MTLIndirectCommandBuffer for zero-dispatch overhead.
    public func fusedDecodePassOptSyncICB(
        hiddenBuf: MTLBuffer,
        currentPos: Int,
        paramsBuffer: MTLBuffer
    ) throws -> MTLBuffer {
        let totalKVLen = currentPos + 1
        let paramsBase = paramsBuffer.contents()

        // Per-call dynamic parameters: update mega-kernel startPos + kvSeqLen
        (paramsBase + 256 + 12).storeBytes(of: UInt32(currentPos), as: UInt32.self)
        (paramsBase + 256 + 28).storeBytes(of: UInt32(totalKVLen), as: UInt32.self)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create command buffer")
        }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw GenerationError.modelLoadFailed(reason: "Failed to create compute encoder")
        }

        // Initialize ICB once
        if decoderState._decodeICB == nil {
            decoderState._decodeICB = buildDecodeICB(hiddenBuf: hiddenBuf)
            decoderState._decodeICBCommandCount = (preloadedWeights.lmHeadRaw != nil && !decodeDebugOptions.disableFusedFinalNormLMHead) ? (config.layerCount * 5 + 1) : (config.layerCount * 5 + 2)
        }

        let icb = decoderState._decodeICB!
        let cmdCount = decoderState._decodeICBCommandCount

        // Memory residency mapping for Metal ICB resources
        if let m4 = metal4State {
            enc.useResidencySet(m4.residencySet)
        } else {
            // Optional: fallback logic for legacy OS missing Metal4State
            // enc.useResources(allResources, usage: [.read, .write]) 
        }

        enc.executeCommandsInBuffer(icb, range: 0..<cmdCount)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }

        return scratch.logits
    }
}
