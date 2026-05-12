import Metal
import Testing
@testable import EdgeRunnerMetal

private func cpuGemmaRMSNorm(
    _ input: [Float],
    weight: [Float],
    rows: Int,
    cols: Int,
    eps: Float
) -> [Float] {
    var output = [Float](repeating: 0, count: input.count)
    for row in 0..<rows {
        let offset = row * cols
        var meanSquare: Float = 0
        for col in 0..<cols {
            let value = input[offset + col]
            meanSquare += value * value
        }
        let scale = 1 / sqrt(meanSquare / Float(cols) + eps)
        for col in 0..<cols {
            output[offset + col] = input[offset + col] * scale * weight[col]
        }
    }
    return output
}

private func cpuGemmaResidualRMSNormAddRows(
    residual: [Float],
    input: [Float],
    weight: [Float],
    rows: Int,
    cols: Int,
    eps: Float
) -> [Float] {
    var output = [Float](repeating: 0, count: input.count)
    for row in 0..<rows {
        let offset = row * cols
        var meanSquare: Float = 0
        for col in 0..<cols {
            let value = input[offset + col]
            meanSquare += value * value
        }
        let scale = 1 / sqrt(meanSquare / Float(cols) + eps)
        for col in 0..<cols {
            output[offset + col] = residual[offset + col] + input[offset + col] * scale * weight[col]
        }
    }
    return output
}

@Suite("Gemma4DecodeKernels")
struct Gemma4DecodeKernelTests {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw TestError.noMetal
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    @Test("Gemma RMSNorm uses direct weight convention")
    func gemmaRMSNormMatchesCPU() async throws {
        let rows = 3
        let cols = 257
        let input = (0..<(rows * cols)).map { Float($0 % 19 - 9) / 7.0 }
        let weight = (0..<cols).map { Float($0 % 11 - 5) / 23.0 }
        let expected = cpuGemmaRMSNorm(input, weight: weight, rows: rows, cols: cols, eps: 1e-6)

        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runRMSNorm(
            input: input,
            weight: weight,
            rows: rows,
            cols: cols,
            eps: 1e-6,
            commandQueue: commandQueue
        )

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Gemma RMSNorm encodes into caller-owned scratch buffers")
    func gemmaRMSNormEncodesIntoExistingCommandBuffer() async throws {
        let rows = 1
        let cols = 2560
        let input = (0..<(rows * cols)).map { Float($0 % 31 - 15) / 29.0 }
        let weight = (0..<cols).map { Float($0 % 13 - 6) / 17.0 }
        let expected = cpuGemmaRMSNorm(input, weight: weight, rows: rows, cols: cols, eps: 1e-6)

        guard let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let weightBuffer = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernels = try Gemma4DecodeKernels(device: device)
        try kernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            outputBuffer: outputBuffer,
            rows: rows,
            cols: cols,
            eps: 1e-6
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let actual = Array(UnsafeBufferPointer(start: pointer, count: input.count))
        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Scalar multiply applies layer output scale in place")
    func scalarMultiplyInPlace() throws {
        let values: [Float] = [1, -2, 3.5, -4.25, 0]
        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try kernels.runMulScalar(values: values, scale: 0.25)
        #expect(actual == [0.25, -0.5, 0.875, -1.0625, 0])
    }

    @Test("Residual RMSNorm add matches CPU")
    func residualRMSNormAddMatchesCPU() async throws {
        let count = 2560
        let residual = (0..<count).map { Float($0 % 23 - 11) / 31.0 }
        let input = (0..<count).map { Float($0 % 29 - 14) / 37.0 }
        let weight = (0..<count).map { Float($0 % 17 - 8) / 19.0 }
        var meanSquare: Float = 0
        for value in input {
            meanSquare += value * value
        }
        let scale = 1 / sqrt(meanSquare / Float(count) + 1e-6)
        let expected = input.indices.map { residual[$0] + input[$0] * scale * weight[$0] }

        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runResidualRMSNormAdd(
            residual: residual,
            input: input,
            weight: weight,
            eps: 1e-6,
            commandQueue: commandQueue
        )

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Residual RMSNorm add rows matches per-row CPU")
    func residualRMSNormAddRowsMatchesPerRowCPU() async throws {
        let rows = 3
        let cols = 257
        let count = rows * cols
        let residual = (0..<count).map { Float($0 % 23 - 11) / 31.0 }
        let input = (0..<count).map { Float($0 % 29 - 14) / 37.0 }
        let weight = (0..<cols).map { Float($0 % 17 - 8) / 19.0 }
        let expected = cpuGemmaResidualRMSNormAddRows(
            residual: residual,
            input: input,
            weight: weight,
            rows: rows,
            cols: cols,
            eps: 1e-6
        )

        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runResidualRMSNormAddRows(
            residual: residual,
            input: input,
            weight: weight,
            rows: rows,
            cols: cols,
            eps: 1e-6,
            commandQueue: commandQueue
        )

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }

    @Test("F32 to F16 store supports output byte offsets")
    func storeF32ToF16SupportsOutputOffset() async throws {
        let values: [Float] = [-2.5, -0.25, 0, 0.5, 3.75]
        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runStoreF32ToF16(
            values: values,
            outputOffset: 3 * MemoryLayout<Float16>.stride
        )

        #expect(actual == values.map(Float16.init))
    }

    @Test("F32 to F16 store supports input and output byte offsets")
    func storeF32ToF16SupportsInputAndOutputOffsets() async throws {
        guard let inputBuffer = device.makeBuffer(
            bytes: [Float](arrayLiteral: 100, 200, -1.5, 0.25, 3.5),
            length: 5 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: 6 * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernels = try Gemma4DecodeKernels(device: device)
        try kernels.encodeStoreF32ToF16(
            commandBuffer: commandBuffer,
            inputBuffer: inputBuffer,
            inputOffset: 2 * MemoryLayout<Float>.stride,
            outputBuffer: outputBuffer,
            outputOffset: 1 * MemoryLayout<Float16>.stride,
            count: 3
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: 6)
        let actual = Array(UnsafeBufferPointer(start: pointer.advanced(by: 1), count: 3))
        #expect(actual == [-1.5, 0.25, 3.5].map(Float16.init))
    }

    @Test("F32 to F16 store supports chunked KV row writes")
    func storeF32ToF16SupportsChunkedKVRowWrites() async throws {
        let rowWidth = 4
        let sourceRows: [Float] = [
            10, 11, 12, 13,
            20, 21, 22, 23,
            30, 31, 32, 33
        ]
        guard let inputBuffer = device.makeBuffer(
            bytes: sourceRows,
            length: sourceRows.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let kvCacheBuffer = device.makeBuffer(
            length: 8 * rowWidth * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        let kernels = try Gemma4DecodeKernels(device: device)
        for (sourceRow, cacheRow) in zip([0, 1, 2], [1, 3, 6]) {
            try kernels.encodeStoreF32ToF16(
                commandBuffer: commandBuffer,
                inputBuffer: inputBuffer,
                inputOffset: sourceRow * rowWidth * MemoryLayout<Float>.stride,
                outputBuffer: kvCacheBuffer,
                outputOffset: cacheRow * rowWidth * MemoryLayout<Float16>.stride,
                count: rowWidth
            )
        }

        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = kvCacheBuffer.contents().bindMemory(to: Float16.self, capacity: 8 * rowWidth)
        for (sourceRow, cacheRow) in zip([0, 1, 2], [1, 3, 6]) {
            let actual = Array(
                UnsafeBufferPointer(start: pointer.advanced(by: cacheRow * rowWidth), count: rowWidth)
            )
            let expected = sourceRows[(sourceRow * rowWidth)..<((sourceRow + 1) * rowWidth)].map(Float16.init)
            #expect(actual == expected)
        }
    }

    @Test("GeGLU kernel remains finite for large gates")
    func gegluLargeGatesStayFinite() throws {
        let kernel = try GeGLUKernel(device: device)
        let actual = try kernel.run(gate: [-100, 100], up: [3, 4])
        #expect(actual == [0, 400])
    }

    @Test("Q6_K token embedding gather dequantizes selected rows")
    func q6KTokenEmbeddingGather() async throws {
        let rowWidth = 512
        let rowStrideBytes = (rowWidth / 256) * 210
        let table = makeQ6KTable(rowCount: 3, rowWidth: rowWidth)
        let tokens: [Int] = [2, 0]
        let expected = tokens.flatMap { token in
            cpuQ6KRow(table, row: token, rowStrideBytes: rowStrideBytes, rowWidth: rowWidth, scale: 2.0)
        }

        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runGatherQ6KTokenEmbedding(
            table: table,
            tokenIDs: tokens,
            rowWidth: rowWidth,
            rowStrideBytes: rowStrideBytes,
            tableByteOffset: 0,
            scale: 2.0,
            commandQueue: commandQueue
        )

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Windowed F16 KV GQA matches CPU for local and global Gemma head dims")
    func windowedF16KVGQA() async throws {
        for headDim in [256, 512] {
            let numHeads = 8
            let numKVHeads = 2
            let capacity = 8
            let kvStart = 5
            let kvCount = 6
            let q = (0..<(numHeads * headDim)).map { Float(($0 % 13) - 6) / 17.0 }
            let k = (0..<(capacity * numKVHeads * headDim)).map { Float16(Float(($0 % 11) - 5) / 19.0) }
            let v = (0..<(capacity * numKVHeads * headDim)).map { Float16(Float(($0 % 7) - 3) / 11.0) }
            let expected = cpuWindowedGQA(
                q: q,
                k: k,
                v: v,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                kvStart: kvStart,
                kvCount: kvCount,
                capacity: capacity,
                attentionScale: 1.0 / sqrt(Float(headDim))
            )

            let kernels = try Gemma4DecodeKernels(device: device)
            let actual = try await kernels.runDecodeGQAF16KVWindowed(
                q: q,
                k: k,
                v: v,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                kvStart: kvStart,
                kvCount: kvCount,
                kvCapacity: capacity,
                attentionScale: 1.0 / sqrt(Float(headDim)),
                commandQueue: commandQueue
            )

            for index in actual.indices {
                #expect(abs(actual[index] - expected[index]) < 2e-3)
            }
        }
    }

    @Test("Windowed F16 KV GQA supports Gemma 4 unscaled attention")
    func windowedF16KVGQAUnscaledAttention() async throws {
        let numHeads = 8
        let numKVHeads = 2
        let headDim = 256
        let capacity = 4
        let kvStart = 0
        let kvCount = 4
        let q = (0..<(numHeads * headDim)).map { Float(($0 % 17) - 8) / 11.0 }
        let k = (0..<(capacity * numKVHeads * headDim)).map { Float16(Float(($0 % 13) - 6) / 13.0) }
        let v = (0..<(capacity * numKVHeads * headDim)).map { Float16(Float(($0 % 7) - 3) / 7.0) }
        let expected = cpuWindowedGQA(
            q: q,
            k: k,
            v: v,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            capacity: capacity,
            attentionScale: 1.0
        )

        let kernels = try Gemma4DecodeKernels(device: device)
        let actual = try await kernels.runDecodeGQAF16KVWindowed(
            q: q,
            k: k,
            v: v,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            kvCapacity: capacity,
            attentionScale: 1.0,
            commandQueue: commandQueue
        )

        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < 2e-3)
        }
    }

    @Test("Fast windowed F16 KV GQA matches CPU with real Gemma cache capacities")
    func fastWindowedF16KVGQARealCacheCapacities() async throws {
        for shape in [
            (headDim: 256, capacity: 512, kvStart: 0, kvCount: 512),
            (headDim: 256, capacity: 512, kvStart: 33, kvCount: 512),
            (headDim: 512, capacity: 4096, kvStart: 127, kvCount: 384),
            (headDim: 512, capacity: 4096, kvStart: 1536, kvCount: 512),
            (headDim: 512, capacity: 4096, kvStart: 3584, kvCount: 512),
        ] {
            let numHeads = 8
            let numKVHeads = 2
            let q = (0..<(numHeads * shape.headDim)).map { Float(($0 % 29) - 14) / 31.0 }
            let k = (0..<(shape.capacity * numKVHeads * shape.headDim)).map {
                Float16(Float(($0 % 23) - 11) / 37.0)
            }
            let v = (0..<(shape.capacity * numKVHeads * shape.headDim)).map {
                Float16(Float(($0 % 19) - 9) / 23.0)
            }
            let expected = cpuWindowedGQA(
                q: q,
                k: k,
                v: v,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: shape.headDim,
                kvStart: shape.kvStart,
                kvCount: shape.kvCount,
                capacity: shape.capacity,
                attentionScale: 1.0
            )

            let actual = try await runFastWindowedGQA(
                q: q,
                k: k,
                v: v,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                headDim: shape.headDim,
                kvStart: shape.kvStart,
                kvCount: shape.kvCount,
                kvCapacity: shape.capacity,
                attentionScale: 1.0
            )

            for index in actual.indices {
                #expect(abs(actual[index] - expected[index]) < 2e-3)
            }
        }
    }

    @Test("Cache-backed F16 KV GQA rejects undersized real cache buffers")
    func windowedF16KVGQARejectsUndersizedCacheBuffers() async throws {
        let kernels = try Gemma4DecodeKernels(device: device)
        let q = [Float](repeating: 0.25, count: 8 * 512)
        guard let keyBuffer = device.makeBuffer(
            length: 4095 * 2 * 512 * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ),
        let valueBuffer = device.makeBuffer(
            length: 4096 * 2 * 512 * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw TestError.noMetal
        }

        await #expect(throws: Gemma4DecodeKernelError.invalidBufferShape) {
            _ = try await kernels.runDecodeGQAF16KVWindowed(
                q: q,
                keyCacheBuffer: keyBuffer,
                valueCacheBuffer: valueBuffer,
                numHeads: 8,
                numKVHeads: 2,
                headDim: 512,
                kvStart: 0,
                kvCount: 512,
                kvCapacity: 4096,
                attentionScale: 1.0,
                commandQueue: commandQueue
            )
        }
    }

    private func makeQ6KTable(rowCount: Int, rowWidth: Int) -> [UInt8] {
        let rowStrideBytes = (rowWidth / 256) * 210
        var table = [UInt8](repeating: 0, count: rowCount * rowStrideBytes)
        for row in 0..<rowCount {
            for block in 0..<(rowWidth / 256) {
                let start = row * rowStrideBytes + block * 210
                fillQ6KBlock(&table, offset: start, seed: UInt8(row * 17 + block * 11))
            }
        }
        return table
    }

    private func fillQ6KBlock(_ table: inout [UInt8], offset: Int, seed: UInt8) {
        for halfBlock in 0..<2 {
            let outBase = halfBlock * 128
            let qlBase = offset + halfBlock * 64
            let qhBase = offset + 128 + halfBlock * 32
            for lane in 0..<32 {
                let q1 = UInt8((Int(seed) + outBase + lane) % 64)
                let q2 = UInt8((Int(seed) + outBase + 32 + lane) % 64)
                let q3 = UInt8((Int(seed) + outBase + 64 + lane) % 64)
                let q4 = UInt8((Int(seed) + outBase + 96 + lane) % 64)

                table[qlBase + lane] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
                table[qlBase + 32 + lane] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
                table[qhBase + lane] = ((q1 >> 4) & 0x03)
                    | (((q2 >> 4) & 0x03) << 2)
                    | (((q3 >> 4) & 0x03) << 4)
                    | (((q4 >> 4) & 0x03) << 6)
            }
        }
        for group in 0..<16 {
            table[offset + 192 + group] = UInt8(bitPattern: Int8((group % 5) - 2))
        }
        let bits = Float16(0.125).bitPattern
        table[offset + 208] = UInt8(bits & 0x00FF)
        table[offset + 209] = UInt8(bits >> 8)
    }

    private func cpuQ6KRow(
        _ table: [UInt8],
        row: Int,
        rowStrideBytes: Int,
        rowWidth: Int,
        scale outputScale: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rowWidth)
        for block in 0..<(rowWidth / 256) {
            let start = row * rowStrideBytes + block * 210
            let dBits = UInt16(table[start + 208]) | (UInt16(table[start + 209]) << 8)
            let d = Float(Float16(bitPattern: dBits))
            for halfBlock in 0..<2 {
                let outBase = halfBlock * 128
                let qlBase = start + halfBlock * 64
                let qhBase = start + 128 + halfBlock * 32
                let scaleBase = start + 192 + halfBlock * 8
                for lane in 0..<32 {
                    let scaleOffset = lane / 16
                    let q1 = Int((table[qlBase + lane] & 0x0F) | (((table[qhBase + lane] >> 0) & 0x03) << 4)) - 32
                    let q2 = Int((table[qlBase + 32 + lane] & 0x0F) | (((table[qhBase + lane] >> 2) & 0x03) << 4)) - 32
                    let q3 = Int((table[qlBase + lane] >> 4) | (((table[qhBase + lane] >> 4) & 0x03) << 4)) - 32
                    let q4 = Int((table[qlBase + 32 + lane] >> 4) | (((table[qhBase + lane] >> 6) & 0x03) << 4)) - 32
                    let s1 = Int8(bitPattern: table[scaleBase + scaleOffset + 0])
                    let s2 = Int8(bitPattern: table[scaleBase + scaleOffset + 2])
                    let s3 = Int8(bitPattern: table[scaleBase + scaleOffset + 4])
                    let s4 = Int8(bitPattern: table[scaleBase + scaleOffset + 6])
                    output[block * 256 + outBase + lane] = d * Float(s1) * Float(q1) * outputScale
                    output[block * 256 + outBase + 32 + lane] = d * Float(s2) * Float(q2) * outputScale
                    output[block * 256 + outBase + 64 + lane] = d * Float(s3) * Float(q3) * outputScale
                    output[block * 256 + outBase + 96 + lane] = d * Float(s4) * Float(q4) * outputScale
                }
            }
        }
        return output
    }

    private func cpuWindowedGQA(
        q: [Float],
        k: [Float16],
        v: [Float16],
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        capacity: Int,
        attentionScale: Float
    ) -> [Float] {
        let groupSize = numHeads / numKVHeads
        let kvStride = numKVHeads * headDim
        var output = [Float](repeating: 0, count: numHeads * headDim)

        for head in 0..<numHeads {
            let kvHead = head / groupSize
            var scores = [Float](repeating: 0, count: kvCount)
            for kvIndex in 0..<kvCount {
                let physical = (kvStart + kvIndex) % capacity
                let kBase = physical * kvStride + kvHead * headDim
                let qBase = head * headDim
                var dot: Float = 0
                for dim in 0..<headDim {
                    dot += q[qBase + dim] * Float(k[kBase + dim])
                }
                scores[kvIndex] = dot * attentionScale
            }
            let maxScore = scores.max() ?? 0
            let exps = scores.map { Foundation.exp($0 - maxScore) }
            let sum = exps.reduce(0, +)
            for dim in 0..<headDim {
                var value: Float = 0
                for kvIndex in 0..<kvCount {
                    let physical = (kvStart + kvIndex) % capacity
                    let vBase = physical * kvStride + kvHead * headDim
                    value += (exps[kvIndex] / sum) * Float(v[vBase + dim])
                }
                output[head * headDim + dim] = value
            }
        }
        return output
    }

    private func runFastWindowedGQA(
        q: [Float],
        k: [Float16],
        v: [Float16],
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        kvStart: Int,
        kvCount: Int,
        kvCapacity: Int,
        attentionScale: Float
    ) async throws -> [Float] {
        let kernels = try Gemma4DecodeKernels(device: device)
        guard let qBuffer = device.makeBuffer(
            bytes: q,
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let kBuffer = device.makeBuffer(
            bytes: k,
            length: k.count * MemoryLayout<Float16>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ),
        let vBuffer = device.makeBuffer(
            bytes: v,
            length: v.count * MemoryLayout<Float16>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ),
        let outputBuffer = device.makeBuffer(
            length: q.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw TestError.noMetal
        }

        try kernels.encodeDecodeGQAF16KVWindowedFast(
            commandBuffer: commandBuffer,
            qBuffer: qBuffer,
            keyCacheBuffer: kBuffer,
            valueCacheBuffer: vBuffer,
            outputBuffer: outputBuffer,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            kvStart: kvStart,
            kvCount: kvCount,
            kvCapacity: kvCapacity,
            attentionScale: attentionScale
        )
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: q.count)
        return Array(UnsafeBufferPointer(start: pointer, count: q.count))
    }

}
