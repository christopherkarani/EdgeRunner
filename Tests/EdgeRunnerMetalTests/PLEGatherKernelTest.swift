import Testing
import Metal
import Foundation
@testable import EdgeRunnerMetal

@Suite("PLE Q8_0 gather")
struct PLEGatherKernelTests {
    @Test("Gathers single PLE row and scales by sqrt(P)")
    func gathersAndScales() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let P = 256
        let L = 42
        let vocab = 8
        let totalElems = vocab * L * P
        let rows = (0..<totalElems).map { i in Float(i % 137) / 137.0 - 0.5 }

        let (q8Bytes, dequant) = Self.quantizeToQ8_0(floats: rows, blockSize: 32)
        // Sanity: dequantized floats should match rows within Q8_0 tolerance.

        let kernel = try PLEGatherKernel(device: device)
        let tokens: [Int32] = [3, 7, 1]
        let out = try kernel.run(q8Table: q8Bytes, tokens: tokens, perLayerDim: P, numLayers: L)

        #expect(out.count == tokens.count * L * P)

        let sqrtP = Float(16.0)  // sqrt(256)
        for (tIdx, tok) in tokens.enumerated() {
            for ell in 0..<L {
                for p in 0..<P {
                    let srcIdx = Int(tok) * L * P + ell * P + p
                    let outIdx = tIdx * L * P + ell * P + p
                    let expected = dequant[srcIdx] * sqrtP
                    #expect(abs(out[outIdx] - expected) < 1e-2, "mismatch at t=\(tIdx) ell=\(ell) p=\(p)")
                }
            }
        }
    }

    @Test("Rejects negative and out-of-range token ids")
    func rejectsInvalidTokenIds() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let P = 256
        let L = 42
        let vocab = 4
        let totalElems = vocab * L * P
        let rows = [Float](repeating: 0.1, count: totalElems)
        let (q8Bytes, _) = Self.quantizeToQ8_0(floats: rows, blockSize: 32)

        let kernel = try PLEGatherKernel(device: device)

        #expect(throws: (any Error).self) {
            _ = try kernel.run(q8Table: q8Bytes, tokens: [-1, 0], perLayerDim: P, numLayers: L)
        }
        #expect(throws: (any Error).self) {
            _ = try kernel.run(q8Table: q8Bytes, tokens: [0, Int32(vocab)], perLayerDim: P, numLayers: L)
        }
    }

    @Test("Encodes gather into caller-owned command buffer")
    func encodesIntoCommandBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            Issue.record("Metal device unavailable")
            return
        }

        let P = 8
        let L = 4
        let vocab = 5
        let totalElems = vocab * L * P
        let rows = (0..<totalElems).map { Float($0 % 17) / 8.0 - 1.0 }
        let (q8Bytes, dequant) = Self.quantizeToQ8_0(floats: rows, blockSize: 32)
        let rowStrideBytes = (L * P / 32) * 34
        let tokens: [Int32] = [2, 4]
        let outputCount = tokens.count * L * P

        guard let tableBuffer = device.makeBuffer(bytes: q8Bytes, length: q8Bytes.count),
              let tokenBuffer = device.makeBuffer(
                bytes: tokens,
                length: tokens.count * MemoryLayout<Int32>.stride
              ),
              let outputBuffer = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride)
        else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        let kernel = try PLEGatherKernel(device: device)
        try kernel.encode(
            commandBuffer: commandBuffer,
            q8TableBuffer: tableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            perLayerDim: P,
            numLayers: L,
            numTokens: tokens.count,
            rowStrideBytes: rowStrideBytes
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let output = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount),
                count: outputCount
            )
        )
        let sqrtP = sqrt(Float(P))
        for (tIdx, tok) in tokens.enumerated() {
            for elem in 0..<(L * P) {
                let expected = dequant[Int(tok) * L * P + elem] * sqrtP
                let actual = output[tIdx * L * P + elem]
                #expect(abs(actual - expected) < 1e-2, "mismatch at token \(tIdx), elem \(elem)")
            }
        }
    }

    @Test("Gathers Q6_K PLE rows and scales by sqrt(P)")
    func gathersQ6KAndScales() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let P = 256
        let L = 42
        let vocab = 8
        let totalElems = vocab * L * P
        let rows = (0..<totalElems).map { i in Float(i % 197) / 31.0 - 3.0 }

        let (q6Bytes, dequant) = Self.quantizeToQ6KRows(floats: rows)
        let kernel = try PLEGatherKernel(device: device)
        let tokens: [Int32] = [3, 7, 1]
        let out = try kernel.runQ6K(q6KTable: q6Bytes, tokens: tokens, perLayerDim: P, numLayers: L)

        #expect(out.count == tokens.count * L * P)

        let sqrtP = sqrt(Float(P))
        for (tIdx, tok) in tokens.enumerated() {
            for elem in 0..<(L * P) {
                let srcIdx = Int(tok) * L * P + elem
                let outIdx = tIdx * L * P + elem
                let expected = dequant[srcIdx] * sqrtP
                #expect(abs(out[outIdx] - expected) < 1e-2, "mismatch at t=\(tIdx) elem=\(elem)")
            }
        }
    }

    @Test("Encodes Q6_K gather into caller-owned command buffer")
    func encodesQ6KIntoCommandBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            Issue.record("Metal device unavailable")
            return
        }

        let P = 64
        let L = 4
        let vocab = 5
        let totalElems = vocab * L * P
        let rows = (0..<totalElems).map { Float($0 % 23) / 5.0 - 2.0 }
        let (q6Bytes, dequant) = Self.quantizeToQ6KRows(floats: rows)
        let rowStrideBytes = (L * P / 256) * 210
        let tokens: [Int32] = [2, 4]
        let outputCount = tokens.count * L * P

        guard let tableBuffer = device.makeBuffer(bytes: q6Bytes, length: q6Bytes.count),
              let tokenBuffer = device.makeBuffer(
                bytes: tokens,
                length: tokens.count * MemoryLayout<Int32>.stride
              ),
              let outputBuffer = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride)
        else {
            Issue.record("Metal buffer allocation failed")
            return
        }

        let kernel = try PLEGatherKernel(device: device)
        try kernel.encodeQ6K(
            commandBuffer: commandBuffer,
            q6KTableBuffer: tableBuffer,
            tokenBuffer: tokenBuffer,
            outputBuffer: outputBuffer,
            perLayerDim: P,
            numLayers: L,
            numTokens: tokens.count,
            rowStrideBytes: rowStrideBytes
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let output = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount),
                count: outputCount
            )
        )
        let sqrtP = sqrt(Float(P))
        for (tIdx, tok) in tokens.enumerated() {
            for elem in 0..<(L * P) {
                let expected = dequant[Int(tok) * L * P + elem] * sqrtP
                let actual = output[tIdx * L * P + elem]
                #expect(abs(actual - expected) < 1e-2, "mismatch at token \(tIdx), elem \(elem)")
            }
        }
    }

    @Test("Rejects invalid Q6_K shape and token ids")
    func rejectsInvalidQ6KInputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("Metal device unavailable")
            return
        }
        let kernel = try PLEGatherKernel(device: device)
        let rows = [Float](repeating: 0.1, count: 256 * 2)
        let (q6Bytes, _) = Self.quantizeToQ6KRows(floats: rows)

        #expect(throws: (any Error).self) {
            _ = try kernel.runQ6K(q6KTable: q6Bytes, tokens: [0], perLayerDim: 32, numLayers: 3)
        }
        #expect(throws: (any Error).self) {
            _ = try kernel.runQ6K(q6KTable: q6Bytes, tokens: [-1], perLayerDim: 64, numLayers: 4)
        }
        #expect(throws: (any Error).self) {
            _ = try kernel.runQ6K(q6KTable: q6Bytes, tokens: [2], perLayerDim: 64, numLayers: 4)
        }
    }

    // Test helper: Q8_0 quantize per block of 32 elements.
    // Returns packed bytes + the reference-dequantized floats (for expected-value comparison,
    // so tolerance accounts for Q8 rounding).
    static func quantizeToQ8_0(floats: [Float], blockSize: Int) -> (bytes: [UInt8], dequant: [Float]) {
        precondition(floats.count % blockSize == 0)
        var bytes: [UInt8] = []
        bytes.reserveCapacity((floats.count / blockSize) * 34)
        var dequant = [Float](repeating: 0, count: floats.count)
        for blockStart in stride(from: 0, to: floats.count, by: blockSize) {
            let slice = floats[blockStart..<(blockStart + blockSize)]
            let amax = slice.map { abs($0) }.max() ?? 0
            let scale: Float = amax == 0 ? 0 : amax / 127
            let scaleHalf = Float16(scale)
            let scaleBits = scaleHalf.bitPattern
            bytes.append(UInt8(scaleBits & 0xFF))
            bytes.append(UInt8((scaleBits >> 8) & 0xFF))
            for (i, v) in slice.enumerated() {
                let q = scale == 0 ? Int8(0) : Int8(clamping: Int(round(v / scale)))
                bytes.append(UInt8(bitPattern: q))
                dequant[blockStart + i] = Float(scaleHalf) * Float(q)
            }
        }
        return (bytes, dequant)
    }

    static func quantizeToQ6KRows(floats: [Float]) -> (bytes: [UInt8], dequant: [Float]) {
        precondition(floats.count % 256 == 0)
        var bytes: [UInt8] = []
        bytes.reserveCapacity((floats.count / 256) * 210)
        var dequant: [Float] = []
        dequant.reserveCapacity(floats.count)

        for blockStart in stride(from: 0, to: floats.count, by: 256) {
            let values = Array(floats[blockStart..<(blockStart + 256)])
            let block = packQ6KBlock(values: values)
            bytes.append(contentsOf: block)
            dequant.append(contentsOf: dequantQ6KBlock(blockData: block))
        }
        return (bytes, dequant)
    }

    static func packQ6KBlock(values: [Float]) -> [UInt8] {
        precondition(values.count == 256)
        var block = [UInt8](repeating: 0, count: 210)

        let absMax = values.map { abs($0) }.max() ?? 0
        let d: Float = absMax > 0 ? absMax / (63.0 * 31.0) : 0
        let dF16 = Float16(d)
        let dFloat = Float(dF16)

        var quantised = [Int](repeating: 0, count: 256)
        var scaleBytes = [Int8](repeating: 0, count: 16)

        for sub in 0..<16 {
            let base = sub * 16
            let subValues = Array(values[base..<(base + 16)])
            let subAbsMax = subValues.map { abs($0) }.max() ?? 0

            if dFloat > 0 && subAbsMax > 0 {
                let idealScale = subAbsMax / (dFloat * 31.0)
                let clampedScale = min(127, max(-128, Int(idealScale.rounded())))
                scaleBytes[sub] = Int8(clamping: clampedScale)
            } else {
                scaleBytes[sub] = 0
            }

            let scaleFloat = dFloat * Float(scaleBytes[sub])
            for i in 0..<16 {
                let idx = base + i
                if abs(scaleFloat) > 0 {
                    let raw = Int((values[idx] / scaleFloat).rounded()) + 32
                    quantised[idx] = min(63, max(0, raw))
                } else {
                    quantised[idx] = 32
                }
            }
        }

        for halfBlock in 0..<2 {
            let outBase = halfBlock * 128
            let qlBase = halfBlock * 64
            let qhBase = 128 + halfBlock * 32
            for lane in 0..<32 {
                let q1 = UInt8(quantised[outBase + lane])
                let q2 = UInt8(quantised[outBase + 32 + lane])
                let q3 = UInt8(quantised[outBase + 64 + lane])
                let q4 = UInt8(quantised[outBase + 96 + lane])

                block[qlBase + lane] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
                block[qlBase + 32 + lane] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
                block[qhBase + lane] = ((q1 >> 4) & 0x03)
                    | (((q2 >> 4) & 0x03) << 2)
                    | (((q3 >> 4) & 0x03) << 4)
                    | (((q4 >> 4) & 0x03) << 6)
            }
        }

        for sub in 0..<16 {
            block[192 + sub] = UInt8(bitPattern: scaleBytes[sub])
        }

        let dBits = dF16.bitPattern.littleEndian
        withUnsafeBytes(of: dBits) { pointer in
            block[208] = pointer[0]
            block[209] = pointer[1]
        }

        return block
    }

    static func dequantQ6KBlock(blockData: [UInt8]) -> [Float] {
        precondition(blockData.count == 210)
        let dBits = UInt16(blockData[208]) | (UInt16(blockData[209]) << 8)
        let d = Float(Float16(bitPattern: dBits))

        var result = [Float](repeating: 0, count: 256)
        for halfBlock in 0..<2 {
            let outBase = halfBlock * 128
            let qlBase = halfBlock * 64
            let qhBase = 128 + halfBlock * 32
            let scaleBase = 192 + halfBlock * 8
            for lane in 0..<32 {
                let scaleOffset = lane / 16
                let q1 = Int((blockData[qlBase + lane] & 0x0F) | (((blockData[qhBase + lane] >> 0) & 0x03) << 4)) - 32
                let q2 = Int((blockData[qlBase + 32 + lane] & 0x0F) | (((blockData[qhBase + lane] >> 2) & 0x03) << 4)) - 32
                let q3 = Int((blockData[qlBase + lane] >> 4) | (((blockData[qhBase + lane] >> 4) & 0x03) << 4)) - 32
                let q4 = Int((blockData[qlBase + 32 + lane] >> 4) | (((blockData[qhBase + lane] >> 6) & 0x03) << 4)) - 32
                let s1 = Int8(bitPattern: blockData[scaleBase + scaleOffset + 0])
                let s2 = Int8(bitPattern: blockData[scaleBase + scaleOffset + 2])
                let s3 = Int8(bitPattern: blockData[scaleBase + scaleOffset + 4])
                let s4 = Int8(bitPattern: blockData[scaleBase + scaleOffset + 6])
                result[outBase + lane] = d * Float(s1) * Float(q1)
                result[outBase + 32 + lane] = d * Float(s2) * Float(q2)
                result[outBase + 64 + lane] = d * Float(s3) * Float(q3)
                result[outBase + 96 + lane] = d * Float(s4) * Float(q4)
            }
        }
        return result
    }
}
