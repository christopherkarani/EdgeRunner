#if canImport(IOSurface)
import Testing
import Metal
import IOSurface
import Synchronization
@testable import EspressoEdgeRunner
import EdgeRunnerMetal

// MARK: - Mock IO for testing

/// Call-order based mock: first read returns Q data, second returns K data.
/// First write is treated as Q, second as K.
final class MockANEIOSurfaceIO: ANEIOSurfaceIO, Sendable {
    let qReadData: [UInt16]
    let kReadData: [UInt16]
    private let _readCount: Mutex<Int> = .init(0)
    private let _writes: Mutex<[(String, [UInt16])]> = .init([])

    init(qReadData: [UInt16], kReadData: [UInt16]? = nil) {
        self.qReadData = qReadData
        self.kReadData = kReadData ?? qReadData
    }

    func readFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32) -> [UInt16] {
        let index = _readCount.withLock { val -> Int in
            let current = val
            val += 1
            return current
        }
        return index == 0 ? qReadData : kReadData
    }

    func writeFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32, data: [UInt16]) {
        _writes.withLock { writes in
            let label = writes.isEmpty ? "Q" : "K"
            writes.append((label, data))
        }
    }

    var writes: [(String, [UInt16])] {
        _writes.withLock { $0 }
    }

    var writeCount: Int {
        _writes.withLock { $0.count }
    }
}

private func makeIOSurface(elementCount: Int) -> IOSurface? {
    let props: [IOSurfacePropertyKey: Any] = [
        .width: elementCount,
        .height: 1,
        .bytesPerElement: 2,
        .bytesPerRow: elementCount * 2,
        .allocSize: elementCount * 2,
        .pixelFormat: 0,
    ]
    return IOSurface(properties: props as [IOSurfacePropertyKey: Any])
}

@Suite("RoPEBridge")
struct RoPEBridgeTests {

    @Test("FP16 to Float conversion helpers")
    func fp16FloatConversion() {
        let original: Float = 3.14
        let fp16 = Float16(original)
        let bits = fp16.bitPattern
        let recovered = Float(Float16(bitPattern: bits))
        #expect(abs(recovered - original) < 1e-2)
    }

    @Test("Mock IO verifies read and write are called with separate Q/K surfaces")
    func mockIOReadWrite() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EspressoError.metalDeviceUnavailable
        }

        let numHeads = 1
        let numKVHeads = 1
        let headDim = 2
        let seqLen = 1
        let elementCount = seqLen * numHeads * headDim

        let qValues: [Float] = [1.0, 0.0]
        let kValues: [Float] = [0.5, 0.5]
        let qFP16 = qValues.map { Float16($0).bitPattern }
        let kFP16 = kValues.map { Float16($0).bitPattern }

        let mockIO = MockANEIOSurfaceIO(qReadData: qFP16, kReadData: kFP16)
        let bridge = try RoPEBridge(device: device, io: mockIO)

        guard let qSurface = makeIOSurface(elementCount: elementCount),
              let kSurface = makeIOSurface(elementCount: elementCount) else {
            Issue.record("Could not create IOSurface")
            return
        }

        try await bridge.applyRoPE(
            qSurface: unsafeBitCast(qSurface, to: IOSurfaceRef.self),
            kSurface: unsafeBitCast(kSurface, to: IOSurfaceRef.self),
            channelOffset: 0,
            width: Int32(elementCount),
            height: 1,
            seqLen: seqLen,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            startPos: 0,
            theta: 10000.0
        )

        let writes = mockIO.writes
        #expect(writes.count == 2)
        #expect(writes[0].0 == "Q")
        #expect(writes[1].0 == "K")
    }

    @Test("RoPE modifies values (non-identity rotation)")
    func ropeModifiesValues() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EspressoError.metalDeviceUnavailable
        }

        let numHeads = 1
        let numKVHeads = 1
        let headDim = 4
        let seqLen = 1
        let elementCount = seqLen * numHeads * headDim

        let qValues: [Float] = [1.0, 0.0, 1.0, 0.0]
        let kValues: [Float] = [0.5, 0.5, 0.5, 0.5]
        let qFP16 = qValues.map { Float16($0).bitPattern }
        let kFP16 = kValues.map { Float16($0).bitPattern }

        let mockIO = MockANEIOSurfaceIO(qReadData: qFP16, kReadData: kFP16)
        let bridge = try RoPEBridge(device: device, io: mockIO)

        guard let qSurface = makeIOSurface(elementCount: elementCount),
              let kSurface = makeIOSurface(elementCount: elementCount) else {
            Issue.record("Could not create IOSurface")
            return
        }

        try await bridge.applyRoPE(
            qSurface: unsafeBitCast(qSurface, to: IOSurfaceRef.self),
            kSurface: unsafeBitCast(kSurface, to: IOSurfaceRef.self),
            channelOffset: 0,
            width: Int32(elementCount),
            height: 1,
            seqLen: seqLen,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            startPos: 5,
            theta: 10000.0
        )

        let writes = mockIO.writes
        #expect(writes.count >= 1)

        let qWrite = writes.first { $0.0 == "Q" }
        #expect(qWrite != nil)
        let qOut = qWrite!.1.map { Float(Float16(bitPattern: $0)) }
        let changed = zip(qValues, qOut).contains { abs($0 - $1) > 1e-4 }
        #expect(changed, "RoPE should modify Q values at startPos=5")
    }

    @Test("Integration with real IOSurface read/write")
    func realIOSurfaceIntegration() throws {
        let elementCount = 4

        guard let surface = makeIOSurface(elementCount: elementCount) else {
            Issue.record("Could not create IOSurface")
            return
        }
        let surfaceRef = unsafeBitCast(surface, to: IOSurfaceRef.self)

        let io = DefaultANEIOSurfaceIO()

        let values: [UInt16] = [
            Float16(1.0).bitPattern,
            Float16(2.0).bitPattern,
            Float16(3.0).bitPattern,
            Float16(4.0).bitPattern,
        ]
        io.writeFP16(surface: surfaceRef, channelOffset: 0, width: Int32(elementCount), height: 1, data: values)

        let readBack = io.readFP16(surface: surfaceRef, channelOffset: 0, width: Int32(elementCount), height: 1)
        #expect(readBack == values)
    }

    @Test("Empty data write does not crash")
    func emptyDataWriteNoCrash() {
        let io = DefaultANEIOSurfaceIO()
        guard let surface = makeIOSurface(elementCount: 4) else {
            Issue.record("Could not create IOSurface")
            return
        }
        let surfaceRef = unsafeBitCast(surface, to: IOSurfaceRef.self)
        io.writeFP16(surface: surfaceRef, channelOffset: 0, width: 0, height: 0, data: [])
    }
}

#endif
