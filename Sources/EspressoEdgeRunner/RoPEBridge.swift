import Metal
import IOSurface
import EdgeRunnerMetal
import ANEInteropIO

/// Protocol for reading/writing fp16 data to/from IOSurfaces, enabling testability.
public protocol ANEIOSurfaceIO: Sendable {
    func readFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32) -> [UInt16]
    func writeFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32, data: [UInt16])
}

/// Default implementation that calls the C shim.
public struct DefaultANEIOSurfaceIO: ANEIOSurfaceIO, Sendable {
    public init() {}

    public func readFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32) -> [UInt16] {
        guard width > 0, height > 0 else { return [] }
        var buffer = [UInt16](repeating: 0, count: Int(width) * Int(height))
        ane_interop_io_read_fp16(surface, channelOffset, width, height, &buffer)
        return buffer
    }

    public func writeFP16(surface: IOSurfaceRef, channelOffset: Int32, width: Int32, height: Int32, data: [UInt16]) {
        guard !data.isEmpty else { return }
        data.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            ane_interop_io_write_fp16(surface, channelOffset, width, height, base)
        }
    }
}

/// Bridges RoPE computation between ANE IOSurfaces and EdgeRunner's GPU kernel.
public struct RoPEBridge: Sendable {
    private let ropeKernel: RoPEKernel
    private let io: ANEIOSurfaceIO
    private let commandQueue: MTLCommandQueue

    public init(device: MTLDevice, io: ANEIOSurfaceIO? = nil) throws {
        guard let queue = device.makeCommandQueue() else {
            throw EspressoError.metalDeviceUnavailable
        }
        self.ropeKernel = try RoPEKernel(device: device)
        self.io = io ?? DefaultANEIOSurfaceIO()
        self.commandQueue = queue
    }

    /// Applies RoPE to Q and K tensors stored in IOSurfaces.
    public func applyRoPE(
        qSurface: IOSurfaceRef,
        kSurface: IOSurfaceRef,
        channelOffset: Int32,
        width: Int32,
        height: Int32,
        seqLen: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        theta: Float,
        scalingFactor: Float = 1
    ) async throws {
        // 1. Read Q and K from IOSurfaces
        let qFP16 = io.readFP16(surface: qSurface, channelOffset: channelOffset, width: width, height: height)
        let kFP16 = io.readFP16(surface: kSurface, channelOffset: channelOffset, width: width, height: height)

        // 2. Convert fp16 -> Float
        let qFloats = qFP16.map { Float(Float16(bitPattern: $0)) }
        let kFloats = kFP16.map { Float(Float16(bitPattern: $0)) }

        // 3. Apply RoPE
        let (rotatedQ, rotatedK) = try await ropeKernel.applyToQK(
            q: qFloats,
            k: kFloats,
            seqLen: seqLen,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            startPos: startPos,
            theta: theta,
            scalingFactor: scalingFactor,
            commandQueue: commandQueue
        )

        // 4. Convert Float -> fp16
        let qOut = rotatedQ.map { Float16($0).bitPattern }
        let kOut = rotatedK.map { Float16($0).bitPattern }

        // 5. Write back
        io.writeFP16(surface: qSurface, channelOffset: channelOffset, width: width, height: height, data: qOut)
        io.writeFP16(surface: kSurface, channelOffset: channelOffset, width: width, height: height, data: kOut)
    }
}
