import Metal

enum Gemma4ScratchError: Error, Equatable, Sendable {
    case allocationFailed(String)
    case invalidHiddenShape(expected: Int, got: Int)
    case invalidPLEInputShape(expected: Int, got: Int)
    case invalidFFNInputShape(expected: Int, got: Int)
}

/// Persistent Gemma 4 decode scratch buffers.
///
/// This is the foundation for a GPU-resident layer runner. It deliberately owns
/// maximum local/global layer shapes so a single allocation set can be reused
/// across all decoder layers.
final class Gemma4Scratch: @unchecked Sendable {
    let hiddenA: MTLBuffer
    let hiddenB: MTLBuffer
    let normed: MTLBuffer
    let attention: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let qRotated: MTLBuffer
    let kRotated: MTLBuffer
    let ffnInput: MTLBuffer
    let ffnGate: MTLBuffer
    let ffnUp: MTLBuffer
    let ffnActivated: MTLBuffer
    let ffnDown: MTLBuffer
    let pleInput: MTLBuffer
    let pleProjectionInput: MTLBuffer
    let pleGate: MTLBuffer
    let pleActivated: MTLBuffer
    let pleProjection: MTLBuffer
    let logits: MTLBuffer

    private let hiddenSize: Int
    private let perLayerDim: Int
    private var currentHiddenIsA = true

    var currentHidden: MTLBuffer {
        currentHiddenIsA ? hiddenA : hiddenB
    }

    var nextHidden: MTLBuffer {
        currentHiddenIsA ? hiddenB : hiddenA
    }

    init(device: MTLDevice, config: Gemma4ModelConfig) throws {
        self.hiddenSize = config.hiddenSize
        self.perLayerDim = config.perLayerDim
        let maxQRows = config.numAttentionHeads * max(config.headDim, config.globalHeadDim)
        let maxKVRows = config.numKeyValueHeads * max(config.headDim, config.globalHeadDim)
        let f32 = MemoryLayout<Float>.stride

        self.hiddenA = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "hiddenA")
        self.hiddenB = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "hiddenB")
        self.normed = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "normed")
        self.attention = try Self.makeBuffer(device: device, floats: maxQRows, label: "attention")
        self.q = try Self.makeBuffer(device: device, floats: maxQRows, label: "q")
        self.k = try Self.makeBuffer(device: device, floats: maxKVRows, label: "k")
        self.v = try Self.makeBuffer(device: device, floats: maxKVRows, label: "v")
        self.qRotated = try Self.makeBuffer(device: device, floats: maxQRows, label: "qRotated")
        self.kRotated = try Self.makeBuffer(device: device, floats: maxKVRows, label: "kRotated")
        self.ffnInput = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "ffnInput")
        self.ffnGate = try Self.makeBuffer(device: device, floats: config.intermediateSize, label: "ffnGate")
        self.ffnUp = try Self.makeBuffer(device: device, floats: config.intermediateSize, label: "ffnUp")
        self.ffnActivated = try Self.makeBuffer(device: device, floats: config.intermediateSize, label: "ffnActivated")
        self.ffnDown = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "ffnDown")
        self.pleInput = try Self.makeBuffer(device: device, floats: config.perLayerDim, label: "pleInput")
        self.pleProjectionInput = try Self.makeBuffer(
            device: device,
            floats: config.numHiddenLayers * config.perLayerDim,
            label: "pleProjectionInput"
        )
        self.pleGate = try Self.makeBuffer(device: device, floats: config.perLayerDim, label: "pleGate")
        self.pleActivated = try Self.makeBuffer(device: device, floats: config.perLayerDim, label: "pleActivated")
        self.pleProjection = try Self.makeBuffer(device: device, floats: config.hiddenSize, label: "pleProjection")
        self.logits = try Self.makeBuffer(device: device, floats: config.vocabSize, label: "logits")

        precondition(hiddenA.length == config.hiddenSize * f32)
    }

    func swapHiddenBuffers() {
        currentHiddenIsA.toggle()
    }

    func copyHidden(_ values: [Float]) throws {
        guard values.count == hiddenSize else {
            throw Gemma4ScratchError.invalidHiddenShape(expected: hiddenSize, got: values.count)
        }
        copy(values, to: currentHidden)
    }

    func readHidden() throws -> [Float] {
        let pointer = currentHidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
        return Array(UnsafeBufferPointer(start: pointer, count: hiddenSize))
    }

    func copyPLEInput(_ values: [Float]) throws {
        guard values.count == perLayerDim else {
            throw Gemma4ScratchError.invalidPLEInputShape(expected: perLayerDim, got: values.count)
        }
        copy(values, to: pleInput)
    }

    func copyFFNInput(_ values: [Float]) throws {
        guard values.count == hiddenSize else {
            throw Gemma4ScratchError.invalidFFNInputShape(expected: hiddenSize, got: values.count)
        }
        copy(values, to: ffnInput)
    }

    private func copy(_ values: [Float], to buffer: MTLBuffer) {
        values.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }

    private static func makeBuffer(
        device: MTLDevice,
        floats count: Int,
        label: String
    ) throws -> MTLBuffer {
        guard count > 0,
              let buffer = device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ) else {
            throw Gemma4ScratchError.allocationFailed(label)
        }
        buffer.label = "Gemma4Scratch.\(label)"
        return buffer
    }
}
