import Metal
import EdgeRunnerMetal

final class TensorStorage: @unchecked Sendable {
    // @unchecked: MTLBuffer is Obj-C protocol, not marked Sendable,
    // but thread-safe for concurrent reads. Mutations via actor-isolated MetalBackend.
    let buffer: MTLBuffer
    let byteCount: Int

    init(buffer: MTLBuffer) {
        self.buffer = buffer
        self.byteCount = buffer.length
    }

    static func from<T: TensorScalar>(_ data: [T]) -> TensorStorage {
        let byteCount = data.count * T.byteSize
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            fatalError("Failed to allocate buffer of size \(byteCount)")
        }
        return TensorStorage(buffer: buffer)
    }

    static func zeros(byteCount: Int) -> TensorStorage {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        guard let buffer = device.makeBuffer(
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            fatalError("Failed to allocate buffer of size \(byteCount)")
        }
        return TensorStorage(buffer: buffer)
    }

    func toArray<T: TensorScalar>(count: Int) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func copy() -> TensorStorage {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        guard let newBuffer = device.makeBuffer(
            bytes: buffer.contents(),
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            fatalError("Failed to copy buffer")
        }
        return TensorStorage(buffer: newBuffer)
    }
}
