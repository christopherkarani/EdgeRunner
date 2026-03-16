import Metal
import EdgeRunnerMetal

public enum TensorStorageError: Error, Sendable {
    case metalNotAvailable
    case allocationFailed(byteCount: Int)
    case copyFailed
}

final class TensorStorage: Sendable {
    let buffer: MetalBufferHandle
    let byteCount: Int

    init(buffer: MetalBufferHandle) {
        self.buffer = buffer
        self.byteCount = buffer.length
    }

    static func from<T: TensorScalar>(_ data: [T]) throws -> TensorStorage {
        let byteCount = data.count * T.byteSize
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageError.metalNotAvailable
        }
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw TensorStorageError.allocationFailed(byteCount: byteCount)
        }
        return TensorStorage(buffer: MetalBufferHandle(buffer))
    }

    static func zeros(byteCount: Int) throws -> TensorStorage {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageError.metalNotAvailable
        }
        guard let buffer = device.makeBuffer(
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw TensorStorageError.allocationFailed(byteCount: byteCount)
        }
        return TensorStorage(buffer: MetalBufferHandle(buffer))
    }

    func toArray<T: TensorScalar>(count: Int) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func copy() throws -> TensorStorage {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageError.metalNotAvailable
        }
        guard let newBuffer = device.makeBuffer(
            bytes: buffer.contents(),
            length: byteCount,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw TensorStorageError.copyFailed
        }
        return TensorStorage(buffer: MetalBufferHandle(newBuffer))
    }
}
