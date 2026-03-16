import Metal
import Synchronization

package struct MetalBufferHandle: @unchecked Sendable {
    // @unchecked Sendable is limited to the Metal protocol wrapper.
    // The wrapped MTLBuffer never leaves package-internal APIs without actor or mutex protection.
    package let rawValue: MTLBuffer

    package init(_ rawValue: MTLBuffer) {
        self.rawValue = rawValue
    }

    package var length: Int { rawValue.length }

    package func contents() -> UnsafeMutableRawPointer {
        rawValue.contents()
    }
}

/// LRU buffer cache for reusing Metal buffers of similar sizes.
/// Thread-safe via Mutex.
final class BufferCache {
    private let state: Mutex<CacheState>
    private let device: MTLDevice

    private struct CacheState: ~Copyable, Sendable {
        var buckets: [Int: [MetalBufferHandle]] = [:]
        var totalBytes: Int = 0
        let maxBytes: Int
    }

    init(device: MTLDevice, maxBytes: Int) {
        self.device = device
        self.state = Mutex(CacheState(maxBytes: maxBytes))
    }

    var totalCachedBytes: Int {
        state.withLock { $0.totalBytes }
    }

    /// Returns a cached buffer whose length is in [size, size*2], or nil if none available.
    func reuse(size: Int) -> MTLBuffer? {
        state.withLock { state in
            let candidates = state.buckets.keys
                .filter { $0 >= size && $0 <= size * 2 }
                .sorted()

            guard let bucketSize = candidates.first,
                  var buffers = state.buckets[bucketSize],
                  !buffers.isEmpty else {
                return nil
            }

            let buffer = buffers.removeFirst()
            if buffers.isEmpty {
                state.buckets.removeValue(forKey: bucketSize)
            } else {
                state.buckets[bucketSize] = buffers
            }
            state.totalBytes -= buffer.length
            return buffer.rawValue
        }
    }

    /// Returns a buffer to the cache, evicting LRU entries to stay within maxBytes.
    func recycle(_ buffer: MTLBuffer) {
        let length = buffer.length
        state.withLock { state in
            // Evict until we have room or the cache is empty
            while state.totalBytes + length > state.maxBytes {
                guard let bucketSize = state.buckets.keys
                    .first(where: { state.buckets[$0]?.isEmpty == false }),
                      var buffers = state.buckets[bucketSize] else {
                    break
                }
                let evicted = buffers.removeLast()
                if buffers.isEmpty {
                    state.buckets.removeValue(forKey: bucketSize)
                } else {
                    state.buckets[bucketSize] = buffers
                }
                state.totalBytes -= evicted.length
            }

            if state.totalBytes + length <= state.maxBytes {
                state.buckets[length, default: []].append(MetalBufferHandle(buffer))
                state.totalBytes += length
            }
        }
    }

    /// Returns a cached buffer if one fits, otherwise allocates a new one from the device.
    func acquire(size: Int) -> MTLBuffer {
        if let cached = reuse(size: size) {
            return cached
        }
        guard let buffer = device.makeBuffer(
            length: size,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            fatalError("Failed to allocate Metal buffer of size \(size)")
        }
        return buffer
    }
}
