import Metal
import Synchronization

/// LRU buffer cache for reusing Metal buffers of similar sizes.
/// Thread-safe via Mutex.
public final class BufferCache: Sendable {
    private let state: Mutex<CacheState>
    private let device: MTLDevice

    // MTLBuffer is not Sendable but Mutex<CacheState> ensures exclusive access,
    // so it is safe to mark CacheState as @unchecked Sendable.
    struct CacheState: ~Copyable, @unchecked Sendable {
        var buckets: [Int: [MTLBuffer]] = [:]
        var totalBytes: Int = 0
        let maxBytes: Int
    }

    public init(device: MTLDevice, maxBytes: Int) {
        self.device = device
        self.state = Mutex(CacheState(maxBytes: maxBytes))
    }

    public var totalCachedBytes: Int {
        state.withLock { $0.totalBytes }
    }

    /// Returns a cached buffer whose length is in [size, size*2], or nil if none available.
    public func reuse(size: Int) -> MTLBuffer? {
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
            return buffer
        }
    }

    /// Returns a buffer to the cache, evicting LRU entries to stay within maxBytes.
    public func recycle(_ buffer: MTLBuffer) {
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
                state.buckets[length, default: []].append(buffer)
                state.totalBytes += length
            }
        }
    }

    /// Returns a cached buffer if one fits, otherwise allocates a new one from the device.
    public func acquire(size: Int) -> MTLBuffer {
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
