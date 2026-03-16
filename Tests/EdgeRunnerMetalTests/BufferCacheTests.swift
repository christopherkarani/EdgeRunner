import Testing
import Metal
@testable import EdgeRunnerMetal

@Suite("BufferCache")
struct BufferCacheTests {

    let device: MTLDevice

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetal
        }
        self.device = d
    }

    @Test func reuseExactSize() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 256, options: .storageModeShared)!
        cache.recycle(buf)
        let reused = cache.reuse(size: 256)
        #expect(reused != nil)
        #expect(reused!.length >= 256)
    }

    @Test func reuseSlightlyLarger() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 300, options: .storageModeShared)!
        cache.recycle(buf)
        let reused = cache.reuse(size: 256)
        #expect(reused != nil)
        #expect(reused!.length >= 256)
    }

    @Test func noReuseTooLarge() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = device.makeBuffer(length: 1024, options: .storageModeShared)!
        cache.recycle(buf)
        let reused = cache.reuse(size: 256)
        #expect(reused == nil)
    }

    @Test func emptyCache() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let reused = cache.reuse(size: 256)
        #expect(reused == nil)
    }

    @Test func eviction() throws {
        let cache = BufferCache(device: device, maxBytes: 512)
        let buf1 = device.makeBuffer(length: 256, options: .storageModeShared)!
        let buf2 = device.makeBuffer(length: 256, options: .storageModeShared)!
        let buf3 = device.makeBuffer(length: 256, options: .storageModeShared)!
        cache.recycle(buf1)
        cache.recycle(buf2)
        cache.recycle(buf3)
        #expect(cache.totalCachedBytes <= 512)
    }

    @Test func allocateNew() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let buf = try cache.acquire(size: 256)
        #expect(buf.length >= 256)
    }

    @Test func allocateReusesFromCache() throws {
        let cache = BufferCache(device: device, maxBytes: 1024 * 1024)
        let original = device.makeBuffer(length: 256, options: .storageModeShared)!
        cache.recycle(original)
        let acquired = try cache.acquire(size: 256)
        #expect(acquired.length >= 256)
    }
}

enum TestError: Error {
    case noMetal
}
