import Darwin
import Foundation
import Metal

public final class MemoryMappedFile: @unchecked Sendable {
    // The mapped region is immutable after creation and is released only in deinit.
    public let basePointer: UnsafeRawPointer
    public let size: Int

    public var data: UnsafeBufferPointer<UInt8> {
        UnsafeBufferPointer(start: basePointer.assumingMemoryBound(to: UInt8.self), count: size)
    }

    let rawPointer: UnsafeMutableRawPointer

    public init(url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw WeightLoaderError.fileNotFound(url)
        }
        defer { close(fd) }

        var fileStat = stat()
        guard fstat(fd, &fileStat) == 0 else {
            throw WeightLoaderError.mmapFailed(errno: errno)
        }

        let fileSize = Int(fileStat.st_size)
        guard fileSize > 0 else {
            throw WeightLoaderError.invalidFormat("Cannot mmap empty file: \(url.lastPathComponent)")
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: -1)
        else {
            throw WeightLoaderError.mmapFailed(errno: errno)
        }

        self.rawPointer = mapped
        self.basePointer = UnsafeRawPointer(mapped)
        self.size = fileSize
    }

    deinit {
        munmap(rawPointer, size)
    }

    var mappedData: Data {
        Data(bytesNoCopy: rawPointer, count: size, deallocator: .none)
    }

    public func slice(offset: Int, length: Int) -> UnsafeBufferPointer<UInt8> {
        precondition(offset >= 0 && length >= 0 && offset + length <= size)
        let pointer = basePointer
            .advanced(by: offset)
            .assumingMemoryBound(to: UInt8.self)
        return UnsafeBufferPointer(start: pointer, count: length)
    }

    public func makeMetalBuffer(
        device: MTLDevice,
        offset: Int,
        length: Int
    ) throws -> MTLBuffer {
        let region = try makeMetalBufferRegion(device: device, offset: offset, length: length)
        if region.offset == 0 {
            return region.buffer
        }
        guard let copyBuffer = device.makeBuffer(
            bytes: rawPointer.advanced(by: offset),
            length: length,
            options: .storageModeShared
        ) else {
            throw WeightLoaderError.allocationFailed(byteCount: length)
        }
        return copyBuffer
    }

    func makeMetalBufferRegion(
        device: MTLDevice,
        offset: Int,
        length: Int
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        guard offset >= 0, length >= 0, offset + length <= size else {
            throw WeightLoaderError.invalidFormat(
                "Mapped region out of bounds: offset=\(offset) length=\(length) size=\(size)"
            )
        }

        let pageSize = Int(getpagesize())
        let pageStart = (offset / pageSize) * pageSize
        let pageDelta = offset - pageStart
        let mappedLength = pageDelta + length
        let pointer = rawPointer.advanced(by: pageStart)

        if let buffer = device.makeBuffer(
            bytesNoCopy: pointer,
            length: mappedLength,
            options: .storageModeShared,
            deallocator: nil
        ) {
            return (buffer, pageDelta)
        }

        guard let copiedBuffer = device.makeBuffer(
            bytes: rawPointer.advanced(by: offset),
            length: length,
            options: .storageModeShared
        ) else {
            throw WeightLoaderError.allocationFailed(byteCount: length)
        }
        return (copiedBuffer, 0)
    }
}
