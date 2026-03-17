#include "ane_interop_io.h"
#include <string.h>

void er_ane_interop_io_read_fp16(
    IOSurfaceRef surface,
    int32_t channelOffset,
    int32_t width,
    int32_t height,
    uint16_t *outBuffer
) {
    if (!surface || !outBuffer || channelOffset < 0 || width <= 0 || height <= 0) return;

    kern_return_t status = IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    if (status != kIOReturnSuccess) return;

    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return;
    }

    size_t bytesPerRow = IOSurfaceGetBytesPerRow(surface);
    size_t surfaceHeight = IOSurfaceGetHeight(surface);
    size_t copyBytes = (size_t)width * sizeof(uint16_t);

    // Bounds check: ensure access stays within the surface
    if ((size_t)height > surfaceHeight) {
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return;
    }
    if ((size_t)channelOffset + copyBytes > bytesPerRow) {
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return;
    }

    for (int32_t row = 0; row < height; row++) {
        uint8_t *src = base + (size_t)row * bytesPerRow + (size_t)channelOffset;
        uint16_t *dst = outBuffer + (size_t)row * (size_t)width;
        memcpy(dst, src, copyBytes);
    }

    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

void er_ane_interop_io_write_fp16(
    IOSurfaceRef surface,
    int32_t channelOffset,
    int32_t width,
    int32_t height,
    const uint16_t *inBuffer
) {
    if (!surface || !inBuffer || channelOffset < 0 || width <= 0 || height <= 0) return;

    kern_return_t status = IOSurfaceLock(surface, 0, NULL);
    if (status != kIOReturnSuccess) return;

    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        return;
    }

    size_t bytesPerRow = IOSurfaceGetBytesPerRow(surface);
    size_t surfaceHeight = IOSurfaceGetHeight(surface);
    size_t copyBytes = (size_t)width * sizeof(uint16_t);

    // Bounds check: ensure access stays within the surface
    if ((size_t)height > surfaceHeight) {
        IOSurfaceUnlock(surface, 0, NULL);
        return;
    }
    if ((size_t)channelOffset + copyBytes > bytesPerRow) {
        IOSurfaceUnlock(surface, 0, NULL);
        return;
    }

    for (int32_t row = 0; row < height; row++) {
        uint8_t *dst = base + (size_t)row * bytesPerRow + (size_t)channelOffset;
        const uint16_t *src = inBuffer + (size_t)row * (size_t)width;
        memcpy(dst, src, copyBytes);
    }

    IOSurfaceUnlock(surface, 0, NULL);
}
