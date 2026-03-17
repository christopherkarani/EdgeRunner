#ifndef ANE_INTEROP_IO_H
#define ANE_INTEROP_IO_H

#include <IOSurface/IOSurface.h>
#include <stdint.h>

/// Reads fp16 data from an IOSurface plane into `outBuffer`.
/// `channelOffset` is the byte offset into each row for the starting channel.
/// `width` and `height` describe the region to read (in fp16 elements per row / rows).
void ane_interop_io_read_fp16(
    IOSurfaceRef surface,
    int32_t channelOffset,
    int32_t width,
    int32_t height,
    uint16_t *outBuffer
);

/// Writes fp16 data from `inBuffer` into an IOSurface plane.
void ane_interop_io_write_fp16(
    IOSurfaceRef surface,
    int32_t channelOffset,
    int32_t width,
    int32_t height,
    const uint16_t *inBuffer
);

#endif /* ANE_INTEROP_IO_H */
