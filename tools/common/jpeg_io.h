/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#ifndef KEOGRAM_JPEG_IO_H
#define KEOGRAM_JPEG_IO_H

#include <stdint.h>

/**
 * Decode a JPEG as 8-bit RGB, reduced by scale_denom (1, 2, 4 or 8) while
 * decoding (libjpeg DCT scaling, much faster than decoding at full size).
 * A corrupt or truncated file (still being written) returns NULL.
 * The caller frees the buffer.
 */
uint8_t *jpeg_read_rgb_scaled(const char *path, int scale_denom, int *width, int *height);

/** Size of a JPEG without decoding it; 0 on success. */
int jpeg_read_size(const char *path, int *width, int *height);

/**
 * Write 8-bit RGB as JPEG; written to path.tmp and renamed, so a reader never
 * sees a half-written file. 0 on success.
 */
int jpeg_write_rgb(const char *path, const uint8_t *rgb, int width, int height, int quality);

#endif /* KEOGRAM_JPEG_IO_H */
