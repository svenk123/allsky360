/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license 
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef IMAGE_CHECK_H
#define IMAGE_CHECK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reads only the JPEG header. 0 on success, -1 if the file is missing or
 * not a readable JPEG (never exits, unlike libjpeg's default handler). */
int get_jpeg_dimensions(const char *filename, int *width, int *height);

/**
 * Letterbox: the largest rectangle with the aspect ratio of src_w x src_h
 * that fits into dst_w x dst_h, centred. The rest of the frame stays black.
 */
void fit_rect(int src_w, int src_h, int dst_w, int dst_h,
              int *out_x, int *out_y, int *out_w, int *out_h);

#ifdef __cplusplus
}
#endif

#endif
