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
#ifndef DEBAYER_NNI_H
#define DEBAYER_NNI_H

#include <stdint.h>

#define CHANNELS 4	// RGBA

/**
 * Performs nearest-neighbor debayering on 16-bit Bayer data and converts it to RGBA 16-bit output.
 * A black border is added to the right and bottom edges to avoid out-of-bounds reads.
 *
 * @param raw: pointer to the input Bayer image (16-bit, 1 channel per pixel).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param rgb: pointer to the output RGBA image buffer (4 channels per pixel, 16-bit).
 * @param x_offset: X-offset used for Bayer pattern alignment.
 * @param y_offset: Y-offset used for Bayer pattern alignment.
 * @param bayer_pattern: Bayer pattern as string ("RGGB", "GRBG", "GBRG", "BGGR", etc.).
 * @return: 0 on success, 1 if the Bayer pattern is invalid.
 */
int debayer_nearest_neighbor_rgb16(const uint16_t *raw, int width, int height, uint16_t *rgb, 
                              int x_offset, int y_offset, const char *bayer_pattern);

/**
 * Performs nearest-neighbor debayering on 16-bit Bayer data and converts it to RGBA float output.
 * A black border is added to the right and bottom edges to avoid out-of-bounds reads.
 *
 * @param raw: pointer to the input Bayer image (16-bit, 1 channel per pixel).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param rgbf: pointer to the output RGBA image buffer (4 channels per pixel, float).
 * @param x_offset: X-offset used for Bayer pattern alignment.
 * @param y_offset: Y-offset used for Bayer pattern alignment.
 * @param bayer_pattern: Bayer pattern as string ("RGGB", "GRBG", "GBRG", "BGGR", etc.).
 * @return: 0 on success, 1 if the Bayer pattern is invalid.
 */
int debayer_nearest_neighbor_rgbf(const uint16_t *raw, int width, int height, float *rgbf,
                                    int x_offset, int y_offset, const char *bayer_pattern);

#endif // DEBAYER_NNI_H
