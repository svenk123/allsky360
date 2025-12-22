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

#ifndef JPEG_FUNCTIONS_H
#define JPEG_FUNCTIONS_H

/**
 * Saves an RGB float image as JPEG file.
 * The input is an RGBA float array, but only RGB is used.
 * Input range: 0.0–1.0 (normalized)
 *
 * @param rgba: pointer to float[width * height * 4], range 0.0–1.0
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param compression_ratio: JPEG compression level (0 = no compression, 100 =
 * max compression)
 * @param scale: scaling factor (0.0–1.0)
 * @param filename: output filename (path to JPEG file)
 * @return: 0 on success, non-zero on error
 */
int save_jpeg_rgbf1(const float *rgba, int width, int height,
  int compression_ratio, float scale, const char *filename);

#ifdef USE_CUDA
#include "jpeg_functions_cuda.h"

#define save_jpeg_rgbf16(rgba, width, height, compression_ratio, scale, filename) \
  save_jpeg_rgbf16_cuda_Wrapper(rgba, width, height, compression_ratio, scale, filename)

#define tonemap_rgbf1_to_rgbf16(rgbf, width, height)                           \
  tonemap_rgbf1_to_rgbf16_cuda_Wrapper(rgbf, width, height)
#else
#define CHANNELS 4


/**
 * Saves an RGB float image (16-bit per channel) as JPEG file.
 * The input is an RGBA float array, but only RGB is used.
 * Input range: 0.0–65535.0 (16-bit)
 *
 * @param rgba: pointer to float[width * height * 4], range 0.0–65535.0
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param compression_ratio: JPEG compression level (0 = no compression, 100 =
 * max compression)
 * @param scale: scaling factor (0.0–1.0)
 * @param filename: output filename (path to JPEG file)
 * @return: 0 on success, non-zero on error
 */
int save_jpeg_rgbf16(const float *rgba, int width, int height,
                     int compression_ratio, float scale, const char *filename);

/**
 * Tonemap an HDR float RGBA image (range 0.0–1.0) in-place to 16-bit float
 * values (range 0.0–65535.0).
 *
 * This function performs per-channel min-max normalization and scales the RGB
 * values of the image to a 16-bit float range suitable for display or encoding.
 * The alpha channel is set to 65535.0f for all pixels. The transformation is
 * performed in-place.
 *
 * @param rgbf: pointer to the RGBA float image data. Must be an array of size
 * (width * height * CHANNELS). The input values are assumed to be in the range
 * [0.0, 1.0]. After the function completes, the array will be modified
 * in-place.
 * @param width: width of the image in pixels.
 * @param height: height of the image in pixels.
 * @return int Returns 0 on success, 1 on error (e.g., invalid input).
 */
int tonemap_rgbf1_to_rgbf16(float *rgbf, int width, int height);
#endif

#endif // JPEG_FUNCTIONS_H