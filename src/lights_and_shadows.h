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
#ifndef LIGHTS_AND_SHADOWS_H
#define LIGHTS_AND_SHADOWS_H

#ifdef USE_CUDA
#include "lights_and_shadows_cuda.h"

#define adjust_black_point_rgbf1(rgbf, width, height, min_shift_pct,          \
                                  max_shift_pct, dark_threshold)               \
  adjust_black_point_rgbf1_cuda_Wrapper(rgbf, width, height, min_shift_pct,   \
                                         max_shift_pct, dark_threshold)
#define autostretch_rgbf1(hdr_image, width, height, min_val, max_val)          \
  autostretch_rgbf1_cuda_Wrapper(hdr_image, width, height, min_val, max_val)
#else
#define CHANNELS 4

/**
 * @brief Adaptive black point adjustment with histogram analysis (OpenMP)
 *
 * This function analyzes the luminance distribution of an RGB float image
 * (range 0.0–1.0) and adaptively shifts the black point only when needed.
 * Masked pixels (all RGB = 0) are ignored. The algorithm computes approximate
 * histogram quantiles (1% and 99%) and adjusts the black point to ensure
 * proper contrast without clipping shadows or highlights.
 *
 * @param rgbf: pointer to RGB float image array (interleaved: R,G,B)
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param min_shift_pct: minimum black-point shift (e.g., 0.005)
 * @param max_shift_pct: maximum black-point shift (e.g., 0.02)
 * @param dark_threshold: minimum average luminance (below = no adjustment)
 * @return: 0 on success, 1 on error or no valid pixels found
 */
int adjust_black_point_rgbf1(float *rgbf, int width, int height,
                              double min_shift_pct, double max_shift_pct,
                              double dark_threshold);

/**
 * Applies percentile-based automatic stretching to a float RGB image.
 * Pixel values are normalized to the range 0.0 – 1.0 based on the given
 * percentiles.
 *
 * @param hdr_image: input/output image in float RGBA format (4 channels per
 * pixel).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param min_val: lower value for black point (e.g. 0.01 for 1%).
 * @param max_val: upper value for white point (e.g. 0.99 for 99%).
 * @return 0 on success, 1 on error (invalid input, too few valid values, or
 * invalid range).
 */
int autostretch_rgbf1(float *hdr_image, int width, int height, float min_val,
                      float max_val);

#endif

#endif // LIGHTS_AND_SHADOWS_H