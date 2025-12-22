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
#ifndef COLOR_CALIBRATION_H
#define COLOR_CALIBRATION_H

#ifdef USE_CUDA
#include "color_calibration_cuda.h"

#define adjust_saturation_rgbf1(rgba, width, height, saturation_factor)        \
  adjust_saturation_rgbf1_cuda_Wrapper(rgba, width, height, saturation_factor)
#define apply_gamma_correction_rgbf1(rgba, width, height, gamma)               \
  apply_gamma_correction_rgbf1_cuda_Wrapper(rgba, width, height, gamma)
#else
#define CHANNELS 4 // RGBA

/**
 * Adjusts the color saturation of an RGBA float image.
 *
 * @param rgba: pointer to float array (RGBA, 4 channels, range 0.0
 * – 1.0).
 * @param width: image width.
 * @param height: image height.
 * @param saturation_factor: factor > 1.0 = more saturated colors, < 1.0 =
 * desaturated.
 */
void adjust_saturation_rgbf1(float *rgba, int width, int height,
                             float saturation_factor);

/**
 * Applies gamma correction to float-RGB data in RGBA format.
 *
 * @param rgba: pointer to float-RGB(A) data array (4 channels, range 0.0
 * – 1.0).
 * @param width: image width.
 * @param height: image height.
 * @param gamma: gamma value (> 0.0).
 */
int apply_gamma_correction_rgbf1(float *rgba, int width, int height,
                                 float gamma);
#endif

/**
 * Applies white balance correction to a 16-bit RGB image by scaling the red and
 * blue channels.
 *
 * @param rgb: pointer to the 16-bit RGBA data array (4 values per pixel,
 * range 0–65535).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param scale_r: scaling factor for the red channel.
 * @param scale_b: scaling factor for the blue channel.
 */
void apply_white_balance_rgb16(uint16_t *rgb, int width, int height,
                               double scale_r, double scale_b);

#endif // COLOR_CALIBRATION_H
