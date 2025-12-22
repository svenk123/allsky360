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
#ifndef DEHAZE_FILTER_H
#define DEHAZE_FILTER_H

// Falls CUDA aktiviert ist, verwende CUDA-Version
#ifdef USE_CUDA
#include "dehaze_filter_cuda.h"
#define perceptual_dehaze_rgbf1_multiscale_full(rgba_image, width, height,     \
                                                amount, haze_percent)          \
  perceptual_dehaze_rgbf1_multiscale_full_cuda_Wrapper(                        \
      rgba_image, width, height, amount, haze_percent)
#else

#define CHANNELS 4

/**
 * Perceptual All-Sky Dehaze (Multi-Scale Laplacian + Global Haze Removal).
 * Works after Filmic / Gamma.
 *
 * In-place on float RGBA 16-bit image.
 *
 * @param rgba_image: float[width * height * 4], range 0.0–65535.0
 * @param width: image width
 * @param height: image height
 * @param amount: dehaze strength (0.3–1.0), typical 0.5–0.7
 * @param haze_percent: percentile for haze estimate (typ. 0.1f → 10%)
 * @return: 0 on success, 1 on error
 */
int perceptual_dehaze_rgbf1_multiscale_full(float *rgba_image, int width,
                                            int height, float amount,
                                            float haze_percent);
#endif

#endif // DEHAZE_FILTER_H
