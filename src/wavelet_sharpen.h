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
#ifndef WAVELET_SHARPEN_H
#define WAVELET_SHARPEN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define CHANNELS 4

#ifdef USE_CUDA

#include "wavelet_sharpen_cuda.h"
#define wavelet_sharpen_rgbf1(rgba, width, height, gain_small, gain_medium, gain_large) wavelet_sharpen_rgbf1_cuda_Wrapper(rgba, width, height, gain_small, gain_medium, gain_large)

#else
/**
 * Sharpen an RGBA float image using wavelet transform.
 * In-place on float RGBA image (0.0–1.0).
 * 
 * @param rgba: pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel)
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param gain_small: gain for small wavelet coefficients
 * @param gain_medium: gain for medium wavelet coefficients
 * @param gain_large: gain for large wavelet coefficients
 * @return: 0 on success, 1 on error (e.g. invalid parameters).
 */
int wavelet_sharpen_rgbf1(float *rgba, int width, int height, float gain_small, float gain_medium, float gain_large);
#endif

#endif // WAVELET_SHARPEN_H