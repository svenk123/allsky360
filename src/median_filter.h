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
#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

// Falls CUDA aktiviert ist, verwende CUDA-Version
#ifdef USE_CUDA
    #include "median_filter_cuda.h"
    #define median_filter_rgbf(rgba, width, height, kernel_radius) median_filter_rgbf_cuda_Wrapper(rgba, width, height, kernel_radius)

    #define multiscale_median_filter_rgbf1(rgba, width, height, max_scale, blend_factor) multiscale_median_filter_rgbf1_cuda_Wrapper(rgba, width, height, max_scale, blend_factor)
#else
    int median_filter_rgbf(float *rgba, int width, int height, int kernel_radius);


/**
 * Apply a multi-scale median filter on a float RGBA image (0.0–1.0), in-place.
 *
 * Example parameters and their typical effect:
 *
 * max_scale | blend_factor | Effect
 * ----------|--------------|-----------------------------------------------------------
 *    1      |     0.3      | Slight smoothing of fine noise, stars remain sharp.
 *    1      |     1.0      | Strong removal of fine noise structures.
 *    2      |     0.3      | Moderate smoothing, removes small hot pixels, stars stay visible.
 *    2      |     0.7      | Noticeable smoothing, weaker stars may fade.
 *    3      |     0.5      | Strong smoothing, background becomes very uniform, stars soften.
 *    3      |     1.0      | Very strong smoothing, image looks blurred/soft.
 *
 * Notes:
 * - Start conservatively with max_scale = 1–2 and blend_factor = 0.3–0.5.
 * - For very noisy short exposures, higher blend_factor values (e.g., 0.7) may be appropriate.
 * - The higher the max_scale, the larger the median window, leading to stronger smoothing.
 *
 * @param rgba: pointer to float image data (RGBA interleaved, 4 * width * height elements)
 * @param width: image width
 * @param height: image height
 * @param max_scale: number of scales (each scale increases radius by 1)
 * @param blend_factor: strength of filtering (0.0 = no effect, 1.0 = full median)
 * @return: 0 on success, 1 on error
 */
int multiscale_median_filter_rgbf1(float *rgba, int width, int height, int max_scale, float blend_factor);

#endif


#endif // MEDIAN_FILTER_H
