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
#ifndef ACDNR_FILTER_H
#define ACDNR_FILTER_H

/* If CUDA is enabled, use CUDA version */
#ifdef USE_CUDA
    #include "acdnr_filter_cuda.h"

    #define acdnr_filter_rgbf1(rgba, width, height, stddev_l, amount_l, iterations_l, structure_size_l, stddev_c, amount_c, iterations_c, structure_size_c) acdnr_filter_rgbf1_cuda_Wrapper(rgba, width, height, stddev_l, amount_l, iterations_l, structure_size_l, stddev_c, amount_c, iterations_c, structure_size_c)
#else

#define CHANNELS	4

/**
 *  Apply ACDNR filter to a float RGBA image
 * @param rgba: pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel)
 * @param width: image width
 * @param height: image height
 * @param stddev_l: standard deviation for luminance (0.0 to 1.0)
 * @param amount_l: amount for luminance (0.0 to 1.0)
 * @param iterations_l: number of iterations for luminance (0 to 10)
 * @param structure_size_l: structure size for luminance (0 to 10)
 * @param stddev_c: standard deviation for chrominance (0.0 to 1.0)
 * @param amount_c: amount for chrominance (0.0 to 1.0)
 * @param iterations_c: number of iterations for chrominance (0 to 10)
 * @param structure_size_c: structure size for chrominance (0 to 10)
 * @return: 0 on success, 1 on error
 */
int acdnr_filter_rgbf1(float *rgba, int width, int height,
	float stddev_l, float amount_l, int iterations_l, int structure_size_l,
	float stddev_c, float amount_c, int iterations_c, int structure_size_c);

#endif

#endif // ACDNR_FILTER_H
