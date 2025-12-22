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
#include <stdio.h>
#include "median_filter_cuda.h"

#ifdef USE_CUDA
int median_filter_rgbf_cuda_Wrapper(float *rgba, int width, int height, int kernel_radius) {
    return median_filter_rgbf_cuda(rgba, width, height, kernel_radius);
}

int multiscale_median_filter_rgbf1_cuda_Wrapper(
    float *host_rgba, int width, int height, int max_scale,
    float blend_factor) {
    return multiscale_median_filter_rgbf1_cuda(
	host_rgba, width, height, max_scale, blend_factor);
}

#else
// Falls CUDA nicht aktiviert ist, eine Dummy-Funktion bereitstellen
int median_filter_rgbf_cuda_Wrapper(float *rgba, int width, int height, int kernel_radius) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)kernel_radius;
    fprintf(stderr, "Median filter CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int multiscale_median_filter_rgbf1_cuda_Wrapper(
    float *host_rgba, int width, int height, int max_scale,
    float blend_factor) {
    (void)host_rgba;
    (void)width;
    (void)height;
    (void)max_scale;
    (void)blend_factor;
    fprintf(stderr, "Multiscale Median filter CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif
