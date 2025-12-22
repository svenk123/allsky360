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
#ifndef MEDIAN_FILTER_CUDA_H
#define MEDIAN_FILTER_CUDA_H

// CUDA-Median-Filter-Funktion (C-kompatibel)
#ifdef __cplusplus
extern "C" {
#endif

int median_filter_rgbf_cuda(float *rgba, int width, int height, int kernel_radius);
int median_filter_rgbf_cuda_Wrapper(float *rgba, int width, int height, int kernel_radius);

int multiscale_median_filter_rgbf1_cuda(
    float *host_rgba, int width, int height, int max_scale,
    float blend_factor);
int multiscale_median_filter_rgbf1_cuda_Wrapper(
    float *host_rgba, int width, int height, int max_scale,
    float blend_factor);


#ifdef __cplusplus
}
#endif

#endif // MEDIAN_FILTER_CUDA_H
