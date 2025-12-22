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
#ifndef ACDNR_FILTER_CUDA_H
#define ACDNR_FILTER_CUDA_H

/* CUDA-ACDNR-Filter-Function (C-compatible) */
#ifdef __cplusplus
extern "C" {
#endif

int acdnr_filter_rgbf1_cuda(float *rgba, int width, int height,
	float stddev_l, float amount_l, int iterations_l, int structure_size_l,
        float stddev_c, float amount_c, int iterations_c, int structure_size_c);
int acdnr_filter_rgbf1_cuda_Wrapper(float *rgba, int width, int height,
        float stddev_l, float amount_l, int iterations_l, int structure_size_l,
        float stddev_c, float amount_c, int iterations_c, int structure_size_c);

#ifdef __cplusplus
}
#endif

#endif // ACDNR_FILTER_CUDA_H
