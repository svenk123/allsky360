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
#include "acdnr_filter_cuda.h"

#ifdef USE_CUDA
int acdnr_filter_rgbf1_cuda_Wrapper(float *rgba, int width, int height,
        float stddev_l, float amount_l, int iterations_l, int structure_size_l,
        float stddev_c, float amount_c, int iterations_c, int structure_size_c) {
    return acdnr_filter_rgbf1_cuda(rgba, width, height,
        stddev_l, amount_l, iterations_l, structure_size_l,
        stddev_c, amount_c, iterations_c, structure_size_c);
}

#else
// Use dummy function
int acdnr_filter_rgbf1_cuda_Wrapper(float *rgba, int width, int height,
        float stddev_l, float amount_l, int iterations_l, int structure_size_l,
        float stddev_c, float amount_c, int iterations_c, int structure_size_c) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)stddev_l;
    (void)amount_l;
    (void)iterations_l;
    (void)structure_size_l;
    (void)stddev_c;
    (void)amount_c;
    (void)iterations_c;
    (void)structure_size_c;
    fprintf(stderr, "ACDNR filter CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif
