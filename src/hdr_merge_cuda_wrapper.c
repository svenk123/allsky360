/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license 
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include <stdio.h>
#include "hdr_merge_cuda.h"

#if 0
#ifdef USE_CUDA
int adjust_black_point_rgbf1_cuda_Wrapper(float *rgba, int width, int height, double black_shift_pct) {
    return adjust_black_point_rgbf1_cuda(rgba, width, height, black_shift_pct);
}

#else
// Use dummy function
int adjust_black_point_rgbf1_cuda_Wrapper(float *rgba, int width, int height, double black_shift_pct) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)black_shift_pct;
    fprintf(stderr, "HDR merge CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif
#endif
