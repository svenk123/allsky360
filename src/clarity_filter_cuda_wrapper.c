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
#include "clarity_filter_cuda.h"

#ifdef USE_CUDA
int clarity_filter_rgbf_masked_cuda_Wrapper(float *rgba, int width, int height,
                                           float strength, int radius, float midtone_width,
                                           int preserve_highlights, int mask_mode) {
    return clarity_filter_rgbf_masked_cuda(rgba, width, height, strength, radius,
                                          midtone_width, preserve_highlights, mask_mode);
}

#else
// Use dummy functions
int clarity_filter_rgbf_masked_cuda_Wrapper(float *rgba, int width, int height,
                                           float strength, int radius, float midtone_width,
                                           int preserve_highlights, int mask_mode) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)strength;
    (void)radius;
    (void)midtone_width;
    (void)preserve_highlights;
    (void)mask_mode;
    fprintf(stderr, "Clarity filter CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif
