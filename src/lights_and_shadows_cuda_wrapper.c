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
#include "lights_and_shadows_cuda.h"

#ifdef USE_CUDA
int autostretch_rgbf1_cuda_Wrapper(float *hdr_image, int width, int height, float min_val, float max_val) {
    return autostretch_rgbf1_cuda(hdr_image, width, height, min_val, max_val);
}

int adjust_black_point_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                           double min_shift_pct, double max_shift_pct,
                                           double dark_threshold) {
    return adjust_black_point_rgbf1_cuda(rgbf, width, height, min_shift_pct, max_shift_pct, dark_threshold);
}

#else
// Falls CUDA nicht aktiviert ist, Dummy-Funktionen bereitstellen
int autostretch_rgbf1_cuda_Wrapper(float *hdr_image, int width, int height, float min_val, float max_val) {
    (void)hdr_image;
    (void)width;
    (void)height;
    (void)min_val;
    (void)max_val;
    fprintf(stderr, "Autostretch CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int adjust_black_point_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                          double min_shift_pct, double max_shift_pct,
                                          double dark_threshold) {
    (void)rgbf;
    (void)width;
    (void)height;
    (void)min_shift_pct;
    (void)max_shift_pct;
    (void)dark_threshold;
    fprintf(stderr, "Adjust black point CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif

