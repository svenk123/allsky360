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
#include "auto_gamma_cuda.h"

#ifdef USE_CUDA
int reinhard_tonemap_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float white_point) {
    return reinhard_tonemap_rgbf1_cuda(rgbf, width, height, white_point);
}

int apply_tone_curve_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float shadow_lift, float midtone_contrast) {
    return apply_tone_curve_rgbf1_cuda(rgbf, width, height,
                                       shadow_lift, midtone_contrast);
}

#else
/* Dummy functions — CUDA not available */
int reinhard_tonemap_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float white_point) {
    (void)rgbf;
    (void)width;
    (void)height;
    (void)white_point;
    fprintf(stderr, "Reinhard TMO CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int apply_tone_curve_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float shadow_lift, float midtone_contrast) {
    (void)rgbf;
    (void)width;
    (void)height;
    (void)shadow_lift;
    (void)midtone_contrast;
    fprintf(stderr, "Tone Curve CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif
