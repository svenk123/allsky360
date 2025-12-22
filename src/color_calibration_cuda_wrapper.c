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
#include "color_calibration_cuda.h"

#ifdef USE_CUDA

int adjust_saturation_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float saturation_factor) {
    return adjust_saturation_rgbf1_cuda(rgba, width, height, saturation_factor);
}

int apply_gamma_correction_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float gamma) {
    return apply_gamma_correction_rgbf1_cuda(rgba, width, height, gamma);
}
#else
// Dummy functions

int adjust_saturation_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float saturation_factor) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)saturation_factor;
    fprintf(stderr, "Saturation CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int apply_gamma_correction_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float gamma) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)gamma;
    fprintf(stderr, "Gamma CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif
