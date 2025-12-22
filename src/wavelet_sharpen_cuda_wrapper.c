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
#include "wavelet_sharpen_cuda.h"

#ifdef USE_CUDA
int wavelet_sharpen_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float gain_small, float gain_medium, float gain_large) {
    return wavelet_sharpen_rgbf1_cuda(rgba, width, height, gain_small, gain_medium, gain_large);
}

#else
// Use dummy function
int wavelet_sharpen_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float gain_small, float gain_medium, float gain_large) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)gain_small;
    (void)gain_medium;
    (void)gain_large;
    fprintf(stderr, "Wavelet sharpening CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif