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
#include "dehaze_filter_cuda.h"

#ifdef USE_CUDA
int perceptual_dehaze_rgbf1_multiscale_full_cuda_Wrapper(float *rgba_image, int width, int height, float amount, float haze_percent) {
    return perceptual_dehaze_rgbf1_multiscale_full_cuda(rgba_image, width, height, amount, haze_percent);
}

#else
// Use dummy function
int perceptual_dehaze_rgbf1_multiscale_full_cuda_Wrapper(float *rgba_image, int width, int height, float amount, float haze_percent) {
    (void)rgba_image;
    (void)width;
    (void)height;
    (void)amount;
    (void)haze_percent;
    fprintf(stderr, "Dehaze filter CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif
