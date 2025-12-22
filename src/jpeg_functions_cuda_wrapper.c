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
#include <cairo.h>
#include "jpeg_functions_cuda.h"

#ifdef USE_CUDA

int save_jpeg_rgbf16_cuda_Wrapper(const float *rgba, int width, int height, int compression_ratio, float scale, const char *filename) {
    return save_jpeg_rgbf16_cuda(rgba, width, height, compression_ratio, scale, filename);
}

int tonemap_rgbf1_to_rgbf16_cuda_Wrapper(float *rgbf, int width, int height) {
    return tonemap_rgbf1_to_rgbf16_cuda(rgbf, width, height);
}
#else

int save_jpeg_rgbf16_cuda_Wrapper(const float *rgba, int width, int height, int compression_ratio, float scale, const char *filename) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)compression_ratio;
    (void)scale;
    (void)filename;
    fprintf(stderr, "Save JPEG RGBA float 16-bit CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int tonemap_rgbf1_to_rgbf16_cuda_Wrapper(float *rgbf, int width, int height) {
    (void)rgbf;
    (void)width;
    (void)height;
    fprintf(stderr, "Tonemap RGBA float CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif