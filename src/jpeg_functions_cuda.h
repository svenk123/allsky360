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
#ifndef JPEG_FUNCTIONS_CUDA_H
#define JPEG_FUNCTIONS_CUDA_H

/* C-compatible functions */
#ifdef __cplusplus
extern "C" {
#endif

int save_jpeg_rgbf16_cuda(const float *rgba, int width, int height, int compression_ratio, float scale, const char *filename);
int save_jpeg_rgbf16_cuda_Wrapper(const float *rgba, int width, int height, int compression_ratio, float scale, const char *filename);

int tonemap_rgbf1_to_rgbf16_cuda(float *rgbf, int width, int height);
int tonemap_rgbf1_to_rgbf16_cuda_Wrapper(float *rgbf, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // JPEG_FUNCTIONS_CUDA_H