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
#ifndef LIGHTS_AND_SHADOWS_CUDA_H
#define LIGHTS_AND_SHADOWS_CUDA_H

// CUDA-Lights-and-Shadows-Funktionen (C-compatible)
#ifdef __cplusplus
extern "C" {
#endif

int autostretch_rgbf1_cuda(float *hdr_image, int width, int height, float min_val, float max_val);
int autostretch_rgbf1_cuda_Wrapper(float *hdr_image, int width, int height, float min_val, float max_val);

int adjust_black_point_rgbf1_cuda(float *rgbf, int width, int height,
                                   double min_shift_pct, double max_shift_pct,
                                   double dark_threshold);
int adjust_black_point_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                          double min_shift_pct, double max_shift_pct,
                                          double dark_threshold);

#ifdef __cplusplus
}
#endif

#endif // LIGHTS_AND_SHADOWS_CUDA_H

