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
#ifndef CLARITY_FILTER_CUDA_H
#define CLARITY_FILTER_CUDA_H

/* CUDA-Clarity-Filter-Function (C-compatible) */
#ifdef __cplusplus
extern "C" {
#endif

int clarity_filter_rgbf_masked_cuda(float *rgba, int width, int height,
                                   float strength, int radius, float midtone_width,
                                   int preserve_highlights, int mask_mode);
int clarity_filter_rgbf_masked_cuda_Wrapper(float *rgba, int width, int height,
                                           float strength, int radius, float midtone_width,
                                           int preserve_highlights, int mask_mode);


#ifdef __cplusplus
}
#endif

#endif // CLARITY_FILTER_CUDA_H
