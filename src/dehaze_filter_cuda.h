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
#ifndef DEHAZE_FILTER_CUDA_H
#define DEHAZE_FILTER_CUDA_H

/* CUDA-Dehaze-Filter-Function (C-compatible) */
#ifdef __cplusplus
extern "C" {
#endif

int perceptual_dehaze_rgbf1_multiscale_full_cuda(float *rgba_image, int width, int height, float amount, float haze_percent);
int perceptual_dehaze_rgbf1_multiscale_full_cuda_Wrapper(float *rgba_image, int width, int height, float amount, float haze_percent);

#ifdef __cplusplus
}
#endif

#endif // DEHAZE_FILTER_CUDA_H
