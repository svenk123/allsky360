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
#ifndef WHITE_BALANCE_CUDA_H
#define WHITE_BALANCE_CUDA_H

#include "white_balance.h"

/* CUDA-White-Balance-Function (C-compatible) */
#ifdef __cplusplus
extern "C" {
#endif

int auto_color_calibration_rgbf1_cuda(float *rgba, int width, int height, float subframe_percent);
int auto_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent);

SceneType detect_scene_rgbf1_cuda(float *rgba, int width, int height);
SceneType detect_scene_rgbf1_cuda_Wrapper(float *rgba, int width, int height);

int ambience_color_calibration_rgbf1_cuda(float *rgba, int width, int height, float subframe_percent, float luma_clip_high, float sat_clip_high, float sigma_factor, float mix_factor);
int ambience_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent, float luma_clip_high, float sat_clip_high, float sigma_factor, float mix_factor);

#ifdef __cplusplus
}
#endif

#endif // WHITE_BALANCE_CUDA_H
