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
#include "white_balance_cuda.h"

#ifdef USE_CUDA
int auto_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent) {
    return auto_color_calibration_rgbf1_cuda(rgba, width, height, subframe_percent);
}

SceneType detect_scene_rgbf1_cuda_Wrapper(float *rgba, int width, int height) {
    return detect_scene_rgbf1_cuda(rgba, width, height);
}

int ambience_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent, float luma_clip_high, float sat_clip_high, float sigma_factor, float mix_factor) {
    return ambience_color_calibration_rgbf1_cuda(rgba, width, height, subframe_percent, luma_clip_high, sat_clip_high, sigma_factor, mix_factor);
}

#else
// Use dummy function
int auto_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)subframe_percent;
    fprintf(stderr, "Auto Color Calibration CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

SceneType detect_scene_rgbf1_cuda_Wrapper(float *rgba, int width, int height) {
    (void)rgba;
    (void)width;
    (void)height;
    fprintf(stderr, "Scene Detection CUDA is disabled! Falling back to CPU implementation.\n");
    return SCENE_NIGHT_CLEAR;
}

int ambience_color_calibration_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float subframe_percent, float luma_clip_high, float sat_clip_high, float sigma_factor, float mix_factor) {
    (void)rgba;
    (void)width;
    (void)height;
    (void)subframe_percent;
    (void)luma_clip_high;
    (void)sat_clip_high;
    (void)sigma_factor;
    (void)mix_factor;
    fprintf(stderr, "Ambience Color Calibration CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

#endif
