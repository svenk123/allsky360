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
#ifndef WHITE_BALANCE_H
#define WHITE_BALANCE_H

#define CHANNELS 4

typedef enum {
  SCENE_NIGHT_CLEAR = 0,
  SCENE_NIGHT_MOON,
  SCENE_NIGHT_AURORA,
  SCENE_NIGHT_CLOUDY,
  SCENE_TWILIGHT,
  SCENE_DAY_CLEAR,
  SCENE_DAY_CLOUDY
} SceneType;

/**
 * Applies channel-wise scaling to an RGBA image (float) with light protection.
 *
 * @param rgba        Pointer to RGBA image data (uint16_t[width * height * 4])
 * @param width       Image width in pixels
 * @param height      Image height in pixels
 * @param scale_r     Scaling factor for red channel (e.g. 1.1)
 * @param scale_g     Scaling factor for green channel (usually 1.0)
 * @param scale_b     Scaling factor for blue channel (e.g. 1.3)
 * @param lp_strength Strength of light protection (e.g. 0.1), if 0.0 then light protection is disabled
 * @return int         0 on success, 1 on failure.
 */
int white_balance_rgbf1(float *rgba, int width, int height,
                               float scale_r, float scale_g, float scale_b,
                               float lp_strength);

/**
 * Estimates scaling factors for red and blue channels to achieve gray balance
 * based on a selected region in a 16-bit RGBA image.
 *
 * The function computes the average R, G, and B values in the given rectangular
 * region and calculates scale factors so that R and B can be scaled to match G.
 * Overexposed (clipped) pixels near 1.0f are excluded from the analysis.
 *
 * @param rgba     Pointer to the input image buffer (RGBA, uint16_t, 4 channels
 * per pixel).
 * @param width    Image width in pixels.
 * @param height   Image height in pixels.
 * @param x0       Top-left X coordinate of the region.
 * @param y0       Top-left Y coordinate of the region.
 * @param x1       Bottom-right X coordinate of the region.
 * @param y1       Bottom-right Y coordinate of the region.
 * @param scale_r  Output: scaling factor for the red channel (G / R mean).
 * @param scale_b  Output: scaling factor for the blue channel (G / B mean).
 */
int estimate_gray_scaling_rgbf1(const float *rgba, int width, int height,
                                int x0, int y0, int x1, int y1, float *scale_r,
                                float *scale_b);

#ifdef USE_CUDA
#include "white_balance_cuda.h"
#define auto_color_calibration_rgbf1(rgba, width, height, subframe_percent)    \
  auto_color_calibration_rgbf1_cuda_Wrapper(rgba, width, height,               \
                                            subframe_percent)

#define detect_scene_rgbf1(rgba, width, height)                               \
  detect_scene_rgbf1_cuda_Wrapper(rgba, width, height)

#define ambience_color_calibration_rgbf1(rgba, width, height, subframe_percent, luma_clip_high, sat_clip_high, sigma_factor, mix_factor) \
  ambience_color_calibration_rgbf1_cuda_Wrapper(rgba, width, height, subframe_percent, luma_clip_high, sat_clip_high, sigma_factor, mix_factor)
#else

/**
 * @brief Performs automatic white balance and background neutralization on a
 * float RGBA image.
 *
 * This function works in-place and modifies the input `rgba` image.
 *
 * @param rgba: pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel)
 * range).
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param subframe_percent: subframe percentage (e.g. 100.0 = full image, 80.0 = center 80%)
 * @return: 0 on success, 1 on failure.
 */
int auto_color_calibration_rgbf1(float *rgba, int width, int height,
                                 float subframe_percent);


/**
 * Scene detection for HDR/Allsky images.
 * Works on RGBA float image (0.0–1.0).
 *
 * @param rgba: pointer to RGBA float image data (0.0 - 1.0), size = width * height * 4
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @return: scene type (SCENE_NIGHT_CLEAR, SCENE_NIGHT_MOON, SCENE_NIGHT_AURORA, SCENE_NIGHT_CLOUDY, SCENE_TWILIGHT, SCENE_DAY_CLEAR, SCENE_DAY_CLOUDY)
 */
SceneType detect_scene_rgbf1(float *rgba, int width, int height);

/**
 * Ambience-based color calibration
 *
 * Parameters:
 * @param rgba: pointer to RGBA float image data (0.0 - 1.0), size = width * height * 4
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param subframe_percent: percentage of the central image region used for ambience statistics (e.g. 30.0)
 * @param luma_clip_high: ignore pixels with luminance above this threshold (e.g. 0.80)
 * @param sat_clip_high: ignore pixels with saturation above this threshold (e.g. 0.30)
 * @param sigma_factor: only use pixels within ±sigma_factor * stddev(Y) around mean Y (e.g. 1.0 - 1.5)
 * @param mix_factor: blend factor for the correction (0.0 = off, 1.0 = full correction, e.g. 0.2 - 0.4)
 *
 * @return:
 *   0 on success
 *  >0 on error (invalid parameters or not enough valid pixels)
 */
int ambience_color_calibration_rgbf1(float *rgba,
  int width,
  int height,
  float subframe_percent,
  float luma_clip_high,
  float sat_clip_high,
  float sigma_factor,
  float mix_factor);

#endif

#endif // WHITE_BALANCE_H
