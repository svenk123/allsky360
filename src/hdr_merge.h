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
#ifndef HDR_MERGE_H
#define HDR_MERGE_H

#include <stddef.h>

#define CHANNELS 4  // RGBA
#define MAX_IMAGES 5

struct HdrFrames {
  float *image;
  size_t image_size;
  double exposure;
  double gain;
  double median_r;
  double median_g;
  double median_b;
  double brightness;
  float sigma_noise;
};

/**
 * Checks for overexposed pixels in an RGBA float image (value range 0.0 – 65535.0).
 *
 * @param img: pointer to the RGBA float image (4 channels per pixel).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param clipping_threshold: threshold for clipping detection (e.g. 65400.0).
 * @param clip_fraction_threshold: threshold for clipping detection as fraction of all pixels (e.g. 0.00001).
 * @param overexposed: output: set to 1 if clipping is detected, otherwise 0.
 * @return 0 on success, 1 on error (e.g. invalid parameters).
 */
int hdr_check_overexposure_rgbf1(const float *img, int width, int height,
                            float clipping_threshold, float clip_fraction_threshold, int *overexposed);

/**
 * Calculates the mean RGB values of a float RGBA image, considering a clipping threshold range.
 *
 * @param rgba: input image (RGBA, float, normalized range 0.0 – 1.0).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param clip_max: upper clipping threshold (e.g. 0.98f).
 * @param clip_min: lower thresholxd to avoid black-level artifacts (e.g. 0.001f).
 * @param out_mean_r: pointer to double for storing the mean red value.
 * @param out_mean_g: pointer to double for storing the mean green value.
 * @param out_mean_b: pointer to double for storing the mean blue value.
 * @return: 0 on success, 1 on error (e.g. invalid parameters or no valid pixels).
 */
int hdr_compute_mean_rgbf1(const float *rgba, int width, int height, 
                      float clip_min, float clip_max, 
                      double *out_mean_r, double *out_mean_g, double *out_mean_b);


#define MAX_PYRAMID_LEVELS 4

typedef struct {
    int levels;
    float *data[MAX_PYRAMID_LEVELS]; // 0 = full res, 1 = 1/2, 2 = 1/4, ...
    int width[MAX_PYRAMID_LEVELS];
    int height[MAX_PYRAMID_LEVELS];
} Pyramid;

/**
 * Performs Multi-Scale Exposure Fusion on a set of RGBF1 images.
 *
 * @param frames[MAX_IMAGES]: array of HdrFrames containing image data, exposure times, and mean values
 * @param use_max_images: number of images used
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param clipping_threshold: clipping threshold (0..1.0)
 * @param Y_max_expected: expected max Y value (scene dependent). 
 *                        If 0, automatically detected from image data.
 *                        If > 0, uses the specified value.
 * @param output_hdr: output buffer (RGBA float array)
 *
 * @return 0 on success, 1 on error
 */
int hdr_multi_scale_fusion_laplacian_rgbf1(
    const struct HdrFrames frames[MAX_IMAGES], int use_max_images, int width, int height,
    float clipping_threshold, float Y_max_expected,
    float *output_hdr, const char *dump_images_dir, int dump_weight_maps, int dump_weight_stats,
    int chroma_mode, float contrast_weight_strength, int pyramid_levels_override,
    float weight_sigma, float weight_clip_factor,
    float channel_scale_r, float channel_scale_g, float channel_scale_b);

/**
 * Normalizes the HDR RGBF1 image to a target output range.
 *
 * Finds global min/max of the RGB channels, and rescales the values
 * linearly to [target_min .. target_max].
 *
 * @param rgbf: input/output image (RGBA float array, modified in-place)
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param target_min: target minimum value (typically 0.0f)
 * @param target_max: target maximum value (typically 1.0f)
 *
 * @return 0 on success, 1 on error
 */
int hdr_normalize_range_rgbf1(float *rgbf, int width, int height, float target_min, float target_max);


#endif // HDR_MERGE_H
