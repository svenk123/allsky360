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
#include "hdr_merge.h"
#include "allsky.h"
#include "hdr_merge_cuda.h"
#include "jpeg_functions.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hdr_check_overexposure_rgbf1(const float *img, int width, int height,
                                 float clipping_threshold,
                                 float clip_fraction_threshold,
                                 int *overexposed) {
  if (!img || width <= 0 || height <= 0 || !overexposed)
    return 1;

  printf("Overexposure check: clipping threshold: %f, clip fraction threshold: "
         "%f\n",
         clipping_threshold, clip_fraction_threshold);

  size_t pixel_count = width * height;
  *overexposed = 0;
  float max_r = 0.0;
  float max_g = 0.0;
  float max_b = 0.0;

  size_t overexposed_pixels_count = 0;
  size_t clip_limit = (size_t)(clip_fraction_threshold * pixel_count);

  if (clip_limit == 0) {
    clip_limit = 1; // at least 1 pixel required, if clip_fraction_threshold is
                    // very small
  }

  for (size_t i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS; // RGBA
    const float *pixel = &img[idx];

    if (pixel[0] > max_r)
      max_r = pixel[0];
    if (pixel[1] > max_g)
      max_g = pixel[1];
    if (pixel[2] > max_b)
      max_b = pixel[2];

    int clipped =
        (pixel[0] >= clipping_threshold || pixel[1] >= clipping_threshold ||
         pixel[2] >= clipping_threshold);

    if (clipped) {
      overexposed_pixels_count++;

      // Overexposed when more than 0.x of all pixels
      if (overexposed_pixels_count >= clip_limit) {
        *overexposed = 1;
        printf("early: max (red): %.4f, max (green): %.4f, max (blue): %.4f, "
               "clipped pixels: %zu\n",
               max_r, max_g, max_b, overexposed_pixels_count);
        return 0; // Return early
      }
    }
  }

  printf("Overexposure check: ok, max (red): %.4f, max (green): %.4f, max "
         "(blue): %.4f, clipped "
         "pixels: %zu\n",
         max_r, max_g, max_b, overexposed_pixels_count);

  return 0;
}

int hdr_compute_mean_rgbf1(const float *rgba, int width, int height,
                           float clip_min, float clip_max, double *out_mean_r,
                           double *out_mean_g, double *out_mean_b) {
  if (!rgba || width <= 0 || height <= 0 || !out_mean_r || !out_mean_g ||
      !out_mean_b)
    return 1;

  printf("Compute mean brightness: clipping min: %f, clipping max: %f\n",
         clip_min, clip_max);

  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  size_t count_r = 0, count_g = 0, count_b = 0;
  size_t pixel_count = width * height;

#pragma omp parallel for reduction(+ : sum_r, sum_g, sum_b, count_r, count_g,  \
                                       count_b)
  for (size_t i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    float r = rgba[idx + 0];
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    if (r >= clip_min && r < clip_max) {
      sum_r += r;
      count_r++;
    }
    if (g >= clip_min && g < clip_max) {
      sum_g += g;
      count_g++;
    }
    if (b >= clip_min && b < clip_max) {
      sum_b += b;
      count_b++;
    }
  }

  if (count_r == 0 || count_g == 0 || count_b == 0)
    return 1;

  *out_mean_r = sum_r / count_r;
  *out_mean_g = sum_g / count_g;
  *out_mean_b = sum_b / count_b;

  printf("Mean brightness: red: %.6f, green: %.6f, blue: %.6f\n", *out_mean_r,
         *out_mean_g, *out_mean_b);

  return 0;
}

/************************************/

float *convert_weight_map_to_rgbaf1(const float *weight_map, int width,
                                    int height) {
  if (!weight_map || width <= 0 || height <= 0)
    return NULL;

  size_t pixel_count = (size_t)width * height;
  float *rgbaf =
      (float *)allsky_safe_malloc(pixel_count * CHANNELS * sizeof(float));
  if (!rgbaf)
    return NULL;

#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; i++) {
    float v = weight_map[i]; // expected in 0..1.0
    v = clampf1(v);          // clamp to [0..1]

    size_t idx = i * (size_t)CHANNELS;
    rgbaf[idx + 0] = v;    // R
    rgbaf[idx + 1] = v;    // G
    rgbaf[idx + 2] = v;    // B
    rgbaf[idx + 3] = 1.0f; // A
  }

  return rgbaf;
}

void log_weight_map_stats(const float *weight_map, int width, int height,
                          const char *label) {
  if (!weight_map || width <= 0 || height <= 0)
    return;

  size_t pixel_count = (size_t)width * height;

  float min_val = 1e30f, max_val = -1e30f;
  double sum_weights = 0.0;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)     \
    reduction(+ : sum_weights)
  for (size_t i = 0; i < pixel_count; i++) {
    float w = weight_map[i];
    if (w < min_val)
      min_val = w;
    if (w > max_val)
      max_val = w;
    sum_weights += w;
  }

  printf("WeightMap [%s]: min=%.6f  max=%.6f  sum=%.2f  avg=%.6f\n", label,
         min_val, max_val, sum_weights, sum_weights / (double)pixel_count);
}

/************************************/

/* Burt-Adelson 5-tap separable Gaussian kernel [1 4 6 4 1]/16 */
static const float gauss_k5[5] = {
    1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f, 1.f / 16.f};

/* REDUCE: Gaussian filter + downsample by 2 */
static void gaussian_reduce_scalar(const float *src, int sw, int sh, float *dst,
                                   int dw, int dh) {
  for (int oy = 0; oy < dh; oy++) {
    for (int ox = 0; ox < dw; ox++) {
      float sum = 0.0f;
      for (int m = -2; m <= 2; m++) {
        int sy = 2 * oy + m;
        sy = (sy < 0) ? 0 : (sy >= sh ? sh - 1 : sy);
        for (int n = -2; n <= 2; n++) {
          int sx = 2 * ox + n;
          sx = (sx < 0) ? 0 : (sx >= sw ? sw - 1 : sx);
          sum += gauss_k5[m + 2] * gauss_k5[n + 2] * src[sy * sw + sx];
        }
      }
      dst[oy * dw + ox] = sum;
    }
  }
}

/* EXPAND: upsample by 2 using same 5-tap kernel (Burt-Adelson expand) */
static void gaussian_expand_scalar(const float *src, int sw, int sh, float *dst,
                                   int dw, int dh) {
  for (int oy = 0; oy < dh; oy++) {
    for (int ox = 0; ox < dw; ox++) {
      float sum = 0.0f;
      for (int m = -2; m <= 2; m++) {
        int src_y = oy - m;
        if (src_y & 1)
          continue; /* zeros at odd positions in upsampled grid */
        int sy = src_y >> 1;
        sy = (sy < 0) ? 0 : (sy >= sh ? sh - 1 : sy);
        for (int n = -2; n <= 2; n++) {
          int src_x = ox - n;
          if (src_x & 1)
            continue;
          int sx = src_x >> 1;
          sx = (sx < 0) ? 0 : (sx >= sw ? sw - 1 : sx);
          sum += gauss_k5[m + 2] * gauss_k5[n + 2] * src[sy * sw + sx];
        }
      }
      dst[oy * dw + ox] = sum * 4.0f;
    }
  }
}

/* treat pixels with (near) zero RGB or alpha==0 as masked */
static inline int is_masked_pixel(const float *px) {
  const float eps = 1e-6f;

  if (px[3] <= eps)
    return 1; // alpha as mask, if you use it

  return (px[0] <= eps && px[1] <= eps && px[2] <= eps);
}

static void build_gaussian_pyramid(float *input, Pyramid *pyramid,
                                   int base_width, int base_height) {
  pyramid->width[0] = base_width;
  pyramid->height[0] = base_height;
  pyramid->data[0] =
      allsky_safe_malloc(base_width * base_height * sizeof(float));
  if (!pyramid->data[0])
    return;
  memcpy(pyramid->data[0], input, base_width * base_height * sizeof(float));

  for (int level = 1; level < MAX_PYRAMID_LEVELS; level++) {
    int prev_w = pyramid->width[level - 1];
    int prev_h = pyramid->height[level - 1];
    int new_w = (prev_w + 1) / 2;
    int new_h = (prev_h + 1) / 2;

    pyramid->width[level] = new_w;
    pyramid->height[level] = new_h;
    pyramid->data[level] = allsky_safe_malloc(new_w * new_h * sizeof(float));
    if (!pyramid->data[level])
      return;

    gaussian_reduce_scalar(pyramid->data[level - 1], prev_w, prev_h,
                           pyramid->data[level], new_w, new_h);
  }
}

/* Build a Gaussian pyramid for a single channel extracted from an RGBA image
 * with per-channel gain. Level 0 is filled with linearized pixel values. */
static int build_gaussian_pyramid_channel(const float *rgba, int width,
                                          int height, int channel,
                                          float exp_g, float c_gain,
                                          Pyramid *pyramid) {
  pyramid->width[0] = width;
  pyramid->height[0] = height;
  size_t n0 = (size_t)width * height;
  pyramid->data[0] = (float *)allsky_safe_malloc(n0 * sizeof(float));
  if (!pyramid->data[0])
    return 1;

  for (size_t i = 0; i < n0; i++) {
    const float *px = &rgba[i * CHANNELS];
    pyramid->data[0][i] =
        is_masked_pixel(px) ? 0.0f : px[channel] * exp_g * c_gain;
  }

  for (int level = 1; level < MAX_PYRAMID_LEVELS; level++) {
    int prev_w = pyramid->width[level - 1];
    int prev_h = pyramid->height[level - 1];
    int new_w = (prev_w + 1) / 2;
    int new_h = (prev_h + 1) / 2;
    pyramid->width[level] = new_w;
    pyramid->height[level] = new_h;
    pyramid->data[level] =
        (float *)allsky_safe_malloc((size_t)new_w * new_h * sizeof(float));
    if (!pyramid->data[level])
      return 1;
    gaussian_reduce_scalar(pyramid->data[level - 1], prev_w, prev_h,
                           pyramid->data[level], new_w, new_h);
  }
  return 0;
}

static void free_gaussian_pyramid(Pyramid *pyramid) {
  if (!pyramid)
    return;

  for (int level = 0; level < MAX_PYRAMID_LEVELS; level++) {
    if (pyramid->data[level]) {
      allsky_safe_free(pyramid->data[level]);
      pyramid->data[level] = NULL;
    }
    pyramid->width[level] = 0;
    pyramid->height[level] = 0;
  }
}

/**
 * Computes pixel weight for HDR Multi-Scale Fusion using Gaussian-like curve
 * (Debevec/Malik approach). This provides smoother transitions and better
 * balance between different exposures compared to linear/smoothstep methods.
 *
 * @param linR       Linear R channel (after scaling/gain)
 * @param linG       Linear G channel
 * @param linB       Linear B channel
 * @param Y_max_expected   Y normalization factor (auto-detected if 0, otherwise
 * uses specified value)
 * @param opt_exposure_center  Exposure center value (optimal brightness,
 * typically 0.2-0.5)
 * @param weight_sigma  Width of the Gaussian bell curve
 * @return pixel weight [0.0 .. 1.0]
 */
static float compute_pixel_weight(float linR, float linG, float linB,
                                  float Y_max_expected,
                                  float opt_exposure_center,
                                  float weight_sigma) {

  /* 1) Convert RGB to Y (Rec.709) */
  float Y = 0.2126f * linR + 0.7152f * linG + 0.0722f * linB;

  /* 2) Normalize Y to [0..1] range */
  float Y_norm = fminf(Y / Y_max_expected, 1.0f);

  /* 3) Gaussian-like weight function (Debevec/Malik approach)
   * Uses a bell curve centered at opt_exposure_center
   * This provides smoother transitions and better balance between exposures */
  float delta = Y_norm - opt_exposure_center;
  float w = expf(-0.5f * (delta * delta) / (weight_sigma * weight_sigma));

  /* 4) Under-exposure suppression - reduce weight for very dark pixels */
  if (Y_norm < 0.10f) {
    float dark_threshold = 0.10f;
    float dark_factor = Y_norm / dark_threshold;
    /* Use moderate suppression: quadratic falloff for very dark areas */
    w *= dark_factor * dark_factor; // Quadratic falloff for very dark areas
  }

  return w;
}

/* Helper: count how many valid pyramid levels exist (data != NULL and size > 0)
 */
static int get_pyr_levels(const Pyramid *pyr) {
  int L = 0;

  for (int i = 0; i < MAX_PYRAMID_LEVELS; ++i) {
    if (!pyr->data[i])
      break;

    if (pyr->width[i] <= 0 || pyr->height[i] <= 0)
      break;
    L++;
  }
  return L;
}

/* contrast_weight_strength: 0..1; 0 disables local-contrast weighting */
/* noise_fusion_strength: 0..1; 0 disables SNR-based downweight of k>0 */
/* pyramid_levels_override: 0=auto, >0 clamps max levels */
int hdr_multi_scale_fusion_laplacian_rgbf1(
    const struct HdrFrames frames[MAX_IMAGES], int use_max_images, int width,
    int height, float clipping_threshold, float Y_max_expected,
    float *output_hdr, const char *dump_images_dir, int dump_weight_maps,
    int dump_weight_stats, float contrast_weight_strength,
    float noise_fusion_strength, int pyramid_levels_override,
    float weight_sigma, float weight_clip_factor, float wb_r, float wb_b) {
  if (wb_r <= 0.0f)
    wb_r = 1.0f;
  if (wb_b <= 0.0f)
    wb_b = 1.0f;
  if (!frames || !output_hdr || width <= 0 || height <= 0 ||
      use_max_images <= 0)
    return 1;

  const size_t pixel_count = (size_t)width * height;

  /* Check for invalid images */
  for (int k = 0; k < use_max_images; ++k) {
    if (!frames[k].image || frames[k].exposure <= 0.0 ||
        frames[k].median_r <= 0.0 || frames[k].median_g <= 0.0 ||
        frames[k].median_b <= 0.0) {

      /* Don't use invalid images for fusion */
      fprintf(stderr,
              "Invalid image %d: exposure: %.3f, median_r: %.3f, median_g: "
              "%.3f, median_b: %.3f\n",
              k, frames[k].exposure, frames[k].median_r, frames[k].median_g,
              frames[k].median_b);
      return 1;
    }
  }

  /* Precompute gains.
   *
   * These two must express DIFFERENT things, or the correction is applied
   * twice (see review.md, BUG-09):
   *
   *   exp_gain[k]  - the brightness difference between frame 0 and frame k,
   *                  taken from the exposure times.
   *   r/g/b_gain[k] - only the residual COLOUR drift between the frames.
   *
   * The channel gains are therefore normalised against each frame's own green
   * channel, so g_gain is 1 by definition and r/b_gain stay near 1.0. Using
   * the raw median ratio here instead would encode the exposure difference a
   * second time, and px * exp_gain * gain would scale by (t0/tk)^2.
   *
   * Note: the frames are fused on raw sensor values (the white balance
   * follows the fusion); a global per-channel factor would cancel in this
   * ratio of ratios anyway. */
  float exp_gain[MAX_IMAGES] = {0};
  float r_gain[MAX_IMAGES] = {0};
  float g_gain[MAX_IMAGES] = {0};
  float b_gain[MAX_IMAGES] = {0};
  for (int k = 0; k < use_max_images; ++k) {
    exp_gain[k] = (float)(frames[0].exposure / frames[k].exposure);

    /* Colour balance of each frame, relative to its own green channel.
     * The medians are guaranteed > 0 by the validity check above. */
    const double cb_r0 = frames[0].median_r / frames[0].median_g;
    const double cb_b0 = frames[0].median_b / frames[0].median_g;
    const double cb_rk = frames[k].median_r / frames[k].median_g;
    const double cb_bk = frames[k].median_b / frames[k].median_g;

    r_gain[k] = (float)(cb_r0 / cb_rk);
    g_gain[k] = 1.0f;
    b_gain[k] = (float)(cb_b0 / cb_bk);
  }

  /* Per-frame luminance noise std in the same units as luma_maps[], for
   * noise-aware fusion (sigma_noise = MAD background on raw frame k). */
  float sigma_Y_lin[MAX_IMAGES] = {0};
  if (noise_fusion_strength > 1e-6f) {
    int any_sigma = 0;
    for (int k = 0; k < use_max_images; ++k) {
      double sn = frames[k].sigma_noise;
      if (sn > 0.0) {
        sigma_Y_lin[k] = (float)(sn * exp_gain[k] * g_gain[k]);
        any_sigma = 1;
      }
    }
    if (any_sigma) {
      printf("HDR noise-aware fusion: strength=%.2f, sigma_Y linear: ",
             noise_fusion_strength);
      for (int k = 0; k < use_max_images; ++k)
        printf("[%d]=%.5g ", k, sigma_Y_lin[k]);
      printf("\n");
    }
  }

  /* Auto exposure center from green median values */
  double sum_median_Y = 0.0;
  for (int k = 0; k < use_max_images; ++k) {
    /* Compute luminance median from RGB medians (Rec.709) */
    double median_Y = 0.2126 * frames[k].median_r +
                      0.7152 * frames[k].median_g + 0.0722 * frames[k].median_b;
    sum_median_Y += median_Y;
  }
  const double proxy_hdr_brightness = sum_median_Y / use_max_images;

  /* Smooth exposure center: linearly interpolated from scene brightness */
  float base_center = (Y_max_expected >= 1.0f) ? 0.20f : 0.15f;
  float max_center = (Y_max_expected >= 1.0f) ? 0.55f : 0.40f;
  float opt_exposure_center =
      fminf(base_center + (float)proxy_hdr_brightness, max_center);

  /* Allocate maps */
  float *weight_maps[MAX_IMAGES] = {0};
  float *luma_maps[MAX_IMAGES] = {0};
  float *contrast_maps[MAX_IMAGES] = {0}; // optional

  for (int k = 0; k < use_max_images; ++k) {
    weight_maps[k] =
        (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
    if (!weight_maps[k])
      goto OOM_FAIL;

    luma_maps[k] =
        (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
    if (!luma_maps[k])
      goto OOM_FAIL;

    if (contrast_weight_strength > 0.0f) {
      contrast_maps[k] =
          (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
      if (!contrast_maps[k])
        goto OOM_FAIL;
    }
  }

/* Compute linearized luma */
#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; ++i) {
    for (int k = 0; k < use_max_images; ++k) {
      const float *px = &frames[k].image[(size_t)i * CHANNELS];
      if (is_masked_pixel(px)) {
        luma_maps[k][i] = 0.0f;
        continue;
      }

      const float linR = px[0] * exp_gain[k] * r_gain[k];
      const float linG = px[1] * exp_gain[k] * g_gain[k];
      const float linB = px[2] * exp_gain[k] * b_gain[k];

      /* Convert RGB to Y (Rec.709) */
      float Y = 0.2126f * linR + 0.7152f * linG + 0.0722f * linB;
      luma_maps[k][i] = Y;
    }
  }

  /* Auto-detect Y_max_expected if parameter is 0 */
  if (Y_max_expected <= 0.0f) {
    float max_Y_nonclip = 0.0f;
    float max_Y_clipped = 0.0f;

    /* Find maximum Y value separately for clipped and non-clipped pixels */
    for (int k = 0; k < use_max_images; ++k) {
#pragma omp parallel for reduction(max : max_Y_nonclip, max_Y_clipped)
      for (size_t i = 0; i < pixel_count; ++i) {
        const float *px = &frames[k].image[(size_t)i * CHANNELS];
        if (is_masked_pixel(px))
          continue;

        float max_rgb_src = fmaxf(fmaxf(px[0], px[1]), px[2]);
        int is_clipped = (max_rgb_src >= clipping_threshold * 0.95f);
        float Y_val = luma_maps[k][i];

        if (!is_clipped) {
          if (Y_val > max_Y_nonclip)
            max_Y_nonclip = Y_val;
        } else {
          if (Y_val > max_Y_clipped)
            max_Y_clipped = Y_val;
        }
      }
    }

    float max_Y_found = (max_Y_nonclip > 0.0f) ? max_Y_nonclip : max_Y_clipped;
    int found_non_clipped = (max_Y_nonclip > 0.0f);

    /* Apply safety factor (10%) and ensure minimum reasonable value */
    if (max_Y_found > 0.0f) {
      Y_max_expected = max_Y_found * 1.1f; // 10% safety margin

      /* Clamp to reasonable range: if too small or too large, use defaults */
      if (Y_max_expected < 0.5f) {
        Y_max_expected = 1.0f; // Default for very dark scenes
      } else if (Y_max_expected > 5.0f) {
        Y_max_expected = 3.0f; // Default for very bright scenes (sun visible)
      }

      printf("Auto-detected Y_max_expected: %.3f (from max Y: %.3f, "
             "non-clipped: %s)\n",
             Y_max_expected, max_Y_found, found_non_clipped ? "yes" : "no");
    } else {
      /* Fallback if no valid Y values found */
      Y_max_expected = 1.0f;
      printf("Auto-detection failed, using default Y_max_expected: %.3f\n",
             Y_max_expected);
    }
  }

  /* Auto-detect weight_clip_factor from shortest exposure if parameter is 0.
   * Uses partial clipping fraction: pixels where some channels clip but not all.
   * This directly measures the "color fringe risk" (differential channel
   * clipping, e.g. R+G clipped but B not). Higher partial clip fraction →
   * higher weight_clip_factor to suppress the fringe. */
  if (weight_clip_factor <= 0.0f) {
    /* Use shortest exposure (last image) as it has smallest clipped area */
    int k_shortest = use_max_images - 1;
    size_t total_clipped = 0;
    size_t partial_clipped = 0;
    const float clip_lo = clipping_threshold * 0.90f;

    for (size_t i = 0; i < pixel_count; ++i) {
      const float *px = &frames[k_shortest].image[(size_t)i * CHANNELS];
      float max_c = fmaxf(fmaxf(px[0], px[1]), px[2]);
      if (max_c >= clipping_threshold * 0.95f) {
        total_clipped++;
        float min_c = fminf(fminf(px[0], px[1]), px[2]);
        if (min_c < clip_lo)
          partial_clipped++;
      }
    }

    float total_fraction  = (float)total_clipped  / (float)pixel_count;
    float partial_fraction = (float)partial_clipped / (float)pixel_count;

    /* Map partial_fraction → weight_clip_factor.
     * No partial clipping (night/overcast) → 0.5 (soft suppression)
     * Heavy partial clipping (direct sun)  → 0.92 (strong suppression) */
    const float factor_min = 0.50f;
    const float factor_max = 0.92f;
    /* Normalize by total_fraction to get relative partial-clip share */
    float partial_rel = (total_fraction > 1e-6f)
                        ? (partial_fraction / total_fraction) : 0.0f;
    weight_clip_factor = factor_min + (factor_max - factor_min) * partial_rel;

    printf("Auto weight_clip_factor: total_clipped=%.5f partial_clipped=%.5f "
           "partial_rel=%.3f -> weight_clip_factor=%.3f\n",
           total_fraction, partial_fraction, partial_rel, weight_clip_factor);
  }

  /* Optional contrast map (|L - mean3x3| normalized) */
  if (contrast_weight_strength > 0.0f) {
    for (int k = 0; k < use_max_images; ++k) {
      float *C = contrast_maps[k];
      const float *L = luma_maps[k];
      /* compute local deviation */
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int idx = y * width + x;

          const float *p0 = &frames[k].image[(size_t)idx * CHANNELS];
          if (is_masked_pixel(p0)) {
            C[idx] = 0.0f;
            continue;
          }

          float sum = 0.0f;
          int cnt = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            int yy = y + dy;
            if ((unsigned)yy >= (unsigned)height)
              continue;

            for (int dx = -1; dx <= 1; ++dx) {
              int xx = x + dx;

              if ((unsigned)xx >= (unsigned)width)
                continue;

              const float *pn =
                  &frames[k].image[(size_t)(yy * width + xx) * CHANNELS];
              /* Skip masked pixels */
              if (is_masked_pixel(pn))
                continue;

              sum += L[yy * width + xx];
              cnt++;
            }
          }
          /* Skip pixels with no valid neighbors */
          if (cnt == 0) {
            C[idx] = 0.0f;
            continue;
          }

          /* Compute local mean */
          float mean3 = sum / (float)cnt;
          C[idx] = fabsf(L[idx] - mean3);
        }
      }

      /* Normalize by 99th percentile to reduce outlier influence */
      float maxv = 1e-12f;
      for (size_t i = 0; i < pixel_count; ++i)
        if (C[i] > maxv)
          maxv = C[i];

      /* Build 256-bin histogram to find 99th percentile */
      int hist[256] = {0};
      for (size_t i = 0; i < pixel_count; ++i) {
        int bin = (int)(C[i] / maxv * 255.0f);
        hist[(bin < 0) ? 0 : (bin > 255 ? 255 : bin)]++;
      }
      size_t target99 = (size_t)(pixel_count * 0.99);
      size_t cum = 0;
      float p99 = maxv;
      for (int b = 0; b < 256; b++) {
        cum += (size_t)hist[b];
        if (cum >= target99) {
          p99 = maxv * (b + 1.0f) / 256.0f;
          break;
        }
      }
      float inv = 1.0f / fmaxf(p99, 1e-12f);

      for (size_t i = 0; i < pixel_count; ++i) {
        float v = C[i] * inv;
        C[i] = (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);
      }
    }
  }

/* Base weights (exposure/clipping) +/- contrast mix */
#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; ++i) {
    for (int k = 0; k < use_max_images; ++k) {
      const float *px = &frames[k].image[(size_t)i * CHANNELS];

      /* Skip masked pixels */
      if (is_masked_pixel(px)) {
        weight_maps[k][i] = 0.0f;
        continue;
      }

      /* Compute weight based on non-linearized values for fairer distribution
       */
      /* We still need to normalize by exposure to compare fairly across images
       */
      const float normR = px[0] * exp_gain[k];
      const float normG = px[1] * exp_gain[k];
      const float normB = px[2] * exp_gain[k];

      /* Compute pixel weight using normalized (exposure-compensated) but not
       * color-gain-adjusted values */
      float w = compute_pixel_weight(normR, normG, normB, Y_max_expected,
                                     opt_exposure_center, weight_sigma);

      /* Note: Softmask for dark areas in shorter exposures is applied after
       * weight calculation */

      /* Clipping suppression via smoothstep */
      const float *px_src = &frames[k].image[(size_t)i * CHANNELS];
      const float max_rgb_src =
          fmaxf(fmaxf(px_src[0], px_src[1]), px_src[2]);
      const float a = clipping_threshold * 0.95f;
      const float b_clip = clipping_threshold * 1.00f;
      float u = 1.0f;
      if (max_rgb_src >= a) {
        float t = 1.0f - (max_rgb_src - a) / fmaxf(b_clip - a, 1e-6f);
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        u = t * t * (3.f - 2.f * t);
      }
      w *= (1.0f - weight_clip_factor) + weight_clip_factor * u;

      if (contrast_weight_strength > 0.0f) {
        float cw = contrast_maps[k][i]; // [0..1]
        w *= (1.0f - contrast_weight_strength) + contrast_weight_strength * cw;
      }

      /* Shorter exposures: suppress weight where linearized luma is not above
       * noise floor (sigma_Y_lin from background MAD, exposure- and WB-aligned). */
      if (k > 0 && noise_fusion_strength > 1e-6f &&
          sigma_Y_lin[k] > 1e-15f) {
        float Ylin = luma_maps[k][i];
        float snr = Ylin / sigma_Y_lin[k];
        const float snr_lo = 2.5f;
        const float snr_hi = 6.0f;
        float t = (snr - snr_lo) / fmaxf(snr_hi - snr_lo, 1e-6f);
        t = fmaxf(0.0f, fminf(1.0f, t));
        float nf = t * t * (3.0f - 2.0f * t);
        w *= (1.0f - noise_fusion_strength) + noise_fusion_strength * nf;
      }

      w = clampf1(w);
      weight_maps[k][i] = w;
    }
  }

  /* Apply softmask to shorter exposures: mask out very dark areas that are
   * likely noisy */
  /* This prevents 30s, 15s, and shorter images from contributing noise in dark
   * regions */
  for (int k = 1; k < use_max_images; ++k) {

#pragma omp parallel for
    for (size_t i = 0; i < pixel_count; ++i) {
      const float *px = &frames[k].image[(size_t)i * CHANNELS];

      /* Skip masked pixels */
      if (is_masked_pixel(px))
        continue;

      /* Calculate softmask value: 0 = fully masked, 1 = fully visible */
      float mask_value = 1.0f;

      if (k >= 2) {
        /* For very short exposures (k >= 2, e.g. 3.75s): only use bright areas
         */
        /* These images are too short to capture dark areas well, so only bright
         * pixels should contribute */

        /* Define brightness threshold in the short exposure's own raw scale:
         * pixels must be clearly above the sky/noise floor, but do not require
         * near-saturation. */
        float brightness_threshold = 0.0f;

        if (frames[k].median_g > 0.0f) {
          float median_floor = (float)frames[k].median_g * 6.0f;
          float noise_floor = (frames[k].sigma_noise > 0.0f)
                                  ? (float)frames[k].sigma_noise * 6.0f
                                  : 0.0f;
          brightness_threshold =
              fmaxf(fmaxf(median_floor, noise_floor), 0.03f);
        } else {
          brightness_threshold = 0.03f;
        }

        float transition_end = brightness_threshold * 0.70f;
        float transition_start = brightness_threshold;

        /* Gate very short exposures by their own raw brightness. Using
         * exposure-compensated luma here lets amplified sky noise pass through
         * as if it were real signal. */
        float raw_Y = 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];

        if (raw_Y < transition_end) {
          /* Too dark: fully mask out */
          mask_value = 0.0f;
        } else if (raw_Y < transition_start) {
          /* Transition zone: smooth interpolation */
          float t = (raw_Y - transition_end) /
                    (transition_start - transition_end);
          t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp to [0,1]
          /* Use smoothstep for smooth transition */
          mask_value = t * t * (3.0f - 2.0f * t);
        }
        /* else: bright enough, keep full weight (mask_value = 1.0) */
      } else {
        /* For the moderately short exposure (k == 1), use a softer version of
         * the raw short-frame gate: keep real bright signal, but stop
         * exposure-compensated sky noise from entering the fusion. */
        float dark_threshold = 0.0f;
        if (frames[k].median_g > 0.0f) {
          float median_floor = (float)frames[k].median_g * 3.0f;
          float noise_floor = (frames[k].sigma_noise > 0.0f)
                                  ? (float)frames[k].sigma_noise * 4.0f
                                  : 0.0f;
          dark_threshold = fmaxf(fmaxf(median_floor, noise_floor), 0.015f);
        } else {
          dark_threshold = 0.015f;
        }

        float transition_start = dark_threshold;
        float transition_end = dark_threshold * 0.70f;
        float raw_Y = 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];

        if (raw_Y < transition_end) {
          /* Very dark: fully mask out */
          mask_value = 0.0f;
        } else if (raw_Y < transition_start) {
          /* Transition zone: smooth interpolation */
          float t = (raw_Y - transition_end) /
                    (transition_start - transition_end);
          t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp to [0,1]
          /* Use smoothstep for smooth transition */
          mask_value = t * t * (3.0f - 2.0f * t);
        }
        /* else: bright enough, keep full weight (mask_value = 1.0) */
      }

      /* Apply softmask to weight map */
      weight_maps[k][i] *= mask_value;
    }
  }

/* Per-Pixel-normalization */
#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; ++i) {
    float sumw = 0.0f, maxw = 0.0f;
    int maxk = 0;

    /* Where the longest frame is clipped, give the shortest frame a small
     * floor weight. Where every frame is clipped (sun, bright cloud next to
     * it) all weights are ~0, and normalising those tiny leftovers mixed the
     * clipped values of the long frames (linear ~1) with the short ones
     * (~exp_gain): the area came out as a dull grey that looks pink next to
     * the sky. With the floor the shortest frame - the closest to the true
     * value - wins there; elsewhere the floor is negligible. */
    if (use_max_images > 1) {
      const float *p0 = &frames[0].image[(size_t)i * CHANNELS];
      if (!is_masked_pixel(p0)) {
        const float m0 = fmaxf(fmaxf(p0[0], p0[1]), p0[2]);
        const float a = clipping_threshold * 0.95f;
        if (m0 >= a) {
          float t = (m0 - a) / fmaxf(clipping_threshold * 0.05f, 1e-6f);
          t = fminf(1.0f, fmaxf(0.0f, t));
          weight_maps[use_max_images - 1][i] +=
              1e-2f * t * t * (3.0f - 2.0f * t);
        }
      }
    }

    /* Sum weights */
    for (int k = 0; k < use_max_images; ++k) {
      float w = weight_maps[k][i];
      sumw += w;
      if (w > maxw) {
        maxw = w;
        maxk = k;
      }
    }

    /* Normalize weights */
    if (sumw > 1e-12f) {
      float inv = 1.0f / sumw;

      for (int k = 0; k < use_max_images; ++k)
        weight_maps[k][i] *= inv;
    } else if (maxw > 0.0f) {
      for (int k = 0; k < use_max_images; ++k)
        weight_maps[k][i] = (k == maxk) ? 1.0f : 0.0f;
    }
  }

  /* Debug dumps */
  if (dump_weight_stats || dump_weight_maps) {
    for (int k = 0; k < use_max_images; ++k) {
      char label[64];
      snprintf(label, sizeof(label), "Image %d", k);

      /* Log weight map stats */
      if (dump_weight_stats)
        log_weight_map_stats(weight_maps[k], width, height, label);

      /* Dump weight map */
      if (dump_weight_maps) {
        float *weight_rgba =
            convert_weight_map_to_rgbaf1(weight_maps[k], width, height);
        if (weight_rgba) {
          // FIXME Dateiname
          char filename[PATH_MAX + 1];
          if (allsky_safe_snprintf(filename, sizeof(filename),
                                   "%s/hdr_weight_map_img%0d.jpg",
                                   dump_images_dir, k)) {
            fprintf(stderr, "WARNING: String %s truncated\n", filename);
          }

          save_jpeg_rgbf1(weight_rgba, width, height, 50, 0.25f, filename);
          allsky_safe_free(weight_rgba);
        }
      }
    }
  }

  /* Build Gaussian pyramids for R, G, B (per channel) and weights */
  Pyramid gauss_R[MAX_IMAGES], gauss_G[MAX_IMAGES], gauss_B[MAX_IMAGES];
  Pyramid gauss_weight[MAX_IMAGES];
  memset(gauss_R, 0, sizeof(gauss_R));
  memset(gauss_G, 0, sizeof(gauss_G));
  memset(gauss_B, 0, sizeof(gauss_B));
  memset(gauss_weight, 0, sizeof(gauss_weight));

  for (int k = 0; k < use_max_images; ++k) {
    if (build_gaussian_pyramid_channel(frames[k].image, width, height, 0,
                                       exp_gain[k], r_gain[k], &gauss_R[k]))
      goto OOM_FAIL;
    if (build_gaussian_pyramid_channel(frames[k].image, width, height, 1,
                                       exp_gain[k], g_gain[k], &gauss_G[k]))
      goto OOM_FAIL;
    if (build_gaussian_pyramid_channel(frames[k].image, width, height, 2,
                                       exp_gain[k], b_gain[k], &gauss_B[k]))
      goto OOM_FAIL;
    build_gaussian_pyramid(weight_maps[k], &gauss_weight[k], width, height);

    /* Clip repair at pyramid level 0: smooth-blend each clipped pixel from
     * its own (clipped, wrong) color toward a replacement as max(R,G,B)
     * rises from clip_lo to clip_hi. The replacement is the same pixel from
     * the first shorter frame that is not clipped there, linearised exactly
     * like this frame (exp_gain * channel gain), so it carries the true
     * color and brightness. Only where every shorter frame is clipped too
     * (e.g. the sun) the pixel is neutralised to white at its clip level.
     *
     * Why not just neutralise: the clipping weight drops these pixels at
     * full resolution, but the weights are blurred in the coarser pyramid
     * levels, so the low-pass of frame k still contributes along the edge of
     * a clipped area. A clipped or neutralised pixel lacks the channels
     * that clipped first, and that deficit shows up as a colored fringe at
     * cloud edges and towards the horizon. A smoothstep blend also avoids the
     * hard color edge that would ring in the Laplacian.
     *
     * Two ramps: with a replacement frame the ramp is the clipping weight
     * ramp (0.95..1.0 of clipping_threshold), so unclipped bright sky keeps
     * its color. Without one (shortest frame, or all shorter frames clipped)
     * the neutralisation starts early (0.80..0.95): green clips first on
     * this sensor (red/blue stay well below 1.0 even in the sun), and a late
     * ramp leaves a colored fringe around the clipped area.
     *
     * After modifying level 0, the higher Gaussian levels are rebuilt so that
     * lap[i] = gauss[i] - expand(gauss[i+1]) stays self-consistent. */
    {
      /* replacement ramp (same as the clipping weight ramp) */
      const float rep_lo = clipping_threshold * 0.95f;
      const float rep_hi = clipping_threshold * 1.00f;
      /* neutralisation ramp (no unclipped shorter frame) */
      const float neu_lo = clipping_threshold * 0.80f;
      const float neu_hi = clipping_threshold * 0.95f;

#pragma omp parallel for
      for (size_t i = 0; i < pixel_count; ++i) {
        const float *px = &frames[k].image[(size_t)i * CHANNELS];
        float max_rgb = fmaxf(fmaxf(px[0], px[1]), px[2]);
        if (max_rgb < neu_lo || is_masked_pixel(px))
          continue;

        /* First shorter frame without clipping at this pixel */
        int j = -1;
        for (int m = k + 1; m < use_max_images; ++m) {
          const float *pm = &frames[m].image[(size_t)i * CHANNELS];
          if (fmaxf(fmaxf(pm[0], pm[1]), pm[2]) < rep_lo) {
            j = m;
            break;
          }
        }

        const float lo = (j >= 0) ? rep_lo : neu_lo;
        const float hi = (j >= 0) ? rep_hi : neu_hi;
        if (max_rgb < lo)
          continue;
        float t = (max_rgb - lo) / fmaxf(hi - lo, 1e-6f);
        t = fminf(1.0f, fmaxf(0.0f, t));
        t = t * t * (3.0f - 2.0f * t); /* smoothstep */

        float tR, tG, tB;
        if (j >= 0) {
          const float *pj = &frames[j].image[(size_t)i * CHANNELS];
          tR = pj[0] * exp_gain[j] * r_gain[j];
          tG = pj[1] * exp_gain[j] * g_gain[j];
          tB = pj[2] * exp_gain[j] * b_gain[j];
        } else {
          /* No frame has data here: white at the clip level (the largest
           * channel) - a luminance grey would be darker than the unclipped
           * surroundings. White in the white-balanced result: the frames
           * are raw, the white balance follows the fusion, so the target
           * is W / wb per channel. */
          float W = fmaxf(fmaxf(gauss_R[k].data[0][i] * wb_r,
                                gauss_G[k].data[0][i]),
                          gauss_B[k].data[0][i] * wb_b);
          tR = W / wb_r;
          tG = W;
          tB = W / wb_b;
        }
        gauss_R[k].data[0][i] += t * (tR - gauss_R[k].data[0][i]);
        gauss_G[k].data[0][i] += t * (tG - gauss_G[k].data[0][i]);
        gauss_B[k].data[0][i] += t * (tB - gauss_B[k].data[0][i]);
      }

      /* Rebuild higher Gaussian pyramid levels from the corrected level 0 */
      for (int level = 1; level < MAX_PYRAMID_LEVELS; ++level) {
        if (!gauss_R[k].data[level])
          break;
        int pw = gauss_R[k].width[level - 1], ph = gauss_R[k].height[level - 1];
        int nw = gauss_R[k].width[level],     nh = gauss_R[k].height[level];
        gaussian_reduce_scalar(gauss_R[k].data[level - 1], pw, ph,
                               gauss_R[k].data[level], nw, nh);
        gaussian_reduce_scalar(gauss_G[k].data[level - 1], pw, ph,
                               gauss_G[k].data[level], nw, nh);
        gaussian_reduce_scalar(gauss_B[k].data[level - 1], pw, ph,
                               gauss_B[k].data[level], nw, nh);
      }
    }
  }

  /* Determine usable number of levels L (min across all pyramids) */
  int L = get_pyr_levels(&gauss_R[0]);
  if (L <= 0)
    goto OOM_FAIL;
  for (int k = 0; k < use_max_images; ++k) {
    int lk;
    lk = get_pyr_levels(&gauss_R[k]);      if (lk < L) L = lk;
    lk = get_pyr_levels(&gauss_G[k]);      if (lk < L) L = lk;
    lk = get_pyr_levels(&gauss_B[k]);      if (lk < L) L = lk;
    lk = get_pyr_levels(&gauss_weight[k]); if (lk < L) L = lk;
  }
  if (pyramid_levels_override > 0 && pyramid_levels_override < L)
    L = pyramid_levels_override;
  if (L <= 0)
    goto OOM_FAIL;

  /* Build Laplacian pyramids for R, G, B */
  float *lap_R[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};
  float *lap_G[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};
  float *lap_B[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};
  int lap_W[MAX_PYRAMID_LEVELS] = {0};
  int lap_H[MAX_PYRAMID_LEVELS] = {0};

  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < L; ++level) {
      int wL = gauss_R[k].width[level];
      int hL = gauss_R[k].height[level];
      if (k == 0) { lap_W[level] = wL; lap_H[level] = hL; }
      size_t sz = (size_t)wL * hL * sizeof(float);
      lap_R[k][level] = (float *)allsky_safe_malloc(sz);
      lap_G[k][level] = (float *)allsky_safe_malloc(sz);
      lap_B[k][level] = (float *)allsky_safe_malloc(sz);
      if (!lap_R[k][level] || !lap_G[k][level] || !lap_B[k][level])
        goto OOM_FAIL;
    }

    for (int level = 0; level < L - 1; ++level) {
      int wL = gauss_R[k].width[level];
      int hL = gauss_R[k].height[level];
      int wN = gauss_R[k].width[level + 1];
      int hN = gauss_R[k].height[level + 1];
      size_t sz = (size_t)wL * hL;

      float *tmp = (float *)allsky_safe_malloc(sz * sizeof(float));
      if (!tmp)
        goto OOM_FAIL;

      gaussian_expand_scalar(gauss_R[k].data[level + 1], wN, hN, tmp, wL, hL);
#pragma omp parallel for
      for (int i = 0; i < (int)sz; ++i)
        lap_R[k][level][i] = gauss_R[k].data[level][i] - tmp[i];

      gaussian_expand_scalar(gauss_G[k].data[level + 1], wN, hN, tmp, wL, hL);
#pragma omp parallel for
      for (int i = 0; i < (int)sz; ++i)
        lap_G[k][level][i] = gauss_G[k].data[level][i] - tmp[i];

      gaussian_expand_scalar(gauss_B[k].data[level + 1], wN, hN, tmp, wL, hL);
#pragma omp parallel for
      for (int i = 0; i < (int)sz; ++i)
        lap_B[k][level][i] = gauss_B[k].data[level][i] - tmp[i];

      allsky_safe_free(tmp);
    }
    /* Top level = coarsest Gaussian */
    {
      int wTop = gauss_R[k].width[L - 1], hTop = gauss_R[k].height[L - 1];
      size_t sz = (size_t)wTop * hTop;
      memcpy(lap_R[k][L - 1], gauss_R[k].data[L - 1], sz * sizeof(float));
      memcpy(lap_G[k][L - 1], gauss_G[k].data[L - 1], sz * sizeof(float));
      memcpy(lap_B[k][L - 1], gauss_B[k].data[L - 1], sz * sizeof(float));
    }
  }

  /* Fuse Laplacian levels for R, G, B using Gaussian weight pyramids */
  float *fused_R[MAX_PYRAMID_LEVELS] = {0};
  float *fused_G[MAX_PYRAMID_LEVELS] = {0};
  float *fused_B[MAX_PYRAMID_LEVELS] = {0};

  for (int level = 0; level < L; ++level) {
    int wL = lap_W[level];
    int hL = lap_H[level];

    for (int k = 0; k < use_max_images; ++k) {
      if (gauss_weight[k].width[level] != wL ||
          gauss_weight[k].height[level] != hL)
        goto OOM_FAIL;
    }

    size_t sz = (size_t)wL * hL;
    fused_R[level] = (float *)allsky_safe_malloc(sz * sizeof(float));
    fused_G[level] = (float *)allsky_safe_malloc(sz * sizeof(float));
    fused_B[level] = (float *)allsky_safe_malloc(sz * sizeof(float));
    if (!fused_R[level] || !fused_G[level] || !fused_B[level])
      goto OOM_FAIL;

#pragma omp parallel for
    for (int idx = 0; idx < (int)sz; ++idx) {
      float sum_R = 0.f, sum_G = 0.f, sum_B = 0.f, sum_w = 0.f;
      float max_w = 0.f, best_R = 0.f, best_G = 0.f, best_B = 0.f;

      for (int k = 0; k < use_max_images; ++k) {
        float w_i = gauss_weight[k].data[level][idx];
        sum_R += lap_R[k][level][idx] * w_i;
        sum_G += lap_G[k][level][idx] * w_i;
        sum_B += lap_B[k][level][idx] * w_i;
        sum_w += w_i;
        if (w_i > max_w) {
          max_w = w_i;
          best_R = lap_R[k][level][idx];
          best_G = lap_G[k][level][idx];
          best_B = lap_B[k][level][idx];
        }
      }

      /* Weights are already normalized per-pixel, so sum_Lp is the fused value */
      if (sum_w > 1e-8f) {
        fused_R[level][idx] = sum_R;
        fused_G[level][idx] = sum_G;
        fused_B[level][idx] = sum_B;
      } else {
        fused_R[level][idx] = best_R;
        fused_G[level][idx] = best_G;
        fused_B[level][idx] = best_B;
      }
    }
  }

  /* Reconstruct R, G, B from Laplacian pyramids */
  {
    size_t top_sz = (size_t)lap_W[L - 1] * lap_H[L - 1];
    float *cur_R = (float *)allsky_safe_malloc(top_sz * sizeof(float));
    float *cur_G = (float *)allsky_safe_malloc(top_sz * sizeof(float));
    float *cur_B = (float *)allsky_safe_malloc(top_sz * sizeof(float));
    if (!cur_R || !cur_G || !cur_B) {
      allsky_safe_free(cur_R);
      allsky_safe_free(cur_G);
      allsky_safe_free(cur_B);
      goto OOM_FAIL;
    }
    memcpy(cur_R, fused_R[L - 1], top_sz * sizeof(float));
    memcpy(cur_G, fused_G[L - 1], top_sz * sizeof(float));
    memcpy(cur_B, fused_B[L - 1], top_sz * sizeof(float));

    for (int level = L - 2; level >= 0; --level) {
      int wL = lap_W[level];
      int hL = lap_H[level];
      float *up_R = (float *)allsky_safe_malloc((size_t)wL * hL * sizeof(float));
      float *up_G = (float *)allsky_safe_malloc((size_t)wL * hL * sizeof(float));
      float *up_B = (float *)allsky_safe_malloc((size_t)wL * hL * sizeof(float));
      if (!up_R || !up_G || !up_B) {
        allsky_safe_free(up_R); allsky_safe_free(up_G); allsky_safe_free(up_B);
        allsky_safe_free(cur_R); allsky_safe_free(cur_G); allsky_safe_free(cur_B);
        goto OOM_FAIL;
      }
      gaussian_expand_scalar(cur_R, lap_W[level + 1], lap_H[level + 1], up_R, wL, hL);
      gaussian_expand_scalar(cur_G, lap_W[level + 1], lap_H[level + 1], up_G, wL, hL);
      gaussian_expand_scalar(cur_B, lap_W[level + 1], lap_H[level + 1], up_B, wL, hL);
#pragma omp parallel for
      for (int i = 0; i < wL * hL; ++i) {
        up_R[i] += fused_R[level][i];
        up_G[i] += fused_G[level][i];
        up_B[i] += fused_B[level][i];
      }
      allsky_safe_free(cur_R); cur_R = up_R;
      allsky_safe_free(cur_G); cur_G = up_G;
      allsky_safe_free(cur_B); cur_B = up_B;
    }

    /* Write to output, clamping negative values (can occur at sharp edges) */
#pragma omp parallel for
    for (size_t i = 0; i < pixel_count; ++i) {
      float r = cur_R[i]; if (r < 0.f) r = 0.f;
      float g = cur_G[i]; if (g < 0.f) g = 0.f;
      float b = cur_B[i]; if (b < 0.f) b = 0.f;
      output_hdr[(size_t)i * CHANNELS + 0] = r;
      output_hdr[(size_t)i * CHANNELS + 1] = g;
      output_hdr[(size_t)i * CHANNELS + 2] = b;
      output_hdr[(size_t)i * CHANNELS + 3] = 1.0f;
    }
    allsky_safe_free(cur_R);
    allsky_safe_free(cur_G);
    allsky_safe_free(cur_B);
  }

  /* Cleanup */
  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps[k])   allsky_safe_free(weight_maps[k]);
    if (luma_maps[k])     allsky_safe_free(luma_maps[k]);
    if (contrast_maps[k]) allsky_safe_free(contrast_maps[k]);
  }

  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < L; ++level) {
      if (lap_R[k][level]) allsky_safe_free(lap_R[k][level]);
      if (lap_G[k][level]) allsky_safe_free(lap_G[k][level]);
      if (lap_B[k][level]) allsky_safe_free(lap_B[k][level]);
    }
  }

  for (int k = 0; k < use_max_images; ++k) {
    free_gaussian_pyramid(&gauss_R[k]);
    free_gaussian_pyramid(&gauss_G[k]);
    free_gaussian_pyramid(&gauss_B[k]);
    free_gaussian_pyramid(&gauss_weight[k]);
  }

  for (int level = 0; level < L; ++level) {
    if (fused_R[level]) allsky_safe_free(fused_R[level]);
    if (fused_G[level]) allsky_safe_free(fused_G[level]);
    if (fused_B[level]) allsky_safe_free(fused_B[level]);
  }

  printf("HDR Multi Scale Fusion: ok, L=%d, pyramid_levels_override=%d\n", L,
         pyramid_levels_override);

  return 0;

OOM_FAIL:
  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps[k])   { allsky_safe_free(weight_maps[k]);   weight_maps[k]   = NULL; }
    if (luma_maps[k])     { allsky_safe_free(luma_maps[k]);     luma_maps[k]     = NULL; }
    if (contrast_maps[k]) { allsky_safe_free(contrast_maps[k]); contrast_maps[k] = NULL; }
  }

  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < MAX_PYRAMID_LEVELS; ++level) {
      if (lap_R[k][level]) { allsky_safe_free(lap_R[k][level]); lap_R[k][level] = NULL; }
      if (lap_G[k][level]) { allsky_safe_free(lap_G[k][level]); lap_G[k][level] = NULL; }
      if (lap_B[k][level]) { allsky_safe_free(lap_B[k][level]); lap_B[k][level] = NULL; }
    }
  }

  for (int k = 0; k < use_max_images; ++k) {
    free_gaussian_pyramid(&gauss_R[k]);
    free_gaussian_pyramid(&gauss_G[k]);
    free_gaussian_pyramid(&gauss_B[k]);
    free_gaussian_pyramid(&gauss_weight[k]);
  }

  for (int level = 0; level < MAX_PYRAMID_LEVELS; ++level) {
    if (fused_R[level]) { allsky_safe_free(fused_R[level]); fused_R[level] = NULL; }
    if (fused_G[level]) { allsky_safe_free(fused_G[level]); fused_G[level] = NULL; }
    if (fused_B[level]) { allsky_safe_free(fused_B[level]); fused_B[level] = NULL; }
  }

  return 2;
}

/* Extended Reinhard: g(0)=0, g'(0)=1, g(white)=1, monotonic for white>=1 */
static inline float shoulder_reinhard(float u, float white) {
  return u * (1.0f + u / (white * white)) / (1.0f + u);
}

int hdr_normalize_range_rgbf1(float *rgbf, int width, int height,
                              float target_min, float target_max,
                              float knee) {
  if (!rgbf || width <= 0 || height <= 0 || target_max <= target_min)
    return 1;

  size_t pixel_count = (size_t)width * height;

  if (knee > 0.0f) {
    if (knee > 0.99f)
      knee = 0.99f;

    /* Maximum of the largest channel (frame-0 units) */
    float max_val = 0.0f;
#pragma omp parallel for reduction(max : max_val)
    for (size_t i = 0; i < pixel_count; i++) {
      const float *px = &rgbf[i * (size_t)CHANNELS];
      float m = fmaxf(fmaxf(px[0], px[1]), px[2]);
      if (m > max_val)
        max_val = m;
    }

    /* Shoulder: [knee .. max_val] -> [knee .. 1], slope 1 at the knee.
     * Applied to the largest channel, all channels scaled by the same
     * factor, so the hue is kept and nothing exceeds 1. */
    const float range = 1.0f - knee;
    const float white = (max_val - knee) / range; /* in units of range */
    const int compress = (white > 1.0f);

    printf("HDR Normalize: max=%.6f knee=%.2f -> %s, target %.3f .. %.3f\n",
           max_val, knee,
           compress ? "highlight shoulder" : "no compression needed",
           target_min, target_max);

    const float range_out = target_max - target_min;

#pragma omp parallel for
    for (size_t i = 0; i < pixel_count; i++) {
      float *px = &rgbf[i * (size_t)CHANNELS];
      float r = fmaxf(px[0], 0.0f), g = fmaxf(px[1], 0.0f),
            b = fmaxf(px[2], 0.0f);
      float m = fmaxf(fmaxf(r, g), b);
      float scale = 1.0f;
      if (m > knee) {
        float m_out = compress
                          ? knee + range * shoulder_reinhard((m - knee) / range,
                                                             white)
                          : fminf(m, 1.0f);
        scale = m_out / m;
      }
      float v[3] = {r * scale, g * scale, b * scale};
      for (int c = 0; c < 3; c++) {
        float o = target_min + fminf(v[c], 1.0f) * range_out;
        px[c] = o;
      }
      px[3] = 1.0f;
    }

    printf("HDR Normalize: ok\n");
    return 0;
  }

  /* 1) Find global min/max of R, G, B */
  float min_val = 1e30f, max_val = -1e30f;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
  for (size_t i = 0; i < pixel_count; i++) {
    size_t idx = i * (size_t)CHANNELS;

    /* Find global min/max of R, G, B */
    for (int c = 0; c < 3; c++) { // R, G, B only
      float v = rgbf[idx + c];

      if (v < min_val)
        min_val = v;
      if (v > max_val)
        max_val = v;
    }
  }

  /* Avoid division by zero */
  float range_in = max_val - min_val;
  if (range_in < 1e-6f)
    range_in = 1.0f;

  float range_out = target_max - target_min;

  printf("HDR Normalize: min=%.6f max=%.6f -> target %.3f .. %.3f\n", min_val,
         max_val, target_min, target_max);

/* 2) Rescale in-place */
#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; i++) {
    size_t idx = i * (size_t)CHANNELS;

    for (int c = 0; c < 3; c++) {
      float v = rgbf[idx + c];
      v = (v - min_val) / range_in;   // [0..1]
      v = target_min + v * range_out; // [target_min..target_max]

      /* Clamp to target range (optional, but safe) */
      if (v < target_min)
        v = target_min;
      if (v > target_max)
        v = target_max;

      rgbf[idx + c] = v;
    }

    rgbf[idx + 3] = 1.0f; // Alpha fixed
  }

  printf("HDR Normalize: ok\n");

  return 0;
}
