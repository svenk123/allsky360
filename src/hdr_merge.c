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

// Simple YCbCr conversion
static inline void rgb_to_ycbcr(const float R, const float G, const float B,
                                float *Y, float *Cb, float *Cr) {
  /* Rec.709 luminance and chrominance */
  *Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
  *Cb = -0.114572f * R - 0.385428f * G + 0.5f * B;
  *Cr = 0.5f * R - 0.454153f * G - 0.045847f * B;
}

static inline void ycbcr_to_rgb(const float Y, const float Cb, const float Cr,
                                float *R, float *G, float *B) {
  *R = Y + 1.402f * Cr;
  *G = Y - 0.344136f * Cb - 0.714136f * Cr;
  *B = Y + 1.772f * Cb;
}

float *convert_weight_map_to_rgbaf1(const float *weight_map, int width,
                                    int height) {
  if (!weight_map || width <= 0 || height <= 0)
    return NULL;

  int pixel_count = width * height;
  float *rgbaf =
      (float *)allsky_safe_malloc(pixel_count * CHANNELS * sizeof(float));
  if (!rgbaf)
    return NULL;

#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    float v = weight_map[i]; // expected in 0..1.0
    v = clampf1(v);          // clamp to [0..1]

    int idx = i * CHANNELS;
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

  int pixel_count = width * height;

  float min_val = 1e30f, max_val = -1e30f;
  double sum_weights = 0.0;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)     \
    reduction(+ : sum_weights)
  for (int i = 0; i < pixel_count; i++) {
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

static void build_gaussian_pyramid(float *input, Pyramid *pyramid,
                                   int base_width, int base_height) {
  pyramid->width[0] = base_width;
  pyramid->height[0] = base_height;
  pyramid->data[0] =
      allsky_safe_malloc(base_width * base_height * sizeof(float));
  memcpy(pyramid->data[0], input, base_width * base_height * sizeof(float));

  for (int level = 1; level < MAX_PYRAMID_LEVELS; level++) {
    int prev_w = pyramid->width[level - 1];
    int prev_h = pyramid->height[level - 1];
    int new_w = prev_w / 2;
    int new_h = prev_h / 2;

    pyramid->width[level] = new_w;
    pyramid->height[level] = new_h;
    pyramid->data[level] = allsky_safe_malloc(new_w * new_h * sizeof(float));

    // Simple downsample + box blur
    for (int y = 0; y < new_h; y++) {
      for (int x = 0; x < new_w; x++) {
        float sum = 0.0f;
        int count = 0;

        for (int dy = 0; dy < 2; dy++) {
          for (int dx = 0; dx < 2; dx++) {
            int px = x * 2 + dx;
            int py = y * 2 + dy;

            if (px < prev_w && py < prev_h) {
              sum += pyramid->data[level - 1][py * prev_w + px];
              count++;
            }
          }
        }

        pyramid->data[level][y * new_w + x] = sum / (float)count;
      }
    }
  }
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
 * @param clipping_threshold  Clipping threshold (normalized to image range,
 * e.g. 1.0f for 0..1)
 * @return pixel weight [0.0 .. 1.0]
 */
static float compute_pixel_weight(float linR, float linG, float linB,
                                  float Y_max_expected,
                                  float opt_exposure_center,
                                  float clipping_threshold, float weight_sigma,
                                  float weight_clip_factor) {

  /* 1) Convert RGB to Y (Rec.709) */
  float Y = 0.2126f * linR + 0.7152f * linG + 0.0722f * linB;

  /* 2) Normalize Y to [0..1] range */
  float Y_norm = fminf(Y / Y_max_expected, 1.0f);

  /* 3) Gaussian-like weight function (Debevec/Malik approach)
   * Uses a bell curve centered at opt_exposure_center
   * This provides smoother transitions and better balance between exposures */
  float delta = Y_norm - opt_exposure_center;
  float w = expf(-0.5f * (delta * delta) / (weight_sigma * weight_sigma));

  /* 4) Clipping suppression - reduce weight for overexposed pixels */
  float max_rgb = fmaxf(fmaxf(linR, linG), linB);
  float clip_start = clipping_threshold * weight_clip_factor;
  if (max_rgb >= clip_start) {
    float clip_range = clipping_threshold * (1.0f - weight_clip_factor);
    float clip_factor = (max_rgb - clip_start) / fmaxf(clip_range, 1e-6f);
    clip_factor = fminf(clip_factor, 1.0f);
    w *= (1.0f -
          clip_factor * 0.7f); // Reduce weight by up to 70% for clipped pixels
  }
  /* 5) Under-exposure suppression - reduce weight for very dark pixels */
  /* Moderate threshold to suppress noisy dark pixels without making image too
   * dark */
  if (Y_norm < 0.10f) {
    float dark_threshold = 0.10f;
    float dark_factor = Y_norm / dark_threshold;
    /* Use moderate suppression: quadratic falloff for very dark areas */
    w *= dark_factor * dark_factor; // Quadratic falloff for very dark areas
  }

  return w;
}

#if 0
static float compute_pixel_weight(float linR, float linG, float linB,
                                  float Y_max_expected,
                                  float opt_exposure_center,
                                  float clipping_threshold,
                                  float weight_sigma,
                                  float weight_clip_factor) {

  /* 1) Convert to Y */
  float Y, Cb, Cr;
  rgb_to_ycbcr(linR, linG, linB, &Y, &Cb, &Cr);

  /* 2) Normalize Y to [0..1] range */
  float Y_norm = fminf(Y / Y_max_expected, 1.0f);
  
  /* 3) Gaussian-like weight function (Debevec/Malik approach)
   * Uses a bell curve centered at opt_exposure_center
   * This provides smoother transitions and better balance between exposures */
  float delta = Y_norm - opt_exposure_center;
  float w = expf(-0.5f * (delta * delta) / (weight_sigma * weight_sigma));
  
  /* 4) Clipping suppression - reduce weight for overexposed pixels */
  float max_rgb = fmaxf(fmaxf(linR, linG), linB);
  float clip_start = clipping_threshold * weight_clip_factor;
  if (max_rgb >= clip_start) {
    float clip_range = clipping_threshold * (1.0f - weight_clip_factor);
    float clip_factor = (max_rgb - clip_start) / fmaxf(clip_range, 1e-6f);
    clip_factor = fminf(clip_factor, 1.0f);
    w *= (1.0f - clip_factor * 0.7f); // Reduce weight by up to 70% for clipped pixels
  }
  
  /* 5) Under-exposure suppression - reduce weight for very dark pixels */
  /* Increased threshold from 0.05f to 0.20f to better suppress noisy dark images */
  if (Y_norm < 0.20f) {
    float dark_threshold = 0.20f;
    float dark_factor = Y_norm / dark_threshold;
    /* Use much stronger suppression: exponential falloff for very dark areas */
    /* This ensures very dark pixels get almost zero weight */
    float suppression = dark_factor * dark_factor * dark_factor * dark_factor; // 4th power
    w *= suppression;
  }

  return w;
}
#endif

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

/* Helper: simple bilinear upsample for scalar images (dst must be allocated) */
static void upsample_bilinear_scalar(const float *src, int sw, int sh,
                                     float *dst, int dw, int dh) {
  if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
    return;

  if (dw == sw && dh == sh) {
    // Identity copy
    for (int i = 0; i < sw * sh; ++i)
      dst[i] = src[i];

    return;
  }

  for (int y = 0; y < dh; ++y) {
    float gy =
        (sh == 1) ? 0.0f : ((float)y * (float)(sh - 1)) / (float)(dh - 1);
    int y0 = (int)gy;
    int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
    float wy = gy - (float)y0;

    for (int x = 0; x < dw; ++x) {
      float gx =
          (sw == 1) ? 0.0f : ((float)x * (float)(sw - 1)) / (float)(dw - 1);
      int x0 = (int)gx;
      int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
      float wx = gx - (float)x0;

      float v00 = src[y0 * sw + x0];
      float v01 = src[y0 * sw + x1];
      float v10 = src[y1 * sw + x0];
      float v11 = src[y1 * sw + x1];

      float v0 = v00 * (1.0f - wx) + v01 * wx;
      float v1 = v10 * (1.0f - wx) + v11 * wx;
      dst[y * dw + x] = v0 * (1.0f - wy) + v1 * wy;
    }
  }
}

/* Optional: 3x3 box blur for chroma weights (dst may alias src) */
static void box_blur_3x3_scalar(const float *src, float *dst, int width,
                                int height) {
  if (!src || !dst || width <= 0 || height <= 0)
    return;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
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

          sum += src[yy * width + xx];
          cnt++;
        }
      }

      dst[y * width + x] = sum / (float)cnt;
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

/* chroma_mode: 0=best-only, 1=weighted, 2=weighted with 3x3 smoothed weights */
/* contrast_weight_strength: 0..1; 0 disables local-contrast weighting */
/* pyramid_levels_override: 0=auto, >0 clamps max levels */
int hdr_multi_scale_fusion_laplacian_rgbf1(
    const struct HdrFrames frames[MAX_IMAGES], int use_max_images, int width,
    int height, float clipping_threshold, float Y_max_expected,
    float *output_hdr, const char *dump_images_dir, int dump_weight_maps,
    int dump_weight_stats, int chroma_mode, float contrast_weight_strength,
    int pyramid_levels_override, float weight_sigma, float weight_clip_factor,
    float channel_scale_r, float channel_scale_g, float channel_scale_b) {
  if (!frames || !output_hdr || width <= 0 || height <= 0 ||
      use_max_images <= 0)
    return 1;

  const int pixel_count = width * height;

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

  /* Precompute gains */
  float exp_gain[MAX_IMAGES] = {0};
  float r_gain[MAX_IMAGES] = {0};
  float g_gain[MAX_IMAGES] = {0};
  float b_gain[MAX_IMAGES] = {0};
  for (int k = 0; k < use_max_images; ++k) {
    exp_gain[k] = (float)(frames[0].exposure / frames[k].exposure);

    /* Compensate for channel scaling: divide median values by channel scale
     * factors to get original (pre-scaling) values, then compute gains */
    double median_r0_orig = frames[0].median_r / channel_scale_r;
    double median_g0_orig = frames[0].median_g / channel_scale_g;
    double median_b0_orig = frames[0].median_b / channel_scale_b;

    double median_rk_orig = frames[k].median_r / channel_scale_r;
    double median_gk_orig = frames[k].median_g / channel_scale_g;
    double median_bk_orig = frames[k].median_b / channel_scale_b;

    r_gain[k] = (float)(median_r0_orig / median_rk_orig);
    g_gain[k] = (float)(median_g0_orig / median_gk_orig);
    b_gain[k] = (float)(median_b0_orig / median_bk_orig);
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

  float opt_exposure_center;
  // Consider Y_max_expected when determining exposure center
  // For bright scenes (Y_max_expected >= 1.0), use higher center values
  if (Y_max_expected >= 1.0f) {
    // Bright scene - adjust center upward to favor longer exposure
    if (proxy_hdr_brightness < 0.10f)
      opt_exposure_center = 0.20f;
    else if (proxy_hdr_brightness < 0.15f)
      opt_exposure_center = 0.30f;
    else if (proxy_hdr_brightness < 0.25f)
      opt_exposure_center = 0.40f;
    else if (proxy_hdr_brightness < 0.35f)
      opt_exposure_center = 0.50f;
    else
      opt_exposure_center = 0.55f;
  } else {
    // Original logic for darker scenes
    // Increased minimum values to better suppress very dark/noisy images
    if (proxy_hdr_brightness < 0.07f)
      opt_exposure_center = 0.15f; // Increased from 0.07f
    else if (proxy_hdr_brightness < 0.10f)
      opt_exposure_center = 0.20f; // Increased from 0.10f
    else if (proxy_hdr_brightness < 0.15f)
      opt_exposure_center = 0.25f; // Increased from 0.15f
    else if (proxy_hdr_brightness < 0.25f)
      opt_exposure_center = 0.30f; // Increased from 0.25f
    else
      opt_exposure_center = 0.40f; // Increased from 0.35f
  }

  /* Allocate maps */
  float *weight_maps[MAX_IMAGES] = {0};
  float *luma_maps[MAX_IMAGES] = {0};
  float *contrast_maps[MAX_IMAGES] = {0}; // optional
  float *output_Y = NULL;

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
  for (int i = 0; i < pixel_count; ++i) {
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
    float max_Y_found = 0.0f;
    int found_non_clipped = 0;

    /* Find maximum Y value, preferring non-clipped pixels */
    /* This is important because clipped pixels don't represent the true
     * brightness */
    for (int k = 0; k < use_max_images; ++k) {
      for (int i = 0; i < pixel_count; ++i) {
        const float *px = &frames[k].image[(size_t)i * CHANNELS];
        if (is_masked_pixel(px))
          continue;

        /* Check if pixel is clipped in original image (before linearization) */
        float max_rgb_src = fmaxf(fmaxf(px[0], px[1]), px[2]);
        int is_clipped = (max_rgb_src >= clipping_threshold * 0.95f);

        float Y_val = luma_maps[k][i];

        /* Prefer non-clipped pixels, but also consider clipped ones if no
         * better option */
        if (!is_clipped) {
          if (Y_val > max_Y_found) {
            max_Y_found = Y_val;
            found_non_clipped = 1;
          }
        } else if (!found_non_clipped) {
          /* Only use clipped pixels if we haven't found any non-clipped ones */
          /* This handles the case where the sun is clipped in all exposures */
          if (Y_val > max_Y_found)
            max_Y_found = Y_val;
        }
      }
    }

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

      /* Find maximum value */
      float maxv = 1e-12f;
      for (int i = 0; i < pixel_count; ++i)
        if (C[i] > maxv)
          maxv = C[i];
      float inv = 1.0f / maxv;

      /* Normalize to [0,1] */
      for (int i = 0; i < pixel_count; ++i) {
        float v = C[i] * inv;
        C[i] = (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);
      }
    }
  }

/* Base weights (exposure/clipping) +/- contrast mix */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
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
                                     opt_exposure_center, clipping_threshold,
                                     weight_sigma, weight_clip_factor);

      /* Note: Softmask for dark areas in shorter exposures is applied after
       * weight calculation */

#if 1
      const float *px_src = &frames[k].image[(size_t)i * CHANNELS];
      const float max_rgb_src =
          fmaxf(fmaxf(px_src[0], px_src[1]), px_src[2]); // 0..1, pre-gain
      const float a = clipping_threshold * 0.95f;
      const float b = clipping_threshold * 1.00f;
      float u = 1.0f;
      if (max_rgb_src >= a) {
        float t = 1.0f - (max_rgb_src - a) / fmaxf(b - a, 1e-6f);
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        u = t * t * (3.f - 2.f * t); // smoothstep
      }
      const float clip_blend = 0.6f;
      w *= (1.0f - clip_blend) + clip_blend * u;
#endif

      if (contrast_weight_strength > 0.0f) {
        float cw = contrast_maps[k][i]; // [0..1]
        w *= (1.0f - contrast_weight_strength) + contrast_weight_strength * cw;
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
    for (int i = 0; i < pixel_count; ++i) {
      const float *px = &frames[k].image[(size_t)i * CHANNELS];

      /* Skip masked pixels */
      if (is_masked_pixel(px))
        continue;

      /* Get normalized luminance for this pixel (after exposure compensation)
       */
      float Y_val = luma_maps[k][i];

      /* Calculate softmask value: 0 = fully masked, 1 = fully visible */
      float mask_value = 1.0f;

      if (k >= 2) {
        /* For very short exposures (k >= 2, e.g. 3.75s): only use bright areas
         */
        /* These images are too short to capture dark areas well, so only bright
         * pixels should contribute */

        /* Define brightness threshold: pixels must be above this to be
         * considered */
        /* Use the pixel's brightness in the shorter exposure itself (not
         * relative to longest) */
        float brightness_threshold = 0.0f;

        if (frames[k].median_g > 0.0f) {
          /* Threshold based on the image's own median brightness */
          /* Only pixels significantly brighter than the median are considered
           */
          /* This ensures we only use well-exposed bright areas */
          /* ADJUSTABLE: Increase multiplier (e.g. 2.0f, 2.5f) to be more
           * strict, decrease (e.g. 1.2f) to be less strict */
          brightness_threshold =
              frames[k].median_g * 4.0f; // 4.0x median = very bright areas only
                                         // (increased from 3.0f)

          /* Also set a minimum absolute threshold to avoid using very dark
           * images at all */
          /* ADJUSTABLE: Increase (e.g. 0.20f, 0.25f) to mask more, decrease
           * (e.g. 0.10f) to mask less */
          brightness_threshold =
              fmaxf(brightness_threshold, 0.40f); // Increased from 0.30f
        } else {
          /* Fallback: use absolute threshold */
          brightness_threshold = 0.40f; // Increased from 0.20f
        }

        /* Soft transition zone: pixels between brightness_threshold and
         * brightness_threshold * 0.8 get partial masking */
        /* ADJUSTABLE: Decrease factor (e.g. 0.7f, 0.6f) to make transition zone
         * narrower (stricter) */
        float transition_end =
            brightness_threshold *
            0.6f; // Decreased from 0.7f for even stricter masking
        float transition_start = brightness_threshold;

        if (Y_val < transition_end) {
          /* Too dark: fully mask out */
          mask_value = 0.0f;
        } else if (Y_val < transition_start) {
          /* Transition zone: smooth interpolation */
          float t =
              (Y_val - transition_end) / (transition_start - transition_end);
          t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp to [0,1]
          /* Use smoothstep for smooth transition */
          mask_value = t * t * (3.0f - 2.0f * t);
        }
        /* else: bright enough, keep full weight (mask_value = 1.0) */
      } else {
        /* For moderately short exposures (k == 1, e.g. 30s, 15s): mask out very
         * dark areas */
        /* Use threshold relative to longest exposure */
        float dark_threshold = 0.0f;
        if (frames[0].median_g > 0.0f) {
          /* Base threshold: percentage of the longest exposure's median
           * brightness */
          /* ADJUSTABLE: Increase (e.g. 0.15f, 0.20f) to mask more dark areas,
           * decrease (e.g. 0.05f) to mask less */
          dark_threshold = frames[0].median_g * 0.35f; // Increased from 0.25f
        } else {
          /* Fallback: use absolute threshold */
          /* ADJUSTABLE: Increase (e.g. 0.08f, 0.10f) to mask more, decrease
           * (e.g. 0.03f) to mask less */
          dark_threshold = 0.12f; // Increased from 0.08f
        }

        /* Soft transition zone: pixels between dark_threshold and
         * dark_threshold * 1.5 get partial masking */
        /* ADJUSTABLE: Decrease factor (e.g. 1.3f, 1.2f) to make transition zone
         * narrower (stricter) */
        float transition_start = dark_threshold;
        float transition_end =
            dark_threshold *
            1.2f; // Decreased from 1.3f for even stricter masking

        if (Y_val < transition_start) {
          /* Very dark: fully mask out */
          mask_value = 0.0f;
        } else if (Y_val < transition_end) {
          /* Transition zone: smooth interpolation */
          float t =
              (Y_val - transition_start) / (transition_end - transition_start);
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
  for (int i = 0; i < pixel_count; ++i) {
    float sumw = 0.0f, maxw = 0.0f;
    int maxk = 0;

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

  /* Build Gaussian pyramids for Y and weights */
  Pyramid gauss_Y[MAX_IMAGES];
  Pyramid gauss_weight[MAX_IMAGES];
  memset(gauss_Y, 0, sizeof(gauss_Y));
  memset(gauss_weight, 0, sizeof(gauss_weight));

  for (int k = 0; k < use_max_images; ++k) {
    build_gaussian_pyramid(luma_maps[k], &gauss_Y[k], width, height);
    build_gaussian_pyramid(weight_maps[k], &gauss_weight[k], width, height);
  }

  /* Determine usable number of levels L (min across all) */
  int L = get_pyr_levels(&gauss_weight[0]);
  if (L <= 0)
    goto OOM_FAIL;

  /* Find minimum number of levels */
  for (int k = 1; k < use_max_images; ++k) {
    int lk = get_pyr_levels(&gauss_weight[k]);
    if (lk < L)
      L = lk;
  }

  /* Find minimum number of levels */
  for (int k = 0; k < use_max_images; ++k) {
    int lk = get_pyr_levels(&gauss_Y[k]);
    if (lk < L)
      L = lk;
  }

  if (pyramid_levels_override > 0 && pyramid_levels_override < L)
    L = pyramid_levels_override;
  if (L <= 0)
    goto OOM_FAIL;

  /* Build Laplacian pyramids for Y (we avoid using a Pyramid struct here) */
  float *lap_Y[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};
  int lap_W[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};
  int lap_H[MAX_IMAGES][MAX_PYRAMID_LEVELS] = {{0}};

  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < L; ++level) {
      int wL = gauss_Y[k].width[level];
      int hL = gauss_Y[k].height[level];
      lap_W[k][level] = wL;
      lap_H[k][level] = hL;
      lap_Y[k][level] =
          (float *)allsky_safe_malloc((size_t)wL * (size_t)hL * sizeof(float));
      if (!lap_Y[k][level])
        goto OOM_FAIL;
    }

    for (int level = 0; level < L - 1; ++level) {
      int wL = gauss_Y[k].width[level];
      int hL = gauss_Y[k].height[level];
      int wN = gauss_Y[k].width[level + 1];
      int hN = gauss_Y[k].height[level + 1];

      float *tmp =
          (float *)allsky_safe_malloc((size_t)wL * (size_t)hL * sizeof(float));
      if (!tmp)
        goto OOM_FAIL;

      upsample_bilinear_scalar(gauss_Y[k].data[level + 1], wN, hN, tmp, wL, hL);

#pragma omp parallel for
      for (int i = 0; i < wL * hL; ++i)
        lap_Y[k][level][i] = gauss_Y[k].data[level][i] - tmp[i];

      free(tmp);
    }
    /* Top level = coarsest Gaussian */
    {
      int wTop = gauss_Y[k].width[L - 1], hTop = gauss_Y[k].height[L - 1];
#pragma omp parallel for
      for (int i = 0; i < wTop * hTop; ++i)
        lap_Y[k][L - 1][i] = gauss_Y[k].data[L - 1][i];
    }
  }

  /* Fuse Laplacian levels using Gaussian weight pyramids */
  float *fused_Lap[MAX_PYRAMID_LEVELS] = {0};
  int fused_W[MAX_PYRAMID_LEVELS] = {0};
  int fused_H[MAX_PYRAMID_LEVELS] = {0};

  for (int level = 0; level < L; ++level) {
    int wL = gauss_weight[0].width[level];
    int hL = gauss_weight[0].height[level];

    /* Basic consistency check */
    for (int k = 0; k < use_max_images; ++k) {
      if (gauss_weight[k].width[level] != wL ||
          gauss_weight[k].height[level] != hL || lap_W[k][level] != wL ||
          lap_H[k][level] != hL) {
        goto OOM_FAIL;
      }
    }

    fused_W[level] = wL;
    fused_H[level] = hL;
    fused_Lap[level] = (float *)malloc((size_t)wL * (size_t)hL * sizeof(float));
    if (!fused_Lap[level])
      goto OOM_FAIL;

#pragma omp parallel for
    for (int idx = 0; idx < wL * hL; ++idx) {
      float sum_w = 0.0f, sum_Lp = 0.0f;
      float max_w = 0.0f;
      float best_Lp = 0.0f;
      int best_k = -1;

      for (int k = 0; k < use_max_images; ++k) {
        /* Compute weighted sum */
        float w_i = gauss_weight[k].data[level][idx];
        float L_i = lap_Y[k][level][idx];
        sum_Lp += L_i * w_i;
        sum_w += w_i;

        /* Track best weight for fallback */
        if (w_i > max_w) {
          max_w = w_i;
          best_Lp = L_i;
          best_k = k;
        }
      }

      /* Compute fused Laplacian */
      if (sum_w > 1e-8f) {
        fused_Lap[level][idx] = sum_Lp / sum_w;
      } else if (best_k >= 0 && max_w > 0.0f) {
        /* Fallback: use Laplacian from image with highest weight */
        fused_Lap[level][idx] = best_Lp;
      } else {
        /* Last resort: set to zero */
        fused_Lap[level][idx] = 0.0f;
      }
    }
  }

  /* Reconstruct fused Y from Laplacian */
  output_Y = (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
  if (!output_Y) {
    output_Y = NULL;
    goto OOM_FAIL;
  }

  /* Start with top level */
  float *cur = (float *)allsky_safe_malloc(
      (size_t)fused_W[L - 1] * (size_t)fused_H[L - 1] * sizeof(float));
  if (!cur)
    goto OOM_FAIL;
  memcpy(cur, fused_Lap[L - 1],
         (size_t)fused_W[L - 1] * (size_t)fused_H[L - 1] * sizeof(float));

  for (int level = L - 2; level >= 0; --level) {
    int wL = fused_W[level];
    int hL = fused_H[level];

    float *up =
        (float *)allsky_safe_malloc((size_t)wL * (size_t)hL * sizeof(float));
    if (!up) {
      allsky_safe_free(cur);
      goto OOM_FAIL;
    }

    /* Upsample */
    upsample_bilinear_scalar(cur, fused_W[level + 1], fused_H[level + 1], up,
                             wL, hL);

#pragma omp parallel for
    for (int i = 0; i < wL * hL; ++i)
      up[i] += fused_Lap[level][i];

    allsky_safe_free(cur);
    cur = up;
  }

  /* Copy to output. cur now contains full-res fused Y */
  memcpy(output_Y, cur, (size_t)pixel_count * sizeof(float));
  allsky_safe_free(cur);

/* Clamp negative Y values to zero (can happen at sharp edges due to Laplacian
 * fusion) */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    if (output_Y[i] < 0.0f)
      output_Y[i] = 0.0f;
  }

  /* Chroma fusion */
  /* Create separate chroma weight maps that favor the longest exposure for
   * better color balance */
  /* The aggressive softmask for luminance fusion can cause color imbalance, so
   * we use a modified approach for chroma */
  float *weight_maps_chroma[MAX_IMAGES] = {0};
  for (int k = 0; k < use_max_images; ++k) {
    weight_maps_chroma[k] =
        (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
    if (!weight_maps_chroma[k])
      goto OOM_FAIL;
  }

/* Create chroma weights: blend between original weights and a bias towards
 * longest exposure */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    /* Calculate original weight sum for normalization */
    float sum_orig = 0.0f;
    for (int k = 0; k < use_max_images; ++k) {
      sum_orig += weight_maps[k][i];
    }

    if (sum_orig > 1e-12f) {
      /* Blend: 70% original weights, 30% bias towards longest exposure (k=0) */
      /* This ensures the longest exposure contributes to color even if it was
       * masked for luminance */
      const float chroma_bias =
          0.3f; // How much to favor longest exposure for chroma

      for (int k = 0; k < use_max_images; ++k) {
        float orig_weight = weight_maps[k][i];
        float biased_weight;

        if (k == 0) {
          /* Longest exposure: add bias */
          biased_weight = orig_weight * (1.0f - chroma_bias) + chroma_bias;
        } else {
          /* Shorter exposures: reduce by bias amount */
          biased_weight = orig_weight * (1.0f - chroma_bias);
        }

        weight_maps_chroma[k][i] = biased_weight;
      }

      /* Renormalize chroma weights */
      float sum_chroma = 0.0f;
      for (int k = 0; k < use_max_images; ++k) {
        sum_chroma += weight_maps_chroma[k][i];
      }

      if (sum_chroma > 1e-12f) {
        float inv_sum = 1.0f / sum_chroma;
        for (int k = 0; k < use_max_images; ++k) {
          weight_maps_chroma[k][i] *= inv_sum;
        }
      }
    } else {
      /* Fallback: use original weights */
      for (int k = 0; k < use_max_images; ++k) {
        weight_maps_chroma[k][i] = weight_maps[k][i];
      }
    }
  }

  float *weight_maps_smooth[MAX_IMAGES] = {0};
  if (chroma_mode == 2) {
    for (int k = 0; k < use_max_images; ++k) {
      /* Allocate memory for smoothed chroma weights */
      weight_maps_smooth[k] =
          (float *)allsky_safe_malloc((size_t)pixel_count * sizeof(float));
      if (!weight_maps_smooth[k])
        goto OOM_FAIL;

      /* Box blur for chroma weights */
      box_blur_3x3_scalar(weight_maps_chroma[k], weight_maps_smooth[k], width,
                          height);
    }
  }

  /* Chrominance: color preservation from best-weighted image (mode 0) or
   * weighted combination of all images (mode 1, 2).
   * We take RGB from the best image or weighted average (linearized!),
   * then scale so that its Y matches Y_fused. This preserves hue/saturation
   * and avoids grey outputs. */

#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    const float Yf = output_Y[i];

    if (chroma_mode == 0) {
      /* Mode 0: best-only - use image with highest weight */
      float max_w = 0.0f;
      int best_k = 0;

      for (int k = 0; k < use_max_images; ++k) {
        /* For chroma mode 0, use chroma weights that favor longest exposure */
        const float w_i = weight_maps_chroma[k][i]; // Use chroma weights for
                                                    // better color balance
        if (w_i > max_w) {
          max_w = w_i;
          best_k = k;
        }
      }

      if (max_w <= 0.0f) {
        best_k = -1;

        for (int k = 0; k < use_max_images; ++k) {
          const float *px = &frames[k].image[(size_t)i * CHANNELS];

          /* Skip masked pixels */
          if (!is_masked_pixel(px)) {
            best_k = k;
            break;
          }
        }

        /* If best image is masked, set output to black */
        if (best_k < 0) { // komplett maskiert → schwarz
          output_hdr[(size_t)i * CHANNELS + 0] = 0.0f;
          output_hdr[(size_t)i * CHANNELS + 1] = 0.0f;
          output_hdr[(size_t)i * CHANNELS + 2] = 0.0f;
          output_hdr[(size_t)i * CHANNELS + 3] = 1.0f;
          continue;
        }
      }

      /* Take linearized RGB from best_k */
      const float *pxb = &frames[best_k].image[(size_t)i * CHANNELS];
      float Rb = pxb[0] * exp_gain[best_k] * r_gain[best_k];
      float Gb = pxb[1] * exp_gain[best_k] * g_gain[best_k];
      float Bb = pxb[2] * exp_gain[best_k] * b_gain[best_k];

      /* Compute its Y (Rec.709, matching the Y transform used elsewhere) */
      float Yb = 0.2126f * Rb + 0.7152f * Gb + 0.0722f * Bb;

      /* Scale RGB so that Y matches fused Y */
      const float eps = 1e-6f;
      float s;
      if (Yf <= eps || Yb <= eps || Yf < 0.0f) {
        /* If either Y is too small or negative, use original RGB without
         * scaling */
        s = 1.0f;
      } else {
        s = Yf / Yb;
        /* Clamp scaling factor to reasonable range to avoid extreme values */
        if (s < 0.1f)
          s = 0.1f;
        if (s > 10.0f)
          s = 10.0f;
      }
      float R = Rb * s;
      float G = Gb * s;
      float B = Bb * s;

      output_hdr[(size_t)i * CHANNELS + 0] = R;
      output_hdr[(size_t)i * CHANNELS + 1] = G;
      output_hdr[(size_t)i * CHANNELS + 2] = B;
      output_hdr[(size_t)i * CHANNELS + 3] = 1.0f;
    } else {
      /* Mode 1 or 2: weighted combination of all images */
      float sum_w = 0.0f;
      float sum_R = 0.0f;
      float sum_G = 0.0f;
      float sum_B = 0.0f;

      /* Compute weighted average of RGB from all images */
      for (int k = 0; k < use_max_images; ++k) {
        const float *px = &frames[k].image[(size_t)i * CHANNELS];

        /* Skip masked pixels */
        if (is_masked_pixel(px))
          continue;

        /* Get weight for chroma: use separate chroma weights that favor longest
         * exposure */
        float w;
        if (chroma_mode == 2 && weight_maps_smooth[k]) {
          w = weight_maps_smooth[k][i];
        } else {
          w = weight_maps_chroma[k][i]; // Use chroma weights instead of
                                        // luminance weights
        }

        if (w <= 0.0f)
          continue;

        /* Linearize RGB */
        float Rk = px[0] * exp_gain[k] * r_gain[k];
        float Gk = px[1] * exp_gain[k] * g_gain[k];
        float Bk = px[2] * exp_gain[k] * b_gain[k];

        /* Accumulate weighted RGB */
        sum_R += Rk * w;
        sum_G += Gk * w;
        sum_B += Bk * w;
        sum_w += w;
      }

      if (sum_w <= 1e-6f) {
        /* No valid pixels found, set output to black */
        output_hdr[(size_t)i * CHANNELS + 0] = 0.0f;
        output_hdr[(size_t)i * CHANNELS + 1] = 0.0f;
        output_hdr[(size_t)i * CHANNELS + 2] = 0.0f;
        output_hdr[(size_t)i * CHANNELS + 3] = 1.0f;
        continue;
      }

      /* Normalize weighted average */
      float Rb = sum_R / sum_w;
      float Gb = sum_G / sum_w;
      float Bb = sum_B / sum_w;

      /* Compute its Y (Rec.709, matching the Y transform used elsewhere) */
      float Yb = 0.2126f * Rb + 0.7152f * Gb + 0.0722f * Bb;

      /* Scale RGB so that Y matches fused Y */
      const float eps = 1e-6f;
      float s;
      if (Yf <= eps || Yb <= eps || Yf < 0.0f) {
        /* If either Y is too small or negative, use original RGB without
         * scaling */
        s = 1.0f;
      } else {
        s = Yf / Yb;
        /* Clamp scaling factor to reasonable range to avoid extreme values */
        if (s < 0.1f)
          s = 0.1f;
        if (s > 10.0f)
          s = 10.0f;
      }
      float R = Rb * s;
      float G = Gb * s;
      float B = Bb * s;

      output_hdr[(size_t)i * CHANNELS + 0] = R;
      output_hdr[(size_t)i * CHANNELS + 1] = G;
      output_hdr[(size_t)i * CHANNELS + 2] = B;
      output_hdr[(size_t)i * CHANNELS + 3] = 1.0f;
    }
  }

  /* Cleanup */
  allsky_safe_free(output_Y);
  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps_smooth[k])
      allsky_safe_free(weight_maps_smooth[k]);
    if (weight_maps_chroma[k])
      allsky_safe_free(weight_maps_chroma[k]);
  }

  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps[k])
      allsky_safe_free(weight_maps[k]);
    if (luma_maps[k])
      allsky_safe_free(luma_maps[k]);
    if (contrast_maps[k])
      allsky_safe_free(contrast_maps[k]);
  }

  /* Free Laplacian pyramids */
  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < L; ++level) {
      if (lap_Y[k][level])
        allsky_safe_free(lap_Y[k][level]);
    }
  }

  /* Free Gaussian pyramids */
  for (int k = 0; k < use_max_images; ++k) {
    free_gaussian_pyramid(&gauss_Y[k]);
    free_gaussian_pyramid(&gauss_weight[k]);
  }

  /* Free fused Laplacian */
  for (int level = 0; level < L; ++level) {
    if (fused_Lap[level])
      allsky_safe_free(fused_Lap[level]);
  }

  printf("HDR Multi Scale Fusion: ok, L=%d, pyramid_levels_override=%d\n", L,
         pyramid_levels_override);

  return 0;

OOM_FAIL:
  if (output_Y) {
    allsky_safe_free(output_Y);
    output_Y = NULL;
  }
  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps_smooth[k]) {
      allsky_safe_free(weight_maps_smooth[k]);
      weight_maps_smooth[k] = NULL;
    }
    if (weight_maps_chroma[k]) {
      allsky_safe_free(weight_maps_chroma[k]);
      weight_maps_chroma[k] = NULL;
    }
  }

  /* Robust cleanup on failure */
  for (int k = 0; k < use_max_images; ++k) {
    if (weight_maps[k]) {
      allsky_safe_free(weight_maps[k]);
      weight_maps[k] = NULL;
    }

    if (luma_maps[k]) {
      allsky_safe_free(luma_maps[k]);
      luma_maps[k] = NULL;
    }

    if (contrast_maps[k]) {
      allsky_safe_free(contrast_maps[k]);
      contrast_maps[k] = NULL;
    }
  }

  /* Free Laplacian pyramids */
  for (int k = 0; k < use_max_images; ++k) {
    for (int level = 0; level < MAX_PYRAMID_LEVELS; ++level) {
      if (lap_Y[k][level]) {
        allsky_safe_free(lap_Y[k][level]);
        lap_Y[k][level] = NULL;
      }
    }
  }

  /* Free Gaussian pyramids */
  for (int k = 0; k < use_max_images; ++k) {
    free_gaussian_pyramid(&gauss_Y[k]);
    free_gaussian_pyramid(&gauss_weight[k]);
  }

  /* Free fused Laplacian */
  for (int level = 0; level < MAX_PYRAMID_LEVELS; ++level) {
    if (fused_Lap[level]) {
      allsky_safe_free(fused_Lap[level]);
      fused_Lap[level] = NULL;
    }
  }

  return 2;
}

int hdr_normalize_range_rgbf1(float *rgbf, int width, int height,
                              float target_min, float target_max) {
  if (!rgbf || width <= 0 || height <= 0 || target_max <= target_min)
    return 1;

  int pixel_count = width * height;

  /* 1) Find global min/max of R, G, B */
  float min_val = 1e30f, max_val = -1e30f;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;

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
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;

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
