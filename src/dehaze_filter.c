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
#include "dehaze_filter.h"
#include "allsky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA

static void laplacian_filter(const float *src, float *dst, int width,
                             int height) {
#pragma omp parallel for collapse(2)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float center = src[y * width + x];
      float sum = 0.0f;
      int count = 0;

      /* 4-neighbour Laplacian */
      if (x > 0) {
        sum += src[y * width + (x - 1)];
        count++;
      }
      if (x < width - 1) {
        sum += src[y * width + (x + 1)];
        count++;
      }
      if (y > 0) {
        sum += src[(y - 1) * width + x];
        count++;
      }
      if (y < height - 1) {
        sum += src[(y + 1) * width + x];
        count++;
      }

      dst[y * width + x] = center * (float)count - sum;
    }
  }
}

static float compute_global_haze_level(const float *rgba_image, const float *luma,
                                int pixel_count, float haze_percent) {
  int max_valid = pixel_count;
  float *valid_luma = allsky_safe_malloc(max_valid * sizeof(float));
  if (!valid_luma)
    return 0.0f; // fallback

  int count = 0;

/* Build valid Luma array (ignore fully black RGB pixels) */
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < pixel_count; ++i) {
    int idx = i * 4;
    float r = rgba_image[idx + 0];
    float g = rgba_image[idx + 1];
    float b = rgba_image[idx + 2];

    if (!(r == 0.0f && g == 0.0f && b == 0.0f)) {
      int local_idx;
#pragma omp atomic capture
      local_idx = count++;
      valid_luma[local_idx] = luma[i];
    }
  }

  if (count == 0) {
    allsky_safe_free(valid_luma);
    return 0.0f;
  }

  qsort(valid_luma, count, sizeof(float), compare_floats);

  int num_top = (int)(count * haze_percent);
  if (num_top <= 0)
    num_top = 1;

  float haze_sum = 0.0f;
  for (int i = 0; i < num_top; ++i) {
    haze_sum += valid_luma[i];
  }
  float haze_level = haze_sum / num_top;

  allsky_safe_free(valid_luma);

  return haze_level;
}

int perceptual_dehaze_rgbf1_multiscale_full(float *rgba_image, int width,
                                            int height, float amount,
                                            float haze_percent) {
  if (!rgba_image || width <= 0 || height <= 0 || amount <= 0.0f ||
      haze_percent <= 0.0f || haze_percent > 1.0f)
    return 1;

  printf("Perceptual Dehaze: amount=%f, haze_percent=%f\n", amount, haze_percent);

  int pixel_count = width * height;
  float *luma = allsky_safe_malloc(pixel_count * sizeof(float));
  float *lap = allsky_safe_malloc(pixel_count * sizeof(float));
  if (!luma || !lap) {
    allsky_safe_free(luma);
    allsky_safe_free(lap);
    return 1;
  }

/* Step 1: Compute Luminance */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    int idx = i * 4;
    float r = rgba_image[idx + 0];
    float g = rgba_image[idx + 1];
    float b = rgba_image[idx + 2];
    luma[i] = rgb_to_luma(r, g, b);
  }

  /* Step 2: Global haze level estimate */
  float haze_level =
      compute_global_haze_level(rgba_image, luma, pixel_count, haze_percent);

  /* Step 3: Laplacian filter (local contrast) */
  laplacian_filter(luma, lap, width, height);

/* Step 4: Apply both global haze removal + local contrast boost */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    int idx = i * 4;
    float delta = lap[i];
    float grad = fabsf(delta);
    const float epsilon = 5000.0f; // typical value → can be tuned
    float adapt_amount = amount * (grad / (grad + epsilon));

    for (int c = 0; c < 3; ++c) {
      float I = rgba_image[idx + c];

      float J = (I - haze_level) * (1.0f + adapt_amount) + adapt_amount * delta;

      J = clampf1(J);
      rgba_image[idx + c] = J;
    }
    rgba_image[idx + 3] = 1.0f; // Alpha stays 1.0f in 0..1
  }

  allsky_safe_free(luma);
  allsky_safe_free(lap);

  printf("Perceptual Dehaze: ok, haze_level=%.4f\n", haze_level);

  return 0;
}

#endif // !USE_CUDA
